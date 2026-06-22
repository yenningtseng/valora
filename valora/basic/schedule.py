"""Schedule generation and transformation utilities."""

from __future__ import annotations

from copy import copy
from bisect import bisect_left, bisect_right
from typing import (
    List,
    cast,
    Callable,
    Self,
    Iterable,
    Union,
    Optional,
)

import numpy as np

from .date import Date, Period
from .daycount import DayCount
from .calendar import Calendar
from .enum import BusinessDayConvention, PeriodType


class Schedule:
    """Ordered collection of dates generated from a start, end, and tenor."""

    def __init__(
        self,
        beg_date: Date,
        end_date: Date,
        tenor: Period,
        calendar: Calendar,
        convention: BusinessDayConvention = (BusinessDayConvention.FOLLOWING),
        forward_generate: bool = True,
        eom_adjust: bool = True,
        include_begin: bool = True,
        include_end: bool = True,
    ):
        """Create a schedule definition with lazy date generation."""
        self.beg_date = beg_date
        self.end_date = end_date
        self.tenor = tenor
        self.calendar = calendar
        self.convention = convention
        self.forward_generate = forward_generate
        self.eom_adjust = eom_adjust
        self.include_begin = include_begin
        self.include_end = include_end
        self._dates_cache: tuple[Date, ...] | None = None
        self._dates_override: tuple[Date, ...] | None = None

    @property
    def dates(self) -> tuple[Date, ...]:
        """Return generated or overridden schedule dates."""
        if self._dates_override is not None:
            return self._dates_override

        if self._dates_cache is None:
            self._dates_cache = self._generate_dates()

        return self._dates_cache

    def apply_eom(self) -> bool:
        """Return whether end-of-month adjustment should be applied."""
        anchor_date = self.beg_date if self.forward_generate else self.end_date

        return self.eom_adjust and self.calendar.is_eom(anchor_date)

    def _generate_dates(self) -> tuple[Date, ...]:
        """Generate the schedule dates implied by the stored definition."""
        arr: List[Date] = []
        cur: Date = cast(
            Date,
            self.beg_date + self.tenor
            if self.forward_generate
            else self.end_date - self.tenor,
        )
        cond: Callable[[], bool] = (
            (lambda: cur < self.end_date)
            if self.forward_generate
            else (lambda: cur > self.beg_date)
        )
        step: Callable[[], Date] = (
            (lambda: cast(Date, cur + self.tenor))
            if self.forward_generate
            else (lambda: cast(Date, cur - self.tenor))
        )

        while cond():
            arr.append(cur)
            cur = step()

        if self.include_begin:
            arr.append(self.beg_date)
        if self.include_end:
            arr.append(self.end_date)

        if self.apply_eom():
            arr = [self.calendar.end_of_period(a, PeriodType.MONTHLY) for a in arr]
        else:
            arr = [self.calendar.adjust(a, self.convention) for a in arr]

        return tuple(sorted(set(arr)))

    def copy(self) -> Self:
        """Return a shallow copy of the schedule."""
        new = copy(self)
        return new

    def _transform_dates(
        self,
        transform: Callable[[tuple[Date, ...]], Iterable[Date]],
        *,
        inplace: bool,
    ) -> Self:
        """Apply a date transformation and return or mutate a schedule."""
        target = self if inplace else self.copy()

        transformed = transform(target.dates)
        target._dates_override = tuple(sorted(set(transformed)))

        if inplace:
            return None
        return target

    def __len__(self) -> int:
        """Return the number of dates in the schedule."""
        return len(self.dates)

    def __getitem__(self, key):
        """Return dates using normal sequence indexing semantics."""
        return self.dates[key]

    def index(self, ids: Union[int, Iterable[int]]) -> Union[Date, List[Date]]:
        """Return one or more dates by positional index."""
        if isinstance(ids, int):
            return self.dates[ids]
        else:
            indices = list(set(ids))
            return [self.dates[i] for i in indices]

    def date_index(self, dt) -> Optional[int]:
        """Return the position of ``dt`` in the schedule, if present."""
        for i, d in enumerate(self.dates):
            if d == dt:
                return i
        return None

    def next_date(self, dt) -> Optional[Date]:
        """Return the first scheduled date greater than or equal to ``dt``."""
        idx = bisect_left(self.dates, dt)

        if idx == len(self.dates):
            return None

        return self.dates[idx]

    def previous_date(self, dt) -> Optional[Date]:
        """Return the last scheduled date less than or equal to ``dt``."""
        idx = bisect_right(self.dates, dt)

        if idx < 0:
            return None

        return self.dates[idx]

    def append(
        self, new_dt: Union[Date, Iterable[Date]], inplace: bool = False
    ) -> Schedule:
        """Add one or more dates to the schedule."""

        def _append(dates, new_dt):
            if isinstance(new_dt, Date):
                new_dt = [new_dt]
            return sorted(set(dates).union(set(new_dt)))

        return self._transform_dates(
            lambda dates: _append(dates, new_dt), inplace=inplace
        )

    def drop(
        self, new_dt: Union[Date, Iterable[Date]], inplace: bool = False
    ) -> Schedule:
        """Remove one or more dates from the schedule."""

        def _append(dates, new_dt):
            if isinstance(new_dt, Date):
                new_dt = [new_dt]
            return sorted(set(dates) - set(new_dt))

        return self._transform_dates(
            lambda dates: _append(dates, new_dt), inplace=inplace
        )

    def shift(self, period: Period, inplace: bool = False) -> Schedule:
        """Shift every schedule date by the same period."""
        return self._transform_dates(
            lambda dates: (cast(Date, date + period) for date in dates), inplace=inplace
        )

    def clip(
        self,
        lower: Optional[Date] = None,
        upper: Optional[Date] = None,
        include_lower: bool = False,
        inclide_upper: bool = False,
        inplace: bool = False,
    ) -> Schedule:
        """Restrict the schedule to a bounded interval."""

        def _clip(dates, lower, upper, include_lower, include_upper):
            if lower is None:
                lower_sn = dates[0].serial_number
            else:
                lower_sn = lower.serial_number

            if upper is None:
                upper_sn = dates[-1].serial_number
            else:
                upper_sn = upper.serial_number

            dt = np.array([d.serial_number for d in dates])
            new_dt = dt[(dt >= lower_sn) & (dt <= upper_sn)]

            if include_lower:
                new_dt = np.concatenate(([lower_sn], new_dt))

            if include_upper:
                new_dt = np.concatenate((new_dt, [upper_sn]))

            new_dt = np.sort(np.unique(new_dt))
            new_dates = [Date.from_serial_number(n) for n in new_dt]

            return new_dates

        return self._transform_dates(
            lambda dates: _clip(dates, lower, upper, include_lower, inclide_upper),
            inplace=inplace,
        )

    def align_to_period_end(
        self,
        period_type: PeriodType,
        drop_duplicates: bool = True,
        inplace: bool = False,
    ) -> Schedule:
        """Map each date to the end of its containing period."""

        def _find_end_node(dates, period_type, drop_duplicates):
            if period_type == PeriodType.WEEKLY:
                transformed = (date.end_of_week for date in dates)
            if period_type == PeriodType.MONTHLY:
                transformed = (date.end_of_month for date in dates)
            if period_type == PeriodType.YEARLY:
                transformed = (date.end_of_year for date in dates)

            if drop_duplicates:
                return tuple(sorted(set(transformed)))
            else:
                return tuple(sorted(transformed))

        return self._transform_dates(
            lambda x: _find_end_node(x, period_type, drop_duplicates), inplace=inplace
        )

    def align_to_period_begin(
        self,
        period_type: PeriodType,
        drop_duplicates: bool = True,
        inplace: bool = False,
    ) -> Schedule:
        """Map each date to the beginning of its containing period."""

        def _find_begin_node(dates, period_type, drop_duplicates):
            if period_type == PeriodType.WEEKLY:
                transformed = (date.begin_of_week for date in dates)
            if period_type == PeriodType.MONTHLY:
                transformed = (date.begin_of_month for date in dates)
            if period_type == PeriodType.YEARLY:
                transformed = (date.begin_of_year for date in dates)

            if drop_duplicates:
                return tuple(sorted(set(transformed)))
            else:
                return tuple(sorted(transformed))

        return self._transform_dates(
            lambda x: _find_begin_node(x, period_type, drop_duplicates), inplace=inplace
        )

    def adjust_to_business_day(
        self,
        convention: BusinessDayConvention,
        drop_duplicates: bool = True,
        inplace: bool = False,
    ) -> Schedule:
        """Adjust each date using the calendar and convention provided."""

        def _adjust(dates, convention, drop_duplicates):
            transformed = (self.calendar.adjust(d, convention) for d in dates)
            if drop_duplicates:
                return tuple(sorted(set(transformed)))
            else:
                return tuple(sorted(transformed))

        return self._transform_dates(
            lambda x: _adjust(x, convention, drop_duplicates), inplace=inplace
        )

    def get_interval_year_fraction(self, daycount: DayCount) -> list[float]:
        """Return year fractions between consecutive schedule dates.

        Args:
            daycount: Day-count convention used for each interval.

        Returns:
            Tuple of year fractions for adjacent date pairs.
        """
        year_fracs = []
        for d1, d2 in zip(self.dates[:-1], self.dates[1:], strict=True):
            year_frac = daycount.year_fraction(d1, d2)
            year_fracs.append(year_frac)
        return tuple(year_fracs)

    def get_cumulative_year_fraction(self, daycount: DayCount) -> list[float]:
        """Return year fractions from the first date to later schedule dates.

        Args:
            daycount: Day-count convention used for each interval.

        Returns:
            Tuple of year fractions measured from the first date.
        """
        year_fracs = []
        reference = self.dates[0]
        for d in self.dates[1:]:
            year_frac = daycount.year_fraction(reference, d)
            year_fracs.append(year_frac)
        return tuple(year_fracs)
