"""Day-count convention implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from typing import (
    Optional,
    cast,
    Tuple,
)

from .date import Date
from .calendar import Calendar


class DayCount(ABC):
    """Abstract base class for day-count conventions."""

    @staticmethod
    def _check_date(beg_date: Date, end_date: Date) -> None:
        """Validate input dates before day-count calculations."""
        if end_date < beg_date:
            raise ValueError("`beg_date` must be earlier than `end_date`.")
        if not isinstance(beg_date, Date):
            raise ValueError("`beg_date` must be `Date`.")
        if not isinstance(end_date, Date):
            raise ValueError("`end_date` must be `Date`.")

    @staticmethod
    @abstractmethod
    def day_count_between(beg_date: Date, end_date: Date) -> int:
        """Return the convention-specific day count between two dates."""
        ...

    @staticmethod
    @abstractmethod
    def year_fraction(beg_date: Date, end_date: Date) -> int:
        """Return the convention-specific year fraction between two dates."""
        ...


class Business252(DayCount):
    """Business/252 convention based on a configured business calendar."""

    calendar: Optional[Calendar] = None

    @staticmethod
    def set_calendar(calendar: Calendar) -> None:
        """Set the calendar used to count business days."""
        Business252.calendar = calendar

    @staticmethod
    def _check_calendar() -> None:
        """Ensure that a calendar has been configured."""
        if Business252.calendar is None:
            raise ValueError("`calendar` is not set yet.")

    @staticmethod
    def day_count_between(beg_date: Date, end_date: Date) -> int:
        """Return the number of business days between two dates."""
        Business252._check_calendar()
        calendar = Business252.calendar
        return len(calendar.business_day_between(beg_date, end_date))

    @staticmethod
    def year_fraction(beg_date: Date, end_date: Date) -> float:
        """Return the Business/252 year fraction between two dates."""
        Business252._check_date(beg_date, end_date)
        num = Business252.day_count_between(beg_date, end_date)
        denom = 252.0
        return float(num / denom)


class ActActIsda(DayCount):
    """Actual/Actual ISDA convention."""

    @staticmethod
    def day_count_between(beg_date: Date, end_date: Date) -> int:
        """Return actual elapsed calendar days."""
        ActActIsda._check_date(beg_date, end_date)
        return cast(int, end_date - beg_date)

    @staticmethod
    def year_fraction(beg_date: Date, end_date: Date) -> float:
        """Return the ISDA year fraction between two dates."""
        ActActIsda._check_date(beg_date, end_date)
        _, days_in_period, days_in_year = ActActIsda.day_count_each_year(
            beg_date, end_date
        )
        frac = np.sum(days_in_period / days_in_year)
        return frac

    @staticmethod
    def day_count_each_year(
        beg_date: Date, end_date: Date
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split an interval into per-year numerators and denominators."""
        ActActIsda._check_date(beg_date, end_date)
        years = []
        days_in_period = []
        days_in_year = []

        current = beg_date
        while current < end_date:
            seg_end = min(end_date, current.end_of_year + 1)
            actual_days = seg_end - current
            days_total = 366 if Date.is_leap(current.year) else 365

            years.append(current.year)
            days_in_period.append(actual_days)
            days_in_year.append(days_total)

            current = seg_end
        return np.array(years), np.array(days_in_period), np.array(days_in_year)


class Act365Fixed(DayCount):
    """Actual/365 Fixed convention."""

    @staticmethod
    def day_count_between(beg_date: Date, end_date: Date) -> int:
        """Return actual elapsed calendar days."""
        Act365Fixed._check_date(beg_date, end_date)
        return cast(int, end_date - beg_date)

    @staticmethod
    def year_fraction(beg_date: Date, end_date: Date) -> float:
        """Return actual days divided by 365."""
        Act365Fixed._check_date(beg_date, end_date)
        return Act365Fixed.day_count_between(beg_date, end_date) / 365


class Act360(DayCount):
    """Actual/360 convention."""

    @staticmethod
    def day_count_between(beg_date: Date, end_date: Date) -> int:
        """Return actual elapsed calendar days."""
        Act360._check_date(beg_date, end_date)
        return cast(int, end_date - beg_date)

    @staticmethod
    def year_fraction(beg_date: Date, end_date: Date) -> float:
        """Return actual days divided by 360."""
        Act360._check_date(beg_date, end_date)
        return Act360.day_count_between(beg_date, end_date) / 360.0


class Thirty360US(DayCount):
    """30/360 US convention."""

    @staticmethod
    def _is_last_date_of_february(beg_date: Date) -> bool:
        """Return whether a date is the last calendar day of February."""
        if (beg_date.month == 2) and (beg_date.end_of_month == beg_date):
            return True
        return False

    @staticmethod
    def day_count_between(beg_date: Date, end_date: Date, is_eom: bool = True) -> int:
        """Return the 30/360 US day count between two dates."""
        Thirty360US._check_date(beg_date, end_date)
        y1, m1, d1 = beg_date.year, beg_date.month, beg_date.day
        y2, m2, d2 = end_date.year, end_date.month, end_date.day

        # Rule 1: EOM adjustment for February
        if (
            is_eom
            and Thirty360US._is_last_date_of_february(beg_date)
            and Thirty360US._is_last_date_of_february(end_date)
        ):
            d2 = 30

        # Rule 2: EOM adjustment for Date 1
        if is_eom and Thirty360US._is_last_date_of_february(beg_date):
            d1 = 30

        # Rule 3: Adjust d2 based on d1
        if d2 == 31 and d1 >= 30:
            d2 = 30

        # Rule 4: Adjust D1
        if d1 == 31:
            d1 = 30

        return 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)

    @staticmethod
    def year_fraction(beg_date: Date, end_date: Date, is_eom: bool = True) -> float:
        """Return the 30/360 US year fraction between two dates."""
        Thirty360US._check_date(beg_date, end_date)
        return Thirty360US.day_count_between(beg_date, end_date, is_eom) / 360.0


class ThirtyA360(DayCount):
    """30A/360 convention."""

    @staticmethod
    def day_count_between(beg_date: Date, end_date: Date) -> int:
        """Return the 30A/360 day count between two dates."""
        ThirtyA360._check_date(beg_date, end_date)
        y1, m1, d1 = beg_date.year, beg_date.month, beg_date.day
        y2, m2, d2 = end_date.year, end_date.month, end_date.day

        # Rule 1: d1 = min(d1, 30)
        d1 = min(d1, 30)

        # Rule 2: if d1 is later than 29, d2 = min(d2, 30)
        if d1 > 29:
            d2 = min(d2, 30)

        return 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)

    @staticmethod
    def year_fraction(beg_date: Date, end_date: Date) -> float:
        """Return the 30A/360 year fraction between two dates."""
        ThirtyA360._check_date(beg_date, end_date)
        return ThirtyA360.day_count_between(beg_date, end_date) / 360.0


class ThirtyE360(DayCount):
    """30E/360 convention."""

    @staticmethod
    def day_count_between(beg_date: Date, end_date: Date) -> int:
        """Return the 30E/360 day count between two dates."""
        ThirtyE360._check_date(beg_date, end_date)
        y1, m1, d1 = beg_date.year, beg_date.month, beg_date.day
        y2, m2, d2 = end_date.year, end_date.month, end_date.day

        # Rule 1: if d1 is 31, d1 = 30
        if d1 == 31:
            d1 = 30

        # Rule 2: if d2 is 31, d2 = 30
        if d2 == 31:
            d2 = 30

        return 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)

    @staticmethod
    def year_fraction(beg_date: Date, end_date: Date) -> float:
        """Return the 30E/360 year fraction between two dates."""
        ThirtyE360._check_date(beg_date, end_date)
        return ThirtyE360.day_count_between(beg_date, end_date) / 360.0
