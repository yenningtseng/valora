"""Date and period primitives used by the basic valuation layer."""

from __future__ import annotations
from datetime import date, datetime, timedelta

import pandas as pd
from matplotlib.dates import date2num
from typing import Union, List, Self

from .enum import PeriodType


class Date:
    """Wrapper around :class:`datetime.date` with finance-oriented helpers."""

    def __init__(self, year: int, month: int, day: int) -> None:
        """Create a date from calendar year, month, and day."""
        self.year = year
        self.month = month
        self.day = day
        self.base_dt = date(day=1, month=1, year=1900)
        self.date = date(day=day, month=month, year=year)

    @property
    def serial_number(self) -> int:
        """Return the number of days since ``1900-01-01``."""
        return int((self.date - self.base_dt).days)

    @property
    def day_of_week(self) -> int:
        """Return weekday number where Monday is ``1`` and Sunday is ``7``."""
        return self.date.weekday() + 1

    @property
    def day_of_month(self) -> int:
        """Return the day number within the month."""
        return self.day

    @property
    def day_of_year(self) -> int:
        """Return the ordinal day within the year."""
        leap = Date.is_leap(self.year)
        days_map = [31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        return sum(days_map[: self.month - 1]) + self.day

    @property
    def begin_of_year(self) -> Date:
        """Return the first day of the same year."""
        return Date(self.year, 1, 1)

    @property
    def end_of_year(self) -> Date:
        """Return the last day of the same year."""
        return Date(self.year, 12, 31)

    @property
    def begin_of_month(self) -> Date:
        """Return the first day of the same month."""
        return Date(self.year, self.month, 1)

    @property
    def end_of_month(self) -> Date:
        """Return the last day of the same month."""
        leap = Date.is_leap(self.year)
        days_map = [31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        return Date(self.year, self.month, days_map[self.month - 1])

    @property
    def begin_of_week(self) -> Date:
        """Return the Monday of the same week."""
        return self - (self.day_of_week - 1)

    @property
    def end_of_week(self) -> Date:
        """Return the Sunday of the same week."""
        return self + (7 - self.day_of_week)

    @staticmethod
    def from_serial_number(n: Union[int, float]) -> Date:
        """Create a :class:`Date` from a serial number."""
        n = int(n)
        return Date(1900, 1, 1) + n

    @staticmethod
    def today() -> Date:
        """Return today's local calendar date."""
        td = date.today()
        return Date(td.year, td.month, td.day)

    @staticmethod
    def is_leap(y: Union[int, float]) -> bool:
        """Return whether ``y`` is a leap year."""
        if (y % 400 == 0) or ((y % 4 == 0) and (y % 100 != 0)):
            return True
        else:
            return False

    @staticmethod
    def date_range(beg_date: Date, end_date: Date) -> List[Date]:
        """Return all calendar dates between two endpoints, inclusive."""
        if end_date < beg_date:
            raise ValueError("`end_date` must not be earlier than `beg_date`.")
        delta_days = (end_date.date - beg_date.date).days

        return [
            Date(beg_date.year, beg_date.month, beg_date.day) + offset
            for offset in range(delta_days + 1)
        ]

    @classmethod
    def from_timestamp(cls, timestamp: pd.Timestamp) -> Self:
        """Create a :class:`Date` from a :class:`pandas.Timestamp`."""
        return cls(timestamp.year, timestamp.month, timestamp.day)

    @classmethod
    def from_string(cls, value: str) -> Self:
        """Parse a date string from several common formats."""
        formats = [
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%Y.%m.%d",
            "%Y%m%d",
            "%d-%m-%Y",
            "%d/%m/%Y",
            "%d.%m.%Y",
            "%m-%d-%Y",
            "%m/%d/%Y",
            "%d-%m-%y",
            "%m/%d/%y",
            "%b %d, %Y",
            "%B %d, %Y",
        ]

        for date_format in formats:
            try:
                parsed = datetime.strptime(value, date_format)
                return cls(parsed.year, parsed.month, parsed.day)
            except ValueError:
                continue

        raise ValueError(f"Unsupported date format: {value!r}")

    @classmethod
    def from_datetime(cls, date_time: datetime) -> Self:
        """Create a :class:`Date` from a :class:`datetime.datetime`."""
        return cls(date_time.year, date_time.month, date_time.day)

    def __add__(self, other: Union[int, Period]) -> Date:
        """Add calendar days or a :class:`Period` to this date."""
        if isinstance(other, int):
            new_dt = self.date + timedelta(days=other)
            return Date.from_datetime(new_dt)
        if isinstance(other, Period):
            new_dt = other.add_period(self.date)
            return Date.from_datetime(new_dt)

    def __sub__(self, other: Union[int, Date, Period]) -> Union[int, Date]:
        """Subtract calendar days, another date, or a :class:`Period`."""
        if isinstance(other, int):
            new_dt = self.date - timedelta(days=other)
            return Date.from_datetime(new_dt)
        elif isinstance(other, Date):
            return (self.date - other.date).days
        elif isinstance(other, Period):
            reversed_period = Period(other.period_type, -other.value)
            new_dt = reversed_period.add_period(self.date)
            return Date.from_datetime(new_dt)

    def __eq__(self, other: Date) -> bool:
        """Return whether this date is equal to another date."""
        return self.date == other.date

    def __lt__(self, other: Date) -> bool:
        """Return whether this date is earlier than another date."""
        if self.date < other.date:
            return True
        return False

    def __gt__(self, other: Date) -> bool:
        """Return whether this date is later than another date."""
        if self.date > other.date:
            return True
        return False

    def __le__(self, other: Date) -> bool:
        """Return whether this date is earlier than or equal to another date."""
        if self.date <= other.date:
            return True
        return False

    def __ge__(self, other: Date) -> bool:
        """Return whether this date is later than or equal to another date."""
        if self.date >= other.date:
            return True
        return False

    def __hash__(self) -> int:
        """Return in hash."""
        return hash((self.year, self.month, self.day))

    def __str__(self) -> str:
        """Return in string format."""
        return self.date.strftime("%Y%m%d")

    def __repr__(self) -> str:
        """Return in string format."""
        return self.date.strftime("%Y%m%d")

    def __float__(self):
        """Return Matplotlib's floating-date representation."""
        return float(date2num(self.date))


class Period:
    """Signed period expressed in days, weeks, months, or years."""

    def __init__(self, period_type: PeriodType, value: int) -> None:
        """Create a period from a unit and integer step size."""
        self.period_type = period_type
        self.value = value

    def to_timedelta(self) -> timedelta:
        """Convert daily or weekly periods to :class:`datetime.timedelta`."""
        if self.period_type == PeriodType.DAILY:
            return timedelta(days=self.value)
        if self.period_type == PeriodType.WEEKLY:
            return timedelta(weeks=self.value)

    def add_period(self, base_date: date) -> date:
        """Advance ``base_date`` by this period using calendar-aware rules."""
        if self.period_type in [PeriodType.DAILY, PeriodType.WEEKLY]:
            return base_date + self.to_timedelta()

        year = base_date.year
        month = base_date.month
        day = base_date.day

        if self.period_type == PeriodType.MONTHLY:
            month += self.value
            year += (month - 1) // 12
            month = (month - 1) % 12 + 1
            day = min(
                day,
                [
                    31,
                    29 if Date.is_leap(year) else 28,
                    31,
                    30,
                    31,
                    30,
                    31,
                    31,
                    30,
                    31,
                    30,
                    31,
                ][month - 1],
            )
            return date(year, month, day)

        if self.period_type == PeriodType.YEARLY:
            year += self.value
            if month == 2 and day == 29 and not Date.is_leap(year):
                day = 28
            return date(year, month, day)

        raise ValueError(f"Unsupported period_type: {self.period_type}")
