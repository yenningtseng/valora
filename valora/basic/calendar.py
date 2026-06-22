from __future__ import annotations

from abc import ABC
from datetime import date, timedelta
from typing import List, Optional, cast

from .date import Date
from .enum import BusinessDayConvention, PeriodType


class Calendar(ABC):
    def __init__(
        self,
        name: str,
        official_holiday: List[Date],
        added_holiday: List[Date],
        removed_holiday: List[Date],
    ) -> None:
        self.name = name
        self.official_holiday = official_holiday
        self.added_holiday = added_holiday
        self.removed_holiday = removed_holiday
        self.base_date = date(1900, 1, 1)
        self._rebuild_holiday_structures()

    def _rebuild_holiday_structures(self) -> None:
        off = {d.serial_number for d in self.official_holiday}
        add = {d.serial_number for d in self.added_holiday}
        rem = {d.serial_number for d in self.removed_holiday}

        final = (off | add) - rem
        self.holiday_serial = sorted(final)
        self.holiday_set = set(self.holiday_serial)

    def is_business_day(self, dt: Date) -> bool:
        return dt.serial_number not in self.holiday_set

    def is_holiday(self, dt: Date) -> bool:
        return dt.serial_number in self.holiday_set

    def add_holiday(self, dts: List[Date]) -> None:
        self.added_holiday.extend(dts)
        self._rebuild_holiday_structures()

    def remove_holiday(self, dts: List[Date]) -> None:
        self.removed_holiday.extend(dts)
        self._rebuild_holiday_structures()

    def business_date_range(
        self,
        beg_date: Date,
        end_date: Date,
        include_begin: bool = True,
        include_end: bool = False,
    ) -> List[Date]:
        if end_date < beg_date:
            raise ValueError("`beg_date` must be earlier than `end_date`.")

        start = beg_date.serial_number + (0 if include_begin else 1)
        stop = end_date.serial_number + (1 if include_end else 0)

        return [
            Date.from_datetime(self.base_date + timedelta(days=s))
            for s in range(start, stop)
            if s not in self.holiday_set
        ]

    def adjust(
        self,
        dt: Date,
        convention: BusinessDayConvention,
    ) -> Date:
        if convention == BusinessDayConvention.NULLADJUSTMENT:
            return dt
        if self.is_business_day(dt):
            return dt
        if convention == BusinessDayConvention.FOLLOWING:
            cur = dt
            while not self.is_business_day(cur):
                cur = cast(Date, cur + 1)
            return cur
        if convention == BusinessDayConvention.PRECEDING:
            cur = dt
            while not self.is_business_day(cur):
                cur = cast(Date, cur - 1)
            return cur
        if convention == BusinessDayConvention.MODIFIED_FOLLOWING:
            following = self.adjust(dt, BusinessDayConvention.FOLLOWING)
            if following.month != dt.month:
                return self.adjust(dt, BusinessDayConvention.PRECEDING)
            return following
        if convention == BusinessDayConvention.MODIFIED_PRECEDING:
            preceding = self.adjust(dt, BusinessDayConvention.PRECEDING)
            if preceding.month != dt.month:
                return self.adjust(dt, BusinessDayConvention.FOLLOWING)
            return preceding

        raise ValueError(f"Unsupported business-day convention: {convention}")

    def begin_of_period(
        self,
        dt: Date,
        period_type: PeriodType,
    ) -> Date:
        if period_type == PeriodType.WEEKLY:
            init_dt = cast(Date, dt - dt.day_of_week + 1)
        elif period_type == PeriodType.MONTHLY:
            init_dt = dt.begin_of_month
        elif period_type == PeriodType.YEARLY:
            init_dt = dt.begin_of_year
        else:
            raise ValueError(f"Unsupported `period_type`: {period_type}")

        return self.adjust(init_dt, BusinessDayConvention.FOLLOWING)

    def end_of_period(
        self,
        dt: Date,
        period_type: PeriodType,
    ) -> Date:
        if period_type == PeriodType.WEEKLY:
            init_dt = dt + (7 - dt.day_of_week)
        elif period_type == PeriodType.MONTHLY:
            init_dt = dt.end_of_month
        elif period_type == PeriodType.YEARLY:
            init_dt = dt.end_of_year
        else:
            raise ValueError(f"Unsupported `period_type`: {period_type}")

        return self.adjust(init_dt, BusinessDayConvention.PRECEDING)

    def is_eom(self, dt: Date) -> bool:
        if dt == self.end_of_period(dt, PeriodType.MONTHLY):
            return True
        return False

    def add_business_days(
        self,
        dt: Date,
        n: int,
    ) -> Date:
        if n == 0:
            return dt

        cur = dt
        step = 1 if n > 0 else -1
        remaining = abs(n)

        while remaining > 0:
            cur = cast(Date, cur + step)
            if self.is_business_day(cur):
                remaining -= 1

        return cur


class Target2Calendar(Calendar):
    def __init__(
        self,
        official_holiday: Optional[List[Date]] = None,
        added_holiday: Optional[List[Date]] = None,
        removed_holiday: Optional[List[Date]] = None,
        beg_yr: int = 1900,
        end_yr: int = 2100,
    ) -> None:
        if end_yr < beg_yr:
            raise ValueError("`end_yr` must be greater than or equal to `beg_yr`.")

        holidays = (
            official_holiday
            if official_holiday is not None
            else self._generate_official_holidays(beg_yr, end_yr)
        )
        super().__init__(
            name="target_2",
            official_holiday=holidays,
            added_holiday=added_holiday or [],
            removed_holiday=removed_holiday or [],
        )

    @staticmethod
    def _easter_sunday(year: int) -> Date:
        a = year % 19
        b = year // 100
        c = year % 100
        d = b // 4
        e = b % 4
        f = (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d - g + 15) % 30
        i = c // 4
        k = c % 4
        l = (32 + 2 * e + 2 * i - h - k) % 7
        m = (a + 11 * h + 22 * l) // 451

        month = (h + l - 7 * m + 114) // 31
        day = ((h + l - 7 * m + 114) % 31) + 1
        return Date(year, month, day)

    @classmethod
    def _get_full_closure_day(cls, year: int) -> List[Date]:
        return [
            Date(year, 1, 1),
            cast(Date, cls._easter_sunday(year) - 2),
            cast(Date, cls._easter_sunday(year) + 1),
            Date(year, 5, 1),
            Date(year, 12, 25),
            Date(year, 12, 26),
        ]

    @classmethod
    def _generate_official_holidays(cls, beg_yr: int, end_yr: int) -> List[Date]:
        start_dt = max(Date(1900, 1, 1), Date(beg_yr, 1, 1))
        end_dt = Date(end_yr, 12, 31)
        holidays: set[Date] = set()

        cur = start_dt
        while cur <= end_dt:
            if cur.day_of_week > 5:
                holidays.add(cur)
            cur = cast(Date, cur + 1)

        # Include adjacent years so observed holidays crossing year-end are captured.
        for year in range(beg_yr - 1, end_yr + 2):
            for holiday in cls._get_full_closure_day(year):
                if start_dt <= holiday <= end_dt:
                    holidays.add(holiday)
        return sorted(holidays)


class FedWireCalendar(Calendar):
    def __init__(
        self,
        official_holiday: Optional[List[Date]] = None,
        added_holiday: Optional[List[Date]] = None,
        removed_holiday: Optional[List[Date]] = None,
        beg_yr: int = 1900,
        end_yr: int = 2100,
    ) -> None:
        if end_yr < beg_yr:
            raise ValueError("`end_yr` must be greater than or equal to `beg_yr`.")

        holidays = (
            official_holiday
            if official_holiday is not None
            else self._generate_official_holidays(beg_yr, end_yr)
        )
        super().__init__(
            name="fed-wire",
            official_holiday=holidays,
            added_holiday=added_holiday or [],
            removed_holiday=removed_holiday or [],
        )

    @staticmethod
    def _handle_substitute_holidays(holiday: Date) -> Date:
        if holiday.day_of_week == 6:
            return cast(Date, holiday - 1)
        if holiday.day_of_week == 7:
            return cast(Date, holiday + 1)
        return holiday

    @staticmethod
    def _nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> Date:
        first = Date(year, month, 1)
        delta = (weekday - first.day_of_week) % 7
        return cast(Date, first + delta + (n - 1) * 7)

    @staticmethod
    def _last_weekday_of_month(year: int, month: int, weekday: int) -> Date:
        last = Date(year, month, 1).end_of_month
        delta = (last.day_of_week - weekday) % 7
        return cast(Date, last - delta)

    @classmethod
    def _get_full_disclosure_day(cls, year: int) -> List[Date]:
        return [
            cls._handle_substitute_holidays(Date(year, 1, 1)),  # New Year's Day
            cls._nth_weekday_of_month(year, 1, 1, 3),  # Martin Luther King Jr. Day
            cls._nth_weekday_of_month(year, 2, 1, 3),  # Presidents Day
            cls._last_weekday_of_month(year, 5, 1),  # Memorial Day
            cls._handle_substitute_holidays(Date(year, 6, 19)),  # Juneteenth
            cls._handle_substitute_holidays(Date(year, 7, 4)),  # Independence Day
            cls._nth_weekday_of_month(year, 9, 1, 1),  # Labor Day
            cls._nth_weekday_of_month(year, 10, 1, 2),  # Columbus Day
            cls._handle_substitute_holidays(Date(year, 11, 11)),  # Veterans Day
            cls._nth_weekday_of_month(year, 11, 4, 4),  # Thanksgiving
            cls._handle_substitute_holidays(Date(year, 12, 25)),  # Christmas Day
        ]

    @classmethod
    def _generate_official_holidays(cls, beg_yr: int, end_yr: int) -> List[Date]:
        start_dt = max(Date(1900, 1, 1), Date(beg_yr, 1, 1))
        end_dt = Date(end_yr, 12, 31)
        holidays: set[Date] = set()

        cur = start_dt
        while cur <= end_dt:
            if cur.day_of_week > 5:
                holidays.add(cur)
            cur = cast(Date, cur + 1)

        # Include adjacent years so observed holidays crossing year-end are captured.
        for year in range(beg_yr - 1, end_yr + 2):
            for holiday in cls._get_full_disclosure_day(year):
                if start_dt <= holiday <= end_dt:
                    holidays.add(holiday)
        return sorted(holidays)
