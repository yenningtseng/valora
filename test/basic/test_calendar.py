import pytest

from valora.basic.calendar import Calendar, FedWireCalendar, Target2Calendar
from valora.basic.date import Date
from valora.basic.enum import BusinessDayConvention, PeriodType


@pytest.fixture
def calendar():
    return Calendar(
        name="TEST",
        official_holiday=[
            Date(2030, 12, 3),
            Date(2040, 6, 27),
            Date(2025, 9, 6),
            Date(2027, 6, 7),
            Date(2027, 6, 8),
            Date(2027, 6, 9),
            Date(2040, 5, 31),
            Date(2040, 5, 30),
            Date(2027, 6, 1),
        ],
        added_holiday=[],
        removed_holiday=[],
    )


class TestCalendar:
    def test_init_building_holiday_struc(self, calendar):
        assert calendar.name == "TEST"
        assert Date(2030, 12, 3).serial_number in calendar.holiday_set
        assert Date(2040, 6, 27).serial_number in calendar.holiday_set
        assert Date(2025, 9, 6).serial_number in calendar.holiday_set
        assert Date(2027, 6, 7).serial_number in calendar.holiday_set

    def test_is_business_day(self, calendar):
        assert calendar.is_business_day(Date(2030, 12, 4)) is True
        assert calendar.is_business_day(Date(2030, 12, 3)) is False

    def test_is_holiday(self, calendar):
        assert calendar.is_holiday(Date(2030, 12, 4)) is False
        assert calendar.is_holiday(Date(2030, 12, 3)) is True

    def test_add_holiday(self, calendar):
        new_holiday = Date(2018, 12, 12)

        assert calendar.is_business_day(new_holiday) is True

        calendar.add_holiday([new_holiday])

        assert calendar.is_holiday(new_holiday) is True
        assert calendar.is_business_day(new_holiday) is False

    def test_remove_holiday(self, calendar):
        holiday_to_remove = Date(2027, 6, 7)

        assert calendar.is_holiday(holiday_to_remove) is True

        calendar.remove_holiday([holiday_to_remove])

        assert calendar.is_holiday(holiday_to_remove) is False
        assert calendar.is_business_day(holiday_to_remove) is True

    def test_business_date_range(self, calendar):
        with pytest.raises(ValueError):
            beg_date = Date(2027, 6, 7)
            end_date = Date(2027, 1, 5)
            calendar.business_date_range(beg_date, end_date)

        beg_date = Date(2027, 6, 1)
        end_date = Date(2027, 6, 7)
        res_dates = [
            Date(2027, 6, 2),
            Date(2027, 6, 3),
            Date(2027, 6, 4),
            Date(2027, 6, 5),
            Date(2027, 6, 6),
        ]

        assert res_dates == calendar.business_date_range(beg_date, end_date)

    def test_adjust(self, calendar):
        x = Date(2027, 6, 8)
        y = Date(2040, 5, 30)
        z = Date(2027, 6, 1)

        assert Date(2027, 6, 8) == calendar.adjust(
            x, BusinessDayConvention.NULLADJUSTMENT
        )
        assert Date(2027, 6, 10) == calendar.adjust(x, BusinessDayConvention.FOLLOWING)
        assert Date(2027, 6, 6) == calendar.adjust(x, BusinessDayConvention.PRECEDING)
        assert Date(2040, 5, 29) == calendar.adjust(
            y, BusinessDayConvention.MODIFIED_FOLLOWING
        )
        assert Date(2027, 6, 2) == calendar.adjust(
            z, BusinessDayConvention.MODIFIED_PRECEDING
        )

    def test_begin_of_period(self, calendar):
        assert Date(2027, 5, 31) == calendar.begin_of_period(
            Date(2027, 6, 4), PeriodType.WEEKLY,
        )

        assert Date(2018, 12, 1) == calendar.begin_of_period(
            Date(2018, 12, 12), PeriodType.MONTHLY,
        )

        assert Date(2040, 1, 1) == calendar.begin_of_period(
            Date(2040, 5, 1), PeriodType.YEARLY,
        )

    def test_end_of_period(self, calendar):
        assert Date(2027, 6, 6) == calendar.end_of_period(
            Date(2027, 6, 4), PeriodType.WEEKLY,
        )

        assert Date(2018, 12, 31) == calendar.end_of_period(
            Date(2018, 12, 12), PeriodType.MONTHLY,
        )

        assert Date(2040, 12, 31) == calendar.end_of_period(
            Date(2040, 5, 1), PeriodType.YEARLY,
        )

    def test_is_eom(self, calendar):
        assert calendar.is_eom(Date(2040, 12, 31)) is True

    def test_add_business_days(self, calendar):
        assert Date(2027, 6, 6) == calendar.add_business_days(
            Date(2027, 6, 10), -1
        )

        assert Date(2027, 6, 10) == calendar.add_business_days(
            Date(2027, 6, 6), 1
        )
