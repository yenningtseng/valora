import pytest

from valora.basic.date import Date, Period
from valora.basic.enum import PeriodType

class TestDate:

    def test_serial_number(self):
        b = Date(1900, 1, 2)
        assert 1 == b.serial_number

    def test_day_of_week(self):
        a = Date(2026, 6, 8)
        for i in range(0, 6):
            assert (i + 1) == (a + i).day_of_week

    def test_day_of_month(self):
        a = Date(2026, 6, 8)
        for i in range(0, 6):
            assert (i + 8) == (a + i).day_of_month

    def test_day_of_year(self):
        a = Date(2026, 6, 8)
        for i in range(0, 6):
            assert (i + 159) == (a + i).day_of_year

    def test_begin_of_year(self):
        assert Date(2019, 1, 1) == Date(2019, 5, 12).begin_of_year

    def test_end_of_year(self):
        assert Date(2019, 12, 31) == Date(2019, 5, 12).end_of_year

    def test_begin_of_month(self):
        assert Date(2026, 2, 1) == Date(2026, 2, 5).begin_of_month

    def test_end_of_month(self):
        assert Date(2028, 2, 29) == Date(2028, 2, 5).end_of_month

    def test_begin_of_week(self):
        assert Date(2026, 6, 15) == Date(2026, 6, 18).begin_of_week

    def test_end_of_week(self):
        assert Date(2026, 6, 21) == Date(2026, 6, 18).end_of_week

    def test_from_serial_number(self):
        assert Date.from_serial_number(1) == Date(1900, 1, 2)

    def test_today(self):
        from datetime import date
        td = date.today()
        assert Date.from_datetime(td) == Date.today()

    def test_is_leap(self):
        assert Date.is_leap(2028)
        assert not Date.is_leap(2026)

    def test_date_range(self):
        assert [
            Date(2026, 6, 15),
            Date(2026, 6, 16),
            Date(2026, 6, 17),
            Date(2026, 6, 18),
            Date(2026, 6, 19),
            Date(2026, 6, 20),
        ] == Date.date_range(Date(2026, 6, 15), Date(2026, 6, 20))

    def test_from_timestamp(self):
        import pandas as pd
        dt = pd.Timestamp(2026, 6, 21)
        assert Date.from_timestamp(dt) == Date(2026, 6, 21)

    def test_from_datetime(self):
        from datetime import date
        dt = date(2008, 12, 13)
        assert Date.from_datetime(dt) == Date(2008, 12, 13)

    def test_from_string(self):
        assert Date(2009, 12, 31) == Date.from_string("20091231")
        assert Date(2009, 12, 31) == Date.from_string("2009-12-31")
        assert Date(2009, 12, 31) == Date.from_string("2009.12.31")
        assert Date(2009, 12, 31) == Date.from_string("31-12-2009")
        assert Date(2009, 12, 31) == Date.from_string("31/12/2009")

    def test_add(self):
        assert Date(2028, 2, 29) + 1 == Date(2028, 3, 1)
        assert Date(2028, 2, 29) + Period(PeriodType.YEARLY, 4) == Date(2032, 2, 29)
        assert Date(2028, 2, 29) + Period(PeriodType.YEARLY, 1) == Date(2029, 2, 28)
        assert Date(2028, 2, 29) + Period(PeriodType.MONTHLY, 1) == Date(2028, 3, 29)

    def test_sub(self):
        assert Date(2028, 2, 29) - 1 == Date(2028, 2, 28)
        assert Date(2028, 2, 29) - Period(PeriodType.YEARLY, 4) == Date(2024, 2, 29)
        assert Date(2028, 2, 29) - Period(PeriodType.YEARLY, 1) == Date(2027, 2, 28)
        assert Date(2028, 2, 29) - Period(PeriodType.MONTHLY, 1) == Date(2028, 1, 29)

    def test_eq(self):
        assert Date(2022, 12, 3) == Date(2022, 12, 3)

    def test_lt(self):
        assert not Date(2022, 12, 3) < Date(2021, 12, 3)

    def test_gt(self):
        assert Date(2022, 12, 3) > Date(2020, 12, 3)

    def test_le(self):
        assert not Date(2022, 12, 3) <= Date(2020, 12, 3)

    def test_ge(self):
        assert Date(2022, 12, 3) >= Date(2020, 12, 3)

    def test_hash(self):
        date1 = Date(2026, 6, 8)
        date2 = Date(2026, 6, 8)
        date3 = Date(2025, 12, 13)

        assert hash(date1) == hash(date2)
        assert hash(date1) != hash(date3)

    def test_str(self):
        date1 = Date(2026, 6, 8)
        assert str(date1) == "20260608"

    def test_repr(self):
        date1 = Date(2026, 6, 8)
        assert repr(date1) == "20260608"

    def test_float(self):
        from matplotlib.dates import date2num
        date = Date(2026, 6, 8)
        assert float(date) == float(date2num(date.date))


class TestPeriod:

    def test_to_timedelta(self):
        from datetime import timedelta

        prd1 = Period(PeriodType.DAILY, 3)
        prd2 = Period(PeriodType.WEEKLY, 3)

        assert timedelta(days=3) == prd1.to_timedelta()
        assert timedelta(weeks=3) == prd2.to_timedelta()

    def test_add_period(self):
        d1 = Date(2028, 2, 29)

        prd1 = Period(PeriodType.DAILY, 4)
        prd2 = Period(PeriodType.WEEKLY, 4)
        prd3 = Period(PeriodType.MONTHLY, 4)
        prd4 = Period(PeriodType.YEARLY, 4)

        assert Date(2028, 3, 4) == d1 + prd1
        assert Date(2028, 3, 28) == d1 + prd2
        assert Date(2028, 6, 29) == d1 + prd3
        assert Date(2032, 2, 29) == d1 + prd4
