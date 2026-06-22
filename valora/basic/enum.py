"""Enumerations shared across date, calendar, and schedule utilities."""

from enum import Enum


class PeriodType(Enum):
    """Supported units for recurring periods and schedule tenors."""

    DAILY = 252.0
    WEEKLY = 52.0
    MONTHLY = 12.0
    YEARLY = 1.0


class BusinessDayConvention(Enum):
    """Rules for adjusting dates that fall on non-business days."""

    NULLADJUSTMENT = "No Adjusting"
    ACTUAL = "actual"
    FOLLOWING = "following"
    PRECEDING = "preceding"
    MODIFIED_FOLLOWING = "modified_following"
    MODIFIED_PRECEDING = "modified_preceding"


class Compounding(Enum):
    """Compounding conventions used in interest-rate and discounting calculations."""

    SIMPLE = "simple"
    CONTINUOUS = "continuous"
    ANNUAL = 1.0
    SEMIANNUAL = 2.0
    QUARTERLY = 4.0
    MONTHLY = 12.0
    WEEKLY = 52.0
    DAILY360 = 360.0
    DAILY365 = 365.0


class Frequency(Enum):
    """Frequency convention for coupon, principal."""

    ONCE = "once"
    ANNUAL = 1.0
    SEMIANNUAL = 2.0
    QUARTERLY = 4.0
    MONTHLY = 12.0
    WEEKLY = 52.0
    DAILY360 = 360.0
    DAILY365 = 365.0
