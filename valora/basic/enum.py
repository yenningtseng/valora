from enum import Enum

class PeriodType(Enum):
    DAILY = 252.0
    WEEKLY = 52.0
    MONTHLY = 12.0
    YEARLY = 1.0

class BusinessDayConvention(Enum):
    NULLADJUSTMENT = "No Adjusting"
    ACTUAL = "actual"
    FOLLOWING = "following"
    PRECEDING = "preceding"
    MODIFIED_FOLLOWING = "modified_following"
    MODIFIED_PRECEDING = "modified_preceding"
