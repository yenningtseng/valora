"""Coupon-leg containers generated from schedules or existing coupons."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeAlias, Literal
from collections.abc import Iterable

from .cash_flow import Legs
from .coupon import Coupon, FixedRateCoupon, IborCoupon, OvernightIndexCoupon
from .date import Date
from .date import Period
from .calendar import Calendar
from .daycount import DayCount
from .enum import BusinessDayConvention
from .schedule import Schedule
from .index import IborIndex, InterestRateIndex

Scalar: TypeAlias = int | float
DateLike: TypeAlias = Date | int
AverageMethod: TypeAlias = Literal["Compound-Average", "Simple-Average"]


class CouponLegs(Legs, ABC):
    """Abstract base class for legs composed of coupon cash flows.

    Subclasses are responsible for building ``self.coupons`` and exposing
    them as a ``Legs`` view through occurrence dates and coupon amounts.
    """

    def __init__(
        self,
        occur_date: Iterable[DateLike],
        amount: Iterable[Scalar],
        sort: bool = True,
    ) -> None:
        """Initialize a coupon leg from dates and amounts.

        Args:
            occur_date: Sequence of coupon payment dates.
            amount: Sequence of coupon amounts aligned with ``occur_date``.
            sort: Whether to sort and merge entries sharing the same date.
        """
        self.coupons = self.make_coupons()
        super().__init__(occur_date, amount, sort)

    @abstractmethod
    def make_coupons(self) -> list[Coupon]:
        """Build the coupon objects backing this leg."""
        ...

    @classmethod
    def from_coupons(cls, coupons: list[Coupon]) -> CouponLegs:
        """Build a coupon leg directly from preconstructed coupons."""
        obj = cls.__new__(cls)
        obj.coupons = coupons
        Legs.__init__(
            obj,
            [cpn.occur_date for cpn in coupons],
            [cpn.amount for cpn in coupons],
            sort=True,
        )
        return obj

    def keep_unpaid_coupon(self, valuation_date: Date) -> CouponLegs:
        """Return coupons whose payment date is after ``valuation_date``."""
        unpaid = [cpn for cpn in self.coupons if cpn.occur_date > valuation_date]
        return type(self).from_coupons(unpaid)

    def keep_next_coupon(self, valuation_date: Date) -> CouponLegs:
        """Return the coupon whose accrual period contains ``valuation_date``."""
        unpaid = [
            cpn
            for cpn in self.coupons
            if (cpn.accrual_start <= valuation_date)
            & (cpn.accrual_end >= valuation_date)
        ]
        return type(self).from_coupons(unpaid)


class OvernightIndexLegs(CouponLegs):
    """Leg of overnight-index coupons generated from a schedule.

    Each accrual period in the generated schedule produces one
    ``OvernightIndexCoupon``.
    """

    def __init__(
        self,
        effective_date: Date,
        maturity_date: Date,
        payment_period: Period,
        calendar: Calendar,
        daycount: DayCount,
        holiday_rule: BusinessDayConvention,
        payment_lag: int,
        lookback_period: int,
        lockout_period: int,
        observation_shift: bool,
        index: InterestRateIndex,
        average_method: AverageMethod,
        notional: float,
        gearing: float = 1.0,
        spread: float = 0.0,
    ) -> None:
        """Initialize an overnight-index coupon leg.

        Args:
            effective_date: Start date of the leg.
            maturity_date: End date of the leg.
            payment_period: Frequency used to build accrual periods.
            calendar: Calendar used for schedule and payment adjustments.
            daycount: Day-count convention used by each coupon.
            holiday_rule: Business-day adjustment rule for the schedule.
            payment_lag: Number of business days from accrual end to payment.
            lookback_period: Observation lookback in business days.
            lockout_period: Number of final observation dates locked out.
            observation_shift: Whether observation dates drive accrual weights.
            index: Overnight index used to source fixings.
            average_method: Rate aggregation method for each coupon.
            notional: Coupon notional amount.
            gearing: Multiplier applied to coupon rates.
            spread: Fixed spread added to coupon rates.
        """
        self.effective_date = effective_date
        self.maturity_date = maturity_date
        self.payment_period = payment_period
        self.calendar = calendar
        self.holiday_rule = holiday_rule
        self.payment_lag = payment_lag
        self.lookback_period = lookback_period
        self.lockout_period = lockout_period
        self.observation_shift = observation_shift
        self.daycount = daycount
        self.index = index
        self.average_method: AverageMethod = average_method
        self.notional = notional
        self.gearing = gearing
        self.spread = spread

        self.schedule = self.make_schedule()
        self.coupons = self.make_coupons()
        super().__init__(
            [cpn.occur_date for cpn in self.coupons],
            [cpn.amount for cpn in self.coupons],
        )

    def make_schedule(self) -> Schedule:
        """Build the accrual schedule for the leg."""
        return Schedule(
            self.effective_date,
            self.maturity_date,
            self.payment_period,
            self.calendar,
            self.holiday_rule,
        )

    def make_coupons(self) -> list[Coupon]:
        """Build one overnight coupon for each accrual period."""
        cpns: list[Coupon] = []
        mapped_dates: list[Date] = self.schedule.mapped_date
        accrual_starts: list[Date] = mapped_dates[:-1]
        accrual_ends: list[Date] = mapped_dates[1:]
        for beg_date, end_date in zip(accrual_starts, accrual_ends, strict=True):
            cpn = OvernightIndexCoupon(
                beg_date,
                end_date,
                self.payment_lag,
                self.lookback_period,
                self.lockout_period,
                self.observation_shift,
                self.calendar,
                self.daycount,
                self.index,
                self.average_method,
                self.notional,
                self.gearing,
                self.spread,
            )
            cpns.append(cpn)

        return cpns


class FixedRateLegs(CouponLegs):
    """Leg of fixed-rate coupons generated from a schedule."""

    def __init__(
        self,
        effective_date: Date,
        maturity_date: Date,
        payment_period: Period,
        calendar: Calendar,
        daycount: DayCount,
        holiday_rule: BusinessDayConvention,
        payment_lag: int,
        rate: float,
        notional: float,
        pay_in_arrear: bool = True,
    ) -> None:
        """Initialize a fixed-rate coupon leg.

        Args:
            effective_date: Start date of the leg.
            maturity_date: End date of the leg.
            payment_period: Frequency used to build accrual periods.
            calendar: Calendar used for schedule and payment adjustments.
            daycount: Day-count convention used by each coupon.
            holiday_rule: Business-day adjustment rule for the schedule.
            payment_lag: Number of business days from anchor date to payment.
            rate: Fixed coupon rate.
            notional: Coupon notional amount.
            pay_in_arrear: Whether payment is anchored from accrual end.
        """
        self.effective_date = effective_date
        self.maturity_date = maturity_date
        self.payment_period = payment_period
        self.calendar = calendar
        self.holiday_rule = holiday_rule
        self.payment_lag = payment_lag
        self.daycount = daycount
        self.rate = rate
        self.notional = notional
        self.pay_in_arrear = pay_in_arrear

        self.schedule = self.make_schedule()
        self.coupons = self.make_coupons()
        super().__init__(
            [cpn.occur_date for cpn in self.coupons],
            [cpn.amount for cpn in self.coupons],
        )

    def make_schedule(self) -> Schedule:
        """Build the accrual schedule for the leg."""
        return Schedule(
            self.effective_date,
            self.maturity_date,
            self.payment_period,
            self.calendar,
            self.holiday_rule,
        )

    def make_coupons(self) -> list[Coupon]:
        """Build one fixed-rate coupon for each accrual period."""
        cpns: list[Coupon] = []
        mapped_dates: list[Date] = self.schedule.mapped_date
        accrual_starts: list[Date] = mapped_dates[:-1]
        accrual_ends: list[Date] = mapped_dates[1:]

        for beg_date, end_date in zip(accrual_starts, accrual_ends, strict=True):
            cpn = FixedRateCoupon(
                beg_date,
                end_date,
                self.payment_lag,
                self.calendar,
                self.daycount,
                self.rate,
                self.notional,
                self.pay_in_arrear,
            )
            cpns.append(cpn)

        return cpns


class IborLegs(CouponLegs):
    """Leg of IBOR coupons generated from a schedule.

    Each accrual period in the generated schedule produces one
    ``IborCoupon``.
    """

    def __init__(
        self,
        effective_date: Date,
        maturity_date: Date,
        payment_period: Period,
        calendar: Calendar,
        daycount: DayCount,
        holiday_rule: BusinessDayConvention,
        payment_lag: int,
        fixing_lag: int,
        index: IborIndex,
        notional: float,
        gearing: float = 1.0,
        spread: float = 0.0,
        fix_in_advance: bool = True,
    ) -> None:
        """Initialize an IBOR coupon leg.

        Args:
            effective_date: Start date of the leg.
            maturity_date: End date of the leg.
            payment_period: Frequency used to build accrual periods.
            calendar: Calendar used for schedule and payment adjustments.
            daycount: Day-count convention used by each coupon.
            holiday_rule: Business-day adjustment rule for the schedule.
            payment_lag: Number of business days from accrual end to payment.
            fixing_lag: Number of business days between fixing date and reset date.
            index: IBOR index used to source coupon fixings.
            notional: Coupon notional amount.
            gearing: Multiplier applied to coupon rates.
            spread: Fixed spread added to coupon rates.
            fix_in_advance: Whether coupons fix at accrual start or accrual end.
        """
        self.effective_date = effective_date
        self.maturity_date = maturity_date
        self.payment_period = payment_period
        self.calendar = calendar
        self.holiday_rule = holiday_rule
        self.payment_lag = payment_lag
        self.fixing_lag = fixing_lag
        self.daycount = daycount
        self.index = index
        self.notional = notional
        self.fix_in_advance = fix_in_advance
        self.gearing = gearing
        self.spread = spread

        self.schedule = self.make_schedule()
        self.coupons = self.make_coupons()
        super().__init__(
            [cpn.occur_date for cpn in self.coupons],
            [cpn.amount for cpn in self.coupons],
        )

    def make_schedule(self) -> Schedule:
        """Build the accrual schedule for the leg."""
        return Schedule(
            self.effective_date,
            self.maturity_date,
            self.payment_period,
            self.calendar,
            self.holiday_rule,
        )

    def make_coupons(self) -> list[Coupon]:
        """Build one IBOR coupon for each accrual period."""
        cpns: list[Coupon] = []
        mapped_dates: list[Date] = self.schedule.mapped_date
        accrual_starts: list[Date] = mapped_dates[:-1]
        accrual_ends: list[Date] = mapped_dates[1:]

        for beg_date, end_date in zip(accrual_starts, accrual_ends, strict=True):
            cpn = IborCoupon(
                beg_date,
                end_date,
                self.payment_lag,
                self.fixing_lag,
                self.calendar,
                self.daycount,
                self.notional,
                self.index,
                self.gearing,
                self.spread,
                self.fix_in_advance,
            )
            cpns.append(cpn)

        return cpns

class AggregatedLegs(CouponLegs):
    """Adapter that exposes an existing coupon sequence as a coupon leg."""

    def __init__(
        self,
        org_coupons,
    ):
        """Create a coupon leg from an already assembled coupon sequence."""
        self.org_coupons = org_coupons
        self.coupons = self.make_coupons()
        super().__init__(
            [cpn.occur_date for cpn in self.coupons],
            [cpn.amount for cpn in self.coupons],
        )

    def make_coupons(self) -> list[Coupon]:
        """Return the original coupon sequence unchanged."""
        return self.org_coupons
