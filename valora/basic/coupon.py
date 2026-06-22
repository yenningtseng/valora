"""Coupon cash-flow types for fixed, IBOR, and overnight-rate products."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypeAlias, Literal
from collections.abc import Iterable

import numpy as np

from .calendar import Calendar
from .cash_flow import CashFlow
from .date import Date
from .daycount import DayCount
from .enum import BusinessDayConvention

AverageMethod: TypeAlias = Literal["Compound-Average", "Simple-Average"]


class Coupon(CashFlow, ABC):
    """Abstract base class for coupon-style cash flows.

    A coupon is a cash flow defined by an accrual period, payment date,
    day-count convention, and notional. Concrete subclasses are responsible
    for determining the coupon rate and accrued interest.
    """

    rate: float
    tau: float

    def __init__(
        self,
        accrual_start: Date,
        accrual_end: Date,
        payment_date: Date,
        daycount: DayCount,
        notional: float,
    ) -> None:
        """Initialize a coupon and compute its payment amount.

        Args:
            accrual_start: Start date of the accrual period.
            accrual_end: End date of the accrual period.
            payment_date: Coupon payment date.
            daycount: Day-count convention used for accrual calculations.
            notional: Coupon notional amount.
        """
        self.accrual_start = accrual_start
        self.accrual_end = accrual_end
        self.payment_date = payment_date
        self.daycount = daycount
        self.notional = notional

        super().__init__(payment_date, self.get_amount())

    @abstractmethod
    def get_amount(self) -> float:
        """Return the coupon payment amount."""
        ...

    @abstractmethod
    def get_accrued_interest(self, valuation_date: Date) -> float:
        """Return accrued interest as of the valuation date."""
        ...

    def get_npv(self, discount_curve: Any, valuation_date: Date) -> float:
        """Return discounted present value of the coupon.

        Args:
            discount_curve: Curve object exposing ``get_zcb``.
            valuation_date: Date on which the coupon is valued.

        Returns:
            Present value of the coupon. Returns ``0.0`` when the coupon has
            already occurred on or before ``valuation_date``.
        """
        if self.occur_date <= valuation_date:
            return 0.0

        dis_fac = discount_curve.get_zcb(valuation_date, self.occur_date)
        return self.amount * dis_fac


class FixedRateCoupon(Coupon):
    """Fixed-rate coupon with configurable payment timing.

    The coupon amount is determined from a fixed rate, notional, and accrual
    year fraction. Payment can occur either in arrears or in advance,
    depending on ``pay_in_arrear``.
    """

    def __init__(
        self,
        accrual_start: Date,
        accrual_end: Date,
        payment_lag: int,
        calendar: Calendar,
        daycount: DayCount,
        rate: float,
        notional: float,
        pay_in_arrear: bool = True,
    ) -> None:
        """Initialize a fixed-rate coupon.

        Args:
            accrual_start: Start date of the accrual period.
            accrual_end: End date of the accrual period.
            payment_lag: Number of business days between anchor date and payment.
            calendar: Business calendar used for payment-date adjustment.
            daycount: Day-count convention used for accrual calculation.
            rate: Fixed coupon rate.
            notional: Coupon notional amount.
            pay_in_arrear: Whether payment is made from accrual end; if
                ``False``, payment is anchored from accrual start.
        """
        self.accrual_start = accrual_start
        self.accrual_end = accrual_end
        self.payment_lag = payment_lag
        self.calendar = calendar
        self.daycount = daycount
        self.rate = rate
        self.notional = notional
        self.pay_in_arrear = pay_in_arrear

        self.tau = self.get_tau()
        payment_date = self.get_payment_date()

        self._validate()

        super().__init__(accrual_start, accrual_end, payment_date, daycount, notional)

    def _validate(self) -> None:
        """Validate data accuracy."""
        if self.accrual_start >= self.accrual_end:
            raise ValueError("`accrual_start` must be smaller than `accrual_end`.")
        if self.notional < 0:
            raise ValueError("`notional` must be positive.")

    def get_payment_date(self) -> Date:
        """Return the adjusted payment date for the fixed coupon."""
        if self.pay_in_arrear:
            if self.payment_lag == 0:
                return self.accrual_end
            return self.calendar.advance_business_days(
                self.accrual_end,
                self.payment_lag,
                BusinessDayConvention.FOLLOWING,
            )
        if self.payment_lag == 0:
            return self.accrual_start
        return self.calendar.advance_business_days(
            self.accrual_start,
            self.payment_lag,
            BusinessDayConvention.FOLLOWING,
        )

    def get_tau(self) -> float:
        """Return accrual year fraction for the coupon period."""
        return self.daycount.year_fraction(self.accrual_start, self.accrual_end)

    def get_amount(self) -> float:
        """Return the fixed coupon payment amount."""
        return self.rate * self.notional * self.tau

    def get_accrued_interest(self, valuation_date: Date) -> float:
        """Return accrued fixed interest up to ``valuation_date``."""
        tau = min(
            self.daycount.year_fraction(self.accrual_start, valuation_date), self.tau
        )
        return self.rate * self.notional * tau


class IborCoupon(Coupon):
    """Floating-rate coupon linked to an IBOR-style index fixing."""

    def __init__(
        self,
        accrual_start: Date,
        accrual_end: Date,
        payment_lag: int,
        fixing_lag: int,
        calendar: Calendar,
        daycount: DayCount,
        notional: float,
        index: InterestRateIndex,
        gearing: float = 1.0,
        spread: float = 0.0,
        fix_in_advance: bool = True,
    ) -> None:
        """Initialize an IBOR coupon.

        Args:
            accrual_start: Start date of the accrual period.
            accrual_end: End date of the accrual period.
            payment_lag: Number of business days between accrual end and payment.
            fixing_lag: Number of business days between fixing date and reset date.
            calendar: Business calendar used for date adjustments.
            daycount: Day-count convention.
            notional: Coupon notional amount.
            index: Floating-rate index used to source the fixing.
            gearing: Multiplier applied to the fixing rate.
            spread: Fixed spread added to the geared fixing.
            fix_in_advance: Whether fixing happens at period start or end.
        """
        self.index = index
        self.fix_in_advance = fix_in_advance
        self.fixing_lag = fixing_lag
        self.payment_lag = payment_lag
        self.calendar = calendar
        self.notional = notional
        self.daycount = daycount
        self.accrual_start = accrual_start
        self.accrual_end = accrual_end
        self.gearing = gearing
        self.spread = spread
        self.fixing_date = self.get_fixing_date()
        self.rate = self.get_rate()
        self.tau = self.get_tau()
        payment_date = self.get_payment_date()
        self._validate()
        super().__init__(accrual_start, accrual_end, payment_date, daycount, notional)

    def _validate(self) -> None:
        """Validate data accuracy."""
        if self.accrual_start >= self.accrual_end:
            raise ValueError("`accrual_start` must be smaller than `accrual_end`.")
        if self.notional < 0:
            raise ValueError("`notional` must be positive.")

    def get_tau(self) -> float:
        """Return accrual year fraction for the coupon period."""
        return self.daycount.year_fraction(self.accrual_start, self.accrual_end)

    def get_payment_date(self) -> Date:
        """Return the adjusted payment date."""
        if self.payment_lag == 0:
            return self.accrual_end
        return self.calendar.advance_business_days(
            self.accrual_end,
            self.payment_lag,
            BusinessDayConvention.FOLLOWING,
        )

    def get_fixing_date(self) -> Date:
        """Return the date on which the underlying rate is fixed."""
        if self.fix_in_advance:
            if self.fixing_lag == 0:
                return self.accrual_start
            return self.calendar.advance_business_days(
                self.accrual_start,
                -self.fixing_lag,
                BusinessDayConvention.PRECEDING,
            )
        if self.fixing_lag == 0:
            return self.accrual_end
        return self.calendar.advance_business_days(
            self.accrual_end,
            -self.fixing_lag,
            BusinessDayConvention.PRECEDING,
        )

    def get_rate(self) -> float:
        """Return the all-in coupon rate after gearing and spread."""
        fixing_rate = self.index.get_fixing(self.fixing_date)
        return self.gearing * fixing_rate + self.spread

    def get_amount(self) -> float:
        """Return the coupon payment amount."""
        return self.rate * self.notional * self.tau

    def get_accrued_interest(self, valuation_date: Date) -> float:
        """Return accrued interest up to ``valuation_date``."""
        tau = min(
            self.daycount.year_fraction(self.accrual_start, valuation_date), self.tau
        )
        return self.rate * self.notional * tau


class OvernightIndexCoupon(Coupon):
    """Coupon built from a sequence of overnight index fixings."""

    def __init__(
        self,
        accrual_start: Date,
        accrual_end: Date,
        payment_lag: int,
        lookback_period: int,
        lockout_period: int,
        observation_shift: bool,
        calendar: Calendar,
        daycount: DayCount,
        index: InterestRateIndex,
        average_method: AverageMethod,
        notional: float,
        gearing: float = 1.0,
        spread: float = 0.0,
    ) -> None:
        """Initialize an overnight-index coupon.

        Args:
            accrual_start: Start date of the accrual period.
            accrual_end: End date of the accrual period.
            payment_lag: Number of business days between accrual end and payment.
            lookback_period: Business-day lag applied to observation dates.
            lockout_period: Number of trailing fixings to freeze.
            observation_shift: Whether shifted observations also define weights.
            calendar: Business calendar used for schedule operations.
            daycount: Day-count convention.
            index: Overnight index used to source fixings.
            average_method: Rate aggregation method.
            notional: Coupon notional amount.
            gearing: Multiplier applied to the fixing rate.
            spread: Fixed spread added to the geared fixing.
        """
        observe_dates = self.get_observe_dates(
            calendar, accrual_start, accrual_end, lookback_period
        )
        time_weights = self.get_time_weights(
            observation_shift,
            daycount,
            observe_dates,
            calendar,
            accrual_start,
            accrual_end,
        )
        fixing_dates = self.get_fixing_dates(observe_dates, lockout_period)
        fixing_rates = self.get_fixing_rates(fixing_dates, index)
        compound_return = self.get_accumulate_rate(
            fixing_rates, time_weights, average_method
        )
        tau = self.get_tau(accrual_start, accrual_end, daycount)

        payment_date = calendar.advance_business_days(
            accrual_end, payment_lag, BusinessDayConvention.FOLLOWING
        )

        self.compound_return = compound_return
        self.rate = gearing * (compound_return / tau) + spread
        self.tau = tau
        self.time_weights = time_weights
        self.fixing_rates = fixing_rates
        self.observe_dates = observe_dates
        self.gearing = gearing
        self.spread = spread
        self.average_method: AverageMethod = average_method
        self.calendar = calendar

        super().__init__(
            accrual_start=accrual_start,
            accrual_end=accrual_end,
            payment_date=payment_date,
            daycount=daycount,
            notional=notional,
        )

    def get_amount(self) -> float:
        """Return the coupon payment amount."""
        return self.rate * self.notional * self.tau

    def get_accrued_interest(self, valuation_date: Date) -> float:
        """Return accrued coupon interest up to ``valuation_date``."""
        if valuation_date <= self.accrual_start:
            return 0.0

        accrual_days_passed = self.calendar.business_day_between(
            self.accrual_start, valuation_date
        )

        n = len(accrual_days_passed)

        slice_weights = np.array(self.time_weights[:n])
        slice_rates = np.array(self.fixing_rates[:n])

        accrued_return = self.get_accumulate_rate(
            slice_rates, slice_weights, self.average_method
        )

        accrued_tau = min(
            self.daycount.year_fraction(self.accrual_start, valuation_date), self.tau
        )

        return self.notional * (
            self.gearing * accrued_return + self.spread * accrued_tau
        )

    def get_tau(
        self, accrual_start: Date, accrual_end: Date, daycount: DayCount
    ) -> float:
        """Return accrual year fraction for the coupon period."""
        return daycount.year_fraction(accrual_start, accrual_end)

    def get_observe_dates(
        self,
        calendar: Calendar,
        accrual_start: Date,
        accrual_end: Date,
        lookback_period: int,
    ) -> list[Date]:
        """Return observation dates after applying the lookback rule."""
        observation_start = (
            calendar.advance_business_days(
                accrual_start,
                -lookback_period,
                BusinessDayConvention.PRECEDING,
            )
            if lookback_period > 0
            else accrual_start
        )

        observation_end = (
            calendar.advance_business_days(
                accrual_end,
                -lookback_period,
                BusinessDayConvention.PRECEDING,
            )
            if lookback_period > 0
            else accrual_end
        )

        observation_dates = calendar.business_day_between(
            observation_start, observation_end
        )

        return observation_dates

    def get_time_weights(
        self,
        observation_shift: bool,
        daycount: DayCount,
        observe_dates: Iterable[Date],
        calendar: Calendar,
        accrual_start: Date,
        accrual_end: Date,
    ) -> list[float]:
        """Return accrual year-fraction weights for each fixing."""
        time_weights: list[float] = []

        if observation_shift:
            for odate in observe_dates:
                prev_oday = calendar.advance_business_days(
                    odate, -1, BusinessDayConvention.PRECEDING
                )
                year_frac = daycount.year_fraction(prev_oday, odate)

                time_weights.append(year_frac)

        else:
            accrual_days = calendar.business_day_between(accrual_start, accrual_end)
            for adate in accrual_days:
                prev_day = calendar.advance_business_days(
                    adate, -1, BusinessDayConvention.PRECEDING
                )
                year_frac = daycount.year_fraction(prev_day, adate)

                time_weights.append(year_frac)
        return time_weights

    def get_fixing_dates(
        self, observe_dates: list[Date], lockout_period: int
    ) -> list[Date]:
        """Return effective fixing dates after applying lockout repetition."""
        if lockout_period == 0:
            return observe_dates

        lockout_date = observe_dates[-(lockout_period + 1)]
        return observe_dates[:-lockout_period] + [lockout_date] * lockout_period

    def get_fixing_rates(
        self, fixing_dates: Iterable[Date], index: InterestRateIndex
    ) -> list[float]:
        """Return index fixings for each fixing date."""
        return [index.get_fixing(d) for d in fixing_dates]

    def get_accumulate_rate(
        self,
        fixing_rates: Iterable[float],
        time_weight: Iterable[float],
        average_method: AverageMethod,
    ) -> float:
        """Aggregate overnight fixings into one accrual-period return."""
        frates = np.array(fixing_rates)
        tweights = np.array(time_weight)

        if average_method == "Compound-Average":
            return float(np.prod(1 + tweights * frates) - 1)
        if average_method == "Simple-Average":
            return float(np.sum(tweights * frates))
        raise ValueError("Unsupported average_method.")
