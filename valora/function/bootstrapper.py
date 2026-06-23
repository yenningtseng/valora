"""Boostrap interest-rate curve tools."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from scipy.optimize import brentq

from ..basic.term_structure import InterestTermStructure, InterpolatedDiscountCurve
from ..basic.daycount import (
    DayCount,
    Act360,
    Act365Fixed,
    Thirty360US,
    ThirtyA360,
    ThirtyE360,
)
from ..basic.date import Period, Date
from ..basic.enum import Frequency, PeriodType
from .interpolator import Interpolator


class BootstrapEngine:
    """Bootstrap engine for multiple instruments."""

    def __init__(
        self,
        reference_date: Date,
        helpers: list[BootstrapHelper],
        interpolator: Interpolator,
    ) -> None:
        """Initialize BootstrapEngine."""
        self.reference_date = reference_date
        self.helpers = helpers
        self.interpolator = interpolator
        self.tenors = [0.0]
        self.dfactors = [1.0]

    def _check_helpers(self, helper: BootstrapHelper) -> None:
        """Check helper quality."""
        if not isinstance(helper, BootstrapHelper):
            raise TypeError(f"Unsupported helper type: {type(helper)}.")
        if self.reference_date != helper.reference_date:
            raise ValueError(
                f"Reference date don't match, must be: {self.reference_date}."
            )

    def add_helper(self, helper: BootstrapHelper) -> None:
        """Append helper."""
        self.helpers.append(helper)

    def remove_helper(self, idx: int) -> None:
        """Remove helper."""
        del self.helpers[idx]

    def _piecewise_bootstrap(self, helper):
        """Bootstrap curve piecewisely."""
        tenors = self.tenors.copy()
        dfactors = self.dfactors.copy()

        def objective(guess, tenors, dfactors, helper):
            trial_tenors = tenors + [helper.tenor]
            trial_dfactors = dfactors + [guess]

            self.interpolator.fit(trial_tenors, trial_dfactors)

            return helper.error(self.interpolator)

        added_tenor = helper.tenor
        added_dfactor = brentq(objective, 1e-18, 1.0, args=(tenors, dfactors, helper))

        return added_tenor, added_dfactor

    def run(self) -> InterestTermStructure:
        """Run bootstrap."""
        helpers = sorted(self.helpers, key=lambda x: x.tenor)
        cnt = 0
        for helper in helpers:
            self._check_helpers(helper)
            added_tenor, added_dfactor = self._piecewise_bootstrap(helper)
            self.tenors.append(added_tenor)
            self.dfactors.append(added_dfactor)
            cnt += 1

        return InterpolatedDiscountCurve(
            self.tenors,
            self.dfactors,
            self.interpolator,
            self.reference_date,
        )


class BootstrapHelper(ABC):
    """Abstract class for bootstrapped instrument."""

    def __init__(self, quote: float, tenor: float, reference_date: Date) -> None:
        """Initialize BootstrapHelper."""
        self.quote = quote
        self.tenor = tenor
        self.reference_date = reference_date

    @abstractmethod
    def error(self, curve):
        """Compute pricing error."""
        ...


class ParYieldHelper(BootstrapHelper):
    """Bootstrapped par yield instruments."""

    def __init__(
        self,
        quote: float,
        tenor: Period,
        frequency: Frequency,
        reference_date: Date,
        daycount: Optional[DayCount] = None,
    ) -> None:
        """Initialize ParYieldHelper."""
        self._check_data_quality(quote, tenor, frequency, daycount)
        tenor = self._regularize_tenor(daycount, tenor)
        super().__init__(quote, tenor, reference_date)
        self.quote = quote
        self.frequency = frequency
        self.daycount = daycount

    def _check_data_quality(
        self,
        quote: float,
        tenor: Period,
        frequency: Frequency,
        daycount: DayCount,
    ) -> None:
        if np.max(quote) > 1:
            raise ValueError(f"Invalid par yield quote: {quote}.")
        if not isinstance(tenor, Period):
            raise TypeError(f"Unsupported tenor type: {type(tenor)}.")
        if not isinstance(frequency, Frequency):
            raise TypeError(f"Unsupported frequency type: {type(frequency)}.")
        if not isinstance(daycount, DayCount) and daycount is not None:
            raise TypeError(f"Unsupported daycount type: {type(daycount)}.")
        if tenor.period_type == PeriodType.DAILY and daycount is None:
            raise ValueError("daycount must provided while using daily par yield.")

    def _regularize_tenor(self, daycount: DayCount, tenor: Period) -> float:
        if tenor.period_type == PeriodType.DAILY:
            if daycount in [Act365Fixed]:
                return float(tenor.value / 365)
            elif daycount in [Thirty360US, ThirtyA360, ThirtyE360, Act360]:
                return float(tenor.value / 360)
        return float(tenor.value / tenor.period_type.value)

    def _get_implied_price(self, curve: Interpolator) -> float:
        df = curve.predict(self.tenor)

        if self.frequency == Frequency.ONCE:
            return (1 / df - 1) / self.tenor
        else:
            step = 1 / self.frequency.value
            taus = np.arange(step, self.tenor + step, step)
            dfs = np.array([curve.predict(t) for t in taus])
            annuity = np.sum(dfs * step)
            return float((1 - df) / annuity)

    def error(self, curve: Interpolator) -> float:
        """Compute pricing error of par yield."""
        return self._get_implied_price(curve) - self.quote


class FuturesHelper(BootstrapHelper):
    def __init__(
        self,
        quote,
        reference_date,
        beg_date,
        end_date,
        daycount,
        sigma = 0,
    ):
        self.t1, self.t2 = self._regularize_tenor(
            reference_date, beg_date, end_date, daycount
        )
        super().__init__(quote, self.t2, reference_date)
        self.beg_date = beg_date
        self.end_date = end_date
        self.daycount = daycount
        self.sigma = sigma

    def _regularize_tenor(
        self,
        reference_date,
        beg_date,
        end_date,
        daycount,
    ):
        tenor1 = daycount.year_fraction(reference_date, beg_date)
        tenor2 = daycount.year_fraction(reference_date, end_date)

        return tenor1, tenor2

    def _convexity_adjustment(
        self,
        sigma,
        t1,
        t2,
    ):
        return 0.5 * t1 * t2 * sigma**2

    def error(self, curve:Interpolator):
        tau = self.t2 - self.t1
        ca = self._convexity_adjustment(self.sigma, self.t1, self.t2)
        d1 = curve.predict(self.t1)
        d2 = curve.predict(self.t2)

        frate = (1/tau) * (d1/d2-1) - ca
        t_price = (1 - frate) * 100

        return t_price - self.quote


class OisHelper(BootstrapHelper):

    def __init__(
        self,
        quote,
        reference_date: Date,
        tenor: Period,
        fix_frequency: Frequency,
        float_frequency: Frequency,
        daycount: DayCount = None,
    ):
        self._check_data_quality(quote, tenor, daycount)
        tenor = self._regularize_tenor(daycount, tenor)
        super().__init__(quote, tenor, reference_date)
        self.fix_frequency = fix_frequency
        self.float_frequency = float_frequency
        self.daycount = daycount

    def _check_data_quality(
        self,
        quote: float,
        tenor: Period,
        daycount: DayCount,
    ) -> None:
        if np.max(quote) > 1:
            raise ValueError(f"Invalid par yield quote: {quote}.")
        if not isinstance(tenor, Period):
            raise TypeError(f"Unsupported tenor type: {type(tenor)}.")
        if not isinstance(daycount, DayCount) and daycount is not None:
            raise TypeError(f"Unsupported daycount type: {type(daycount)}.")
        if tenor.period_type == PeriodType.DAILY and daycount is None:
            raise ValueError("daycount must provided while using daily par yield.")

    def _regularize_tenor(self, daycount: DayCount, tenor: Period) -> float:
        if tenor.period_type == PeriodType.DAILY:
            if daycount in [Act365Fixed]:
                return float(tenor.value / 365)
            elif daycount in [Thirty360US, ThirtyA360, ThirtyE360, Act360]:
                return float(tenor.value / 360)
        return float(tenor.value / tenor.period_type.value)

    def _get_schedule(self, tenor: float, frequency: Frequency):
        step = 1 / frequency.value
        if step > tenor:
            raise ValueError("term to maturity is less than frequency period.")
        return step, np.arange(step, tenor + step, step)

    def _get_par_swap_rate(self, curve):
        fix_step, fix_schedule = self._get_schedule(self.tenor, self.fix_frequency)
        _, float_schedule = self._get_schedule(self.tenor, self.float_frequency)

        fix_dfactors = np.array([curve.predict(t) for t in fix_schedule])
        float_dfactors = np.array([curve.predict(t) for t in float_schedule])

        denom = np.sum(fix_dfactors * fix_step)
        num = 1 - float_dfactors[-1]

        return float(num/denom)

    def error(self, curve):
        return self._get_par_swap_rate(curve) - self.quote
