"""Interest-rate and credit term-structure models and bootstrap helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import cast

import numpy as np
from scipy.interpolate import interp1d, CubicSpline
from scipy.optimize import brentq

from enum import (
    Compounding,
    PeriodType,
    Frequency,
    GenerationRule,
    BusinessDayConvention,
)

from .date import Date, Period
from .daycount import DayCount, Act360
from .schedule import Schedule
from .calendar import Calendar, FedWireCalendar


def _interp_to_curve(
    interp: str, tenors: list[float], dis_fac: list[float]
) -> Callable[[float], float]:
    """Interpolates to continuous discount factor curve.

    Args:
        interp (str): Interpolation method. Supported values are
            "cubic-spline", "log-linear".
        tenors (list[float]): List of tenors.
        dis_fac (list[float]): List of discount factors.

    Raises:
        ValueError: if `tenors` and `dis_fac` have different length.
        ValueError: if discount factor is negative for log-linear interpolation.
        ValueError: if unknown interpolation method is given.

    Returns:
        Callable[[float], float]: A function maps tenor to discount factor.
    """
    if len(tenors) != len(dis_fac):
        raise ValueError("`tenors` and `dis_fac` must have the same length.")

    if interp == "cubic-spline":
        tmp_curve = CubicSpline(
            tenors, dis_fac, extrapolate=True
        )  # TODO 若客戶反映，可能需要自建

        def curve(t: float) -> float:
            return float(tmp_curve(t))

    elif interp == "log-linear":
        if any(df <= 0 for df in dis_fac):
            raise ValueError(
                "Discount factors must be positive for log-linear interpolation."
            )
        log_dis = [np.log(i) for i in dis_fac]
        tmp_curve = interp1d(
            tenors,
            log_dis,
            kind="linear",
            bounds_error=False,
            fill_value=cast(float, "extrapolate"),
        )  # TODO 若客戶反映，可能需要自建

        def curve(t: float) -> float:
            return float(np.exp(tmp_curve(t)))

    else:
        raise ValueError(f"Unknown interpolation method: {interp}")

    return curve


def _inter_to_curve_by_zero_rate(
    interp: str, tenors: list[float], zero_rates: list[float]
) -> Callable[[float], float]:
    """Build an interpolating function over zero-rate pillars.

    Args:
        interp: Interpolation method. Supported values are ``"cubic-spline"``
            and ``"linear"``.
        tenors: Pillar tenors expressed as year fractions.
        zero_rates: Zero rates associated with ``tenors``.

    Raises:
        ValueError: If ``tenors`` and ``zero_rates`` have different lengths.
        ValueError: If ``interp`` is unsupported.

    Returns:
        Callable mapping a tenor to an interpolated zero rate.
    """
    if len(tenors) != len(zero_rates):
        raise ValueError("`tenors` and `dis_fac` must have the same length.")

    if interp == "cubic-spline":
        tmp_curve = CubicSpline(
            tenors, zero_rates, extrapolate=True
        )  # TODO 若客戶反映，可能需要自建

        def curve(t: float) -> float:
            return float(tmp_curve(t))

    elif interp == "linear":
        tmp_curve = interp1d(
            tenors,
            zero_rates,
            kind="linear",
            bounds_error=False,
            fill_value=cast(float, "extrapolate"),
        )  # TODO 若客戶反映，可能需要自建

        def curve(t: float) -> float:
            return float(tmp_curve(t))

    else:
        raise ValueError(f"Unknown interpolation method: {interp}")

    return curve


class InterestTermStructure(ABC):
    """Basic element of interest rate term structure.

    Attributes:
        reference_date(Date): Observation date of term structure.
        daycount(DayCount): Day count convention.
    """

    def __init__(self, reference_date: Date, daycount: DayCount) -> None:
        """Initialize InterestTermStructure.

        Args:
            reference_date (Date): Observation date of term structure.
            daycount (DayCount): Day count convention.
        """
        self.reference_date = reference_date
        self.daycount = daycount

    def _get_tau(self, dt: Date, dt1: Date) -> float:
        """Computes the year fraction between two given days.

        Args:
            dt (Date): Begin date
            dt1 (Date): End date

        Returns:
            float: Year fraction between two given date.
        """
        return float(self.daycount.year_fraction(dt, dt1))

    @abstractmethod
    def get_zcb_tau(self, t: float, t1: float) -> float:  # pragma: no cover
        """Computes the zero-coupon bond price at notional of 1.

        The price of ZCB at notional of 1 equalizes to discount factor.
        Overidden is needed for sub-classes.

        Args:
            t (float): Year fraction of evaluation date.
            t1 (float): Year fraction of maturity date.

        Raises:
            NotImplementedError: Must be overrided by sub-class.

        Returns:
            float: The price of ZCB or discount factor.
        """
        raise NotImplementedError("Subclasses must implement `get_zcb_tau`.")

    def get_zcb(self, dt: Date, dt1: Date) -> float:
        """Get zero-coupon bond price at notional of 1.

        It equalizes to the discount factor of given begin, end date.

        Args:
            dt (Date): Begin date
            dt1 (Date): End date

        Returns:
            float: The discount factor or zcb price at notional of 1.
        """
        t0 = self._get_tau(self.reference_date, dt)
        t1 = self._get_tau(self.reference_date, dt1)
        return self.get_zcb_tau(t0, t1)

    def get_spot_rate(self, dt: Date, dt1: Date, compound: Compounding) -> float:
        """Get zero (spot) rate of given two dates and compounding rule.

        The rate (R) is derived based on the compounding convention:

        - Continuous:
            R = -ln(P) / T

        - Simple:
            R = (1-P) / (T*P)

        - K-Annually:
            R = k * (P ^(-1/(K*T)) - 1)

        Where P is the price of ZCB between `dt` and `dt1`, T is the year fraction,
        and K represents the compounding frequency (e.g., 2 for semi-annual).

        Args:
            dt (Date): Begin date.
            dt1 (Date): End date.
            compound (Compounding): Compounding rule (Enum).

        Returns:
            float: The calculated zero or spot rate.
        """
        p = self.get_zcb(dt, dt1)
        tau = self._get_tau(dt, dt1)
        if tau <= 0:
            raise ValueError("Year fraction must be positive.")

        if compound == Compounding.CONTINUOUS:
            return float(-np.log(p) / tau)
        elif compound == Compounding.SIMPLE:
            return float((1 - p) / (tau * p))
        else:
            k = compound.value
            return float(k * (p ** (-1 / (k * tau)) - 1))

    def get_forward_rate(
        self, dt: Date, dt1: Date, dt2: Date, compound: Compounding
    ) -> float:
        """Get forward rate of given three days and compounding rule.

        The rate (F) is derived from ZCB prices P(dt, dt1) and P(dt, dt2)
        where dt <= dt1 <= dt2.

        - Continuous:
            F = ln(ratio) * (1/T)

        - Simple:
            F = (ratio-1) * (1/T)

        - K-Annually:
            F = K * (ratio^(1/(K*T)) - 1)

        Where ratio is P(dt, dt1) / P(dt, dt2), T is the year fraction, K represents
        the compounding frequency.

        Args:
            dt (Date): Evaluation date.
            dt1 (Date): Begin date.
            dt2 (Date): End date.
            compound (Compounding): Compounding rule (Enum).

        Returns:
            float: The calculated forward rate.
        """
        p01 = self.get_zcb(dt, dt1)
        p02 = self.get_zcb(dt, dt2)
        ratio = p01 / p02
        tau = self._get_tau(dt1, dt2)
        if tau <= 0:
            raise ValueError("Year fraction must be positive.")

        if compound == Compounding.SIMPLE:
            return float((1 / tau) * (ratio - 1))
        elif compound == Compounding.CONTINUOUS:
            return float((1 / tau) * np.log(ratio))
        else:
            k = compound.value
            return float(k * (ratio ** (1 / (k * tau)) - 1))


class FlatForward(InterestTermStructure):
    """Makes Flat Forward curve.

    Attributes:
        rate (float): Constant forward rate.
        reference_date (Date): Date to refer.
        daycount (DayCount): Day Count convention.
        rate_compounding (Compounding): Compounding convention.
    """

    def __init__(
        self,
        rate: float,
        reference_date: Date,
        daycount: DayCount,
        rate_compounding: Compounding,
    ) -> None:
        """Initialize Flatforward Curve.

        Args:
            rate (float): Constant forward rate.
            reference_date (Date): Date to refer.
            daycount (DayCount): Day Count convention.
            rate_compounding (Compounding): Compounding convention.
        """
        super().__init__(reference_date, daycount)
        self.rate = rate
        self.rate_compounding = rate_compounding

    def get_zcb_tau(self, t: float, t1: float) -> float:
        """Gets zero coupon bond price given year fraction.

        Args:
            t (float): Year fraction of begin date.
            t1 (float): Year fraction of end date.

        Raises:
            ValueError: `t1` must be greater than or equal to `t`.

        Returns:
            float: Price of zero coupon bond under nominal equals to 1.
        """
        tau = t1 - t
        if tau < 0:
            raise ValueError("`t1` must be greater than or equal to `t`.")

        if self.rate_compounding == Compounding.SIMPLE:
            return float(1.0 / (1.0 + self.rate * tau))
        elif self.rate_compounding == Compounding.CONTINUOUS:
            return float(np.exp(-self.rate * tau))
        else:
            k = self.rate_compounding.value
            return float((1.0 + self.rate / k) ** (-k * tau))


class InterpolatedZeroCurve(InterestTermStructure):
    """Generates interest rate term structure by given zero rates.

    Notices that zero rates are compounded differently accross different authorities.
    While the argument of `compond` represents the compounding rule for generated rates.

    Attributes:
        tenors (list[float]): A sequence of year fraction.
        zrates (list[float]): Correponding zero rates for each term.
        interp (str): Interpolation method to applied.
        reference_date (Date): Observation date of the curve.
        daycount (DayCount): Day count convention.
    """

    def __init__(
        self,
        tenors: list[float],
        zrates: list[float],
        interp: str,
        reference_date: Date,
        daycount: DayCount,
        compound: Compounding,
    ) -> None:
        """Initialize InterpolatedZeroCurve.

        Args:
            tenors (list[float]): A sequence of year fraction.
            zrates (list[float]): Correponding zero rates for each term.
            interp (str): Interpolation method to applied.
            reference_date (Date): Observation date of the curve.
            daycount (DayCount): Day count convention.
        """
        self.tenors = tenors
        self.zrates = zrates
        self.interp = interp
        self.compound = compound
        super().__init__(reference_date, daycount)
        self._check_data_quality()
        self.dis_fac: list[float] = self._get_dis_fac()
        self.nodes: dict = dict(zip(self.tenors, self.dis_fac, strict=True))
        self.zero_curve = _inter_to_curve_by_zero_rate(
            self.interp, self.tenors, self.zrates
        )

    def _check_data_quality(self) -> None:
        """Checks inputs data quality.

        Raises:
            ValueError: Numbers of `tenors` and `zrates` do not match.
            ValueError: `tenors` must be larger than 0.
            ValueError: Detect unusual term (>100).
            ValueError: Number of `tenors` is zero.
            ValueError: `tenors` is not strictly increment or is duplicated.
        """
        if len(self.tenors) != len(self.zrates):
            raise ValueError("Numbers of `tenors` and `zrates` do not match.")
        if np.any(np.array(self.tenors) < 0):
            raise ValueError("`tenors` must be larger than 0.")
        if np.any(np.array(self.tenors) > 100):
            raise ValueError("Detect unusual term (>100).")
        if len(self.tenors) == 0:
            raise ValueError("Number of `tenors` is zero.")
        if not all(
            x < y for x, y in zip(self.tenors[:-1], self.tenors[1:], strict=True)
        ):
            raise ValueError("`tenors` is not strictly increment or is duplicated.")

    def _get_dis_fac(self) -> list[float]:
        """Turns given zero rates to discount factors.

        Zero rates (R) turn to discount factors (P) in advance by different compounding.
        while T represents year fraction, and K represents compounding frequency.

        - Simple:
            1 / (1 + T*R)

        - Continuous
            e ^ (-R*T)

        - K-Annually
            (1 + R/K) ^ (-K*T)

        Returns:
            list: The computed discount factor.
        """
        tenor = np.array(self.tenors)
        rates = np.array(self.zrates)

        dis_facs = []
        for t, r in zip(tenor, rates, strict=True):
            if self.compound == Compounding.SIMPLE:
                dis_fac = float(1 / (1 + t * r))
            elif self.compound == Compounding.CONTINUOUS:
                dis_fac = float(np.exp(-r * t))
            else:
                k = self.compound.value
                dis_fac = (1 + r / k) ** (-k * t)

            dis_facs.append(dis_fac)

        return dis_facs

    def get_zcb_tau(self, t: float, t1: float) -> float:
        """Gets zero-coupon bond price at notional of 1 given year fraction.

        Interpolation method is used for computing ZCB price in specific tenors,
        while there are no zero rate data corresponding to those tenors.

        Two interpolation method is provided, one is 'linear',
        the other is 'cubic-spline'.

        Args:
            t (float): Year fraction of evaluation date.
            t1 (float): Year fraction of maturity date.

        Returns:
            float: The calculated price of ZCB.

        Raise:
            ValueError: `t` should be less than `t1`.
        """
        if t > t1:
            raise ValueError("`t` should be less than `t1`.")
        tau = t1 - t

        r = self.zero_curve(tau)

        if self.compound == Compounding.SIMPLE:
            dis_fac = float(1 / (1 + tau * r))
        elif self.compound == Compounding.CONTINUOUS:
            dis_fac = float(np.exp(-r * tau))
        else:
            k = self.compound.value
            dis_fac = (1 + r / k) ** (-k * tau)
        return float(dis_fac)


class BootstrapCurve(InterestTermStructure):
    """Yield curve constructed via bootstrapping from market instruments.

    This class builds a term structure by sequentially solving for discount
    factors that match market quotes of input instruments (BootstrapHelper).
    The curve is constructed iteratively, where each new node is solved such
    that the pricing error of the corresponding helper is zero.

    The resulting curve is represented as an interpolated function mapping
    year fraction (tenor) to discount factors.

    Attributes:
        helpers (list[BootstrapHelper]):
            Market instruments used for bootstrapping.
        interp (str):
            Interpolation method to construct discount curve.
        reference_date (Date):
            Observation date of the curve.
        daycount (DayCount):
            Day count convention used to convert discount curve
            to spot or forward curve.
        curve (Callable[[float], float]):
            Constructed curve.
    """

    def __init__(
        self,
        helpers: list[BootstrapHelper],
        interp: str,
        reference_date: Date,
        daycount: DayCount,
    ) -> None:
        """Initialize BootstrapCurve class.

        Args:
            helpers (list[BootstrapHelper]):
                Market instruments used for bootstrapping.
            interp (str):
                Interpolation method to construct discount curve.
            reference_date (Date):
                Observation date of the curve.
            daycount (DayCount):
                Day count convention used to convert discount curve
                to spot or forward curve.
        """
        super().__init__(reference_date, daycount)
        self.interp = interp
        self.helpers = sorted(helpers, key=lambda x: x.maturity_date)
        self.nodes: dict[float, float] = {0.0: 1.0}
        self._bootstrap()

        tenors = list(self.nodes.keys())
        dis_fac = list(self.nodes.values())
        self.curve: Callable[[float], float] = _interp_to_curve(interp, tenors, dis_fac)

    def _bootstrap(self) -> None:
        """Performs iterative bootstrapping of the curve.

        For each helper, this method:
            1. sets reference date
            2. computes time-to-maturity
            3. defines an objective function based on pricing error
            4. solves for discount factor using brentq

        The interated discount factor is appended to `self.nodes`, and
        subsequent instruments use the updated curve.
        """
        for helper in self.helpers:
            helper._fetch_reference_date(self.reference_date)
            new_tau = self._get_tau(self.reference_date, helper.maturity_date)

            lower = 1e-8
            upper = max(next(reversed(self.nodes.values())) * 1.5, 2.0)

            def object(
                x: float, new_tau: float = new_tau, helper: BootstrapHelper = helper
            ) -> float:
                """Objective function for root-finding.

                Args:
                    x (float):
                        Candidate discount factor at `new_tau`.
                    new_tau (float, optional):
                        Next time-to-maturity. Defaults to `new_tau`.
                    helper (BootstrapHelper, optional):
                        Helpers for bootstrap. Defaults to helper.

                Returns:
                    float: Error between model-implied price and quote.
                """
                taus = list(self.nodes.keys()) + [new_tau]
                dis_fac = list(self.nodes.values()) + [x]

                curve = _interp_to_curve(self.interp, taus, dis_fac)
                return cast(float, helper.error(curve))

            # 檢查 BrentQ 條件
            f_lo = object(lower)
            f_up = object(upper)
            if f_lo * f_up > 0:
                raise ValueError(
                    f"Failed to bracket root for product={helper.product}, "
                    f"maturity={helper.maturity_date}, quote={helper.quote}."
                )

            x_star = cast(float, brentq(object, lower, upper, xtol=1e-12, maxiter=100))
            self.nodes[new_tau] = x_star

    def get_zcb_tau(self, t: float, t1: float) -> float:
        """Gets ZCB price between two times.

        Args:
            t (float): Start time.
            t1 (float): End time.

        Returns:
            float: The discount factor from `t` to `t1`.
        """
        p0 = self.curve(t)
        p1 = self.curve(t1)

        return float(p1 / p0)


class BootstrapHelper(ABC):
    """Abstract base class for bootstrapping helpers.

    A BootstrapHelper represents a market instruments (e.g., deposit, swap, bond)
    used in the construction of a yield curve. Each helper provides an error function
    that measures the mispricing between curve-implied price and market quote.

    Attributes:
        product (str): The name of instrument.
        quote (float): Market quote of instrument.
        daycount (DayCount): Day coount convention of instrument.
        frequency (Frequency): coupon frequency of instrument.
        maturity_date (Date): Maturity date of instrument.
    """

    def __init__(
        self,
        product: str,
        quote: float,
        daycount: DayCount,
        frequency: Frequency,
        maturity_date: Date,
    ) -> None:
        """Initializes BootstrapHelper instance.

        Args:
            product (str): The name of instrument.
            quote (float): Market quote of instrument.
            daycount (DayCount): Day coount convention of instrument.
            frequency (Frequency): coupon frequency of instrument.
            maturity_date (Date): Maturity date of instrument.
        """
        self.product = product
        self.quote = quote
        self.daycount = daycount
        self.frequency = frequency
        self.maturity_date = maturity_date

    def _fetch_reference_date(self, reference_date: Date) -> None:
        """Assigns the reference date for each helper.

        This method would be called in the initialization stage at
        the `BootstrapCurve`.

        Args:
            reference_date (Date):
                Reference date of constructed yield curve.
        """
        self.reference_date: Date = reference_date

    @abstractmethod
    def error(self, curve: Callable[[float], float]) -> float:
        """Computes the error between curve-implied and market price.

        Args:
            curve (Callable[[float], float]):
                A function mapping year fraction to discount factor from curve.

        Returns:
            float: Error between model-implied price and quote.

        Raises:
            NotImplementedError: Implemented is needed.
        """
        raise NotImplementedError


class ZeroCouponBondHelper(BootstrapHelper):
    """Bootstrap helper for zero-coupon instruments quoted by yield.

    The helper converts a quoted zero-coupon yield into an implied pricing
    equation for the discount factor at maturity. The compounding convention
    is inferred from the tenor unit:

    - non-yearly tenors use simple compounding
    - yearly tenors use annual compounding
    """

    def __init__(
        self,
        quote: float,
        quote_date: Date,
        tenor: Period,
        daycount: DayCount,
    ):
        """Initialize a zero-coupon bond bootstrap helper.

        Args:
            quote: Market-quoted zero-coupon yield.
            quote_date: Observation date of the quote.
            tenor: Time from quote date to maturity.
            daycount: Day-count convention used to interpret the quote.
        """
        maturity_date = quote_date + tenor
        self.tenor_year = tenor.number / tenor.period_type.value
        self.quote_date = quote_date
        if tenor.period_type != PeriodType.YEARLY:
            self.compound = Compounding.SIMPLE
        else:
            self.compound = Compounding.ANNUAL
        super().__init__(
            "zero-coupon-bond", quote, daycount, Frequency.ONCE, maturity_date
        )

    def error(self, curve: Callable[[float], float]) -> float:
        """Return the difference between market quote and model-implied yield."""
        tau = self.daycount.year_fraction(self.reference_date, self.maturity_date)
        dis_fac = float(curve(tau))

        if self.compound == Compounding.SIMPLE:
            theo_price = (1 / tau) * (1 / dis_fac - 1)
        if self.compound == Compounding.ANNUAL:
            theo_price = dis_fac ** (-1 / tau) - 1

        return self.quote - theo_price


class ParYieldHelper(BootstrapHelper):
    """Bootstrap helper for instruments quoted with par yield.

    This helper represents a fix-rate par bond, where coupon rate equals the
    market-quoted par yield and the bond price is assumed to be par (1.0).

    The helper constructs a coupon schedule based on a given tenor (in years)
    and payment frequency, and computes the pricing error as the difference
    between model-implied bond price and par.

    Notes:
        - Assumes regular coupon schedule (no stub period).
        - Assumes integer year tenors.
        - Assumes constant accrual per period: alpha = 1 / frequency
        - Final cash flow included both coupon and principal.
        - The `quote` is interpreted as an annualized par yield in decimal form.

    Attributes:
        quote (float): Annualized par yield (in decimal form).
        quote_date (Date): Observation date of market quote.
        tenor (int): Maturities in year. Must be integer.
        frequency (Frequency): Coupon frequency convention.
    """

    def __init__(
        self,
        quote: float,
        quote_date: Date,
        daycount: DayCount,
        tenor: int,
        frequency: Frequency,
    ) -> None:
        """Initializes ParYieldHelper instance.

        Args:
            quote (float): Annualized par yield (in decimal form).
            quote_date (Date): Observation date of market quote.
            daycount (DayCount): Day Count convention of par rate.
            tenor (int): Maturities in year. Must be integer.
            frequency (Frequency): Coupon frequency convention.

        Raises:
            TypeError: `frequency` is not `Frequency` enum.
            ValueError: `Tenor` less than 1.
        """
        product = "Par-Yield"
        maturity_date = quote_date + Period(PeriodType.YEARLY, tenor)

        if not isinstance(frequency, Frequency):
            raise TypeError("Incorrect `frequency`, must be Frequency enum.")

        super().__init__(product, quote, daycount, frequency, maturity_date)

        if tenor < 1:
            raise ValueError(
                "`Tenor` less than 1, uses `DepositHelper` as alternative."
            )
        self.tenor = tenor

    def handle_coupon_schedule(self) -> list[float]:
        """Computes year fraction of coupon pay date.

        Raises:
            ValueError: Par yield at Frequency.ONCE can be viewes as zero-coupon bond.

        Returns:
            list[float]: Year fraction after computation.
        """
        if self.frequency == Frequency.ONCE:
            raise ValueError(
                "Coupon occurs only at maturity is seemed as zero-coupon bond, "
                "no needs to bootstrap."
            )
        elif self.frequency == Frequency.ANNUAL:
            scheduled_tau = [float(i) for i in range(1, self.tenor)]
        else:
            k: float = self.frequency.value
            scheduled_tau = np.arange(1 / k, self.tenor, 1 / k, dtype=float)

        return list(scheduled_tau)

    def error(self, curve: Callable[[float], float]) -> float:
        """Computes the difference of model-implied price and quote.

        Args:
            curve (Callable[[float], float]): Curve model for theortical price.

        Returns:
            float: Error.

        Raises:
            ValueError: Par yield at Frequency.ONCE can be viewes as zero-coupon bond.
        """
        scheduled_tau = self.handle_coupon_schedule()

        if self.frequency == Frequency.ONCE:
            raise ValueError(
                "Coupon occurs only at maturity is seemed as zero-coupon bond, "
                "no needs to bootstrap."
            )
        elif self.frequency == Frequency.ANNUAL:
            alpha = 1.0
        else:
            alpha = 1.0 / self.frequency.value

        pv_coupon = sum(self.quote * alpha * curve(t) for t in scheduled_tau)
        pv_notional = (1.0 + self.quote * alpha) * curve(self.tenor)
        model_price = float(pv_coupon + pv_notional)
        market_price = 1.0

        return market_price - model_price


class FuturesRateHelper(BootstrapHelper):
    """Bootstrap helper for futures pricing.

    This helper interprets a futures quote as an implied forward rate over
    a period [start_date, end_date], and converts it into a constraint on
    the discount curve.

    A futures constract implies a forward rate over [t, T]:

        F = (1 / tau) * (DF(0, t)/ DF(0, T) - 1)

    where:
        - t = start_date.
        - T = end_date.
        - tau = year fraction between t, T.

    With convexity adjustment:

        futures_rate = F + convex_adj

    The market quote is:

        quote_rate = (100 - quote) / 100

    The helper returns pricing error:

        model_futures_rate - market_quote_rate.

    Attributes:
        convex_adj (float):
            Convexity adjustments to convert forward rate to futures rate.
        quote (float):
            Market quote of futures price, quoted in hundred dollar.
        daycount (DayCount):
            Day count convention of futures (SOFR: Act360).
        start_date (Date):
            Start of the accrual period (t).
        end_date (Date):
            End of the accrual period (T).
    """

    def __init__(
        self,
        convex_adj: float,
        quote: float,
        daycount: DayCount,
        start_date: Date,
        end_date: Date,
    ) -> None:
        """Initialize FuturesRateHelper.

        Args:
            convex_adj (float):
                Convexity adjustments to convert forward rate to futures rate.
            quote (float):
                Market quote of futures price, quoted in hundred dollar.
            daycount (DayCount):
                Day count convention of futures (SOFR: Act360).
            start_date (Date):
                Start of the accrual period (t), IMM start date.
            end_date (Date):
                End of the accrual period (T), IMM end date.

        Raises:
            ValueError: `start_date` should be earlier than `end_date`.
        """
        product = "Futures"
        frequency = Frequency.ONCE
        maturity_date = end_date
        super().__init__(product, quote, daycount, frequency, maturity_date)

        if start_date >= end_date:
            raise ValueError("`start_date` should be ealier than `end_date`.")

        self.convex_adj = convex_adj
        self.start_date = start_date
        self.end_date = end_date

    def error(self, curve: Callable[[float], float]) -> float:
        """Computes the error between model-implied futures rate and market quote.

        Args:
            curve (Callable[[float], float]):
                Curve model for theortical price.

        Returns:
            float: Error.
        """
        tau1 = self.daycount.year_fraction(self.reference_date, self.start_date)
        tau2 = self.daycount.year_fraction(self.reference_date, self.end_date)
        tau = self.daycount.year_fraction(self.start_date, self.end_date)

        df1 = float(curve(tau1))
        df2 = float(curve(tau2))
        ratio = df1 / df2

        if tau <= 0:
            raise ValueError("Year fraction must be positive.")

        forward_rate = float((1 / tau) * (ratio - 1))
        futures_rate = forward_rate + self.convex_adj
        quote_rate = (100.0 - self.quote) / 100.0

        return futures_rate - quote_rate


class DepositRateHelper(BootstrapHelper):
    """Bootstrap helper for deposit.

    Includes O/N, T/N rate.

    Attributes:
        quote (float): The market rate of deposit.
        daycount (DayCount): Day count convention.
        start_date (Date): Accrual begin date.
        end_date (Date): Accrual end date.
    """

    def __init__(
        self, quote: float, daycount: DayCount, start_date: Date, end_date: Date
    ) -> None:
        """Initializes DepositRateHelper.

        Args:
            quote (float): The market rate of deposit.
            daycount (DayCount): Day count convention.
            start_date (Date): Begin date.
            end_date (Date): End date.
        """
        product = "Deposit"
        frequency = Frequency.ONCE
        self.start_date = start_date
        self.end_date = end_date
        super().__init__(product, quote, daycount, frequency, end_date)

    def error(self, curve: Callable[[float], float]) -> float:
        """Computes the error between model-implied deposit rate and market quotes.

        Args:
            curve (Callable[[float], float]):
                Curve model for theoretical price.

        Returns:
            float: Error.
        """
        if self.start_date == self.reference_date:
            self.product = "O/N"
            tau = self.daycount.year_fraction(self.reference_date, self.end_date)
            dis_fac = float(curve(tau))
            price = (1 / dis_fac - 1) / tau

            return price - self.quote

        tau = self.daycount.year_fraction(self.start_date, self.end_date)
        tau1 = self.daycount.year_fraction(self.reference_date, self.start_date)
        tau2 = self.daycount.year_fraction(self.reference_date, self.end_date)

        dis_fac_1 = float(curve(tau1))
        dis_fac_2 = float(curve(tau2))
        dis_fac_f = dis_fac_2 / dis_fac_1
        price = (1 / dis_fac_f - 1) / tau

        return price - self.quote


class OisRateHelper(BootstrapHelper):
    """Bootstrap helper for OIS par-rate instruments.

    This helper builds the accrual and payment schedules of an overnight
    indexed swap (OIS) from trade-date conventions, then computes the
    pricing error between the curve-implied par fixed rate and the market
    quote.

    The fixed leg is valued as the discounted sum of coupon accrual factors,
    while the floating leg is approximated by ``1 - P(0, T)`` under standard
    single-curve bootstrapping assumptions.
    """

    def __init__(
        self,
        quote: float,
        daycount: DayCount,
        frequency: Frequency,
        tenor: Period,
        calendar: Calendar,
        trade_date: Date,
        spot_lag: int,
        payment_lag: int,
        holiday_rule: BusinessDayConvention,
    ) -> None:
        """Initialize an OIS bootstrap helper.

        The helper constructs accrual and payment schedules from trade-date
        conventions, then uses those schedules to price an OIS instrument
        during curve bootstrapping.

        Args:
            quote (float): Market-quoted OIS fixed rate in decimal form.
            daycount (DayCount): Day count convention used for accrual fractions.
            frequency (Frequency): Coupon payment frequency of the fixed leg.
            tenor (Period): Overall swap tenor.
            calendar (Calendar): Calendar used for business-day adjustments.
            trade_date (Date): Trade date of the quoted OIS instrument.
            spot_lag (int): Number of business days from trade date to accrual start.
            payment_lag (int): Number of business days from accrual end dates to
                payment dates.
            holiday_rule (BusinessDayConvention): Business-day adjustment rule
                applied when generating schedules.
        """
        self.tenor = tenor
        self.calendar = calendar
        self.trade_date = trade_date
        self.spot_lag = spot_lag
        self.payment_lag = payment_lag
        self.holiday_rule = holiday_rule

        if not spot_lag == 0:
            accrual_start = calendar.advance_business_days(
                trade_date, spot_lag, holiday_rule
            )
        else:
            accrual_start = calendar.advance(trade_date, holiday_rule)
        accrual_end = calendar.advance(accrual_start + tenor, holiday_rule)
        coupon_tenor = self._handle_coupon_tenor(frequency)

        accrual_schedule = Schedule(
            begin_date=accrual_start,
            end_date=accrual_end,
            tenor=coupon_tenor,
            calendar=calendar,
            holiday_rule=holiday_rule,
            generation_rule=GenerationRule.FORWARD,
            end_of_month_rule=False,
            begin_end_inclusion=(True, True),
        )

        if payment_lag == 0:
            payment_schedule = accrual_schedule
        else:
            payment_schedule = accrual_schedule.advance_business_days(
                payment_lag, holiday_rule
            )
        payment_schedule.drop(payment_schedule[0], inplace=True)
        maturity_date = cast(Date, payment_schedule[-1])

        super().__init__(
            product="OIS",
            quote=quote,
            daycount=daycount,
            frequency=frequency,
            maturity_date=maturity_date,
        )

        self.payment_schedule = payment_schedule
        self.accrual_schedule = accrual_schedule

    def _handle_coupon_tenor(self, frequency: Frequency) -> Period:
        """Convert coupon frequency into the schedule tenor used for coupons.

        Args:
            frequency (Frequency): Coupon frequency of the OIS fixed leg.

        Returns:
            Period: Interval between consecutive accrual dates.

        Raises:
            ValueError: If the input frequency is not supported.
        """
        if frequency == Frequency.ONCE:
            return self.tenor
        if frequency == Frequency.ANNUAL:
            return Period(PeriodType.YEARLY, 1)
        if frequency in (Frequency.SEMIANNUAL, Frequency.QUARTERLY, Frequency.MONTHLY):
            return Period(PeriodType.MONTHLY, int(12 / frequency.value))
        if frequency == Frequency.WEEKLY:
            return Period(PeriodType.WEEKLY, 1)
        raise ValueError("Input frequency is not supported.")

    def error(self, curve: Callable[[float], float]) -> float:
        """Compute the pricing error of the OIS helper.

        Args:
            curve: Function mapping year fraction from the reference date to a
                discount factor.

        Returns:
            float: Curve-implied par OIS fixed rate minus the market quote.

        Raises:
            ValueError: If the generated accrual schedule is inconsistent with
                ``Frequency.ONCE``.
        """
        accrual_schedule = self.accrual_schedule
        payment_schedule = self.payment_schedule

        accrual_dates: list[Date] = accrual_schedule.mapped_date
        payment_dates: list[Date] = payment_schedule.mapped_date

        if self.frequency == Frequency.ONCE:
            if len(accrual_dates) > 2:
                raise ValueError(
                    "Length of `accrual_schedule` must be 2 under `Frequency.ONCE`."
                )

        tau_coupons = [
            self.daycount.year_fraction(i, j)
            for i, j in zip(accrual_dates[:-1], accrual_dates[1:], strict=True)
        ]
        tau_maturitys = [
            self.daycount.year_fraction(self.reference_date, i) for i in payment_dates
        ]
        dis_facs = [curve(i) for i in tau_maturitys]
        denom = sum(i * j for i, j in zip(tau_coupons, dis_facs, strict=True))

        theo_price = (1 - dis_facs[-1]) / denom

        return theo_price - self.quote


class SofrOisRateHelper(OisRateHelper):
    """Bootstrap helper for Sofr Ois.

    Market convention of SOFR OIS:
        1. spot lag = 0
        2. payment lag = T+2
        3. Day count = Actual/360
        4. Calendar = Fed
        5. Business day convention = Modified following

    Attributes:
        quote (float): Market quote of swap rate.
        frequency (Frequency): Coupon frequency of swap.
        tenor (Period): Swap Tenor.
        trade_date (Date): Trade date of swap.
    """

    def __init__(
        self,
        quote: float,
        frequency: Frequency,
        tenor: Period,
        trade_date: Date,
    ) -> None:
        """Initialize SofrOisRateHelper.

        Args:
            quote (float): Market quote of swap rate.
            frequency (Frequency): Coupon frequency of swap.
            tenor (Period): Swap Tenor.
            trade_date (Date): Trade date of swap.
        """
        super().__init__(
            quote,
            Act360(),
            frequency,
            tenor,
            FedWireCalendar(),
            trade_date,
            0,
            2,
            BusinessDayConvention.MODIFIED_FOLLOWING,
        )


class CreditSpreadedCurve(InterestTermStructure):
    """Credit-spreaded Yield curve construction.

    Combines a risk-free interest rate term structure with a credit term structure
    ro produce a credit-adjusted discount factor, based on reduced-from pricing:

        P_risky(t, t1) = P_rf(t, t1) * [R + (1-R) * S(t, t1)]

    Attributes:
        interest_curve (InterestTermStructure): Risk-free interest rate
            term structure used to compute the base discount factor.
        credit_curve (CreditTermStructure): Credit term structure used
            to compute the survival probability.
        recovery_rate (float): Fraction of face value recovered upon
            default, between 0 and 1. Defaults to 0.4 (40%).
    """

    def __init__(
        self,
        interest_curve: InterestTermStructure,
        credit_curve: CreditTermStructure,
        recovery_rate: float = 0.4,
    ) -> None:
        """Initialize CreditSpreadedCurve.

        Args:
            interest_curve (InterestTermStructure): Risk-free interest rate
            term structure used to compute the base discount factor.
            credit_curve (CreditTermStructure): Credit term structure used
            to compute the survival probability.
            recovery_rate (float, optional): Fraction of face value recovered upon
            default, between 0 and 1. Defaults to 0.4 (40%).

        Raises:
            ValueError: `reference_date` of both curves must be identical.
            ValueError: `recovery_rate` must be between 0 to 1.
        """
        if interest_curve.reference_date != credit_curve.reference_date:
            raise ValueError("`reference_date` of both curves must be identical.")
        if (recovery_rate > 1) | (recovery_rate < 0):
            raise ValueError("`recovery_rate` must be between 0 to 1.")
        self.interest_curve = interest_curve
        self.credit_curve = credit_curve
        self.recovery_rate = recovery_rate
        super().__init__(interest_curve.reference_date, interest_curve.daycount)

    def get_zcb_tau(self, t: float, t1: float) -> float:
        """Compute the credit-adjusted ZCB price given year fractions.

        Applies the recovery-of-face-value formula:

            P_risky(t, t1) = P_rf(t, t1) * [R + (1 - R) * S(t, t1)]

        Args:
            t (float): Year fraction of the begin date from reference date.
            t1 (float): Year fraction of the maturity date from reference date.

        Returns:
            float: Credit-adjusted ZCB price, equivalent to the risky
                discount factor between t and t1.
        """
        base_zcb = self.interest_curve.get_zcb_tau(t, t1)
        surv_prob = self.credit_curve.get_survive_prob_tau(t, t1)

        return base_zcb * (self.recovery_rate + (1 - self.recovery_rate) * surv_prob)


class CreditTermStructure(ABC):
    """Abstract base class for credit term structure.

    Provides the interface and common methods for computing survival
    probabilities, default probabilities, and marginal default probabilities
    from a credit term structure. Subclasses must implement
    `get_survive_prob_tau` to define the underlying credit model.

    The survival probability S(t, t1) represents the risk-neutral probability
    that a counterparty does not default between year fractions t and t1,
    following the reduced-form framework of Jarrow & Turnbull (1995) and
    Duffie & Singleton (1999):

        S(t, t1) = exp(-integral from t to t1 of lambda(s) ds)

    where lambda(s) is the hazard rate process.

    Attributes:
        reference_date (Date): Observation date of the term structure.
        daycount (DayCount): Day count convention used for year fraction
            calculation.
    """

    def __init__(self, reference_date: Date, daycount: DayCount) -> None:
        """Initialize CreditTermStructure.

        Args:
            reference_date (Date): Observation date of term structure.
            daycount (DayCount): Day count convention used for year fraction
            calculation.
        """
        self.reference_date = reference_date
        self.daycount = daycount

    def _get_tau(self, dt: Date, dt1: Date) -> float:
        """Computes the year fraction between two given days.

        Args:
            dt (Date): Begin date
            dt1 (Date): End date

        Returns:
            float: Year fraction between two given date.
        """
        return float(self.daycount.year_fraction(dt, dt1))

    @abstractmethod
    def get_survive_prob_tau(self, t: float, t1: float) -> float:  # pragma: no cover
        """Computes the survival probability under year fraction.

        Overidden is needed for sub-classes.

        Args:
            t (float): Year fraction of evaluation date.
            t1 (float): Year fraction of maturity date.

        Raises:
            NotImplementedError: Must be overrided by sub-class.

        Returns:
            float: The survival probability.
        """
        raise NotImplementedError("Subclasses must implement `get_zcb_tau`.")

    def get_survive_prob(self, dt: Date, dt1: Date) -> float:
        """Computes the survival probability under calendar date.

        Args:
            dt (Date): Begin date
            dt1 (Date): End date

        Returns:
            float: The survival probability.
        """
        t0 = self._get_tau(self.reference_date, dt)
        t1 = self._get_tau(self.reference_date, dt1)
        return self.get_survive_prob_tau(t0, t1)

    def get_default_prob(self, dt: Date, dt1: Date) -> float:
        """Computes the default probability under calendar date.

        Args:
            dt (Date): Begin date.
            dt1 (Date): End date.

        Returns:
            float: The default probability
        """
        survive_prob = self.get_survive_prob(dt, dt1)
        return 1.0 - survive_prob

    def get_marginal_default_prob(self, dt: Date, dt1: Date, dt2: Date) -> float:
        """Computes the marginal default probability under calendar date.

        marginal_PD(dt1, dt2) = S(dt, dt1) - S(dt, dt2)

        Args:
            dt (Date): The reference date.
            dt1 (Date): The begin date.
            dt2 (Date): The end date.

        Returns:
            float: Marginal default rate.
        """
        t = self._get_tau(self.reference_date, dt)
        t1 = self._get_tau(self.reference_date, dt1)
        t2 = self._get_tau(self.reference_date, dt2)

        s1 = self.get_survive_prob_tau(t, t1)
        s2 = self.get_survive_prob_tau(t, t2)

        return s1 - s2


class FlatHazard(CreditTermStructure):
    """Credit term structure with a constant (flat) hazard rate.

    Implements the simplest reduced-form credit model where the hazard rate
    lambda is assumed to be constant over time. The survival probability is
    then given analytically by:

        S(t, t1) = exp(-lambda * (t1 - t))

    Attributes:
        hazard_rate (float): Constant hazard rate lambda, representing the
            instantaneous conditional default intensity.
        reference_date (Date): Observation date of the term structure.
        daycount (DayCount): Day count convention used for year fraction
            calculation.
    """

    def __init__(
        self,
        hazard_rate: float,
        reference_date: Date,
        daycount: DayCount,
    ):
        """Initialize FlatHazardCurve.

        Args:
            hazard_rate (float): Constant hazard rate lambda. Must be
                non-negative.
            reference_date (Date): Observation date of the term structure.
            daycount (DayCount): Day count convention.
        """
        self.hazard_rate = hazard_rate
        super().__init__(reference_date, daycount)

    def get_survive_prob_tau(self, t: float, t1: float) -> float:
        """Compute the survival probability given year fractions.

        Under a constant hazard rate assumption, the survival probability
        is given by the exponential decay formula:

            S(t, t1) = exp(-lambda * (t1 - t))

        Args:
            t (float): Year fraction of the begin date from reference date.
            t1 (float): Year fraction of the maturity date from reference date.

        Returns:
            float: Survival probability between year fractions t and t1.
        """
        lda = self.hazard_rate
        return np.exp(-lda * (t1 - t))


class InterpolatedDefaultCurve(CreditTermStructure):
    """Credit term structure built from interpolated cumulative default probabilities.

    Constructs a survival probability curve by interpolating over a set of
    cumulative default probabilities at discrete tenors. The interpolation
    method is configurable, allowing different assumptions about the shape
    of the hazard rate curve between pillars.

    A anchor point of (0, 0) is automatically prepended to the cumulative
    default probabilities if the minimum tenor is greater than 0, ensuring
    S(0) = 1.

    Attributes:
        tenors (list[float]): Year fractions of the pillar points. A zero
            anchor point is prepended automatically if not provided.
        c_pds (list[float]): Cumulative default probabilities at each pillar
            tenor. A zero anchor point is prepended automatically if not
            provided.
        interp (str): Interpolation method identifier passed to
            `_interp_to_curve`. Determines the shape of the hazard rate
            curve between pillars (e.g., log-linear, cubic spline).
        reference_date (Date): Observation date of the term structure.
        daycount (DayCount): Day count convention used for year fraction
            calculation.
        survival_rate (list[float]): Survival probabilities derived from
            cumulative default probabilities as S(T) = 1 - PD(0, T).
        curve (callable): Interpolated curve function mapping year fraction
            tau to survival probability S(tau).
    """

    def __init__(
        self,
        tenors: list[float],
        c_pds: list[float],
        interp: str,
        reference_date: Date,
        daycount: DayCount,
    ) -> None:
        """Initialize InterpolatedDefaultCurve.

        Args:
            tenors (list[float]): Year fractions of pillar points in strictly
                ascending order. Must be non-negative and less than 100.
            c_pds (list[float]): Cumulative default probabilities PD(0, T)
                at each pillar tenor. Must have the same length as tenors.
            interp (str): Interpolation method identifier. Passed to
                `_interp_to_curve` to construct the survival probability curve.
            reference_date (Date): Observation date of the term structure.
            daycount (DayCount): Day count convention.

        Raises:
            ValueError: If tenors and c_pds have different lengths.
            ValueError: If any tenor is negative.
            ValueError: If any tenor exceeds 100.
            ValueError: If tenors is empty.
            ValueError: If tenors are not strictly increasing or contain
                duplicates.
        """
        self.tenors = tenors
        self.c_pds = c_pds
        self.interp = interp
        super().__init__(reference_date, daycount)

        self._validate()

        self.survival_rate = self._get_survival_rate()
        self.curve = _interp_to_curve(self.interp, self.tenors, self.survival_rate)

    def _validate(self) -> None:
        """Validate inputs and prepend zero anchor point if necessary.

        Ensures tenors and c_pds are consistent in length, tenors are
        strictly increasing, and all values are within expected bounds.
        If the minimum tenor is greater than 0, prepends (0, 0) to both
        tenors and c_pds to anchor S(0) = 1.

        Raises:
            ValueError: If any validation condition is violated.
        """
        if len(self.tenors) != len(self.c_pds):
            raise ValueError("Numbers of `tenors` and `c_pds` do not match.")
        if np.any(np.array(self.tenors) < 0):
            raise ValueError("`tenors` must be larger than 0.")
        if np.any(np.array(self.tenors) > 100):
            raise ValueError("Detect unusual tenors (>100).")
        if len(self.tenors) == 0:
            raise ValueError("Number of `tenors` is zero.")
        if not all(
            x < y for x, y in zip(self.tenors[:-1], self.tenors[1:], strict=True)
        ):
            raise ValueError("`tenors` is not strictly increment or is duplicated.")
        if np.min(self.tenors) > 0.0:
            self.tenors = [0.0] + self.tenors
            self.c_pds = [0.0] + self.c_pds

    def _get_survival_rate(self) -> list[float]:
        """Convert cumulative default probabilities to survival probabilities.

        Applies S(T) = 1 - PD(0, T) to each pillar point.

        Returns:
            list[float]: Survival probabilities corresponding to each tenor.
        """
        return [1.0 - c_pd for c_pd in self.c_pds]

    def get_survive_prob_tau(self, t: float, t1: float) -> float:
        """Compute the survival probability given year fractions.

        Evaluates the interpolated survival probability curve at the
        forward tenor tau = t1 - t.

        Args:
            t (float): Year fraction of the begin date from reference date.
            t1 (float): Year fraction of the maturity date from reference date.

        Returns:
            float: Survival probability between year fractions t and t1.
        """
        tau = t1 - t
        return self.curve(tau)
