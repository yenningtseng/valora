"""Interest-rate and credit term-structure models and bootstrap helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import cast, Literal

import numpy as np
from scipy.interpolate import interp1d, CubicSpline

from .enum import Compounding
from .date import Date
from .daycount import DayCount
from ..function.Interpolator import Interpolator


InterMethod = Literal["linear", "cublic-spline"]

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
        tmp_curve = CubicSpline(tenors, zero_rates, extrapolate=True)

        def curve(t: float) -> float:
            return float(tmp_curve(t))

    elif interp == "linear":
        tmp_curve = interp1d(
            tenors,
            zero_rates,
            kind="linear",
            bounds_error=False,
            fill_value=cast(float, "extrapolate"),
        )

        def curve(t: float) -> float:
            return float(tmp_curve(t))

    else:
        raise ValueError(f"Unknown interpolation method: {interp}")

    return curve


class InterestTermStructure(ABC):
    """Basic element of interest rate term structure."""

    def __init__(self, reference_date: Date) -> None:
        """Initialize InterestTermStructure."""
        self.reference_date = reference_date

    def _get_tau(self, daycount: DayCount, dt1: Date, dt2: Date) -> float:
        """Computes the year fraction between two given days."""
        return float(daycount.year_fraction(dt1, dt2))

    @abstractmethod
    def get_zcb_tau(self, t1: float, t2: float) -> float:
        """Computes the zero-coupon bond price given two year fractions."""
        ...

    def get_zcb(self, daycount: DayCount, dt1: Date, dt2: Date) -> float:
        """Computes zcb price given two dates.

        Args:
            daycount (DayCount): Day count convention
            dt1 (Date): Begin date
            dt2 (Date): End date

        Returns:
            float: The discount factor or zcb price at notional of 1.
        """
        t1 = self._get_tau(daycount, self.reference_date, dt1)
        t2 = self._get_tau(daycount, self.reference_date, dt2)
        return self.get_zcb_tau(t1, t2)

    def get_spot_rate_tau(self, t1: float, compound: Compounding) -> float:
        """Get spot rate given two year fractions.

            - Continuous:
                R = -ln(P) / T
            - Simple:
                R = (1-P) / (T*P)
            - K-Annually:
                R = k * (P ^(-1/(K*T)) - 1)

        Args:
            t1 (float): year fraction of spot rate
            compound (Compounding): compound rule
        Returns:
            float: The calculated zero or spot rate.
        """
        p = self.get_zcb_tau(0, t1)
        if compound == Compounding.CONTINUOUS:
            return float(-np.log(p) / t1)
        elif compound == Compounding.SIMPLE:
            return float((1 - p) / (t1 * p))
        else:
            k = compound.value
            return float(k * (p ** (-1 / (k * t1)) - 1))

    def get_spot_rate(
        self, daycount: DayCount, dt1: Date, compound: Compounding
    ) -> float:
        """Get spot rate given a certain date.

            - Continuous:
                R = -ln(P) / T
            - Simple:
                R = (1-P) / (T*P)
            - K-Annually:
                R = k * (P ^(-1/(K*T)) - 1)

        Args:
            daycount (DayCount): Day count convention
            dt1 (float): year fraction of spot rate
            compound (Compounding): compound rule
        Returns:
            float: The calculated zero or spot rate.
        """
        p = self.get_zcb(self.reference_date, dt1)
        tau = daycount.year_fraction(self.reference_date, dt1)
        if tau <= 0:
            raise ValueError("Year fraction must be positive.")

        if compound == Compounding.CONTINUOUS:
            return float(-np.log(p) / tau)
        elif compound == Compounding.SIMPLE:
            return float((1 - p) / (tau * p))
        else:
            k = compound.value
            return float(k * (p ** (-1 / (k * tau)) - 1))

    def get_forward_rate_tau(self, t1: float, t2: float, compound: Compounding) -> float:
        """Computes forward rate given two year fraction.

        - Continuous:
            F = ln(ratio) * (1/T)

        - Simple:
            F = (ratio-1) * (1/T)

        - K-Annually:
            F = K * (ratio^(1/(K*T)) - 1)

        Args:
            t1 (float): year fraction 1
            t2 (float): year fraction 2
            compound (Compounding): compound rule
        Returns:
            float: Forward rate.
        """
        p01 = self.get_zcb_tau(0, t1)
        p02 = self.get_zcb_tau(0, t2)
        ratio = p01 / p02
        tau = t2 - t1
        if tau <= 0:
            raise ValueError("Year fraction must be positive.")

        if compound == Compounding.SIMPLE:
            return float((1 / tau) * (ratio - 1))
        elif compound == Compounding.CONTINUOUS:
            return float((1 / tau) * np.log(ratio))
        else:
            k = compound.value
            return float(k * (ratio ** (1 / (k * tau)) - 1))


    def get_forward_rate(
        self, daycount: DayCount, dt1: Date, dt2: Date, compound: Compounding
    ) -> float:
        """Get forward rate of given 2 days.

        - Continuous:
            F = ln(ratio) * (1/T)

        - Simple:
            F = (ratio-1) * (1/T)

        - K-Annually:
            F = K * (ratio^(1/(K*T)) - 1)

        Args:
            daycount (DayCount): Day count convention
            dt1 (Date): begin date
            dt2 (Date): end date
            compound (Compounding): compounding rule

        Returns:
            float: The calculated forward rate.
        """
        p01 = self.get_zcb(daycount, self.reference_date, dt1)
        p02 = self.get_zcb(daycount, self.reference_date, dt2)
        ratio = p01 / p02
        tau = self._get_tau(daycount, dt1, dt2)
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
    """Flat Forward interest curve."""

    def __init__(self, reference_date: Date, rate: float, compound: Compounding):
        """Initialize FlatForward."""
        self.rate = rate
        self.compound = compound
        super().__init__(reference_date)

    def get_zcb_tau(self, t1, t2):
        """Computes ZCB price.

        Args:
            t1 (float): year fraction of 1
            t2 (float): year fraction of 2
        Returns:
            float: ZCB price or discount factor.
        """
        tau = t2 - t1
        if tau < 0:
            raise ValueError("`t2` must be greater than or equal to `t1`.")

        if self.compound == Compounding.SIMPLE:
            return float(1.0 / (1.0 + self.rate * tau))
        elif self.compound == Compounding.CONTINUOUS:
            return float(np.exp(-self.rate * tau))
        else:
            k = self.compound.value
            return float((1.0 + self.rate / k) ** (-k * tau))


class InterpolatedZeroCurve(InterestTermStructure):

    def __init__(
        self,
        tenors: list[float],
        zrates: list[float],
        interp_method: Interpolator,
        reference_date: Date,
        compound: Compounding,
    ):
        self.tenors = tenors
        self.zrates = zrates

class InterpolatedZeroCurve(InterestTermStructure):
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
            compound (Compound): Compound method.
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
