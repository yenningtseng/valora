"""Interest-rate and credit term-structure models and bootstrap helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np

from .enum import Compounding, Frequency
from .date import Date
from .daycount import DayCount
from ..function.interpolator import Interpolator


InterMethod = Literal["linear", "cublic-spline"]


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
        p = self.get_zcb(daycount, self.reference_date, dt1)
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

    def get_forward_rate_tau(
        self, t1: float, t2: float, compound: Compounding
    ) -> float:
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
    """Curve composed of interpolated zero rates."""

    def __init__(
        self,
        tenors: list[float],
        zrates: list[float],
        interpator: Interpolator,
        reference_date: Date,
        compound: Compounding,
    ) -> None:
        self._check_data_quality(tenors, zrates, interpator, reference_date, compound)
        super().__init__(reference_date)
        self.tenors = tenors
        self.zrates = zrates
        self.compound = compound
        self.interpator = self._interpolate(tenors, zrates, interpator)
        self.nodes = self._build_nodes(tenors, zrates)

    def get_zcb_tau(self, t1: float, t2: float) -> float:
        """Compute discount factor given two year fractions.

        Args:
            t1: year fraction of dt1
            t2: year fraction of dt2
        Returns:
            float: Discount factor.
        """
        if t1 > t2:
            raise ValueError("t1 should be less than t2.")
        r1 = self.interpator.predict(t1)
        r2 = self.interpator.predict(t2)

        if self.compound == Compounding.SIMPLE:
            d1 = 1 / (1 + t1 * r1)
            d2 = 1 / (1 + t2 * r2)
        elif self.compound == Compounding.CONTINUOUS:
            d1 = np.exp(-r1 * t1)
            d2 = np.exp(-r2 * t2)
        else:
            k = self.compound.value
            d1 = (1 + r1 / k) ** (-k * t1)
            d2 = (1 + r2 / k) ** (-k * t2)
        return float(d2 / d1)

    def _interpolate(
        self, tenors: list[float], zrates: list[float], interpator: Interpolator
    ) -> Interpolator:
        """Interpolate zero rates."""
        return interpator.fit(tenors, zrates)

    def _build_nodes(
        self,
        tenors,
        zrates,
    ) -> dict:
        """Build node with tenors and zrates."""
        return dict(zip(tenors, zrates, strict=True))

    def _check_data_quality(
        self,
        tenors: list[float],
        zrates: list[float],
        interpator: Interpolator,
        reference_date: Date,
        compound: Compounding,
    ) -> None:
        """Ensure data quality."""
        if len(tenors) != len(zrates):
            raise ValueError("tenors and zrates must be the same length.")
        if np.min(tenors) < 0:
            raise ValueError(f"Invalid tenor: {np.min(tenors)}.")
        if np.max(tenors) > 100:
            raise ValueError(f"Invalid tenor: {np.max(tenors)}.")
        if np.max(zrates) > 1:
            raise ValueError(f"zrates must be below 1, get: {np.max(zrates)}.")
        if not all(a < b for a, b in zip(tenors[:-1], tenors[1:], strict=True)):
            raise ValueError("tenors must be ascending order.")
        if not isinstance(interpator, Interpolator):
            raise TypeError(f"Unsupported type of interp_method: {type(interpator)}.")
        if not isinstance(reference_date, Date):
            raise TypeError(
                f"Unsupported type of reference_date: {type(reference_date)}."
            )
        if not isinstance(compound, Compounding):
            raise TypeError(f"Unsupported type of compound: {type(compound)}.")


class InterpolatedDiscountCurve(InterestTermStructure):
    """Curve composed of interpolated discount factors."""

    def __init__(
        self,
        tenors: list[float],
        dfactors: list[float],
        interpator: Interpolator,
        reference_date: Date,
    ) -> None:
        self._check_data_quality(tenors, dfactors, interpator, reference_date)
        super().__init__(reference_date)
        self.tenors = tenors
        self.dfactors = dfactors
        self.interpator = self._interpolate(tenors, dfactors, interpator)
        self.nodes = self._build_nodes(tenors, dfactors)

    def get_zcb_tau(self, t1: float, t2: float) -> float:
        """Compute discount factor given two year fractions.

        Args:
            t1: year fraction of dt1
            t2: year fraction of dt2
        Returns:
            float: Discount factor.
        """
        if t1 > t2:
            raise ValueError("t1 should be less than t2.")
        d1 = self.interpator.predict(t1)
        d2 = self.interpator.predict(t2)
        return float(d2 / d1)

    def _interpolate(
        self, tenors: list[float], dfactors: list[float], interpator: Interpolator
    ) -> Interpolator:
        """Interpolate discount factors."""
        return interpator.fit(tenors, dfactors)

    def _build_nodes(
        self,
        tenors,
        dfactors,
    ) -> dict:
        """Build node with tenors and dfactors."""
        return dict(zip(tenors, dfactors, strict=True))

    def _check_data_quality(
        self,
        tenors: list[float],
        dfactors: list[float],
        interpator: Interpolator,
        reference_date: Date,
    ) -> None:
        """Ensure data quality."""
        if len(tenors) != len(dfactors):
            raise ValueError("tenors and zrates must be the same length.")
        if np.min(tenors) < 0:
            raise ValueError(f"Invalid tenor: {np.min(tenors)}.")
        if np.max(tenors) > 100:
            raise ValueError(f"Invalid tenor: {np.max(tenors)}.")
        if np.max(dfactors) > 1:
            raise ValueError(f"zrates must be below 1, get: {np.max(dfactors)}.")
        if not all(a < b for a, b in zip(tenors[:-1], tenors[1:], strict=True)):
            raise ValueError("tenors must be ascending order.")
        if not isinstance(interpator, Interpolator):
            raise TypeError(f"Unsupported type of interp_method: {type(interpator)}.")
        if not isinstance(reference_date, Date):
            raise TypeError(
                f"Unsupported type of reference_date: {type(reference_date)}."
            )
