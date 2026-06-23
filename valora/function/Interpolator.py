"""Interpolator method."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Literal

import numpy as np
from scipy.interpolate import CubicSpline

InputX = int | float | Sequence[float]
BcType = Literal["not-a-knot", "natural", "clamped"]


class Interpolator(ABC):
    """Abstract class for Interpolation."""

    def fit(self, x: Sequence[float], y: Sequence[float]) -> Interpolator:
        """Fits data."""
        if len(x) != len(y):
            raise ValueError("x and y must have the same length.")
        if len(x) < 2:
            raise ValueError("At least 2 samples are required.")
        if list(x) != sorted(x):
            raise ValueError("x must be in ascending order.")

        self.x = np.array(x)
        self.y = np.array(y)
        self._fitted = True
        self._fit(self.x, self.y)
        return self

    @abstractmethod
    def _fit(self, x: Sequence[float], y: Sequence[float]) -> None: ...

    @abstractmethod
    def _predict(self, x_pred: float) -> None: ...

    def predict(self, x_pred: InputX):
        """Pedict y given x."""
        if not getattr(self, "_fitted", False):
            raise RuntimeError("Interpolator is not fitted yet.")
        if isinstance(x_pred, int) | isinstance(x_pred, float):
            return self._predict(x_pred)
        return [self._predict(xs) for xs in x_pred]


class LinearInterpolator(Interpolator):
    """Linear interpolation."""

    def __init__(self, extrapolate: bool = True):
        """Initialize LinearInterpolate."""
        self._fitted = False
        self.extrapolate = extrapolate

    def _fit(self, x: Sequence[float], y: Sequence[float]):
        pass

    def _predict(self, x_pred: float):
        x = self.x
        y = self.y
        extrapolate = self.extrapolate

        idx_exact = np.where(x == x_pred)[0]
        if len(idx_exact) > 0:
            return float(y[idx_exact[0]])

        # Extrapolate at left
        if x_pred < x[0]:
            if not extrapolate:
                raise ValueError(f"{x_pred} is below minimum x: {x[0]}")
            slope = (y[1] - y[0]) / (x[1] - x[0])
            return float(y[0] + slope * (x_pred - x[0]))

        # Extrapolate at right
        if x_pred > x[-1]:
            if not extrapolate:
                raise ValueError(f"{x_pred} is above maximum x: {x[-1]}")
            slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
            return float(y[-1] + slope * (x_pred - x[-1]))

        # Interpolate
        idx = int(np.searchsorted(x, x_pred, side="right")) - 1
        x0, x1 = x[idx], x[idx + 1]
        y0, y1 = y[idx], y[idx + 1]
        slope = (x_pred - x0) / (x1 - x0)
        return float(y0 + (y1 - y0) * slope)


class LogLinearInterpolator(Interpolator):
    """Log linear interpolation."""

    def __init__(self, extrapolate: bool = True):
        """Initialize LogLinearInterpolate."""
        self._fitted = False
        self.extrapolate = extrapolate

    def _fit(self, x: Sequence[float], y: Sequence[float]) -> None:
        if np.any(y <= 0) :
            raise ValueError("Log linear Interpolation requires ll y >= 0.")
        self.x = x
        self.log_y = np.log(y)

    def _predict(self, x_pred):
        x = self.x
        y = self.log_y
        extrapolate = self.extrapolate

        idx_exact = np.where(x == x_pred)[0]
        if len(idx_exact) > 0:
            return float(np.exp(y[idx_exact[0]]))

        # Extrapolate at left
        if x_pred < x[0]:
            if not extrapolate:
                raise ValueError(f"{x_pred} is below minimum x: {x[0]}")
            slope = (y[1] - y[0]) / (x[1] - x[0])
            return float(np.exp(y[0] + slope * (x_pred - x[0])))

        # Extrapolate at right
        if x_pred > x[-1]:
            if not extrapolate:
                raise ValueError(f"{x_pred} is above maximum x: {x[-1]}")
            slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
            return float(np.exp(y[-1] + slope * (x_pred - x[-1])))

        # Interpolate
        idx = int(np.searchsorted(x, x_pred, side="right")) - 1
        x0, x1 = x[idx], x[idx + 1]
        y0, y1 = y[idx], y[idx + 1]
        slope = (x_pred - x0) / (x1 - x0)
        return float(np.exp(y0 + (y1 - y0) * slope))


class CubicSplineInterpolator(Interpolator):
    """Cubic Spline Interpolation."""

    def __init__(self, bc_type: BcType = "natural", extrapolate: bool = True) -> None:
        """Initialize Cubic spline interpolator."""
        self._fitted = False
        self.bc_type = bc_type
        self.extrapolate = extrapolate

    def _fit(self, x: Sequence[float], y: Sequence[float]) -> float:
        """Fit y given x."""
        self._cs = CubicSpline(x, y, bc_type=self.bc_type, extrapolate=self.extrapolate)

    def _predict(self, x_pred) -> float:
        """Predict y given x."""
        return float(self._cs(x_pred))

    def derivative_1st(self, x_pred) -> float:
        """Compute 1st order of derivative given x.

        Args:
            x_pred: x given for 1st order derivative

        Returns:
            float: First order derivative
        """
        return float(self._cs(x_pred, 1))

    def derivative_2nd(self, x_pred) -> float:
        """Compute 2nd order of derivative given x.

        Args:
            x_pred: x given for 2nd order derivative

        Returns:
            float: Second order derivative
        """
        return float(self._cs(x_pred, 2))
