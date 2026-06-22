"""Cash-flow primitives and vectorised leg transformations."""
from __future__ import annotations

from typing import TypeAlias, Any, Literal, cast
from collections.abc import Callable, Iterable, Iterator, Mapping

import numpy as np

from .date import Date
from .schedule import Schedule

Scalar: TypeAlias = int | float
DateLike: TypeAlias = Date | int
AmountLike: TypeAlias = Scalar | Iterable[Scalar] | np.ndarray
DateUpdateLike: TypeAlias = Schedule | Iterable[DateLike]

class CashFlow:
    """Single dated cash flow.

    Occurrence dates are stored internally as integer serial numbers and
    exposed externally as :class:`Date` objects.

    Attributes:
        _occur_date_int: Serial number of the occurrence date.
        amount: Cash amount paid on that date.
    """

    @staticmethod
    def _to_int(dt: DateLike) -> int:
        """Convert a date-like value to its serial number.

        Args:
            dt: Date object or serial number.

        Returns:
            Serial number representation of ``dt``.
        """
        if isinstance(dt, Date):
            return dt.serial_number
        return int(dt)

    @staticmethod
    def _to_date(v: int) -> Date:
        """Convert a serial number to a ``Date`` object.

        Args:
            v: Serial number to convert.

        Returns:
            ``Date`` object represented by ``v``.
        """
        base = Date(1, 1, 1950)
        real = base + int(v)
        return real

    def __init__(self, dt: DateLike, amount: float) -> None:
        """Initialize a cash flow.

        Args:
            dt: Occurrence date.
            amount: Cash amount paid on ``dt``.
        """
        self._occur_date_int = CashFlow._to_int(dt)
        self.amount = float(amount)

    @property
    def occur_date(self) -> Date:
        """Return the occurrence date of the cash flow as a ``Date`` object.

        Returns:
            Occurrence date in ``Date`` form.
        """
        return CashFlow._to_date(self._occur_date_int)

    def __add__(self, other: CashFlow | Legs) -> Legs:
        """Combine this cash flow with another cash flow or a leg object.

        If 'other' is a CashFlow, a new Legs object containing both cash flows.
        When two cash flows have the same occur date, their cash amount are summed.

        If 'other' is a Legs, this cash flow is appended to it.
        Cash flows with identical occur dates are aggregating by summing their amounts.

        The resulting Legs object is sorted by the occur date.

        Args:
            other: Cash flow container to combine with this cash flow.

        Returns:
            A Legs object sorted by occur date.

        Raises:
            TypeError: Arises when the type of 'other' is neither CashFlow nor Legs.
        """
        if isinstance(other, CashFlow):
            return Legs(
                [self._occur_date_int, other._occur_date_int],
                [self.amount, other.amount],
                sort=True,
            )
        if isinstance(other, Legs):
            return Legs(
                np.append([self._occur_date_int], other._occur_date_int),
                np.append([self.amount], other.amount),
                sort=True,
            )
        raise TypeError

    def __sub__(self, other: CashFlow | Legs) -> Legs:
        """Subtract another cash flow or leg object from this cash flow.

        If `other` is a CashFlow, a new Legs object is returned containing
        this cash flow and the negated cash flow of `other`.
        If both cash flows share the same occurrence date, their amounts
        are combined accordingly.

        If `other` is a Legs object, the cash flows in `other` are negated
        and merged with this cash flow. Cash flows with identical occurrence
        dates are aggregated by summing their amounts. The resulting Legs
        object is sorted by occurrence date.

        Args:
            other: Cash flow container to subtract from this cash flow.

        Returns:
            A Legs object sorted by occur date.

        Raises:
            TypeError: Arises when the type of 'other' is neither CashFlow nor Legs.
        """
        if isinstance(other, CashFlow):
            return Legs(
                [self._occur_date_int, other._occur_date_int],
                [self.amount, -other.amount],
                sort=True,
            )
        if isinstance(other, Legs):
            return Legs(
                np.append([self._occur_date_int], other._occur_date_int),
                np.append([self.amount], -other.amount),
                sort=True,
            )
        raise TypeError

    def __radd__(self, other: CashFlow | Legs | int) -> Legs:
        """Define right-handed addition for cash flow objects.

        This enables usage with built-in functions such as ``sum()``.
        When ``other`` is zero, this cash flow is converted into a ``Legs``
        object. Otherwise, addition is delegated to ``__add__``.

        Args:
            other: The left-hand operand.

        Returns:
            A Legs object sorted by occur date.
        """
        if other == 0:
            return Legs([self._occur_date_int], [self.amount], sort=False)
        if isinstance(other, int):
            raise TypeError("`other` must be 0, CashFlow, or Legs.")
        return self.__add__(other)


class Legs:
    """Vector-like container of dated cash flows.

    Duplicate dates can be merged by summing amounts. The object supports
    NumPy-style transformations over amounts while preserving date alignment.

    Attributes:
        _occur_date_int: A sequence of serial numbers transformed from occur dates.
        amount: A sequence of cash amounts correspond to occur dates.
    """

    @staticmethod
    def _to_int(dt: DateLike) -> int:
        """Convert a date-like value to its serial number.

        Args:
            dt: Date object or serial number.

        Returns:
            Serial number representation of ``dt``.
        """
        if isinstance(dt, Date):
            return dt.serial_number
        return dt

    @staticmethod
    def _to_date(v: int) -> Date:
        """Convert a serial number to a ``Date`` object.

        Args:
            v: Serial number to convert.

        Returns:
            ``Date`` object represented by ``v``.
        """
        base = Date(1, 1, 1950)
        real = base + int(v)
        return real

    def __init__(
        self,
        occur_date: Iterable[DateLike],
        amount: Iterable[Scalar],
        sort: bool = True,
    ) -> None:
        """Create a leg object from occur dates and cash amounts.

        Args:
            occur_date: Sequence of occurrence dates.
            amount: Sequence of cash amounts corresponding to ``occur_date``.
            sort: Whether to sort and merge entries with the same occurrence date.

        Raises:
            ValueError: If ``occur_date`` and ``amount`` have different lengths.
        """
        self._occur_date_int = np.asarray(
            [Legs._to_int(x) for x in occur_date], dtype=np.int64
        )
        self.amount = np.asarray(amount, dtype=float)

        if len(self._occur_date_int) != len(self.amount):
            raise ValueError

        if sort:
            self._merge_same_date()

    def _merge_same_date(self) -> None:
        """Sort by date and sum amounts sharing the same occurrence date."""
        order = np.argsort(self._occur_date_int)
        d = self._occur_date_int[order]
        a = self.amount[order]

        uniq, idx = np.unique(d, return_inverse=True)
        summed = np.bincount(idx, weights=a)

        self._occur_date_int = uniq
        self.amount = summed

    @property
    def occur_date(self) -> list[Date]:
        """Return occurrence dates as ``Date`` objects.

        Returns:
            Occurrence dates in ``Date`` form.
        """
        return [Legs._to_date(v) for v in self._occur_date_int]

    def __len__(self) -> int:
        """Compute the length of this Legs object.

        Returns:
            The total number of occur dates in this Legs object.
        """
        return len(self._occur_date_int)

    def __array__(self, dtype: np.dtype[Any] | None = None) -> np.ndarray:
        """Return a NumPy array view of the cash amounts.

        It enables implicit conversion of a Legs object to a NumPy
        array, allowing it to be used directly in NumPy operations.

        Args:
            dtype: Desired NumPy data type.

        Returns:
            A NumPy array with cash amounts.
        """
        return self.amount.astype(dtype) if dtype else self.amount

    def __getitem__(self, idx: int) -> CashFlow:
        """Return the cash flow at the specified index.

        Indexing a Legs object yields a CashFlow instance corresponding
        to the selected occur date and cash amount.

        Args:
            idx: Index of the desired cash flow.

        Returns:
            Cash flow at ``idx``.
        """
        return CashFlow(Legs._to_date(self._occur_date_int[idx]), self.amount[idx])

    def __iter__(self) -> Iterator[CashFlow]:
        """Iterate over the cash flows in this Legs object.

        Each iteration yields a CashFlow instance containing the
        occurrence date and corresponding cash amount.

        Yields:
            CashFlow objects in occurrence-date order.
        """
        for d, a in zip(self._occur_date_int, self.amount, strict=True):
            yield CashFlow(Legs._to_date(d), a)

    def __add__(self, other: Legs | CashFlow) -> Legs:
        """Combine this object with another cash-flow container.

        Args:
            other: ``Legs`` or ``CashFlow`` to merge into this object.

        Returns:
            New ``Legs`` instance sorted and merged by occurrence date.
        """
        other_dt = np.atleast_1d(other._occur_date_int)
        other_amt = np.atleast_1d(other.amount)
        return Legs(
            np.concatenate([self._occur_date_int, other_dt]),
            np.concatenate([self.amount, other_amt]),
            True,
        )

    def __sub__(self, other: Legs | CashFlow) -> Legs:
        """Subtract another cash-flow container from this object.

        Args:
            other: ``Legs`` or ``CashFlow`` to subtract.

        Returns:
            New ``Legs`` instance sorted and merged by occurrence date.
        """
        other_dt = np.atleast_1d(other._occur_date_int)
        other_amt = np.atleast_1d(-other.amount)
        return Legs(
            np.concatenate([self._occur_date_int, other_dt]),
            np.concatenate([self.amount, other_amt]),
            True,
        )

    def __radd__(self, other: Legs | CashFlow | int) -> Legs:
        """Support right-hand addition, including ``sum()``.

        Args:
            other: The left-hand operand.

        Returns:
            Result of combining ``other`` with this object.
        """
        if other == 0:
            return self
        if isinstance(other, int):
            raise TypeError("`other` must be 0, Legs, or CashFlow.")
        return self.__add__(other)

    @staticmethod
    def inplacement(func: Callable[..., Legs | float]) -> Callable[..., Legs | float]:
        """Decorate methods that optionally modify ``Legs`` in place.

        This decorator wraps a method that normally returns a new ``Legs``
        object and adds support for an ``inplace`` keyword argument. When
        ``inplace=True``, the current instance is updated in place.

        Args:
            func: Method returning either a new ``Legs`` object or a scalar.

        Returns:
            Wrapped callable supporting an ``inplace`` keyword argument.
        """

        def wrapper(
            self, *args: Any, inplace: bool = False, **kwargs: Any
        ) -> Legs | float:
            """Execute the wrapped operation with optional in-place behavior.

            Args:
                self: Current Legs instance.
                *args: Positional arguments passed to the wrapped method.
                inplace: Whether to modify the current instance.
                **kwargs: Keyword arguments to the wrapped method.

            Returns:
                Either a scalar result or a ``Legs`` object.
            """
            new_leg = func(self, *args, **kwargs)
            if inplace:
                if not isinstance(new_leg, Legs):
                    raise TypeError(
                        "`inplace=True` is only supported for Legs results."
                    )
                self.amount = new_leg.amount
                # pylint: disable=protected-access
                self._occur_date_int = new_leg._occur_date_int
                return self
            return new_leg

        return wrapper

    @inplacement
    def add(self, value: Scalar) -> Legs:
        """Add the same scalar to every amount.

        Args:
            value: Scalar value to add.

        Returns:
            Updated ``Legs`` instance.
        """
        return Legs(self._occur_date_int, self.amount + value, False)

    @inplacement
    def subtract(self, value: Scalar) -> Legs:
        """Subtract the same scalar from every amount.

        Args:
            value: Scalar value to subtract.

        Returns:
            Updated ``Legs`` instance.
        """
        return Legs(self._occur_date_int, self.amount - value, False)

    @inplacement
    def multiply(self, value: Scalar) -> Legs:
        """Multiply every amount by the same scalar.

        Args:
            value: Scalar value to multiply.

        Returns:
            Updated ``Legs`` instance.
        """
        return Legs(self._occur_date_int, self.amount * value, False)

    @inplacement
    def divide(self, value: Scalar) -> Legs:
        """Divide every amount by the same scalar.

        Args:
            value: Scalar value to divide by.

        Returns:
            Updated ``Legs`` instance.
        """
        return Legs(self._occur_date_int, self.amount / value, False)

    @inplacement
    def max(self, array: AmountLike | None = None) -> float | Legs:
        """Return a scalar maximum or element-wise maxima.

        # noqa: DAR003

        If ``array`` is omitted, the maximum amount in the leg is returned.
        Otherwise an element-wise comparison is performed.

        Args:
            array: Values to compare against each amount. If ``None``,
                the maximum cash amount is returned.

        Returns:
            Maximum cash amount if ``array`` is ``None``; otherwise a new
            ``Legs`` object containing the element-wise maximum.
        """
        if array is None:
            return float(np.max(self.amount))
        compare = np.asarray(array, dtype=float)
        new_amount = np.maximum(self.amount, compare)
        return Legs(self._occur_date_int, new_amount, False)

    @inplacement
    def min(self, array: AmountLike | None = None) -> float | Legs:
        """Return a scalar minimum or element-wise minima.

        If ``array`` is omitted, the minimum amount in the leg is returned.
        Otherwise an element-wise comparison is performed.

        # noqa: DAR003

        Args:
            array: Values to compare against each amount. If ``None``,
                the minimum cash amount is returned.

        Returns:
            Minimum cash amount if ``array`` is ``None``; otherwise a new
            ``Legs`` object containing the element-wise minimum.
        """
        if array is None:
            return float(np.min(self.amount))
        compare = np.asarray(array, dtype=float)
        new_amount = np.minimum(self.amount, compare)
        return Legs(self._occur_date_int, new_amount, False)

    @inplacement
    def cumsum(self) -> Legs:
        """Return cumulative sums of the leg amounts.

        Returns:
            Updated ``Legs`` instance.
        """
        new_amount = np.add.accumulate(self.amount)
        return Legs(self._occur_date_int, new_amount, False)

    @inplacement
    def cumprod(self) -> Legs:
        """Return cumulative products of the leg amounts.

        Returns:
            Updated ``Legs`` instance.
        """
        new_amount = np.multiply.accumulate(self.amount)
        return Legs(self._occur_date_int, new_amount, False)

    @inplacement
    def update(
        self,
        new: AmountLike | DateUpdateLike | Callable[[Any], Any],
        by_amount: bool = True,
    ) -> Legs:
        """Update amounts or occurrence dates.

        When ``by_amount`` is ``True``, ``new`` is interpreted as amount
        data or a callable applied to amounts. Otherwise ``new`` must
        provide replacement occurrence dates.

        Args:
            new:
                The update specification. Its interpretation depends on
                the value of `by_amount`.
            by_amount:
                Determines whether to update cash amounts or occurrence dates.
                If True (default), updates cash amounts.
                If False, updates occurrence dates.

        Returns:
            New ``Legs`` object with updated amounts or dates.
        """
        if by_amount:
            if callable(new):
                f = np.frompyfunc(new, 1, 1)
                amt = f(self.amount).astype(float)  # type: ignore
            else:
                amt = np.asarray(new, dtype=float)
            return Legs(self._occur_date_int, amt, False)

        if isinstance(new, Schedule):
            new_dt = [Legs._to_int(d) for d in new.mapped_date]
        elif callable(new):
            f = np.frompyfunc(new, 1, 1)
            new_dt = f(self._occur_date_int).astype(np.int64)  # type: ignore
        elif isinstance(new, Iterable):
            new_dates = cast(Iterable[DateLike], new)
            new_dt = [Legs._to_int(d) for d in new_dates]
        else:
            raise TypeError("`new` must be a schedule, callable, or iterable date set.")
        return Legs(new_dt, self.amount, True)

    @inplacement
    def where(
        self,
        cond: Iterable[bool],
        proc_true: Scalar,
        proc_false: Scalar,
    ) -> Legs:
        """Replace amounts using :func:`numpy.where` semantics.

        Args:
            cond:
                A boolean condition array detemining which
                values to select.
            proc_true:
                values to use where 'cond' is true.
            proc_false:
                values to use where 'cond' is false.

        Returns:
            A new Legs object with updated cash flow.
        """
        cond_arr = np.asarray(cond, dtype=bool)
        new_amt = np.where(cond_arr, proc_true, proc_false)
        return Legs(self._occur_date_int, new_amt, False)

    @inplacement
    def replace(
        self,
        to_replace: Mapping[Scalar, Scalar] | Iterable[Scalar],
        value: Scalar,
    ) -> Legs:
        """Replace specified cash amounts with a new value.

        This method performs an element-wise replacement on the cash amounts.

        - If `to_replace` is a mapping, each key-value pair specifies a
        replacement rule where occurrences of the key are replaced by
        the corresponding value.
        - If `to_replace` is not a mapping, it is interpreted as a collection
        of values to be replaced by `value`.

        Args:
            to_replace:
                Mapping of old-to-new values, or iterable of values that
                should all be replaced by ``value``.
            value:
                Replacement value used when ``to_replace`` is not a mapping.

        Returns:
            Updated ``Legs`` instance.
        """
        arr = self.amount.copy()
        if isinstance(to_replace, dict):
            for old, new in to_replace.items():
                arr[arr == old] = new
        else:
            values = np.asarray(list(to_replace))
            arr[np.isin(arr, values)] = value

        return Legs(self._occur_date_int, arr, False)

    @inplacement
    def expand(
        self,
        schedule: Schedule | Iterable[DateLike],
        fillna: Literal["ffill", "bfill", "keep", "zero"],
    ) -> Legs:
        """Reindex the leg onto a new set of dates.

        Args:
            schedule:
                Target dates, either as a :class:`Schedule` or iterable.
            fillna:
                Strategy for dates missing from the original leg.

        Returns:
            Updated ``Legs`` instance.

        Raises:
            ValueError:
                If `fillna` is not one of the supported strategies.
        """
        dates = schedule.mapped_date if isinstance(schedule, Schedule) else schedule
        new_dt = np.array([Legs._to_int(d) for d in dates], dtype=np.int64)
        base_dt = self._occur_date_int
        base_amt = self.amount

        idx = np.searchsorted(base_dt, new_dt)

        if fillna == "ffill":
            idx = np.clip(idx - 1, 0, len(base_amt) - 1)
            return Legs(new_dt, base_amt[idx], False)

        if fillna == "bfill":
            idx = np.clip(idx, 0, len(base_amt) - 1)
            return Legs(new_dt, base_amt[idx], False)

        if fillna == "keep":
            out = np.full(len(new_dt), np.nan)
            pos = np.searchsorted(base_dt, new_dt)
            valid = pos < len(base_dt)
            match = np.zeros(len(new_dt), dtype=bool)
            match[valid] = base_dt[pos[valid]] == new_dt[valid]
            out[match] = base_amt[pos[match]]
            return Legs(new_dt, out, False)

        if fillna == "zero":
            out = np.zeros(len(new_dt))
            pos = np.searchsorted(base_dt, new_dt)
            valid = pos < len(base_dt)
            match = np.zeros(len(new_dt), dtype=bool)
            match[valid] = base_dt[pos[valid]] == new_dt[valid]
            out[match] = base_amt[pos[match]]
            return Legs(new_dt, out, False)

        raise ValueError

    def find_nearest_date(
        self, date: DateLike, side: Literal["left", "right"] = "left"
    ) -> Date | None:
        """Find the nearest stored date on the requested side of ``date``.

        Args:
            date:
                Target date as a :class:`Date` or serial number.
            side:
                ``"left"`` returns the last date not after ``date`` and
                ``"right"`` returns the first date not before ``date``.

        Returns:
            Matching date, or ``None`` when no such date exists.

        Raises:
            ValueError:
                If `side` is not one of the supported values.
        """
        tg = Legs._to_int(date)
        arr = self._occur_date_int

        if len(arr) == 0:
            return None

        if side == "left":
            idx = np.searchsorted(arr, tg, side="right") - 1
            if idx < 0:
                return None
            return Legs._to_date(arr[idx])

        if side == "right":
            idx = np.searchsorted(arr, tg, side="left")
            if idx >= len(arr):
                return None
            return Legs._to_date(arr[idx])

        raise ValueError

    def get_amount_by_date(self, date: DateLike) -> float | Legs:
        """Return amount information for cash flows occurring on ``date``.

        Args:
            date:
                Query date as a :class:`Date` or serial number.

        Returns:
            ``0.0`` when there is no match, a scalar when there is one
            match, or a ``Legs`` object when multiple entries match.
        """
        dt_int = Legs._to_int(date)
        mask = self._occur_date_int == dt_int
        if not np.any(mask):
            return 0.0

        matched = self.amount[mask]
        if len(matched) == 1:
            return matched[0]
        return Legs([dt_int] * len(matched), matched)
