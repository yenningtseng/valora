"""ORM-based data access and normalization helpers for market data."""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import date, datetime
from typing import Any, TypeAlias

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.ext.automap import automap_base

from .data_engine import bill_engine, engine, twn_engine
from .date import Date, Period
from .daycount import Act360, Act365Fixed, DayCount
from .enum import Compounding, PeriodType
from .term_structure import InterpolatedZeroCurve

Numeric: TypeAlias = int | float
NumericIterable: TypeAlias = Iterable[Numeric]
RateInput: TypeAlias = Numeric | str | pd.Series | np.ndarray | NumericIterable
DateStringInput: TypeAlias = str | Iterable[str] | pd.Series
DateTimeInput: TypeAlias = (
    datetime | date | pd.Series | np.ndarray | Iterable[datetime | date]
)

Base = automap_base()
Base.prepare(autoload_with=engine)

BillBase = automap_base()
BillBase.prepare(autoload_with=bill_engine, schema="dbo")

TwnBase = automap_base()
TwnBase.prepare(autoload_with=twn_engine, schema="stk")


class OrmLoader:
    """Helpers for loading reflected ORM tables into pandas data frames."""

    @staticmethod
    def select_table(
        model: Any,
        eg: Engine = engine,
        order_by: str | None = None,
        ascending: bool = True,
        **filters: Any,
    ) -> pd.DataFrame:
        """Query a reflected ORM model and return the result as a data frame.

        Args:
            model: SQLAlchemy reflected ORM model.
            eg: SQLAlchemy engine used to execute the query.
            order_by: Column name used for ordering.
            ascending: Whether ordering is ascending.
            **filters: Field filters using ``field__operator=value`` syntax.

        Returns:
            Query result loaded into a :class:`pandas.DataFrame`.
        """
        stmt = select(model)

        for key, value in filters.items():
            if value is None:
                continue

            if "__" in key:
                field, op = key.split("__", 1)
            else:
                field, op = key, "eq"

            col = getattr(model, field)

            if op == "eq":
                stmt = stmt.where(col == value)
            elif op == "ne":
                stmt = stmt.where(col != value)
            elif op == "gt":
                stmt = stmt.where(col > value)
            elif op == "gte":
                stmt = stmt.where(col >= value)
            elif op == "lt":
                stmt = stmt.where(col < value)
            elif op == "lte":
                stmt = stmt.where(col <= value)
            elif op == "in":
                stmt = stmt.where(col.in_(value))
            elif op == "like":
                stmt = stmt.where(col.like(value))
            else:
                raise ValueError(f"Unsupported operator: {op}")

        if order_by is not None:
            col = getattr(model, order_by)
            stmt = stmt.order_by(col.asc() if ascending else col.desc())

        return pd.read_sql(stmt, eg)

    @staticmethod
    def get_wrnt_exercise(**filters: Any) -> pd.DataFrame:
        """Load warrant exercise records."""
        return OrmLoader.select_table(
            Base.classes.wrnt_exercise,
            eg=engine,
            **filters,
        )

    @staticmethod
    def get_twn_calendar(**filters: Any) -> pd.DataFrame:
        """Load Taiwan trading-calendar records."""
        return OrmLoader.select_table(
            TwnBase.classes.attr_tradingday,
            eg=twn_engine,
            **filters,
        )

    @staticmethod
    def get_stock_prc(**filters: Any) -> pd.DataFrame:
        """Load stock price records."""
        return OrmLoader.select_table(
            Base.classes.stock_prc,
            **filters,
        )

    @staticmethod
    def get_var_2f_id(**filters: Any) -> pd.DataFrame:
        """Load ``var2f_id`` records from the billing database."""
        return OrmLoader.select_table(
            BillBase.classes.var2f_id,
            eg=bill_engine,
            **filters,
        )

    @staticmethod
    def get_exchange_crossfx(**filters: Any) -> pd.DataFrame:
        """Load FX cross-rate data."""
        return OrmLoader.select_table(
            Base.classes.exchange_crossfx,
            **filters,
        )

    @staticmethod
    def get_bond_yield_gb(**filters: Any) -> pd.DataFrame:
        """Load government bond yield data."""
        return OrmLoader.select_table(
            Base.classes.bond_yield_gb,
            **filters,
        )

    @staticmethod
    def get_rate_float_benchmark(**filters: Any) -> pd.DataFrame:
        """Load floating-rate benchmark metadata."""
        return OrmLoader.select_table(
            Base.classes.rate_float_benchmark,
            **filters,
        )

    @staticmethod
    def get_bond_prc(**filters: Any) -> pd.DataFrame:
        """Load bond pricing data from the billing database."""
        return OrmLoader.select_table(
            BillBase.classes.var2,
            eg=bill_engine,
            **filters,
        )

    @staticmethod
    def get_attr_bond_cf(**filters: Any) -> pd.DataFrame:
        """Load bond cash-flow attributes."""
        return OrmLoader.select_table(
            Base.classes.attr_bond_cf,
            **filters,
        )


class Normalizer:
    """Converters used to normalize raw database fields into domain objects."""

    @staticmethod
    def standardize_to_year(tnrs: Iterable[str]) -> list[float]:
        """Convert tenor strings such as ``6M`` or ``2Y`` to year fractions."""
        stand_tnrs: list[float] = []
        for tnr in tnrs:
            match = re.match(r"(\d+)([DMY])$", tnr)
            value = int(match.group(1))
            unit = match.group(2)

            if unit == "D":
                stand_tnrs.append(value / 365)
            if unit == "M":
                stand_tnrs.append(value / 12)
            if unit == "Y":
                stand_tnrs.append(value / 1)
        return stand_tnrs

    @staticmethod
    def standardize_day_to_year(tnrs: Iterable[Numeric]) -> list[float]:
        """Convert day counts to year fractions on a 360-day basis."""
        return [i / 360 for i in tnrs]

    @staticmethod
    def get_discount_curve(
        rtype: Any,
        crate: Any,
        basis_currency: Any,
        valuation_date: Date,
        daycount: DayCount,
        fix_rate_markup: Numeric | None,
        risk_premium: Numeric | None,
        compound: Compounding = Compounding.ANNUAL,
    ) -> InterpolatedZeroCurve | None:
        """Construct a discount curve from database-sourced market quotes.

        Args:
            rtype: Curve type identifier.
            crate: Unused placeholder kept for compatibility with callers.
            basis_currency: Currency used to filter source market data.
            valuation_date: Curve valuation date.
            daycount: Day-count convention for the output curve.
            fix_rate_markup: Additional fixed-rate spread in percentage points.
            risk_premium: Additional risk premium in percentage points.
            compound: Compounding convention used by the curve.

        Returns:
            Interpolated zero curve when ``rtype == "6"``; otherwise ``None``.
        """
        del crate
        rtype = str(rtype)
        basis_currency = str(basis_currency)

        if rtype == "6":
            gbs = OrmLoader.get_bond_yield_gb(
                currency_issue=basis_currency,
                zdate=str(valuation_date),
            )

            tenors = Normalizer.standardize_day_to_year(gbs["day_cnt"])
            zrates = Normalizer.convert_to_rate(
                gbs["rate"], fix_rate_markup, risk_premium
            )

            return InterpolatedZeroCurve(
                tenors,
                zrates,
                "linear",
                valuation_date,
                daycount,
                compound,
            )

        return None

    @staticmethod
    def get_calendar(input: Any):
        """Return the default calendar used by loader utilities."""
        from .calendar import FedWireCalendar
        del input
        return FedWireCalendar()

    @staticmethod
    def convert_thousand_to_dollar(
        input: Numeric | pd.Series | np.ndarray,
    ) -> Numeric | list[Numeric] | None:
        """Convert amounts quoted in thousands into whole-dollar amounts."""
        if isinstance(input, pd.Series):
            return (input * 1000).tolist()
        if isinstance(input, np.ndarray):
            return list(input * 1000)
        if isinstance(input, (float, int)):
            return input * 1000
        return None

    @staticmethod
    def convert_to_period(input: Any) -> Period:
        """Convert a month count into a monthly :class:`Period`."""
        input = int(input)
        return Period(PeriodType.MONTHLY, input)

    @staticmethod
    def convert_to_daycnt(input: Any) -> type[Act365Fixed] | type[Act360] | None:
        """Map raw day-count codes to day-count classes."""
        input = str(input)
        if input == "365":
            return Act365Fixed
        if input == "360":
            return Act360
        return None

    @staticmethod
    def convert_to_rate(
        input: RateInput,
        fix_rate_markup: Numeric | None = None,
        risk_premium: Numeric | None = None,
    ) -> float | list[float] | None:
        """Convert percentage-form quotes into decimal-form rates."""
        if fix_rate_markup is None:
            fix_rate_markup = 0

        if risk_premium is None:
            risk_premium = 0

        if isinstance(input, (int, float)):
            return (input + fix_rate_markup + risk_premium) / 100

        if isinstance(input, pd.Series):
            return ((input + fix_rate_markup + risk_premium) / 100).tolist()

        if isinstance(input, np.ndarray):
            return list((input + fix_rate_markup + risk_premium) / 100)

        if isinstance(input, str):
            return float(input + fix_rate_markup + risk_premium) / 100

        if isinstance(input, Iterable):
            return [i + fix_rate_markup + risk_premium / 100 for i in input]

        return None

    @staticmethod
    def convert_str_to_date(input: DateStringInput) -> Date | list[Date] | None:
        """Convert string-form dates into :class:`Date` objects."""
        if isinstance(input, str):
            return Date.from_string(input)
        if isinstance(input, pd.Series):
            return [Date.from_string(i) for i in input]
        if isinstance(input, Iterable):
            return [Date.from_string(i) for i in input]
        return None

    @staticmethod
    def convert_date_to_date(input: DateTimeInput) -> Date | list[Date] | None:
        """Convert Python or pandas date objects into :class:`Date` objects."""
        if isinstance(input, datetime):
            return Date.from_datetime(input)
        if isinstance(input, date):
            return Date.from_datetime(input)
        if isinstance(input, pd.Series):
            return [Date.from_datetime(i) for i in input]
        if isinstance(input, np.ndarray):
            return [Date.from_datetime(i) for i in input]
        if isinstance(input, Iterable):
            return [Date.from_datetime(i) for i in input]
        return None

    @staticmethod
    def fix_float_rate_index(string: str) -> str | None:
        """Map raw floating-rate index labels to internal identifiers."""
        mapping = {
            "TIBIR": "TAIBIR02",
            "CP": "CP",
            "BA": "BA",
        }

        return mapping.get(string)

    @staticmethod
    def fix_float_rate_tenor(string: str | None) -> str | None:
        """Extract trailing tenor text such as ``3M`` or ``1Y`` from a label."""
        if string is None:
            return None

        match = re.search(r"(\d+[DWMY])$", string)
        if match:
            return match.group(1)
        return None
