"""Interest-rate index models for term and overnight benchmarks."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Optional

from .enum import BusinessDayConvention, Compounding, PeriodType
from .term_structure import InterestTermStructure
from .calendar import Calendar, FedWireCalendar, Target2Calendar
from .date import Date, Period


class InterestRateIndex(ABC):
    """Base class for interest rate indices.

    An index stores historical fixing data and provides a unified interface for
    retrieving fixings. If a fixing is unavailable historically and the
    requested date is not in the past, the index forecasts the fixing from its
    associated term structure.

    Attributes:
        name: Name of the index.
        curve: Term structure used for forecasting.
        fixings: Historical fixings keyed by fixing date.
    """

    def __init__(
        self,
        name: str,
        curve: InterestTermStructure,
        fixings: Optional[dict[Date, float]] = None,
    ) -> None:
        """Initialize an interest rate index.

        Args:
            name: Name of the index.
            curve: Term structure used for forecasting future fixings.
            fixings: Optional historical fixing mapping keyed by fixing date.
        """
        self.name = name
        self.curve = curve
        self.fixings = dict(fixings) if fixings is not None else {}

    @abstractmethod
    def get_forecast_fixing(self, fixing_date: Date) -> float:
        """Forecast the fixing rate for a given date.

        Args:
            fixing_date: Fixing date to forecast.

        Raises:
            NotImplementedError: Implementation is required in subclasses.

        Returns:
            Forecast fixing rate.
        """
        raise NotImplementedError

    def add_fixing(self, fixing_date: Date, fixing_rate: float) -> None:
        """Store a historical fixing.

        Args:
            fixing_date: Fixing date.
            fixing_rate: Observed market rate.

        Raises:
            TypeError: If ``fixing_date`` is not a ``Date`` instance.
            ValueError: If ``fixing_rate`` is not finite.
        """
        if not isinstance(fixing_date, Date):
            raise TypeError("`fixing_date` must be Date.")
        rate = float(fixing_rate)
        if not math.isfinite(rate):
            raise ValueError("`fixing_rate` must be finite.")
        self.fixings[fixing_date] = rate

    def get_fixing(self, fixing_date: Date) -> float:
        """Get the fixing for a given fixing date.

        The method behaves as follows:
        - If a historical fixing exists, return it.
        - If the fixing date is in the past and no fixing exists, raise an
          error.
        - Otherwise, forecast the fixing from the associated term structure.

        Args:
            fixing_date: Date for which the fixing is requested.

        Raises:
            ValueError: If the fixing is historical but unavailable.

        Returns:
            Historical or forecast fixing rate.
        """
        if fixing_date in self.fixings:
            return self.fixings[fixing_date]
        if fixing_date < self.curve.reference_date:
            raise ValueError(f"Missing historical fixing for {fixing_date}.")
        return self.get_forecast_fixing(fixing_date)


class IborIndex(InterestRateIndex):
    """Represents a term IBOR-style index.

    This class models a term rate index with a spot lag, a spot date, and a
    tenor-based maturity date derived under a given business-day convention.

    Attributes:
        tenor: Tenor of the index.
        fixing_calendar: Calendar used to adjust fixing-related dates.
        holiday_rule: Business-day convention used for date adjustment.
        spot_lag: Number of business days between fixing date and spot date.
        eom_adjustment: Whether end-of-month handling is enabled.
    """

    def __init__(
        self,
        name: str,
        tenor: Period,
        curve: InterestTermStructure,
        fixing_calendar: Calendar,
        holiday_rule: BusinessDayConvention,
        spot_lag: int,
        eom_adjustment: bool,
        fixings: Optional[dict[Date, float]] = None,
    ) -> None:
        """Initialize an IBOR index.

        Args:
            name: Name of the index.
            tenor: Tenor of the index.
            curve: Term structure used for forecasting.
            fixing_calendar: Calendar used to adjust fixing-related dates.
            holiday_rule: Business-day convention used for date adjustment.
            spot_lag: Number of business days between fixing date and value
                date.
            eom_adjustment: Whether end-of-month handling is enabled.
            fixings: Optional historical fixing mapping.

        Raises:
            ValueError: If ``spot_lag`` is negative.
        """
        super().__init__(name, curve, fixings)
        self.tenor = tenor
        self.fixing_calendar = fixing_calendar
        self.holiday_rule = holiday_rule
        self.spot_lag = spot_lag
        self.eom_adjustment = eom_adjustment

        if spot_lag < 0:
            raise ValueError("`spot_lag` must be >= 0.")

    def get_spot_date(self, fixing_date: Date) -> Date:
        """Get the spot date implied by a fixing date.

        Args:
            fixing_date: Fixing date used to determine the accrual start.

        Returns:
            Accrual start date.
        """
        if self.spot_lag == 0:
            return self.fixing_calendar.advance(fixing_date, self.holiday_rule)
        return self.fixing_calendar.advance_business_days(
            fixing_date, self.spot_lag, self.holiday_rule
        )

    def get_maturity_date(self, spot_date: Date) -> Date:
        """Get the maturity date implied by a spot date.

        Args:
            spot_date: Accrual start date.

        Returns:
            Accrual end date.
        """
        last_business_day = self.fixing_calendar.advance(
            spot_date.end_of_month, BusinessDayConvention.PRECEDING
        )
        use_eom = self.eom_adjustment and (spot_date == last_business_day)
        base = (
            (spot_date + self.tenor).end_of_month
            if use_eom
            else (spot_date + self.tenor)
        )
        return self.fixing_calendar.advance(base, self.holiday_rule)

    def get_forecast_fixing(self, fixing_date: Date) -> float:
        """Forecast the IBOR fixing from the term structure.

        The fixing is forecast as the simple-compounded forward rate over the
        interval from the spot date to the maturity date.

        Args:
            fixing_date: Date to fix the interest rate.

        Returns:
            Forecast IBOR fixing.
        """
        start = self.get_spot_date(fixing_date)
        end = self.get_maturity_date(start)

        refer_date = self.fixing_calendar.advance(
            self.curve.reference_date, self.holiday_rule
        )

        f_rate = self.curve.get_forward_rate(refer_date, start, end, Compounding.SIMPLE)

        return float(f_rate)


class OvernightIndex(InterestRateIndex):
    """Represents an overnight index with publication-lag support.

    This class models an overnight index over a one-business-day accrual
    period. The fixing input is interpreted as the value date, while the
    publication date can be derived by advancing the value date by
    ``publication_lag`` business days.

    Attributes:
        calendar: Calendar used for fixing-related date adjustment.
        holiday_rule: Business-day convention used for date adjustment.
        publication_lag: Number of business days between value date and
            publication date.
    """

    def __init__(
        self,
        name: str,
        curve: InterestTermStructure,
        calendar: Calendar,
        holiday_rule: BusinessDayConvention,
        publication_lag: int = 0,
        fixings: Optional[dict[Date, float]] = None,
    ) -> None:
        """Initialize an overnight index.

        Args:
            name: Name of the overnight index.
            curve: Term structure used for forecasting.
            calendar: Calendar used for fixing-related date adjustment.
            holiday_rule: Business-day convention used for date adjustment.
            publication_lag: Number of business days between publication date
                and effective date.
            fixings: Optional historical fixing mapping.

        Raises:
            ValueError: If ``publication_lag`` is negative.
        """
        if publication_lag < 0:
            raise ValueError("`publication_lag` must be >= 0.")
        self.calendar = calendar
        self.holiday_rule = holiday_rule
        self.publication_lag = publication_lag
        super().__init__(name, curve, fixings)

    def get_publication_date(self, value_date: Date) -> Date:
        """Get the publication date implied by a value date.

        Args:
            value_date: Value date of the overnight fixing.

        Returns:
            Publication date associated with the fixing.
        """
        return self.calendar.advance_business_days(
            value_date, self.publication_lag, BusinessDayConvention.FOLLOWING
        )

    def get_forecast_fixing(self, fixing_date: Date) -> float:
        """Forecast the overnight fixing from the term structure.

        The input date is interpreted as the value date of the overnight
        accrual period. The accrual end date is the next business day after
        the value date.

        Args:
            fixing_date: Value date used to determine the fixing.

        Raises:
            TypeError: If ``value_date`` is not a ``Date`` instance.
            ValueError: If ``value_date`` is not a business day.

        Returns:
            Forecast overnight fixing.
        """
        if not isinstance(fixing_date, Date):
            raise TypeError("Fixing date must be Date.")
        if not self.calendar.is_business_day(fixing_date):
            raise ValueError("Fixing date must be a business day.")
        start = fixing_date
        end = self.calendar.advance_business_days(
            start, 1, BusinessDayConvention.FOLLOWING
        )
        refer_date = self.curve.reference_date
        f_rate = self.curve.get_forward_rate(refer_date, start, end, Compounding.SIMPLE)

        return float(f_rate)


class SOFR(OvernightIndex):
    """Secured Overnight Financing Rate overnight index.

    This index uses the Fedwire calendar, the following business-day
    convention, and a one-business-day publication lag.
    """

    def __init__(
        self,
        curve: InterestTermStructure,
        fixings: Optional[dict[Date, float]] = None,
    ) -> None:
        """Initialize a SOFR index.

        Args:
            curve: Term structure used for forecasting.
            fixings: Optional historical fixing mapping.
        """
        super().__init__(
            name="sofr",
            curve=curve,
            calendar=FedWireCalendar(),
            holiday_rule=BusinessDayConvention.FOLLOWING,
            publication_lag=1,
            fixings=fixings,
        )


class Estr(OvernightIndex):
    """Euro short-term rate overnight index.

    This index uses the TARGET2 calendar, the following business-day
    convention, and a one-business-day publication lag.
    """

    def __init__(
        self,
        curve: InterestTermStructure,
        fixings: Optional[dict[Date, float]] = None,
    ) -> None:
        """Initialize an ESTR index.

        Args:
            curve: Term structure used for forecasting.
            fixings: Optional historical fixing mapping.
        """
        super().__init__(
            name="estr",
            curve=curve,
            calendar=Target2Calendar(),
            holiday_rule=BusinessDayConvention.FOLLOWING,
            publication_lag=1,
            fixings=fixings,
        )


class Euribor(IborIndex):
    """Base class for EURIBOR term indices.

    This class centralizes the shared conventions used by different EURIBOR
    tenors, such as calendar, spot lag, holiday adjustment, and end-of-month
    handling.
    """

    def __init__(
        self,
        tenor: Period,
        curve: InterestTermStructure,
        fixings: Optional[dict[Date, float]] = None,
    ) -> None:
        """Initialize a EURIBOR index for a given tenor.

        Args:
            tenor: EURIBOR tenor.
            curve: Term structure used for forecasting.
            fixings: Optional historical fixing mapping.
        """
        super().__init__(
            name=f"euribor_{tenor.number}",
            curve=curve,
            tenor=tenor,
            fixing_calendar=Target2Calendar(),
            holiday_rule=BusinessDayConvention.MODIFIED_FOLLOWING,
            spot_lag=2,
            eom_adjustment=True,
            fixings=fixings,
        )


class Euribor3m(Euribor):
    """Three-month EURIBOR index."""

    def __init__(
        self,
        curve: InterestTermStructure,
        fixings: Optional[dict[Date, float]] = None,
    ) -> None:
        """Initialize a 3M EURIBOR index.

        Args:
            curve: Term structure used for forecasting.
            fixings: Optional historical fixing mapping.
        """
        super().__init__(Period(PeriodType.MONTHLY, 3), curve, fixings)


class Euribor6m(Euribor):
    """Six-month EURIBOR index."""

    def __init__(
        self,
        curve: InterestTermStructure,
        fixings: Optional[dict[Date, float]] = None,
    ) -> None:
        """Initialize a 6M EURIBOR index.

        Args:
            curve: Term structure used for forecasting.
            fixings: Optional historical fixing mapping.
        """
        super().__init__(Period(PeriodType.MONTHLY, 6), curve, fixings)


class Taibor(IborIndex):
    """Base class for TAIBOR term indices.

    This class centralizes the shared conventions used by different TAIBOR
    tenors, such as calendar, spot lag, holiday adjustment, and end-of-month
    handling.
    """

    def __init__(
        self,
        tenor: Period,
        curve: InterestTermStructure,
        fixings: Optional[dict[Date, float]] = None,
    ) -> None:
        """Initialize a TAIBOR index for a given tenor.

        Args:
            tenor: TAIBOR tenor.
            curve: Term structure used for forecasting.
            fixings: Optional historical fixing mapping.
        """
        super().__init__(
            name=f"taibor_{tenor.number}",
            curve=curve,
            tenor=tenor,
            fixing_calendar=TwseCalendar(),
            spot_lag=2,
            eom_adjustment=True,
            holiday_rule=BusinessDayConvention.MODIFIED_FOLLOWING,
            fixings=fixings,
        )


class Taibor3m(Taibor):
    """Three-month TAIBOR index."""

    def __init__(
        self,
        curve: InterestTermStructure,
        fixings: Optional[dict[Date, float]] = None,
    ) -> None:
        """Initialize a 3M TAIBOR index.

        Args:
            curve: Term structure used for forecasting.
            fixings: Optional historical fixing mapping.
        """
        super().__init__(
            Period(PeriodType.MONTHLY, 3),
            curve,
            fixings,
        )


class Taibir(IborIndex):
    """Base class for TAIBIR term indices.

    This class centralizes the shared conventions used by different TAIBIR
    tenors, such as calendar, spot lag, holiday adjustment, and end-of-month
    handling.
    """

    def __init__(
        self,
        tenor: Period,
        curve: InterestTermStructure,
        fixings: Optional[dict[Date, float]] = None,
    ) -> None:
        """Initialize a TAIBIR index for a given tenor.

        Args:
            tenor: TAIBIR tenor.
            curve: Term structure used for forecasting.
            fixings: Optional historical fixing mapping.
        """
        super().__init__(
            name="taibir",
            curve=curve,
            tenor=tenor,
            fixing_calendar=FedWireCalendar(),
            holiday_rule=BusinessDayConvention.MODIFIED_FOLLOWING,
            spot_lag=0,
            eom_adjustment=True,
            fixings=fixings
        )
