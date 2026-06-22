"""Database engine factories and default application connections."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


class Connector:
    """Factory methods for creating SQLAlchemy database engines."""

    @staticmethod
    def make_engine(server: str, db: str, user: str, pwd: str) -> Engine:
        """Create a SQL Server engine.

        Args:
            server: SQL Server host name or address.
            db: Database name.
            user: Database login user.
            pwd: Database login password.

        Returns:
            SQLAlchemy engine configured for the requested SQL Server database.
        """
        conn_string = (
            f"mssql+pyodbc://{user}:{pwd}"
            f"@{server}/{db}"
            "?driver=ODBC+Driver+18+for+SQL+Server"
            "&TrustServerCertificate=yes"
        )
        return create_engine(conn_string)

    @staticmethod
    def make_postgre_engine(server: str, db: str, user: str, pwd: str) -> Engine:
        """Create a PostgreSQL engine.

        Args:
            server: PostgreSQL host name or address.
            db: Database name.
            user: Database login user.
            pwd: Database login password.

        Returns:
            SQLAlchemy engine configured for the requested PostgreSQL database.
        """
        conn_string = f"postgresql+psycopg2://{user}:{pwd}@{server}:5432/{db}"
        return create_engine(conn_string)


load_dotenv(override=True)

server: str | None = os.getenv("SERVER")
db: str | None = os.getenv("DB")
user: str | None = os.getenv("DB_USER")
pwd: str | None = os.getenv("DB_PWD")

if server is None or db is None or user is None or pwd is None:
    raise ValueError(
        "Database environment variables SERVER, DB, DB_USER, and DB_PWD must be set."
    )

engine: Engine = Connector.make_engine(server, db, user, pwd)
"""Default SQL Server engine loaded from environment variables."""

bill_engine: Engine = Connector.make_engine(
    "10.10.10.110",
    "billdb",
    "billuser",
    "billpasswd",
)
"""SQL Server engine for the billing database."""

twn_engine: Engine = Connector.make_postgre_engine(
    "itportal.tejwin.com",
    "twn",
    "2023020607",
    ")P:?9ol.",
)
"""PostgreSQL engine for the ``twn`` database."""
