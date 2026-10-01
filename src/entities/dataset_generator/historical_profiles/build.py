from __future__ import annotations

import pandas as pd

from src.shared.phases import WARMUP, belongs_to
from src.shared.tables import MULTIVALUE_SEPARATOR

COLUMNS = (
    "operator_id",
    "window_opens_at",
    "window_closes_at",
    "observed_ips",
    "session_count",
)


def ruler_period(log: pd.DataFrame) -> pd.DataFrame:
    """As linhas do log que caem no aquecimento; as de outra fase ficam de fora."""
    return log[belongs_to(WARMUP, log["timestamp"])]


def session_openings(log: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por sessao: quem abriu, quando e de onde, tomados do primeiro evento.
    """
    moments = pd.to_datetime(log["timestamp"])
    dated = log.assign(moment=moments)

    grouped = dated.sort_values("moment", kind="stable").groupby("session_id")

    return pd.DataFrame({
        "operator_id": grouped["operator_id"].first(),
        "source_ip": grouped["source_ip"].first(),
        "moment": grouped["moment"].min(),
    })


HOUR_FORMAT = "%H:%M:%S"


def hour_of_day(moments: pd.Series) -> pd.Series:
    """A hora do dia como texto `HH:MM:SS`, sem a data."""
    return moments.dt.strftime(HOUR_FORMAT)


def window_width_hours(profiles: pd.DataFrame) -> pd.Series:
    """A largura da janela em horas: converte para numero as pontas, que sao texto
    `HH:MM:SS`.
    """
    opens = pd.to_timedelta(profiles["window_opens_at"])
    closes = pd.to_timedelta(profiles["window_closes_at"])

    return (closes - opens).dt.total_seconds() / 3600


def joined_distinct(addresses: pd.Series) -> str:
    """As origens distintas de um operador, ordenadas e unidas por separador."""
    distinct = sorted(set(addresses))

    return MULTIVALUE_SEPARATOR.join(distinct)


def profile_of(openings: pd.DataFrame) -> pd.DataFrame:
    """A janela e as origens de cada operador, do log do aquecimento."""
    hours = hour_of_day(openings["moment"])
    dated = openings.assign(hour=hours)

    grouped = dated.groupby("operator_id")

    addresses = grouped["source_ip"].apply(joined_distinct)

    return pd.DataFrame({
        "operator_id": grouped.size().index,
        "window_opens_at": grouped["hour"].min().values,
        "window_closes_at": grouped["hour"].max().values,
        "observed_ips": addresses.values,
        "session_count": grouped.size().values,
    })


def build_profiles(log: pd.DataFrame) -> pd.DataFrame:
    """Do log do aquecimento ao perfil historico de cada operador."""
    openings = session_openings(ruler_period(log))
    profiles = profile_of(openings)

    return profiles.sort_values("operator_id", ignore_index=True)[list(COLUMNS)]
