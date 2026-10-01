from __future__ import annotations

import pandas as pd

from src.entities.dataset_generator.dataset.parameters import (
    LABEL_ABSENT,
    LABEL_PRESENT,
    SHORTEST_MEASURABLE_MINUTES,
)
from src.shared.phases import EVALUATED
from src.shared.tables import MULTIVALUE_SEPARATOR
from src.entities.dataset_generator.historical_profiles.build import hour_of_day
from src.entities.kms.policy import DENIED_BY_POLICY, SUCCESS

IDENTIFIERS = ("session_id", "operator_id", "opened_at")

ATTRIBUTES = (
    "events",
    "duration_minutes",
    "requests_per_minute",
    "distinct_keys",
    "failures_per_event",
    "denials_per_event",
    "atypical_hour",
    "new_source_ip",
)

LABEL = "compromised"


def require_label(sessions: pd.DataFrame) -> None:
    """Recusa um conjunto sem rotulo, como o do aquecimento."""
    has_label = LABEL in sessions.columns

    if not has_label:
        raise ValueError(
            "o conjunto recebido nao tem rotulo; e o do aquecimento? "
            "So o periodo avaliado tem rotulo"
        )


def columns_of(phase: str) -> tuple[str, ...]:
    """As colunas do `sessions.csv` daquela fase: os identificadores e os oito
    atributos, e na fase avaliada tambem o rotulo.
    """
    is_evaluated = phase == EVALUATED

    if is_evaluated:
        return IDENTIFIERS + ATTRIBUTES + (LABEL,)

    return IDENTIFIERS + ATTRIBUTES


def count_failures(outcomes: pd.Series) -> int:
    """Quantos desfechos da sessao nao foram sucesso."""
    return (outcomes != SUCCESS).sum()


def count_denials(outcomes: pd.Series) -> int:
    """Quantos desfechos da sessao foram negacao por politica."""
    return (outcomes == DENIED_BY_POLICY).sum()


def per_session(log: pd.DataFrame) -> pd.DataFrame:
    """As grandezas brutas de cada sessao, antes de virarem atributo.

    O `apply` roda a funcao uma vez por sessao, com os desfechos daquela sessao.
    """
    dated = log.assign(moment=pd.to_datetime(log["timestamp"]))
    grouped = dated.groupby("session_id", sort=True)

    return pd.DataFrame({
        "operator_id": grouped["operator_id"].first(),
        "source_ip": grouped["source_ip"].first(),
        "opened_at": grouped["moment"].min(),
        "closed_at": grouped["moment"].max(),
        "events": grouped.size(),
        "distinct_keys": grouped["key_id"].nunique(),
        "failures": grouped["outcome"].apply(count_failures),
        "denials": grouped["outcome"].apply(count_denials),
    })


def with_rate_attributes(sessions: pd.DataFrame) -> pd.DataFrame:
    """Duracao, taxa e as duas razoes de falha."""
    spans = (sessions["closed_at"] - sessions["opened_at"]).dt.total_seconds() / 60
    measurable = spans.clip(lower=SHORTEST_MEASURABLE_MINUTES)

    return sessions.assign(
        duration_minutes=spans.round(4),
        requests_per_minute=(sessions["events"] / measurable).round(4),
        failures_per_event=(sessions["failures"] / sessions["events"]).round(4),
        denials_per_event=(sessions["denials"] / sessions["events"]).round(4),
    )


def as_address_set(joined: str) -> frozenset[str]:
    """As origens observadas, de texto unido por separador para conjunto."""
    return frozenset(str(joined).split(MULTIVALUE_SEPARATOR)) # |


def read_profiles(profiles: pd.DataFrame) -> pd.DataFrame:
    """Indexa o perfil por operador, com as origens ja como conjunto."""
    indexed = profiles.set_index("operator_id")
    addresses = indexed["observed_ips"].apply(as_address_set)

    return indexed.assign(seen_addresses=addresses)


def with_history_attributes(
    sessions: pd.DataFrame, profiles: pd.DataFrame
) -> pd.DataFrame:
    """Os dois atributos binarios, lidos contra o perfil do aquecimento daquele
    operador.

    Operador ausente do perfil levanta `KeyError`.
    """
    history = read_profiles(profiles)

    hours = hour_of_day(sessions["opened_at"])
    opens = sessions["operator_id"].map(history["window_opens_at"])
    closes = sessions["operator_id"].map(history["window_closes_at"])

    # Comparacao de texto: as tres pontas sao `HH:MM:SS` de largura fixa,
    # entao a ordem alfabetica e a cronologica.
    is_inside = (hours >= opens) & (hours <= closes)

    is_known = []

    for row in sessions.itertuples():
        seen_addresses = history.loc[row.operator_id, "seen_addresses"]
        is_known.append(row.source_ip in seen_addresses)

    return sessions.assign(
        atypical_hour=(~is_inside).astype(int),
        new_source_ip=(~pd.Series(is_known, index=sessions.index)).astype(int),
    )


def with_label(sessions: pd.DataFrame, compromised: pd.DataFrame) -> pd.DataFrame:
    """Marca as sessoes que a campanha abriu: estar na lista e o rotulo."""
    marked = frozenset(compromised["session_id"])
    belongs = sessions.index.isin(marked)

    return sessions.assign(
        compromised=pd.Series(belongs, index=sessions.index).map(
            {True: LABEL_PRESENT, False: LABEL_ABSENT}
        )
    )


def build_dataset(
    log: pd.DataFrame,
    profiles: pd.DataFrame,
    phase: str,
    compromised: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Do log de uma fase ao conjunto de sessoes daquela fase.

    `compromised` e obrigatorio na fase avaliada e recusado no aquecimento.
    """
    is_evaluated = phase == EVALUATED

    if is_evaluated and compromised is None:
        raise ValueError("a fase avaliada exige `compromised_sessions.csv`")

    if not is_evaluated and compromised is not None:
        raise ValueError(
            "o aquecimento nao recebe rotulo: ele alimenta so o M8, e "
            "calibracao por percentil nao usa rotulo"
        )

    sessions = with_rate_attributes(per_session(log))
    described = with_history_attributes(sessions, profiles)

    if is_evaluated:
        described = with_label(described, compromised)

    dated = described.assign(
        opened_at=described["opened_at"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    )

    return dated.reset_index()[list(columns_of(phase))]
