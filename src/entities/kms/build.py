from __future__ import annotations

import pandas as pd

from src.shared.phases import belongs_to
from src.entities.kms.policy import COLUMNS, outcome_of, read_repository


def requests_of_phase(requests: pd.DataFrame, phase: str) -> pd.DataFrame:
    """As requisicoes que caem na fase indicada."""
    return requests[belongs_to(phase, requests["timestamp"])].reset_index(drop=True)


def build_outcomes(
    requests: pd.DataFrame,
    keys: pd.DataFrame,
    operators: pd.DataFrame,
    phase: str,
) -> pd.DataFrame:
    """O desfecho de cada tentativa da fase, uma linha por evento. Nao sorteia nada."""
    of_phase = requests_of_phase(requests, phase)
    repository = read_repository(keys, operators)

    outcomes = []

    for row in of_phase.itertuples():
        outcome = outcome_of(row.operator_id, row.key_id, repository)
        outcomes.append(outcome)

    return pd.DataFrame(
        {"event_id": of_phase["event_id"], "outcome": outcomes},
        columns=list(COLUMNS),
    )
