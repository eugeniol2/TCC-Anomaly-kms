from __future__ import annotations

import pandas as pd

from src.shared.phases import belongs_to

COLUMNS = (
    "event_id",
    "session_id",
    "operator_id",
    "timestamp",
    "source_ip",
    "operation",
    "key_id",
    "outcome",
)


def build_log(
    requests: pd.DataFrame, outcomes: pd.DataFrame, phase: str
) -> pd.DataFrame:
    """Junta as tentativas da fase aos desfechos, por `event_id`.

    Juncao interna: o que nao casa some, em vez de virar celula vazia. O teste
    confere que nada some.
    """
    of_phase = requests[belongs_to(phase, requests["timestamp"])]

    joined = of_phase.merge(outcomes, on="event_id", how="inner")

    return joined[list(COLUMNS)].reset_index(drop=True)
