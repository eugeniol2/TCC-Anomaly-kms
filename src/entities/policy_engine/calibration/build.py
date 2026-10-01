from __future__ import annotations

import numpy as np
import pandas as pd

from src.entities.policy_engine.calibration.parameters import (
    THRESHOLD_ATTRIBUTES,
    THRESHOLD_PERCENTILE,
)
from src.shared.phases import WARMUP, belongs_to

COLUMNS = ("attribute", "threshold", "percentile", "sessions")


def ruler_period(sessions: pd.DataFrame) -> pd.DataFrame:
    """As sessoes que abriram no aquecimento; as de outra fase ficam de fora."""
    return sessions[belongs_to(WARMUP, sessions["opened_at"])]


def threshold_of(values: pd.Series) -> float:
    """O valor acima do qual a regra dispara."""
    return float(np.percentile(values, THRESHOLD_PERCENTILE))


def build_thresholds(sessions: pd.DataFrame) -> pd.DataFrame:
    """Do conjunto de sessoes do aquecimento aos seis limiares da semente."""
    calibration = ruler_period(sessions)

    is_empty = len(calibration) == 0

    if is_empty:
        raise ValueError(
            "nenhuma sessao do aquecimento; "
            "o `sessions.csv` recebido e da fase avaliada?"
        )

    rows = []

    for attribute in THRESHOLD_ATTRIBUTES:
        threshold = round(threshold_of(calibration[attribute]), 4)
        rows.append({
            "attribute": attribute,
            "threshold": threshold,
            "percentile": THRESHOLD_PERCENTILE,
            "sessions": len(calibration),
        })

    return pd.DataFrame(rows)[list(COLUMNS)]
