"""O teste pareado e a correcao para comparacoes multiplas (D-027, D-111)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


def signed_rank_test(
    differences: pd.Series, zero_method: str, alternative: str
) -> tuple[float, float]:
    """Wilcoxon de postos sinalizados sobre diferencas pareadas: estatistica e p.

    Quando todas as diferencas sao zero nao ha o que testar: estatistica 0 e p 1.
    Com menos de dois pares o teste nao existe, e os dois valores ficam indefinidos.
    """
    is_partial = len(differences) < 2

    if is_partial:
        return np.nan, np.nan

    has_difference = bool((differences != 0).any())

    if not has_difference:
        return 0.0, 1.0

    result = wilcoxon(differences, zero_method=zero_method, alternative=alternative)

    return float(result.statistic), float(result.pvalue)


def holm(p_values: pd.Series) -> np.ndarray:
    """Os p corrigidos por Holm, na mesma ordem dos recebidos."""
    return multipletests(p_values, method="holm")[1]
