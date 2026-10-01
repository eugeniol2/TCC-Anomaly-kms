from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
from numpy.random import Generator

from src.formulas.rounding import round_half_up
from src.entities.dataset_generator.dataset.build import LABEL, require_label
from src.entities.dataset_generator.dataset.parameters import LABEL_PRESENT
from src.shared.rng import PARTITION, stream
from src.entities.dataset_generator.partition.parameters import HOLDOUT_SHARE

STABLE_KEY = ("opened_at", "operator_id", "tie")


class Partition(NamedTuple):
    train: pd.DataFrame
    holdout: pd.DataFrame


def holdout_count(total: int) -> int:
    """Quantas sessoes de uma classe vao para o holdout: `total x HOLDOUT_SHARE` no
    inteiro mais proximo, com o meio para cima.
    """
    return round_half_up(total * HOLDOUT_SHARE)


def in_stable_order(sessions: pd.DataFrame) -> pd.DataFrame:
    """As sessoes de uma classe numa ordem que nao depende de sigma.

    Sessoes do mesmo operador abertas no mesmo segundo se desempatam pela ordem do
    arquivo.
    """
    by_file = sessions.sort_values("session_id", kind="mergesort")
    tie = by_file.groupby(["operator_id", "opened_at"], sort=False).cumcount()

    ordered = by_file.assign(tie=tie).sort_values(list(STABLE_KEY), kind="mergesort")

    return ordered.drop(columns="tie").reset_index(drop=True)


def holdout_mask(rng: Generator, total: int) -> np.ndarray:
    """Quais posicoes de uma classe vao para o holdout: as primeiras de uma permutacao,
    que depende so de quantas sessoes a classe tem.
    """
    chosen = rng.permutation(total)[: holdout_count(total)]
    mask = np.zeros(total, dtype=bool)
    mask[chosen] = True

    return mask


def split_class(rng: Generator, sessions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Uma classe dividida em treino e holdout."""
    ordered = in_stable_order(sessions)
    mask = holdout_mask(rng, len(ordered))

    return ordered[~mask], ordered[mask]


def in_file_order(parts: list[pd.DataFrame], columns: list[str]) -> pd.DataFrame:
    """Junta as classes de um lado e volta a ordem cronologica do arquivo."""
    joined = pd.concat(parts, ignore_index=True)

    return joined.sort_values("session_id", kind="mergesort", ignore_index=True)[columns]


def build_partition(seed: int, sessions: pd.DataFrame) -> Partition:
    """Do conjunto avaliado de uma condicao aos dois lados da divisao.

    Sorteia primeiro as legitimas e depois as do atacante, do mesmo fluxo: trocar a
    ordem muda a particao.
    """
    require_label(sessions)

    rng = stream(seed, PARTITION)
    is_positive = sessions[LABEL] == LABEL_PRESENT

    legitimate_train, legitimate_holdout = split_class(rng, sessions[~is_positive])
    attack_train, attack_holdout = split_class(rng, sessions[is_positive])

    columns = list(sessions.columns)

    return Partition(
        train=in_file_order([legitimate_train, attack_train], columns),
        holdout=in_file_order([legitimate_holdout, attack_holdout], columns),
    )
