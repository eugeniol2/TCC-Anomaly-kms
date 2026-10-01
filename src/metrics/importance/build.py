from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from src.entities.dataset_generator.dataset.build import ATTRIBUTES, LABEL
from src.entities.models.build import fitted, seed_of, training_seeds
from src.entities.models.parameters import MODEL_NAMES
from src.entities.policy_engine.baseline.build import MECHANISM as RULES
from src.entities.policy_engine.baseline.build import decide, magnitude_cutoffs
from src.formulas.classification import confusion, rates
from src.metrics.evaluation.build import read_table
from src.metrics.importance.parameters import PERMUTATION_REPEATS
from src.shared import layout
from src.shared.rng import IMPORTANCE, stream

COLUMNS = ("seed", "sigma", "mechanism", "attribute", "f1_drop")


class ImportanceRun(NamedTuple):
    seed: int
    sigma: float
    train: pd.DataFrame
    holdout: pd.DataFrame
    thresholds: pd.DataFrame


class Holdout(NamedTuple):
    features: pd.DataFrame
    truth: pd.Series


class RulesDecision(NamedTuple):
    cutoffs: np.ndarray

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return decide(features, self.cutoffs)


def read_importance_run(root: Path, seed: int, sigma: float) -> ImportanceRun:
    """O treino e o holdout da execucao, e os limiares da semente."""
    directory = layout.run_directory(root, seed, sigma)

    return ImportanceRun(
        seed=seed,
        sigma=sigma,
        train=read_table(directory, layout.TRAIN),
        holdout=read_table(directory, layout.HOLDOUT),
        thresholds=read_table(layout.seed_directory(root, seed), layout.THRESHOLDS),
    )


def mechanisms_of(run: ImportanceRun, configuration: dict[str, dict]) -> dict[str, object]:
    """Os tres mecanismos da execucao: as regras com os limiares, e os modelos retreinados."""
    mechanisms = {RULES: RulesDecision(magnitude_cutoffs(run.thresholds))}
    seeds = training_seeds(run.seed)

    for name in MODEL_NAMES:
        random_state = seed_of(seeds, name)
        mechanisms[name] = fitted(name, configuration[name], run.train, random_state)

    return mechanisms


def permutation_orders(seed: int, rows: int) -> dict[str, list[np.ndarray]]:
    """As permutacoes de cada atributo, sorteadas uma vez e usadas pelos tres
    mecanismos.

    Saem do fluxo `IMPORTANCE`, que depende so da semente.
    """
    rng = stream(seed, IMPORTANCE)
    orders = {}

    for attribute in ATTRIBUTES:
        draws = []

        for _ in range(PERMUTATION_REPEATS):
            draws.append(rng.permutation(rows))

        orders[attribute] = draws

    return orders


def f1_of(truth: pd.Series, decided) -> float:
    return rates(confusion(truth, decided))["f1"]


def with_shuffled(features: pd.DataFrame, attribute: str, order: np.ndarray) -> pd.DataFrame:
    """Os atributos com uma coluna so embaralhada, e as outras intactas."""
    shuffled = features.copy()
    values = features[attribute].to_numpy()
    shuffled[attribute] = values[order]

    return shuffled


def mechanism_drops(
    mechanism, holdout: Holdout, orders: dict[str, list[np.ndarray]]
) -> dict[str, float]:
    """Quanto o F1 de um mecanismo cai, em media, com cada atributo embaralhado."""
    original = f1_of(holdout.truth, mechanism.predict(holdout.features))
    drops = {}

    for attribute in ATTRIBUTES:
        scores = []

        for order in orders[attribute]:
            shuffled = with_shuffled(holdout.features, attribute, order)
            scores.append(f1_of(holdout.truth, mechanism.predict(shuffled)))

        drops[attribute] = original - float(np.mean(scores))

    return drops


def run_importance(run: ImportanceRun, configuration: dict[str, dict]) -> pd.DataFrame:
    """A importancia de cada atributo para cada mecanismo, numa execucao."""
    holdout = Holdout(run.holdout[list(ATTRIBUTES)], run.holdout[LABEL])
    orders = permutation_orders(run.seed, len(run.holdout))
    rows = []

    for name, mechanism in mechanisms_of(run, configuration).items():
        drops = mechanism_drops(mechanism, holdout, orders)

        for attribute in ATTRIBUTES:
            rows.append({
                "seed": run.seed,
                "sigma": run.sigma,
                "mechanism": name,
                "attribute": attribute,
                "f1_drop": round(drops[attribute], 4),
            })

    return pd.DataFrame(rows)[list(COLUMNS)]
