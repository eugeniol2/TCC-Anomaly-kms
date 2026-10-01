"""Testes da importancia por permutacao (D-124).

O que se confere e que a medida mede o que diz: os mecanismos retreinados decidem
como os da grade, um atributo ignorado custa exatamente zero, e o sorteio das
permutacoes nao desloca nenhum fluxo que ja existia.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.random import PCG64, Generator, SeedSequence

from src.entities.dataset_generator.dataset.build import ATTRIBUTES
from src.entities.models.parameters import MODEL_NAMES
from src.entities.policy_engine.baseline.build import MECHANISM as RULES
from src.metrics.evaluation.build import MECHANISMS
from src.metrics.importance.build import (
    COLUMNS,
    Holdout,
    mechanism_drops,
    mechanisms_of,
    permutation_orders,
    read_importance_run,
    run_importance,
)
from src.pipeline.build import Specifications, run_seed_branch, run_sigma_branch
from src.shared import layout
from src.shared.rng import STREAM_COUNT, stream

SEED = 1
SIGMA = 0.5

TEST_CONFIGURATION = {
    "random_forest": {
        "n_estimators": 10, "max_depth": None, "min_samples_leaf": 1,
        "max_features": "sqrt", "class_weight": None,
    },
    "xgboost": {
        "n_estimators": 10, "max_depth": 3, "learning_rate": 0.1,
        "subsample": 1.0, "colsample_bytree": 1.0, "class_weight": "balanced",
    },
}
"""Uma configuracao pequena, so para os testes rodarem rapido. A de verdade sai da 902."""


@pytest.fixture(scope="module")
def data_root(tmp_path_factory) -> Path:
    """Uma raiz de dados com uma execucao inteira, como o comando grava."""
    root = tmp_path_factory.mktemp("data")
    branch = run_seed_branch(SEED, root, Specifications(models=TEST_CONFIGURATION))
    run_sigma_branch(branch, SIGMA)

    return root


class OnlyDistinctKeys:
    """Um mecanismo de mentira, que so olha um atributo."""

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return (features["distinct_keys"] > 10).to_numpy()


def test_the_retrained_mechanisms_decide_as_the_grid(data_root: Path) -> None:
    """A importancia mede os mesmos modelos que decidiram na grade, e nao outros."""
    run = read_importance_run(data_root, SEED, SIGMA)
    mechanisms = mechanisms_of(run, TEST_CONFIGURATION)
    features = run.holdout[list(ATTRIBUTES)]

    directory = layout.run_directory(data_root, SEED, SIGMA)
    rules = pd.read_csv(directory / layout.PREDICTIONS_RULES)
    models = pd.read_csv(directory / layout.PREDICTIONS_ML)

    decided = mechanisms[RULES].predict(features).astype(int)

    assert np.array_equal(decided, rules["predicted"].to_numpy())

    for name in MODEL_NAMES:
        decided = mechanisms[name].predict(features).astype(int)

        assert np.array_equal(decided, models[name].to_numpy()), name


def test_one_row_per_mechanism_and_attribute(data_root: Path) -> None:
    table = run_importance(read_importance_run(data_root, SEED, SIGMA), TEST_CONFIGURATION)

    assert tuple(table.columns) == COLUMNS
    assert len(table) == len(MECHANISMS) * len(ATTRIBUTES)
    assert set(table["mechanism"]) == set(MECHANISMS)
    assert not table["f1_drop"].isna().any()


def test_the_same_run_gives_the_same_importance(data_root: Path) -> None:
    run = read_importance_run(data_root, SEED, SIGMA)

    first = run_importance(run, TEST_CONFIGURATION)
    second = run_importance(run, TEST_CONFIGURATION)

    assert first.equals(second)


def test_an_ignored_attribute_costs_nothing() -> None:
    """Embaralhar o que o mecanismo nao le nao muda decisao nenhuma: queda zero, exata."""
    rng = np.random.default_rng(0)
    rows = 400
    features = pd.DataFrame(rng.integers(0, 20, size=(rows, len(ATTRIBUTES))), columns=list(ATTRIBUTES))
    truth = pd.Series((features["distinct_keys"] > 10).astype(int))

    drops = mechanism_drops(OnlyDistinctKeys(), Holdout(features, truth), permutation_orders(SEED, rows))

    for attribute in ATTRIBUTES:
        is_the_one_it_reads = attribute == "distinct_keys"

        if is_the_one_it_reads:
            assert drops[attribute] > 0.3
        else:
            assert drops[attribute] == 0.0, attribute


def test_the_permutations_depend_only_on_the_seed() -> None:
    """As mesmas permutacoes para a mesma semente, e cada uma e de fato uma permutacao."""
    first = permutation_orders(SEED, 50)
    again = permutation_orders(SEED, 50)
    other = permutation_orders(SEED + 1, 50)

    for attribute in ATTRIBUTES:
        for order, repeated, different in zip(first[attribute], again[attribute], other[attribute]):
            assert np.array_equal(order, repeated)
            assert not np.array_equal(order, different)
            assert np.array_equal(np.sort(order), np.arange(50))


def test_the_sixth_stream_leaves_the_other_five_alone() -> None:
    """Acrescentar o fluxo da importancia nao muda nenhum dado ja gerado."""
    five_streams = SeedSequence(SEED).spawn(5)

    assert STREAM_COUNT == 6

    for subsystem in range(5):
        before = Generator(PCG64(five_streams[subsystem])).random(8)

        assert np.array_equal(stream(SEED, subsystem).random(8), before), subsystem
