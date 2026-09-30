"""Testes do M11: os modelos, a busca e a configuracao.

Como no baseline, nenhum teste mede acerto no holdout. Eles conferem que o
mecanismo e o descrito: treina so com os oito atributos, nao consulta o rotulo do
holdout, sorteia do fluxo proprio, e a busca escolhe pela regra da D-103.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd
import pytest
from numpy.random import PCG64, Generator, SeedSequence
from sklearn.model_selection import ParameterGrid

import src.models.build as models_build
from src.attack.build import build_attack
from src.attack.parameters import AttackSpecification
from src.audit_logger.build import build_log
from src.dataset.build import IDENTIFIERS, LABEL, build_dataset
from src.globals.phases import EVALUATED, WARMUP
from src.globals.rng import PARTITION, stream
from src.globals.tables import write_csv
from src.globals.timing import COLUMNS as TIMING_COLUMNS
from src.historical_profiles.build import build_profiles
from src.kms.build import build_outcomes
from src.models.build import (
    COLUMNS,
    build_models,
    chosen_configuration,
    configuration_scores,
    read_configuration,
    training_seeds,
    xgboost,
)
from src.models.parameters import GRIDS, MODEL_NAMES
from src.partition.build import Partition, build_partition
from src.population.build import build_population
from src.population.parameters import KeyRepositorySpecification
from src.traffic.build import build_traffic
from src.traffic.parameters import TrafficSpecification

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


@lru_cache(maxsize=None)
def partition(seed: int) -> Partition:
    """Treino e holdout de uma semente, em sigma 0,5."""
    tables = build_population(seed, KeyRepositorySpecification())
    traffic = build_traffic(seed, tables.operators, tables.keys, TrafficSpecification())

    warmup_outcomes = build_outcomes(traffic, tables.keys, tables.operators, WARMUP)
    profiles = build_profiles(build_log(traffic, warmup_outcomes, WARMUP))

    campaign = build_attack(
        seed, 0.5, tables.operators, tables.keys, traffic,
        TrafficSpecification(), AttackSpecification(),
    )
    outcomes = build_outcomes(campaign.requests, tables.keys, tables.operators, EVALUATED)
    log = build_log(campaign.requests, outcomes, EVALUATED)

    return build_partition(seed, build_dataset(log, profiles, EVALUATED, campaign.compromised))


def predictions(seed: int, train: pd.DataFrame, holdout: pd.DataFrame) -> pd.DataFrame:
    return build_models(seed, train, holdout, TEST_CONFIGURATION).predictions


# As decisoes.


def test_one_decision_per_holdout_session_and_model() -> None:
    sides = partition(1)
    decided = predictions(1, sides.train, sides.holdout)

    assert tuple(decided.columns) == COLUMNS
    assert list(decided["session_id"]) == list(sides.holdout["session_id"])

    for name in MODEL_NAMES:
        assert set(decided[name]) <= {0, 1}


def test_the_holdout_label_is_carried_but_never_consulted() -> None:
    """Zerar ou inverter o rotulo do holdout nao muda decisao nenhuma (D-113)."""
    sides = partition(1)
    reference = predictions(1, sides.train, sides.holdout)

    zeroed = sides.holdout.assign(**{LABEL: 0})
    inverted = sides.holdout.assign(**{LABEL: 1 - sides.holdout[LABEL]})

    for relabeled in (zeroed, inverted):
        decided = predictions(1, sides.train, relabeled)

        assert decided[list(MODEL_NAMES)].equals(reference[list(MODEL_NAMES)])
        assert list(decided[LABEL]) == list(relabeled[LABEL])


def test_the_identifiers_never_reach_the_models() -> None:
    """Trocar identificadores no treino e no holdout nao muda decisao (D-015)."""
    sides = partition(1)
    reference = predictions(1, sides.train, sides.holdout)

    def disguised(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.assign(**{column: "x" for column in IDENTIFIERS})

    decided = predictions(1, disguised(sides.train), disguised(sides.holdout))

    assert decided[list(MODEL_NAMES)].equals(reference[list(MODEL_NAMES)])


def test_the_same_seed_trains_the_same_models() -> None:
    sides = partition(1)

    first = predictions(1, sides.train, sides.holdout)
    second = predictions(1, sides.train, sides.holdout)

    assert first.equals(second)


def test_the_timing_has_one_row_per_model() -> None:
    sides = partition(1)
    timing = build_models(1, sides.train, sides.holdout, TEST_CONFIGURATION).timing

    assert tuple(timing.columns) == TIMING_COLUMNS
    assert list(timing["mechanism"]) == list(MODEL_NAMES)
    assert (timing["median_nanoseconds"] > 0).all()


# As sementes.


def test_the_training_seeds_depend_only_on_the_seed() -> None:
    """Mesma replica, mesmas sementes: o modelo treina igual nos onze sigmas (D-104)."""
    assert training_seeds(1) == training_seeds(1)
    assert training_seeds(1) != training_seeds(2)


@pytest.mark.parametrize("seed", (1, 15, 30))
def test_the_fifth_stream_did_not_move_the_partition_stream(seed: int) -> None:
    """O quinto fluxo nao desloca o quarto: nenhuma particao ja gerada muda."""
    four_streams = SeedSequence(seed).spawn(4)
    before = Generator(PCG64(four_streams[PARTITION])).random(8)

    assert list(stream(seed, PARTITION).random(8)) == list(before)


# O peso da classe rara (D-105).


def test_the_balanced_weight_is_negatives_over_positives_of_the_train_in_use() -> None:
    labels = pd.Series([0] * 30 + [1] * 3)
    parameters = TEST_CONFIGURATION["xgboost"]

    weighted = xgboost(parameters, labels, random_state=0)
    unweighted = xgboost({**parameters, "class_weight": None}, labels, random_state=0)

    assert weighted.get_params()["scale_pos_weight"] == 10.0
    assert unweighted.get_params()["scale_pos_weight"] == 1.0


# A busca e a configuracao.


def test_the_choice_prefers_the_higher_mean_then_the_lower_spread() -> None:
    """D-103: maior F1 medio; no empate, menor desvio; persistindo, a primeira da grade."""
    scores = pd.DataFrame([
        {"model": "random_forest", "position": 0, "mean_f1": 0.80, "std_f1": 0.05},
        {"model": "random_forest", "position": 1, "mean_f1": 0.90, "std_f1": 0.09},
        {"model": "random_forest", "position": 2, "mean_f1": 0.90, "std_f1": 0.02},
        {"model": "xgboost", "position": 3, "mean_f1": 0.70, "std_f1": 0.01},
        {"model": "xgboost", "position": 1, "mean_f1": 0.70, "std_f1": 0.01},
    ])

    chosen = chosen_configuration(scores)

    forest = dict(ParameterGrid(GRIDS["random_forest"])[2])
    boosting = dict(ParameterGrid(GRIDS["xgboost"])[1])

    assert chosen[chosen["model"] == "random_forest"].set_index("parameter")["value"].to_dict() == {
        name: str(value) for name, value in forest.items()
    }
    assert chosen[chosen["model"] == "xgboost"].set_index("parameter")["value"].to_dict() == {
        name: str(value) for name, value in boosting.items()
    }


def test_the_configuration_survives_the_csv_with_its_types(tmp_path: Path) -> None:
    """None, inteiro, fracionario e texto voltam do CSV como eram."""
    scores = pd.DataFrame([
        {"model": name, "position": 0, "mean_f1": 0.5, "std_f1": 0.1}
        for name in MODEL_NAMES
    ])
    write_csv(chosen_configuration(scores), tmp_path / "config.csv")

    configuration = read_configuration(tmp_path / "config.csv")

    for name in MODEL_NAMES:
        assert configuration[name] == dict(ParameterGrid(GRIDS[name])[0])


def test_a_missing_configuration_points_to_the_search(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="--search"):
        read_configuration(tmp_path / "config.csv")


def test_the_search_scores_every_configuration_of_the_grid(monkeypatch) -> None:
    """Com uma grade minima: uma nota por configuracao, e so no treino dado."""
    tiny = {
        "random_forest": {**{k: [v] for k, v in TEST_CONFIGURATION["random_forest"].items()},
                          "n_estimators": [5, 10]},
        "xgboost": {k: [v] for k, v in TEST_CONFIGURATION["xgboost"].items()},
    }
    monkeypatch.setattr(models_build, "GRIDS", tiny)

    scores = list(configuration_scores(1, partition(1).train))

    assert [score["model"] for score in scores] == ["random_forest", "random_forest", "xgboost"]

    for score in scores:
        assert 0.0 <= score["mean_f1"] <= 1.0
        assert score["std_f1"] >= 0.0
