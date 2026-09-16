"""Testes da grade experimental.

A invariante que importa aqui e a separacao entre as sementes das replicas e as
das preparacoes. Se ela se romper, o baseline passa a ser calibrado sobre dado
que depois julga, e a configuracao de hiperparametros passa a ser escolhida sobre
uma populacao que reaparece entre as avaliadas — as duas sem erro em lugar nenhum.
"""

from __future__ import annotations

import pytest

from src.shared.experiment import RESERVED_SEEDS, SEEDS, SIGMAS, TOTAL_RUNS


@pytest.mark.parametrize("seed", RESERVED_SEEDS)
def test_reserved_seeds_are_outside_the_replication_grid(seed: int) -> None:
    assert seed not in SEEDS


def test_reserved_seeds_are_distinct_from_each_other() -> None:
    assert len(set(RESERVED_SEEDS)) == len(RESERVED_SEEDS)


def test_grid_has_the_declared_size() -> None:
    assert len(SEEDS) == 30
    assert len(SIGMAS) == 11
    assert TOTAL_RUNS == 330


def test_sigma_covers_the_whole_interval() -> None:
    assert SIGMAS[0] == 0.0
    assert SIGMAS[-1] == 1.0
