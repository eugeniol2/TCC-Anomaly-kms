from __future__ import annotations

from typing import NamedTuple

import pandas as pd

from src.entities.kms.repository.keys import build_key_repository
from src.entities.kms.repository.parameters import KeyRepositorySpecification
from src.entities.kms.repository.scopes import scope_pool
from src.entities.scenario_engine.population.operators import build_operators_covering_pool
from src.shared.rng import POPULATION, stream


class Population(NamedTuple):
    operators: pd.DataFrame
    keys: pd.DataFrame


def build_population(seed: int, specification: KeyRepositorySpecification) -> Population:
    """Da semente as duas tabelas.

    Trocar a ordem das duas chamadas muda a saida inteira, mesmo com a mesma
    semente: as duas consomem sorteios do mesmo fluxo.
    """
    rng = stream(seed, POPULATION)
    pool = scope_pool(specification.scope_count)

    operators = build_operators_covering_pool(rng, pool) # cria os operadores que cobrem todos os escopos
    keys = build_key_repository(rng, pool, specification)

    return Population(operators, keys)
