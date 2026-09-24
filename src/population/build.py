"""Composicao do M1: da semente as duas tabelas estaticas.

Mora separado da linha de comando para que o teste exercite exatamente o que a
execucao real exercita, e nao uma copia da sequencia de chamadas.
"""

from __future__ import annotations

from typing import NamedTuple

import pandas as pd

from src.population.keys import build_keys, disable_random_sample, split_keys_by_scope
from src.population.operators import build_operators_covering_pool
from src.population.scopes import scope_pool
from src.population.parameters import KeyRepositorySpecification
from src.globals.rng import POPULATION, stream
from src.globals.tables import shuffle_rows


class Population(NamedTuple):
    """As duas tabelas estaticas que o M1 produz."""

    operators: pd.DataFrame
    keys: pd.DataFrame


def build_population(seed: int, specification: KeyRepositorySpecification) -> Population:
    """Da semente as duas tabelas.

    A ordem das chamadas faz parte do resultado: todas consomem sorteios do
    mesmo fluxo, entao trocar duas de lugar muda a saida inteira mesmo com a
    mesma semente.
    """
    rng = stream(seed, POPULATION)
    pool = scope_pool(specification.scope_count)

    operators = build_operators_covering_pool(rng, pool) # cria os operadores que cobrem todos os escopos

    scope_sizes = split_keys_by_scope(rng, pool, specification) # cria um dicionario que mapeia cada escopo para a quantidade de chaves que ele deve ter
    keys_in_scope_order = build_keys(rng, scope_sizes)
    keys_with_status = disable_random_sample(
        rng, keys_in_scope_order, specification.disabled_rate
    )
    keys = shuffle_rows(rng, keys_with_status)

    return Population(operators, keys)
