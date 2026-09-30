"""Construcao de `keys.csv`: o repositorio de chaves com estado."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.random import Generator

from src.kms.repository.parameters import KeyRepositorySpecification
from src.shared.tables import shuffle_rows


def largest_remainder(weights: np.ndarray, total: int) -> np.ndarray:
    """Reparte `total` em inteiros proporcionais a `weights`, somando exatamente `total`."""
    exact = weights * total
    allocated = np.floor(exact).astype(int)
    leftover = total - int(allocated.sum())

    has_leftover = leftover > 0

    if has_leftover:
        priority = np.argsort(-(exact - allocated))
        allocated[priority[:leftover]] += 1

    return allocated


def split_keys_by_scope(
    rng: Generator, pool: list[str], specification: KeyRepositorySpecification
) -> dict[str, int]:
    """Reparte as chaves entre escopos de forma deliberadamente desigual.

    Escopos de tamanho uniforme fariam o total de chaves distintas acessadas
    variar pouco entre operadores legitimos, e o atacante ficaria destacavel por
    esse atributo isolado. O piso garante que nenhum escopo fique vazio.
    """
    floor = specification.scope_floor
    total = specification.total_keys
    reserved = floor * len(pool)

    floor_exceeds_total = reserved > total

    if floor_exceeds_total:
        raise ValueError(f"piso de {floor} por escopo nao cabe em {total} chaves")

    weights = rng.dirichlet(np.full(len(pool), specification.concentration)) # gera uma distribuicao de Dirichlet para determinar a proporcao de chaves por escopo
    extra = largest_remainder(weights, total - reserved)

    sizes = {}

    for scope, count in zip(pool, extra):
        sizes[scope] = floor + int(count)

    return sizes


def draw_key_ids(rng: Generator, quantity: int) -> list[str]:
    """Identificadores aleatorios de 48 bits, nunca sequenciais (D-009).

    Com identificador sequencial, a enumeracao do atacante produz progressao
    aritmetica e qualquer atributo de distancia separa as classes sozinho.
    """
    seen: set[str] = set()
    identifiers: list[str] = []

    while len(identifiers) < quantity:
        random_value = int(rng.integers(0, 2**48))
        candidate = f"k_{random_value:012x}"

        is_repeated = candidate in seen

        if not is_repeated:
            seen.add(candidate)
            identifiers.append(candidate)

    return identifiers


def build_keys(rng: Generator, sizes: dict[str, int]) -> pd.DataFrame:
    """Repositorio de chaves, todas ativas, agrupadas por escopo.

    **A chave nao tem dono** (D-099). Quem alcanca uma chave e quem detem o
    escopo dela, e o escopo e detido por varios operadores, entao propriedade
    nao decide acesso, nem aqui nem no M4. A tabela teve uma coluna `owner` ate
    24/09; ela nunca foi consumida por modulo nenhum.
    """
    identifiers = iter(draw_key_ids(rng, sum(sizes.values())))

    rows = []

    for scope in sorted(sizes):
        for _ in range(sizes[scope]):
            row = {"key_id": next(identifiers), "scope": scope, "status": "active"}
            rows.append(row)

    return pd.DataFrame(rows)


def disable_random_sample(
    rng: Generator, keys: pd.DataFrame, rate: float
) -> pd.DataFrame:
    """Marca como desabilitada uma fracao das chaves, sem tocar na tabela recebida."""
    draw = rng.random(len(keys))
    is_disabled = draw < rate

    updated = keys.copy()
    updated.loc[is_disabled, "status"] = "disabled"

    return updated


def build_key_repository(
    rng: Generator, pool: list[str], specification: KeyRepositorySpecification
) -> pd.DataFrame:
    """O repositorio inteiro: quantas chaves por escopo, os identificadores, as
    desabilitadas, e as linhas embaralhadas para nao ficarem agrupadas por escopo.

    A ordem dos sorteios faz parte do resultado.
    """
    scope_sizes = split_keys_by_scope(rng, pool, specification) # cria um dicionario que mapeia cada escopo para a quantidade de chaves que ele deve ter
    keys_in_scope_order = build_keys(rng, scope_sizes)
    keys_with_status = disable_random_sample(
        rng, keys_in_scope_order, specification.disabled_rate
    )

    return shuffle_rows(rng, keys_with_status)
