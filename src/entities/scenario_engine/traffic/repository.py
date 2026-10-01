from __future__ import annotations

from typing import NamedTuple

import pandas as pd

from src.entities.scenario_engine.traffic.operators import Operator


class OperatorKeys(NamedTuple):
    in_reach: tuple[str, ...]

    out_of_reach: tuple[str, ...]

    existing: frozenset[str]


def keys_by_scope(keys: pd.DataFrame) -> dict[str, list[str]]:
    """Indice inverso da tabela: de cada escopo para as chaves que ele contem,
    ordenadas.
    """
    grouped: dict[str, list[str]] = {}

    for row in keys.itertuples():
        grouped.setdefault(row.scope, []).append(row.key_id)

    ordered = {}

    for scope, identifiers in grouped.items():
        ordered[scope] = sorted(identifiers)

    return ordered


def reach_of(operator: Operator, by_scope: dict[str, list[str]]) -> tuple[str, ...]:
    """As chaves que um operador alcanca, reunindo os escopos que ele detem."""
    reachable: list[str] = []

    for scope in operator.scopes:
        reachable.extend(by_scope.get(scope, []))

    return tuple(sorted(reachable))


def build_repository(
    keys: pd.DataFrame, operators: list[Operator]
) -> dict[str, OperatorKeys]:
    """ o que cada operador alcança, o que não alcança, e o que existe """
    by_scope = keys_by_scope(keys)  # separa as chaves por escopo
    existing = frozenset(keys["key_id"]) # conjunto de todas as chaves existentes, para forjar identificadores inexistentes sem colisao

    repository: dict[str, OperatorKeys] = {} # dicionario que mapeia cada operador para suas chaves in_reach, out_of_reach e existing

    for operator in operators:
        in_reach = reach_of(operator, by_scope) # chaves que o operador pode acessar
        repository[operator.operator_id] = OperatorKeys(
            in_reach=in_reach,
            out_of_reach=tuple(sorted(existing - set(in_reach))),
            existing=existing,
        )

    return repository
