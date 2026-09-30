"""Leitura de `keys.csv`: o que o M2 precisa saber sobre as chaves.

Montado uma vez por execucao e consultado em toda sessao. A atividade legitima
e dirigida por **escopo**, e nao ha o que a dirija de outro jeito: a chave nao
tem dono (D-099). A tabela teve uma coluna `owner` ate 24/09, e a D-042 ja a
declarava nao consumida: a coluna so descrevia um mecanismo que o pipeline
nunca usou.
"""

from __future__ import annotations

from typing import NamedTuple

import pandas as pd

from src.scenario_engine.traffic.operators import Operator


class OperatorKeys(NamedTuple):
    """As tres visoes do repositorio, do ponto de vista de um operador."""

    in_reach: tuple[str, ...]
    """Chaves dos escopos que ele detem. Alvo normal das requisicoes."""

    out_of_reach: tuple[str, ...]
    """Chaves que existem e estao fora dos escopos dele.

    Alvo da referencia a escopo obsoleto, que o M4 nega por politica (D-056).
    """

    existing: frozenset[str]
    """Todo identificador do repositorio.

    Serve para forjar identificador inexistente sem colidir com chave real.
    Compartilhado entre todos os operadores; e o repositorio inteiro.
    """


def keys_by_scope(keys: pd.DataFrame) -> dict[str, list[str]]:
    """Indice inverso da tabela: de cada escopo para as chaves que ele contem.

    Ordenado para nao herdar o embaralhamento de linhas do M1. O alcance de um
    operador passa a depender so de quais chaves estao no escopo, e nao da
    ordem em que foram gravadas.
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
