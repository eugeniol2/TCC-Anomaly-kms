from __future__ import annotations

from typing import NamedTuple

import pandas as pd

SUCCESS = "success"
DENIED_BY_POLICY = "denied_by_policy"
DISABLED_KEY = "disabled_key"
UNKNOWN_KEY = "unknown_key"

OUTCOMES = (SUCCESS, DENIED_BY_POLICY, DISABLED_KEY, UNKNOWN_KEY)

COLUMNS = ("event_id", "outcome")


class Repository(NamedTuple):
    scope_by_key: dict[str, str]

    disabled: frozenset[str]

    scopes_by_operator: dict[str, frozenset[str]]


def read_repository(keys: pd.DataFrame, operators: pd.DataFrame) -> Repository:
    """Indexa as duas tabelas estaticas para consulta por requisicao."""
    from src.shared.tables import MULTIVALUE_SEPARATOR

    scopes_by_operator = {}

    for row in operators.itertuples():
        scopes = row.scopes.split(MULTIVALUE_SEPARATOR)
        scopes_by_operator[row.operator_id] = frozenset(scopes)

    return Repository(
        scope_by_key=dict(zip(keys["key_id"], keys["scope"])),
        disabled=frozenset(keys.loc[keys["status"] == "disabled", "key_id"]),
        scopes_by_operator=scopes_by_operator,
    )


def outcome_of(operator_id: str, key_id: str, repository: Repository) -> str:
    """O desfecho de uma tentativa, verificado nesta ordem:

        1. o identificador nao existe   ->  unknown_key
        2. fora dos escopos do operador ->  denied_by_policy
        3. a chave esta desabilitada    ->  disabled_key
        4. nenhum dos anteriores        ->  success
    """
    scope = repository.scope_by_key.get(key_id)

    is_unknown = scope is None

    if is_unknown:
        return UNKNOWN_KEY

    is_out_of_scope = scope not in repository.scopes_by_operator[operator_id]

    if is_out_of_scope:
        return DENIED_BY_POLICY

    is_disabled = key_id in repository.disabled

    if is_disabled:
        return DISABLED_KEY

    return SUCCESS
