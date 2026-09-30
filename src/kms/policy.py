"""A politica do KMS: de uma tentativa ao desfecho.

Esta e a peca de que depende o argumento central do trabalho. O Scenario
Engine emite **tentativas** e nao sabe se elas serao autorizadas; quem
decide e este modulo, avaliando a politica por chave (D-013). Se o gerador
escrevesse o desfecho, o rotulo seria inventado em vez de derivado da
politica, e a comparacao nao sustentaria nada.

Nenhum numero governa este modulo: a decisao e estrutural, nao parametrizada.
Por isso ele nao tem `parameters.py`.
"""

from __future__ import annotations

from typing import NamedTuple

import pandas as pd

SUCCESS = "success"
DENIED_BY_POLICY = "denied_by_policy"
DISABLED_KEY = "disabled_key"
UNKNOWN_KEY = "unknown_key"

OUTCOMES = (SUCCESS, DENIED_BY_POLICY, DISABLED_KEY, UNKNOWN_KEY)
"""Os quatro desfechos possiveis (D-013, grafia pela D-064)."""

COLUMNS = ("event_id", "outcome")
"""As duas colunas de `outcomes.csv` (D-078).

Nada alem disso: o M5 casa por `event_id` e acrescenta `outcome` as sete do
`requests.csv`, fechando as oito do `log.csv`. Coluna de diagnostico aqui
seria coluna que o M5 teria de lembrar de descartar, e o escopo violado e
exatamente o que a D-064 mantem fora do log.
"""


class Repository(NamedTuple):
    """O que o KMS precisa saber para decidir, ja indexado."""

    scope_by_key: dict[str, str]
    """De cada chave existente para o escopo dela."""

    disabled: frozenset[str]
    """As chaves que nascem desabilitadas (D-038)."""

    scopes_by_operator: dict[str, frozenset[str]]
    """De cada operador para os escopos que ele detem."""


def read_repository(keys: pd.DataFrame, operators: pd.DataFrame) -> Repository:
    """Indexa as duas tabelas estaticas para consulta por requisicao.

    Feito uma vez por execucao: a alternativa seria varrer as tabelas a cada
    uma das dezenas de milhares de requisicoes.
    """
    from src.shared.tables import MULTIVALUE_SEPARATOR

    return Repository(
        scope_by_key=dict(zip(keys["key_id"], keys["scope"])),
        disabled=frozenset(keys.loc[keys["status"] == "disabled", "key_id"]),
        scopes_by_operator={
            row.operator_id: frozenset(row.scopes.split(MULTIVALUE_SEPARATOR))
            for row in operators.itertuples()
        },
    )


def outcome_of(operator_id: str, key_id: str, repository: Repository) -> str:
    """O desfecho de uma tentativa, na ordem que a D-077 fixou.

        1. o identificador nao existe   ->  unknown_key
        2. fora dos escopos do operador ->  denied_by_policy
        3. a chave esta desabilitada    ->  disabled_key
        4. nenhum dos anteriores        ->  success

    **Autorizacao antes de estado.** Responder `disabled_key` a quem nao
    detem o escopo confirmaria que a chave existe e revelaria em que estado
    ela esta, que e vazamento por mensagem de erro. Negar sem qualificar e a
    pratica correta, e por acaso e tambem a que acrescenta 4 % a contagem de
    `denied_by_policy`, que e o desfecho escasso.
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
