"""Composicao do M4: das requisicoes de uma fase aos desfechos.

Mora separado da linha de comando para que o teste exercite exatamente o que
a execucao real exercita, e nao uma copia da sequencia de chamadas.
"""

from __future__ import annotations

import pandas as pd

from src.shared.phases import belongs_to
from src.entities.kms.policy import COLUMNS, outcome_of, read_repository


def requests_of_phase(requests: pd.DataFrame, phase: str) -> pd.DataFrame:
    """As requisicoes que caem na fase indicada.

    O `requests.csv` do ramo da semente traz as oito semanas, entao o
    aquecimento precisa recortar as quatro primeiras. O do ramo de sigma ja vem
    so com as semanas 5 a 8, e ali o filtro nao remove nada: aplicar nos dois
    casos custa pouco e evita que o modulo dependa de qual arquivo recebeu.
    """
    return requests[belongs_to(phase, requests["timestamp"])].reset_index(drop=True)


def build_outcomes(
    requests: pd.DataFrame,
    keys: pd.DataFrame,
    operators: pd.DataFrame,
    phase: str,
) -> pd.DataFrame:
    """O desfecho de cada tentativa da fase, uma linha por evento.

    Nao sorteia nada: o KMS e deterministico por construcao. Duas execucoes
    sobre a mesma entrada dao a mesma saida sem depender de semente, e por
    isso este modulo nao recebe nenhuma.
    """
    of_phase = requests_of_phase(requests, phase)
    repository = read_repository(keys, operators)

    outcomes = []

    for row in of_phase.itertuples():
        outcome = outcome_of(row.operator_id, row.key_id, repository)
        outcomes.append(outcome)

    return pd.DataFrame(
        {"event_id": of_phase["event_id"], "outcome": outcomes},
        columns=list(COLUMNS),
    )
