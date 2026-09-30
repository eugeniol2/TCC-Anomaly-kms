"""Leitura de `operators.csv`: o que o M2 consome da tabela de populacao.

A fronteira entre modulos e o arquivo, entao o M2 le o CSV do M1 em vez de
importar o codigo que o produziu. Esta e a unica funcao que conhece o formato
daquela tabela; o resto do modulo trabalha sobre `Operator`.
"""

from __future__ import annotations

from typing import NamedTuple

import pandas as pd

from src.shared.tables import MULTIVALUE_SEPARATOR


class Operator(NamedTuple):
    """Uma linha de `operators.csv`, no que o M2 consome dela.

    `profile` entra apenas para escolher a mistura de operacoes; quem decide
    ritmo e `regime`, pelo motivo registrado em `specification.REGIMES`.
    """

    operator_id: str
    profile: str
    regime: str
    scopes: tuple[str, ...]
    usual_ips: tuple[str, ...]


def read_operators(frame: pd.DataFrame) -> list[Operator]:
    """Converte a tabela do M1 na forma que o M2 usa.

    A ordem das linhas e preservada: ela participa do sorteio, entao reordenar
    a tabela mudaria o trafego mesmo com a mesma semente.
    """
    operators = []

    for row in frame.itertuples():
        operators.append(Operator(
            operator_id=row.operator_id,
            profile=row.profile,
            regime=row.regime,
            scopes=tuple(row.scopes.split(MULTIVALUE_SEPARATOR)),
            usual_ips=tuple(row.usual_ips.split(MULTIVALUE_SEPARATOR)),
        ))

    return operators
