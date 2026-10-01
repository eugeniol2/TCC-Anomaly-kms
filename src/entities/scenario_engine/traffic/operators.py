from __future__ import annotations

from typing import NamedTuple

import pandas as pd

from src.shared.tables import MULTIVALUE_SEPARATOR


class Operator(NamedTuple):
    operator_id: str
    profile: str
    regime: str
    scopes: tuple[str, ...]
    usual_ips: tuple[str, ...]


def read_operators(frame: pd.DataFrame) -> list[Operator]:
    """Converte a tabela de operadores na forma que o trafego usa.

    A ordem das linhas e preservada: ela participa do sorteio.
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
