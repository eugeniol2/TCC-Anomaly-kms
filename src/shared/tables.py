"""Operacoes sobre as tabelas do pipeline, antes e no momento de virarem CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from numpy.random import Generator

MULTIVALUE_SEPARATOR = "|"
"""Separa varios valores dentro de uma celula, sem colidir com a virgula do CSV.

Usado por `operators.scopes` e `operators.usual_ips`.
"""


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Escreve CSV com cabecalho e quebra de linha fixa em LF.

    LF explicito porque o padrao no Windows e CRLF, e a saida precisa ser
    identica byte a byte em qualquer plataforma.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def shuffle_rows(rng: Generator, frame: pd.DataFrame) -> pd.DataFrame:
    """Embaralha as linhas para que a ordem do arquivo nao carregue agrupamento."""
    order = rng.permutation(len(frame))
    shuffled = frame.iloc[order]

    return shuffled.reset_index(drop=True)
