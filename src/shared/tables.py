from __future__ import annotations

from pathlib import Path

import pandas as pd
from numpy.random import Generator

MULTIVALUE_SEPARATOR = "|"


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Escreve CSV com cabecalho e quebra de linha fixa em LF, criando a pasta se
    faltar.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def shuffle_rows(rng: Generator, frame: pd.DataFrame) -> pd.DataFrame:
    """Embaralha as linhas para que a ordem do arquivo nao carregue agrupamento."""
    order = rng.permutation(len(frame))
    shuffled = frame.iloc[order]

    return shuffled.reset_index(drop=True)
