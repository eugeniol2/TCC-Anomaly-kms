from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.shared import layout


@dataclass(frozen=True)
class SeedFiles:
    operators: pd.DataFrame
    keys: pd.DataFrame
    requests: pd.DataFrame
    outcomes: pd.DataFrame
    log: pd.DataFrame
    profiles: pd.DataFrame
    sessions: pd.DataFrame
    thresholds: pd.DataFrame


@dataclass(frozen=True)
class RunFiles:
    requests: pd.DataFrame
    compromised: pd.DataFrame
    run: pd.DataFrame
    outcomes: pd.DataFrame
    log: pd.DataFrame
    sessions: pd.DataFrame
    train: pd.DataFrame
    holdout: pd.DataFrame
    predictions_rules: pd.DataFrame
    predictions_ml: pd.DataFrame
    timing: pd.DataFrame


@dataclass(frozen=True)
class GridFiles:
    metrics: pd.DataFrame
    triviality: pd.DataFrame
    comparison: pd.DataFrame
    timing: pd.DataFrame


SEED_NAMES = (
    layout.OPERATORS, layout.KEYS, layout.REQUESTS, layout.OUTCOMES,
    layout.LOG, layout.HISTORICAL_PROFILES, layout.SESSIONS, layout.THRESHOLDS,
)
RUN_NAMES = (
    layout.REQUESTS, layout.COMPROMISED_SESSIONS, layout.RUN, layout.OUTCOMES,
    layout.LOG, layout.SESSIONS, layout.TRAIN, layout.HOLDOUT,
    layout.PREDICTIONS_RULES, layout.PREDICTIONS_ML,
)
GRID_NAMES = (layout.METRICS, layout.TRIVIALITY, layout.COMPARISON, layout.TIMING)


def missing_files(directory: Path, names: tuple[str, ...]) -> list[str]:
    """Os arquivos que faltam numa pasta. Vazio quando o comando ja rodou."""
    absent = []

    for name in names:
        is_missing = not (directory / name).exists()

        if is_missing:
            absent.append(name)

    return absent


def read_all(directory: Path, names: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    """Le os arquivos, cada um pelo nome, com erro claro quando falta algum."""
    absent = missing_files(directory, names)
    has_absent = len(absent) > 0

    if has_absent:
        raise FileNotFoundError(
            f"faltam {', '.join(absent)} em {directory}; rode antes: python -m src.main"
        )

    frames = {}

    for name in names:
        frames[name] = pd.read_csv(directory / name)

    return frames


def read_seed(root: Path, seed: int) -> SeedFiles:
    files = read_all(layout.seed_directory(root, seed), SEED_NAMES)

    return SeedFiles(
        operators=files[layout.OPERATORS],
        keys=files[layout.KEYS],
        requests=files[layout.REQUESTS],
        outcomes=files[layout.OUTCOMES],
        log=files[layout.LOG],
        profiles=files[layout.HISTORICAL_PROFILES],
        sessions=files[layout.SESSIONS],
        thresholds=files[layout.THRESHOLDS],
    )


def read_run(root: Path, seed: int, sigma: float) -> RunFiles:
    directory = layout.run_directory(root, seed, sigma)
    files = read_all(directory, RUN_NAMES + (layout.TIMING_RULES, layout.TIMING_ML))
    timing = pd.concat(
        [files[layout.TIMING_RULES], files[layout.TIMING_ML]], ignore_index=True
    )

    return RunFiles(
        requests=files[layout.REQUESTS],
        compromised=files[layout.COMPROMISED_SESSIONS],
        run=files[layout.RUN],
        outcomes=files[layout.OUTCOMES],
        log=files[layout.LOG],
        sessions=files[layout.SESSIONS],
        train=files[layout.TRAIN],
        holdout=files[layout.HOLDOUT],
        predictions_rules=files[layout.PREDICTIONS_RULES],
        predictions_ml=files[layout.PREDICTIONS_ML],
        timing=timing,
    )


def read_grid(root: Path) -> GridFiles:
    files = read_all(root, GRID_NAMES)

    return GridFiles(
        metrics=files[layout.METRICS],
        triviality=files[layout.TRIVIALITY],
        comparison=files[layout.COMPARISON],
        timing=files[layout.TIMING],
    )


def read_thresholds(root: Path, seeds: tuple[int, ...]) -> pd.DataFrame:
    """Os limiares das sementes que já rodaram, empilhados, com a coluna `seed`.

    Semente sem `thresholds.csv` fica de fora, e a tabela da página diz quantas
    entraram.
    """
    frames = []

    for seed in seeds:
        path = layout.seed_directory(root, seed) / layout.THRESHOLDS

        if path.exists():
            frames.append(pd.read_csv(path).assign(seed=seed))

    has_any = len(frames) > 0

    if not has_any:
        raise FileNotFoundError(
            f"falta {layout.THRESHOLDS} em {root}; rode antes: python -m src.main"
        )

    return pd.concat(frames, ignore_index=True)


def read_importance(root: Path) -> pd.DataFrame:
    """A importancia por permutacao das execucoes, lida da raiz."""
    path = root / layout.IMPORTANCE

    if not path.exists():
        raise FileNotFoundError(
            f"falta {layout.IMPORTANCE} em {root}; rode antes: python -m src.main"
        )

    return pd.read_csv(path)


def figure_paths(root: Path) -> dict[str, Path]:
    """As figuras que o comando desenhou, as que existirem."""
    directory = root / layout.FIGURES
    names = {"f1_sigma": "f1_sigma.png", "roc": "roc.png"}
    paths = {}

    for name, file in names.items():
        path = directory / file

        if path.exists():
            paths[name] = path

    return paths
