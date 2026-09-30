"""M11: Random Forest e XGBoost, treinados no treino e decidindo sobre o holdout.

Cada modelo treina **uma vez** (D-114), com a configuracao que a busca da
preparatoria 902 escolheu (D-032, D-103, D-105) e a semente de treino da replica
(D-104). Como no baseline, o rotulo do holdout vai junto na saida, mas nao e
consultado (D-113).

A busca tambem mora aqui: ela roda so no treino da 902, nunca no holdout (D-045).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, NamedTuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid, RepeatedStratifiedKFold
from xgboost import XGBClassifier

from src.dataset_generator.dataset.build import ATTRIBUTES, IDENTIFIERS, LABEL, require_label
from src.shared.rng import MODELS, stream
from src.shared.timing import time_decision
from src.models.parameters import FOLDS, GRIDS, JOBS, MODEL_NAMES, REPEATS, SEED_CEILING

def score_column(name: str) -> str:
    """A coluna do escore de um modelo: a probabilidade de ataque, de onde sai a curva ROC (D-119)."""
    return f"{name}_score"


def score_columns() -> tuple[str, ...]:
    columns = []

    for name in MODEL_NAMES:
        columns.append(score_column(name))

    return tuple(columns)


SCORE_COLUMNS = score_columns()

COLUMNS = IDENTIFIERS + MODEL_NAMES + SCORE_COLUMNS + (LABEL,)
"""As colunas do `predictions_ml.csv`: a decisao e o escore de cada modelo, e a verdade por ultimo."""

CONFIGURATION_COLUMNS = ("model", "parameter", "value")


class TrainingSeeds(NamedTuple):
    """As sementes que o M11 usa numa replica, todas do fluxo `MODELS`."""

    random_forest: int
    xgboost: int
    folds: int


class Models(NamedTuple):
    """O que o M11 produz: as decisoes e o tempo que elas levaram."""

    predictions: pd.DataFrame
    timing: pd.DataFrame


def training_seeds(seed: int) -> TrainingSeeds:
    """As sementes da replica, sorteadas sempre na mesma ordem (D-104, D-115).

    Dependem so da semente: o mesmo modelo treina com a mesma semente nos onze
    sigmas. A das dobras so e usada na busca.
    """
    rng = stream(seed, MODELS)
    forest, boosting, folds = rng.integers(0, SEED_CEILING, size=3)

    return TrainingSeeds(int(forest), int(boosting), int(folds))


def positive_weight(labels: pd.Series) -> float:
    """Negativas / positivas do treino em uso: o peso da classe rara (D-105)."""
    positives = int(labels.sum())
    negatives = len(labels) - positives

    return negatives / positives


def random_forest(parameters: dict, labels: pd.Series, random_state: int):
    return RandomForestClassifier(**parameters, random_state=random_state, n_jobs=JOBS)


def xgboost(parameters: dict, labels: pd.Series, random_state: int):
    settings = dict(parameters)
    is_weighted = settings.pop("class_weight") == "balanced"
    weight = positive_weight(labels) if is_weighted else 1.0

    return XGBClassifier(
        **settings, scale_pos_weight=weight, random_state=random_state, n_jobs=JOBS
    )


MODEL_FACTORIES = {"random_forest": random_forest, "xgboost": xgboost}


def fitted(name: str, parameters: dict, sessions: pd.DataFrame, random_state: int):
    """Um modelo treinado nas sessoes dadas, so com os oito atributos."""
    labels = sessions[LABEL]
    model = MODEL_FACTORIES[name](parameters, labels, random_state)

    return model.fit(sessions[list(ATTRIBUTES)], labels)


# A busca, so na preparatoria 902.


def described(parameters: dict) -> str:
    """Uma configuracao numa linha legivel, para o arquivo de resultados da busca."""
    parts = []

    for name, value in parameters.items():
        parts.append(f"{name}={value}")

    return "; ".join(parts)


def fold_score(name: str, parameters: dict, train: pd.DataFrame, fold, random_state: int) -> float:
    """O F1 de uma configuracao numa dobra: treina numa parte, mede na outra.

    Dobra que nao preve positiva nenhuma vale F1 zero.
    """
    fit_rows, check_rows = fold
    model = fitted(name, parameters, train.iloc[fit_rows], random_state)
    checked = train.iloc[check_rows]
    decisions = model.predict(checked[list(ATTRIBUTES)])

    return float(f1_score(checked[LABEL], decisions, zero_division=0))


def configuration_scores(seed: int, train: pd.DataFrame) -> Iterator[dict]:
    """A nota de cada configuracao da grade: media e desvio dos 15 F1 (D-103).

    Devolve uma configuracao por vez, para quem chama poder mostrar o progresso.
    """
    require_label(train)
    seeds = training_seeds(seed)
    splitter = RepeatedStratifiedKFold(
        n_splits=FOLDS, n_repeats=REPEATS, random_state=seeds.folds
    )
    folds = list(splitter.split(train, train[LABEL]))

    for name in MODEL_NAMES:
        random_state = getattr(seeds, name)

        for position, parameters in enumerate(ParameterGrid(GRIDS[name])):
            scores = []

            for fold in folds:
                score = fold_score(name, parameters, train, fold, random_state)
                scores.append(score)

            yield {
                "model": name,
                "position": position,
                "parameters": described(parameters),
                "mean_f1": round(float(np.mean(scores)), 4),
                "std_f1": round(float(np.std(scores)), 4),
            }


def chosen_configuration(scores: pd.DataFrame) -> pd.DataFrame:
    """A melhor configuracao de cada modelo: maior F1 medio; no empate, menor desvio (D-103).

    Persistindo o empate, vence a que vem primeiro na grade.
    """
    ranked = scores.sort_values(
        ["model", "mean_f1", "std_f1", "position"],
        ascending=[True, False, True, True],
    )
    best = ranked.groupby("model", sort=False).head(1)

    rows = []

    for name, position in zip(best["model"], best["position"]):
        parameters = ParameterGrid(GRIDS[name])[int(position)]

        for parameter, value in sorted(parameters.items()):
            row = {"model": name, "parameter": parameter, "value": str(value)}
            rows.append(row)

    return pd.DataFrame(rows)[list(CONFIGURATION_COLUMNS)]


def parsed(text: str):
    """O valor de um parametro, lido de volta do CSV com o tipo certo."""
    if text == "None":
        return None

    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            pass

    return text


def configuration_of(frame: pd.DataFrame) -> dict[str, dict]:
    """O `config.csv` como dicionario: modelo -> parametros."""
    configuration = {}

    for name in MODEL_NAMES:
        configuration[name] = {}

    for row in frame.astype(str).itertuples():
        configuration[row.model][row.parameter] = parsed(row.value)

    return configuration


def read_configuration(path: Path) -> dict[str, dict]:
    """Le o `config.csv` da 902, com erro claro quando a busca ainda nao rodou.

    Lido como texto puro: o pandas trataria "None" como celula vazia.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"falta `{path.name}` em {path.parent}"
            "\n       rode o comando sem --sem-busca, para a busca rodar antes:  python -m src.main"
        )

    frame = pd.read_csv(path, dtype=str, keep_default_na=False)

    return configuration_of(frame)


# O treino e a decisao, nas 330 execucoes.


def build_models(
    seed: int, train: pd.DataFrame, holdout: pd.DataFrame, configuration: dict[str, dict]
) -> Models:
    """Treina os dois modelos e decide sobre o holdout, cronometrando so a decisao (D-106).

    Grava tambem o escore, a probabilidade de ataque, que a curva ROC usa (D-119). A
    decisao continua sendo a do `predict`, que corta em 0,5.
    """
    require_label(train)
    require_label(holdout)

    seeds = training_seeds(seed)
    features = holdout[list(ATTRIBUTES)]

    outputs = {}
    timings = []

    for name in MODEL_NAMES:
        model = fitted(name, configuration[name], train, getattr(seeds, name))
        outputs[name] = model.predict(features).astype(int)
        outputs[score_column(name)] = model.predict_proba(features)[:, 1].round(6)
        timings.append(time_decision(name, len(holdout), lambda: model.predict(features)))

    predictions = holdout[list(IDENTIFIERS)].assign(**outputs, **{LABEL: holdout[LABEL]})

    return Models(
        predictions[list(COLUMNS)].reset_index(drop=True),
        pd.concat(timings, ignore_index=True),
    )
