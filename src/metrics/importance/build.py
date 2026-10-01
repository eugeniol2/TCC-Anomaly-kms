"""M12, a importancia por permutacao: de que atributo cada mecanismo depende (D-124).

Em cada execucao, embaralha um atributo do holdout por vez, deixando os outros como
estao, e mede quanto o F1 cai. Atributo de que o mecanismo nao depende pode ser
embaralhado sem custo; atributo de que ele depende derruba o F1. Vale igual para as
regras e para os modelos, e e isso que permite pôr os tres lado a lado.

**Nao e SHAP** (D-059). Nao explica decisao individual: diz, para a execucao inteira,
o quanto cada mecanismo se apoia em cada atributo. Ocupa o lugar da remedicao que a
D-114 tirou, e responde a pergunta que ela deixou para Ameacas a validade: se a
vantagem dos modelos depende do `distinct_keys`.

**Os modelos sao retreinados**, com a configuracao da 902 e a semente de treino da
replica (D-104, D-115). O treino e deterministico, e o modelo e o mesmo que decidiu
na grade: o teste confere a decisao contra o `predictions_ml.csv` gravado.

**Mede o quanto o atributo e indispensavel, e nao o quanto e usado.** Embaralhar um
atributo deixa os outros carregando o que eles tem em comum com ele, e a queda sai
menor do que o uso real. Em sigma 0,0 o atacante se denuncia por varios atributos ao
mesmo tempo, e o Random Forest nao perde nada com nenhum deles embaralhado sozinho.
Pela mesma razao, atributos correlacionados dividem a importancia: `events`,
`duration_minutes` e `requests_per_minute` andam juntos, e no atacante
`distinct_keys` anda com as falhas.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from src.entities.dataset_generator.dataset.build import ATTRIBUTES, LABEL
from src.entities.models.build import fitted, seed_of, training_seeds
from src.entities.models.parameters import MODEL_NAMES
from src.entities.policy_engine.baseline.build import MECHANISM as RULES
from src.entities.policy_engine.baseline.build import decide, magnitude_cutoffs
from src.formulas.classification import confusion, rates
from src.metrics.evaluation.build import read_table
from src.metrics.importance.parameters import REPEATS
from src.shared import layout
from src.shared.rng import IMPORTANCE, stream

COLUMNS = ("seed", "sigma", "mechanism", "attribute", "f1_drop")
"""As colunas do `importance.csv`: uma linha por execucao, mecanismo e atributo.

`f1_drop` e o F1 da execucao menos a media do F1 com o atributo embaralhado. Pode
sair levemente negativo: embaralhar um atributo de que o mecanismo nao depende as
vezes acerta uma sessao a mais, por acaso.
"""


class ImportanceRun(NamedTuple):
    """O que a importancia le de uma execucao (semente, sigma)."""

    seed: int
    sigma: float
    train: pd.DataFrame
    holdout: pd.DataFrame
    thresholds: pd.DataFrame


class Holdout(NamedTuple):
    """O holdout separado no que o mecanismo ve e na verdade, que so a medida usa."""

    features: pd.DataFrame
    truth: pd.Series


class RulesDecision(NamedTuple):
    """O baseline com a mesma forma dos modelos: um `predict` sobre os oito atributos."""

    cutoffs: np.ndarray

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return decide(features, self.cutoffs)


def read_importance_run(root: Path, seed: int, sigma: float) -> ImportanceRun:
    """O treino e o holdout da execucao, e os limiares da semente."""
    directory = layout.run_directory(root, seed, sigma)

    return ImportanceRun(
        seed=seed,
        sigma=sigma,
        train=read_table(directory, layout.TRAIN),
        holdout=read_table(directory, layout.HOLDOUT),
        thresholds=read_table(layout.seed_directory(root, seed), layout.THRESHOLDS),
    )


def mechanisms_of(run: ImportanceRun, configuration: dict[str, dict]) -> dict[str, object]:
    """Os tres mecanismos da execucao: as regras com os limiares, e os modelos retreinados."""
    mechanisms = {RULES: RulesDecision(magnitude_cutoffs(run.thresholds))}
    seeds = training_seeds(run.seed)

    for name in MODEL_NAMES:
        random_state = seed_of(seeds, name)
        mechanisms[name] = fitted(name, configuration[name], run.train, random_state)

    return mechanisms


def permutation_orders(seed: int, rows: int) -> dict[str, list[np.ndarray]]:
    """As permutacoes de cada atributo, sorteadas uma vez e usadas pelos tres mecanismos.

    Saem do fluxo `IMPORTANCE`, que depende so da semente: o mesmo embaralhamento
    serve as regras e aos dois modelos, e a comparacao entre eles fica pareada.
    """
    rng = stream(seed, IMPORTANCE)
    orders = {}

    for attribute in ATTRIBUTES:
        draws = []

        for _ in range(REPEATS):
            draws.append(rng.permutation(rows))

        orders[attribute] = draws

    return orders


def f1_of(truth: pd.Series, decided) -> float:
    return rates(confusion(truth, decided))["f1"]


def with_shuffled(features: pd.DataFrame, attribute: str, order: np.ndarray) -> pd.DataFrame:
    """Os atributos com uma coluna so embaralhada, e as outras intactas."""
    shuffled = features.copy()
    values = features[attribute].to_numpy()
    shuffled[attribute] = values[order]

    return shuffled


def mechanism_drops(
    mechanism, holdout: Holdout, orders: dict[str, list[np.ndarray]]
) -> dict[str, float]:
    """Quanto o F1 de um mecanismo cai, em media, com cada atributo embaralhado."""
    original = f1_of(holdout.truth, mechanism.predict(holdout.features))
    drops = {}

    for attribute in ATTRIBUTES:
        scores = []

        for order in orders[attribute]:
            shuffled = with_shuffled(holdout.features, attribute, order)
            scores.append(f1_of(holdout.truth, mechanism.predict(shuffled)))

        drops[attribute] = original - float(np.mean(scores))

    return drops


def run_importance(run: ImportanceRun, configuration: dict[str, dict]) -> pd.DataFrame:
    """A importancia de cada atributo para cada mecanismo, numa execucao."""
    holdout = Holdout(run.holdout[list(ATTRIBUTES)], run.holdout[LABEL])
    orders = permutation_orders(run.seed, len(run.holdout))
    rows = []

    for name, mechanism in mechanisms_of(run, configuration).items():
        drops = mechanism_drops(mechanism, holdout, orders)

        for attribute in ATTRIBUTES:
            rows.append({
                "seed": run.seed,
                "sigma": run.sigma,
                "mechanism": name,
                "attribute": attribute,
                "f1_drop": round(drops[attribute], 4),
            })

    return pd.DataFrame(rows)[list(COLUMNS)]
