"""M12: a avaliacao das 330 execucoes.

Quatro tabelas, e cada uma responde uma pergunta:

- `metrics.csv`: quanto cada mecanismo acertou em cada execucao, com a matriz de
  confusao inteira ao lado do F1 (D-021, D-022). Duas vezes: sobre o holdout
  inteiro, e so sobre as sessoes de administradores, o desfecho secundario da
  proposta (D-119). Para os modelos, tambem a ROC AUC.
- `triviality.csv`: quantas positivas ha em cada lado da particao (D-023); um
  atributo sozinho ja separa as classes? Ha sessao repetida entre treino e
  holdout? (D-028, D-029)
- `comparison.csv`: em cada sigma, cada modelo difere do baseline? Wilcoxon
  pareado por semente, com a correcao de Holm so sobre as condicoes mantidas
  (D-027, D-111, D-116).
- `timing.csv`: quanto cada mecanismo leva para decidir (D-025, D-106).
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from statsmodels.stats.multitest import multipletests

from src.scenario_engine.attack.build import ADMINISTRATOR
from src.policy_engine.baseline.build import MECHANISM as RULES
from src.dataset_generator.dataset.build import ATTRIBUTES, LABEL
from src.evaluation.parameters import (
    ALTERNATIVE,
    EXCLUSION_F1,
    SIGNIFICANCE,
    STUMP_DEPTH,
    STUMP_RANDOM_STATE,
    ZERO_METHOD,
)
from src.shared import layout
from src.models.build import score_column
from src.models.parameters import MODEL_NAMES

MECHANISMS = (RULES,) + MODEL_NAMES

ALL_SESSIONS = "all"
ADMINISTRATORS = "administrators"
SCOPES = (ALL_SESSIONS, ADMINISTRATORS)
"""Sobre o que a metrica e calculada: o holdout inteiro (desfecho primario) ou so as
sessoes de administradores, o perfil que o atacante personifica (desfecho secundario,
proposta, Subsecao 5.4). No segundo, o atalho de reconhecer o papel some (D-118)."""

METRIC_COLUMNS = (
    "seed", "sigma", "mechanism", "scope", "sessions", "positives",
    "true_positives", "false_positives", "true_negatives", "false_negatives",
    "f1", "precision", "recall", "accuracy", "specificity", "roc_auc",
)

TRIVIALITY_COLUMNS = (
    "seed", "sigma", "train_sessions", "train_positives",
    "holdout_sessions", "holdout_positives",
    "stump_attribute", "stump_threshold", "stump_f1",
    "shared_sessions", "repeated_sessions",
)

COMPARISON_COLUMNS = (
    "sigma", "model", "seeds", "median_f1_model", "median_f1_rules",
    "median_difference", "q1_difference", "q3_difference",
    "statistic", "p_value", "stump_median_f1", "excluded", "p_holm", "significant",
    "median_f1_model_administrators", "median_f1_rules_administrators",
)

TIMING_COLUMNS = (
    "mechanism", "runs", "median_microseconds", "q1_microseconds",
    "q3_microseconds", "median_sessions_per_second",
)


class Run(NamedTuple):
    """Tudo que o M12 le de uma execucao (semente, sigma)."""

    seed: int
    sigma: float
    train: pd.DataFrame
    holdout: pd.DataFrame
    predictions_rules: pd.DataFrame
    predictions_ml: pd.DataFrame
    timing: pd.DataFrame
    administrators: frozenset[str]
    """Os operadores de perfil administrador daquela semente, para o desfecho secundario."""


class Evaluation(NamedTuple):
    metrics: pd.DataFrame
    triviality: pd.DataFrame
    comparison: pd.DataFrame
    timing: pd.DataFrame


# As metricas de uma execucao.


def confusion(truth: pd.Series, decided) -> dict[str, int]:
    """A matriz de confusao, com a sessao comprometida como classe positiva."""
    truth = np.asarray(truth)
    decided = np.asarray(decided)

    return {
        "true_positives": int(((truth == 1) & (decided == 1)).sum()),
        "false_positives": int(((truth == 0) & (decided == 1)).sum()),
        "true_negatives": int(((truth == 0) & (decided == 0)).sum()),
        "false_negatives": int(((truth == 1) & (decided == 0)).sum()),
    }


def rates(counts: dict[str, int]) -> dict[str, float]:
    """F1, precisao, revocacao, acuracia e especificidade de uma matriz.

    Sem alerta nenhum, a precisao nao tem denominador e vale zero (D-116). O F1
    nunca fica indefinido: o holdout sempre tem positivas.
    """
    tp = counts["true_positives"]
    fp = counts["false_positives"]
    tn = counts["true_negatives"]
    fn = counts["false_negatives"]
    alerts = tp + fp

    values = {
        "f1": 2 * tp / (2 * tp + fp + fn),
        "precision": tp / alerts if alerts else 0.0,
        "recall": tp / (tp + fn),
        "accuracy": (tp + tn) / (tp + fp + tn + fn),
        "specificity": tn / (tn + fp),
    }

    rounded = {}

    for name, value in values.items():
        rounded[name] = round(value, 4)

    return rounded


def area_under_curve(truth: pd.Series, score: pd.Series | None) -> float:
    """A ROC AUC de um escore continuo (D-119).

    O baseline nao tem escore, so decisao, e fica sem AUC: nao e falha, e o que ele
    e (proposta, Tabela 9).
    """
    has_score = score is not None

    if not has_score:
        return np.nan

    return round(float(roc_auc_score(truth, score)), 4)


def mechanism_outputs(run: Run) -> dict[str, tuple[pd.Series, pd.Series | None]]:
    """A decisao e o escore de cada mecanismo. O baseline so decide."""
    outputs = {RULES: (run.predictions_rules["predicted"], None)}

    for name in MODEL_NAMES:
        outputs[name] = (run.predictions_ml[name], run.predictions_ml[score_column(name)])

    return outputs


def run_metrics(run: Run) -> pd.DataFrame:
    """Uma linha por mecanismo e escopo: o holdout inteiro e so os administradores."""
    same_sessions = run.predictions_rules["session_id"].equals(run.predictions_ml["session_id"])

    if not same_sessions:
        raise ValueError(f"semente {run.seed}, sigma {run.sigma}: predicoes de sessoes diferentes")

    truth = run.predictions_rules[LABEL]
    is_administrator = run.predictions_rules["operator_id"].isin(run.administrators)
    rows_of = {ALL_SESSIONS: truth.index == truth.index, ADMINISTRATORS: is_administrator}

    rows = []

    for mechanism, (decided, score) in mechanism_outputs(run).items():
        for scope, selected in rows_of.items():
            counts = confusion(truth[selected], decided[selected])
            chosen_score = None if score is None else score[selected]
            rows.append({
                "seed": run.seed, "sigma": run.sigma, "mechanism": mechanism,
                "scope": scope, "sessions": int(selected.sum()),
                "positives": int(truth[selected].sum()),
                **counts, **rates(counts),
                "roc_auc": area_under_curve(truth[selected], chosen_score),
            })

    return pd.DataFrame(rows)[list(METRIC_COLUMNS)]


# A verificacao de trivialidade de uma execucao.


def stump(train: pd.DataFrame, holdout: pd.DataFrame) -> dict[str, object]:
    """A arvore de profundidade 1: treina no treino, mede no holdout (D-028, D-114).

    Diz qual atributo ela escolheu e onde cortou. Quando nenhum corte melhora a
    arvore, ela fica sem atributo e responde "legitima" para tudo.
    """
    tree = DecisionTreeClassifier(max_depth=STUMP_DEPTH, random_state=STUMP_RANDOM_STATE)
    tree.fit(train[list(ATTRIBUTES)], train[LABEL])

    root = int(tree.tree_.feature[0])
    has_split = root >= 0
    decided = tree.predict(holdout[list(ATTRIBUTES)])

    return {
        "stump_attribute": ATTRIBUTES[root] if has_split else "",
        "stump_threshold": round(float(tree.tree_.threshold[0]), 4) if has_split else np.nan,
        "stump_f1": rates(confusion(holdout[LABEL], decided))["f1"],
    }


def duplicates(train: pd.DataFrame, holdout: pd.DataFrame) -> dict[str, int]:
    """Sessoes em comum e sessoes do holdout com atributos identicos a alguma do treino (D-029).

    A primeira tem de ser zero: a sessao e uma linha so, e a particao nao a corta.
    A segunda pode nao ser: duas sessoes diferentes podem ter os mesmos oito numeros.
    """
    shared = set(train["session_id"]) & set(holdout["session_id"])
    seen = set()

    for row in train[list(ATTRIBUTES)].to_numpy():
        seen.add(tuple(row))

    repeated = 0

    for row in holdout[list(ATTRIBUTES)].to_numpy():
        is_repeated = tuple(row) in seen

        if is_repeated:
            repeated += 1

    return {"shared_sessions": len(shared), "repeated_sessions": repeated}


def partition_counts(train: pd.DataFrame, holdout: pd.DataFrame) -> dict[str, int]:
    """Sessoes e positivas de cada lado: a proporcao de anomalias, contada (D-023)."""
    return {
        "train_sessions": len(train), "train_positives": int(train[LABEL].sum()),
        "holdout_sessions": len(holdout), "holdout_positives": int(holdout[LABEL].sum()),
    }


def run_triviality(run: Run) -> pd.DataFrame:
    row = {
        "seed": run.seed, "sigma": run.sigma,
        **partition_counts(run.train, run.holdout),
        **stump(run.train, run.holdout), **duplicates(run.train, run.holdout),
    }

    return pd.DataFrame([row])[list(TRIVIALITY_COLUMNS)]


# A comparacao, sobre a grade inteira.


def of_condition(metrics: pd.DataFrame, sigma: float, scope: str) -> pd.DataFrame:
    return metrics[(metrics["sigma"] == sigma) & (metrics["scope"] == scope)]


def median_f1(metrics: pd.DataFrame, sigma: float, scope: str, mechanism: str) -> float:
    condition = of_condition(metrics, sigma, scope)

    return round(condition.loc[condition["mechanism"] == mechanism, "f1"].median(), 4)


def paired_differences(metrics: pd.DataFrame, sigma: float, model: str) -> pd.Series:
    """F1 do modelo menos F1 do baseline, semente a semente, no holdout inteiro."""
    condition = of_condition(metrics, sigma, ALL_SESSIONS)
    f1 = condition.pivot(index="seed", columns="mechanism", values="f1")

    return f1[model] - f1[RULES]


def signed_rank_test(differences: pd.Series) -> tuple[float, float]:
    """Wilcoxon pareado, bilateral (D-116).

    Quando todas as diferencas sao zero nao ha o que testar: estatistica 0 e p 1.
    Com menos de duas sementes, a execucao e parcial e o teste nao existe.
    """
    is_partial = len(differences) < 2

    if is_partial:
        return np.nan, np.nan

    has_difference = bool((differences != 0).any())

    if not has_difference:
        return 0.0, 1.0

    result = wilcoxon(differences, zero_method=ZERO_METHOD, alternative=ALTERNATIVE)

    return float(result.statistic), float(result.pvalue)


def comparison_row(metrics: pd.DataFrame, stumps: pd.Series, sigma: float, model: str) -> dict:
    """O teste e feito so no desfecho primario; os administradores vao ao lado, descritivos."""
    differences = paired_differences(metrics, sigma, model)
    statistic, p_value = signed_rank_test(differences)
    stump_median = float(stumps[sigma])

    return {
        "sigma": sigma,
        "model": model,
        "seeds": len(differences),
        "median_f1_model": median_f1(metrics, sigma, ALL_SESSIONS, model),
        "median_f1_rules": median_f1(metrics, sigma, ALL_SESSIONS, RULES),
        "median_difference": round(differences.median(), 4),
        "q1_difference": round(differences.quantile(0.25), 4),
        "q3_difference": round(differences.quantile(0.75), 4),
        "statistic": statistic,
        "p_value": round(p_value, 6),
        "stump_median_f1": round(stump_median, 4),
        "excluded": stump_median >= EXCLUSION_F1,
        "median_f1_model_administrators": median_f1(metrics, sigma, ADMINISTRATORS, model),
        "median_f1_rules_administrators": median_f1(metrics, sigma, ADMINISTRATORS, RULES),
    }


def with_holm(rows: pd.DataFrame) -> pd.DataFrame:
    """Holm so sobre as condicoes mantidas (D-111); as excluidas ficam sem teste corrigido."""
    kept = ~rows["excluded"] & rows["p_value"].notna()
    corrected = pd.Series(np.nan, index=rows.index)

    if kept.any():
        corrected[kept] = multipletests(rows.loc[kept, "p_value"], method="holm")[1]

    return rows.assign(
        p_holm=corrected.round(6),
        significant=kept & (corrected < SIGNIFICANCE),
    )


def comparison(metrics: pd.DataFrame, triviality: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por sigma e modelo: a diferenca de F1 contra o baseline, e o teste."""
    stumps = triviality.groupby("sigma")["stump_f1"].median()
    rows = []

    for sigma in sorted(metrics["sigma"].unique()):
        for model in MODEL_NAMES:
            rows.append(comparison_row(metrics, stumps, sigma, model))

    return with_holm(pd.DataFrame(rows))[list(COMPARISON_COLUMNS)]


def timing_summary(timings: pd.DataFrame) -> pd.DataFrame:
    """Mediana e quartis do tempo por sessao de cada mecanismo, sobre as execucoes."""
    rows = []

    for mechanism in MECHANISMS:
        of_mechanism = timings[timings["mechanism"] == mechanism]
        per_session = of_mechanism["microseconds_per_session"]
        rows.append({
            "mechanism": mechanism,
            "runs": len(of_mechanism),
            "median_microseconds": round(per_session.median(), 4),
            "q1_microseconds": round(per_session.quantile(0.25), 4),
            "q3_microseconds": round(per_session.quantile(0.75), 4),
            "median_sessions_per_second": round(of_mechanism["sessions_per_second"].median(), 1),
        })

    return pd.DataFrame(rows)[list(TIMING_COLUMNS)]


def build_evaluation(runs: list[Run]) -> Evaluation:
    """Das execucoes as quatro tabelas."""
    metric_frames = []
    triviality_frames = []
    timing_frames = []

    for run in runs:
        metric_frames.append(run_metrics(run))
        triviality_frames.append(run_triviality(run))
        timing_frames.append(run.timing)

    metrics = pd.concat(metric_frames, ignore_index=True)
    triviality = pd.concat(triviality_frames, ignore_index=True)
    timings = pd.concat(timing_frames, ignore_index=True)

    return Evaluation(
        metrics, triviality, comparison(metrics, triviality), timing_summary(timings)
    )


# A leitura de uma execucao do disco.


def read_administrators(root: Path, seed: int) -> frozenset[str]:
    """Os administradores da semente, lidos da populacao do ramo da semente."""
    path = layout.seed_directory(root, seed) / layout.OPERATORS

    if not path.exists():
        raise FileNotFoundError(f"falta `{layout.OPERATORS}` em {path.parent}")

    operators = pd.read_csv(path)
    is_administrator = operators["profile"] == ADMINISTRATOR

    return frozenset(operators.loc[is_administrator, "operator_id"])


def read_run(root: Path, seed: int, sigma: float) -> Run:
    """Os arquivos de uma execucao, com erro claro quando falta algum."""
    directory = layout.run_directory(root, seed, sigma)

    def table(name: str) -> pd.DataFrame:
        path = directory / name

        if not path.exists():
            raise FileNotFoundError(f"falta `{name}` em {directory}")

        return pd.read_csv(path)

    timing = pd.concat(
        [table(layout.TIMING_RULES), table(layout.TIMING_ML)], ignore_index=True
    )

    return Run(
        seed, sigma, table(layout.TRAIN), table(layout.HOLDOUT),
        table(layout.PREDICTIONS_RULES), table(layout.PREDICTIONS_ML), timing,
        read_administrators(root, seed),
    )
