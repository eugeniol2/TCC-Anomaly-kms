"""Testes do M12: metricas, verificacao de trivialidade, comparacao e tempo.

Tudo aqui roda sobre dados sinteticos, montados a mao para ter resposta
conhecida. Nenhum teste le resultado de deteccao de verdade.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

from src.dataset.build import ATTRIBUTES, LABEL
from src.evaluation.build import (
    ADMINISTRATORS,
    COMPARISON_COLUMNS,
    MECHANISMS,
    METRIC_COLUMNS,
    SCOPES,
    TRIVIALITY_COLUMNS,
    Run,
    build_evaluation,
    comparison,
    confusion,
    duplicates,
    rates,
    read_run,
    run_metrics,
    run_triviality,
    signed_rank_test,
    stump,
    timing_summary,
)
from src.evaluation.parameters import EXCLUSION_F1
from src.models.parameters import MODEL_NAMES


# A matriz de confusao e as taxas.


def test_the_rates_of_a_known_matrix() -> None:
    """Verdade 1 1 0 0 0, decisao 1 0 1 0 0: um de cada erro."""
    counts = confusion([1, 1, 0, 0, 0], [1, 0, 1, 0, 0])

    assert counts == {
        "true_positives": 1, "false_positives": 1,
        "true_negatives": 2, "false_negatives": 1,
    }
    assert rates(counts) == {
        "f1": 0.5, "precision": 0.5, "recall": 0.5,
        "accuracy": 0.6, "specificity": 0.6667,
    }


def test_no_alert_gives_zero_precision_and_zero_f1() -> None:
    """Sem alerta nenhum a precisao nao tem denominador, e vale zero (D-116)."""
    values = rates(confusion([1, 0, 0], [0, 0, 0]))

    assert values["precision"] == 0.0
    assert values["f1"] == 0.0
    assert values["specificity"] == 1.0


# Uma execucao sintetica.


ADMINISTRATOR_IDS = frozenset({"admin_01", "admin_02"})


def synthetic_sessions(count: int, positives: int, prefix: str) -> pd.DataFrame:
    """Sessoes em que so `distinct_keys` separa: alto nas positivas, baixo nas outras.

    As positivas sao todas do `admin_01`; as negativas se dividem entre o `admin_02`
    e um usuario final, como na campanha, que so age sob credencial de administrador.
    """
    rng = np.random.default_rng(0)
    label = np.array([1] * positives + [0] * (count - positives))
    operators = ["admin_01"] * positives + [
        "admin_02" if number % 2 else "user_01" for number in range(count - positives)
    ]
    frame = pd.DataFrame({
        "session_id": [f"{prefix}_{number:05d}" for number in range(count)],
        "operator_id": operators,
        "opened_at": "2026-02-02T10:00:00",
        **{attribute: rng.random(count) for attribute in ATTRIBUTES},
        LABEL: label,
    })
    frame["distinct_keys"] = np.where(label == 1, 40.0, 5.0) + rng.random(count)

    return frame


def synthetic_run(seed: int = 1, sigma: float = 0.5) -> Run:
    """O baseline acerta tudo, o Random Forest tambem, o XGBoost nunca alerta."""
    train = synthetic_sessions(120, 6, "train")
    holdout = synthetic_sessions(80, 4, "holdout")
    truth = holdout[LABEL]

    identifiers = holdout[["session_id", "operator_id"]]
    rules = identifiers.assign(predicted=truth, **{LABEL: truth})
    models = identifiers.assign(
        random_forest=truth, xgboost=0,
        random_forest_score=truth * 0.9 + 0.05, xgboost_score=0.1,
        **{LABEL: truth},
    )
    timing = pd.DataFrame({
        "mechanism": list(MECHANISMS),
        "microseconds_per_session": [1.0, 20.0, 5.0],
        "sessions_per_second": [1e6, 5e4, 2e5],
    })

    return Run(seed, sigma, train, holdout, rules, models, timing, ADMINISTRATOR_IDS)


def test_one_metrics_row_per_mechanism_and_scope() -> None:
    metrics = run_metrics(synthetic_run())

    assert tuple(metrics.columns) == METRIC_COLUMNS
    assert list(metrics["mechanism"]) == [m for m in MECHANISMS for _ in SCOPES]
    assert list(metrics["scope"]) == list(SCOPES) * len(MECHANISMS)
    assert list(metrics["f1"]) == [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]


def test_the_secondary_outcome_counts_only_administrator_sessions() -> None:
    """Proposta, Subsecao 5.4: as sessoes de usuario final saem, as positivas ficam."""
    run = synthetic_run()
    metrics = run_metrics(run)
    administrators = metrics[metrics["scope"] == ADMINISTRATORS].iloc[0]

    expected = int(run.holdout["operator_id"].isin(ADMINISTRATOR_IDS).sum())

    assert administrators["sessions"] == expected
    assert administrators["sessions"] < len(run.holdout)
    assert administrators["positives"] == int(run.holdout[LABEL].sum())


def test_the_roc_auc_exists_for_the_models_and_not_for_the_rules() -> None:
    """Proposta, Tabela 9: o baseline so decide, e nao tem curva (D-119)."""
    metrics = run_metrics(synthetic_run())
    auc = metrics.set_index(["mechanism", "scope"])["roc_auc"]

    assert np.isnan(auc[("rules", "all")])
    assert auc[("random_forest", "all")] == 1.0
    assert auc[("xgboost", "all")] == 0.5


def test_predictions_of_different_sessions_are_refused() -> None:
    run = synthetic_run()
    shuffled = run.predictions_ml.iloc[::-1].reset_index(drop=True)

    with pytest.raises(ValueError, match="sessoes diferentes"):
        run_metrics(run._replace(predictions_ml=shuffled))


# A verificacao de trivialidade.


def test_the_stump_finds_the_attribute_that_separates_alone() -> None:
    run = synthetic_run()
    found = stump(run.train, run.holdout)

    assert found["stump_attribute"] == "distinct_keys"
    assert found["stump_f1"] == 1.0


def test_a_stump_without_a_useful_cut_says_legitimate_to_everything() -> None:
    """Sem atributo informativo a arvore nao corta, e o F1 dela e zero."""
    run = synthetic_run()
    blank = run.train.assign(**{attribute: 0.0 for attribute in ATTRIBUTES})
    found = stump(blank, run.holdout)

    assert found["stump_attribute"] == ""
    assert found["stump_f1"] == 0.0


def test_the_duplicate_check_counts_shared_and_repeated_sessions() -> None:
    run = synthetic_run()
    leaked = pd.concat([run.holdout.iloc[:3], run.train.iloc[:2]], ignore_index=True)

    assert duplicates(run.train, run.holdout) == {"shared_sessions": 0, "repeated_sessions": 0}
    assert duplicates(run.train, leaked) == {"shared_sessions": 2, "repeated_sessions": 2}


def test_the_triviality_row_counts_the_positives_of_each_side() -> None:
    """D-023: a proporcao de anomalias de cada particao, contada."""
    row = run_triviality(synthetic_run()).iloc[0]

    assert tuple(run_triviality(synthetic_run()).columns) == TRIVIALITY_COLUMNS
    assert (row["train_sessions"], row["train_positives"]) == (120, 6)
    assert (row["holdout_sessions"], row["holdout_positives"]) == (80, 4)


# A comparacao.


def comparison_metrics(differences: dict[float, float], seeds: int = 30) -> pd.DataFrame:
    """Metricas em que, em cada sigma, os modelos ganham do baseline pela diferenca dada."""
    rows = []
    rng = np.random.default_rng(1)

    for sigma, difference in differences.items():
        for seed in range(1, seeds + 1):
            base = 0.5 + 0.01 * rng.random()
            f1_of = {"rules": base}

            for model in MODEL_NAMES:
                gain = difference * (1 + rng.random()) if difference else 0.0
                f1_of[model] = base + gain

            for scope in SCOPES:
                rows.extend(
                    {"seed": seed, "sigma": sigma, "mechanism": mechanism,
                     "scope": scope, "f1": f1}
                    for mechanism, f1 in f1_of.items()
                )

    return pd.DataFrame(rows)


def stump_rows(stump_f1: dict[float, float], seeds: int = 30) -> pd.DataFrame:
    return pd.DataFrame([
        {"seed": seed, "sigma": sigma, "stump_f1": value}
        for sigma, value in stump_f1.items()
        for seed in range(1, seeds + 1)
    ])


def test_no_difference_at_all_is_p_one() -> None:
    assert signed_rank_test(pd.Series([0.0] * 30)) == (0.0, 1.0)


def test_a_consistent_gain_is_significant() -> None:
    statistic, p_value = signed_rank_test(pd.Series(np.linspace(0.01, 0.3, 30)))

    assert p_value < 0.001


def test_holm_runs_only_over_the_kept_conditions() -> None:
    """D-111: a condicao excluida pela arvore rasa fica fora da familia de Holm."""
    metrics = comparison_metrics({0.0: 0.2, 0.5: 0.1, 1.0: 0.0})
    stumps = stump_rows({0.0: 1.0, 0.5: 0.6, 1.0: 0.0})

    result = comparison(metrics, stumps)

    assert tuple(result.columns) == COMPARISON_COLUMNS
    assert list(result["excluded"]) == [True, True, False, False, False, False]
    assert result.loc[result["excluded"], "p_holm"].isna().all()
    assert not result.loc[result["excluded"], "significant"].any()

    kept = result[~result["excluded"]]
    expected = multipletests(kept["p_value"], method="holm")[1]

    assert np.allclose(kept["p_holm"], expected, atol=1e-6)


def test_the_exclusion_uses_the_median_of_the_stump() -> None:
    """D-028: F1 mediano da arvore rasa >= 0,95 exclui."""
    metrics = comparison_metrics({0.2: 0.1})
    just_above = stump_rows({0.2: EXCLUSION_F1})
    just_below = stump_rows({0.2: EXCLUSION_F1 - 0.01})

    assert comparison(metrics, just_above)["excluded"].all()
    assert not comparison(metrics, just_below)["excluded"].any()


def test_a_gain_in_every_seed_is_significant_after_holm() -> None:
    result = comparison(comparison_metrics({0.5: 0.1}), stump_rows({0.5: 0.5}))

    assert result["significant"].all()
    assert (result["median_difference"] > 0).all()


# O tempo, e a avaliacao inteira.


def test_the_timing_summary_has_one_row_per_mechanism() -> None:
    timings = pd.concat([synthetic_run().timing] * 3, ignore_index=True)
    summary = timing_summary(timings)

    assert list(summary["mechanism"]) == list(MECHANISMS)
    assert list(summary["runs"]) == [3, 3, 3]
    assert list(summary["median_microseconds"]) == [1.0, 20.0, 5.0]


def test_the_whole_evaluation_from_synthetic_runs() -> None:
    runs = [synthetic_run(seed, sigma) for seed in range(1, 6) for sigma in (0.0, 1.0)]
    evaluation = build_evaluation(runs)

    assert len(evaluation.metrics) == len(runs) * len(MECHANISMS) * len(SCOPES)
    assert len(evaluation.triviality) == len(runs)
    assert len(evaluation.comparison) == 2 * len(MODEL_NAMES)
    assert len(evaluation.timing) == len(MECHANISMS)


def test_a_missing_run_file_is_named(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="timing_rules.csv"):
        read_run(tmp_path, 1, 0.5)


def test_the_administrators_medians_travel_with_the_comparison() -> None:
    """O desfecho secundario vai ao lado do teste, descritivo, sem entrar no Holm."""
    result = comparison(comparison_metrics({0.5: 0.1}), stump_rows({0.5: 0.5}))

    assert result["median_f1_model_administrators"].notna().all()
    assert result["median_f1_rules_administrators"].notna().all()
