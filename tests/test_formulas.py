"""Testes das formulas e da separacao entre contextos (D-123).

As formulas sao testadas com numeros escolhidos a mao, de resposta conhecida. Os
dois ultimos testes conferem a direcao das dependencias: `formulas` nao importa nada
do projeto, e nenhuma entidade importa da avaliacao.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.formulas.apportionment import largest_remainder
from src.formulas.classification import (
    confusion,
    negatives_per_positive,
    rates,
    roc_auc,
    roc_point,
)
from src.formulas.distributions import (
    chance_never_drawn,
    geometric_weights,
    negative_binomial_success,
    share_above,
    share_above_with_tail,
)
from src.formulas.hypothesis_tests import holm, signed_rank_test
from src.formulas.interpolation import interpolate, interpolate_range
from src.formulas.rounding import round_half_up

SOURCE = Path(__file__).resolve().parent.parent / "src"


# Interpolacao.


def test_the_interpolation_returns_the_exact_ends() -> None:
    """A forma com dois pesos devolve o extremo exato, e nao 0,005000000000000004."""
    assert interpolate(0.0, 0.30, 0.005) == 0.30
    assert interpolate(1.0, 0.30, 0.005) == 0.005


def test_the_interpolation_is_linear() -> None:
    assert interpolate(0.5, 10.0, 20.0) == 15.0
    assert interpolate(0.25, 0.0, 8.0) == 2.0


def test_the_range_interpolation_rounds_each_end() -> None:
    assert interpolate_range(0.0, (40, 90), (8, 25)) == (40, 90)
    assert interpolate_range(1.0, (40, 90), (8, 25)) == (8, 25)
    assert interpolate_range(0.5, (40, 90), (8, 26)) == (24, 58)


# Reparticao e arredondamento.


def test_the_largest_remainder_closes_the_total() -> None:
    """Pisos 3, 2 e 1 somam 6; a unidade que falta vai para o maior resto, 0,5."""
    allocated = largest_remainder(np.array([0.5, 0.3, 0.2]), 7)

    assert list(allocated) == [4, 2, 1]
    assert allocated.sum() == 7


def test_the_half_goes_up_whatever_the_parity() -> None:
    assert round_half_up(2.5) == 3
    assert round_half_up(3.5) == 4
    assert round_half_up(2.4) == 2
    assert round_half_up(0.0) == 0


# Distribuicoes.


def test_the_geometric_weights_decay_and_sum_to_one() -> None:
    weights = geometric_weights(4, 0.80)

    assert weights.sum() == pytest.approx(1.0)
    assert weights[1] / weights[0] == pytest.approx(0.20)
    assert list(weights) == sorted(weights, reverse=True)


def test_a_rare_value_can_be_missing() -> None:
    assert chance_never_drawn(0.5, 3) == 0.125
    assert chance_never_drawn(0.0, 30) == 1.0


def test_the_negative_binomial_keeps_the_mean() -> None:
    """Com `p = n / (m + n)`, a media `n(1-p)/p` volta a ser `m`, e a variancia passa dela."""
    mean, dispersion = 2.0, 3.0
    success = negative_binomial_success(mean, dispersion)

    assert dispersion * (1 - success) / success == pytest.approx(mean)
    assert mean / success > mean


def test_without_tail_nothing_passes_the_ceiling() -> None:
    typical = np.arange(8, 26)

    assert share_above(25, typical) == 0.0
    assert share_above(7, typical) == 1.0
    assert share_above_with_tail(25, typical, 0.0, 15.0) == 0.0


def test_the_tail_keeps_a_chance_above_the_ceiling() -> None:
    typical = np.arange(8, 26)

    above = share_above_with_tail(25, typical, 0.05, 15.0)
    further = share_above_with_tail(60, typical, 0.05, 15.0)

    assert 0.0 < further < above < 0.05


# Classificacao.


def test_the_matrix_and_the_rates_of_a_known_case() -> None:
    """Verdade 1 1 0 0 0, decisao 1 0 1 0 0: um de cada erro."""
    counts = confusion([1, 1, 0, 0, 0], [1, 0, 1, 0, 0])

    assert counts == {
        "true_positives": 1, "false_positives": 1,
        "true_negatives": 2, "false_negatives": 1,
    }
    assert rates(counts) == pytest.approx({
        "f1": 0.5, "precision": 0.5, "recall": 0.5,
        "accuracy": 0.6, "specificity": 2 / 3,
    })
    assert roc_point(counts) == pytest.approx((1 / 3, 0.5))


def test_no_alert_gives_zero_precision_and_zero_f1() -> None:
    """Sem alerta nenhum a precisao nao tem denominador, e vale zero (D-116)."""
    values = rates(confusion([1, 0, 0], [0, 0, 0]))

    assert values["precision"] == 0.0
    assert values["f1"] == 0.0
    assert values["specificity"] == 1.0


def test_a_perfect_score_has_area_one() -> None:
    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0


def test_the_weight_of_the_rare_class() -> None:
    assert negatives_per_positive(pd.Series([1, 0, 0, 0])) == 3.0


# Testes de hipotese.


def test_no_difference_at_all_is_p_one() -> None:
    assert signed_rank_test(pd.Series([0.0] * 30), "wilcox", "two-sided") == (0.0, 1.0)


def test_a_single_pair_has_no_test() -> None:
    statistic, p_value = signed_rank_test(pd.Series([0.1]), "wilcox", "two-sided")

    assert np.isnan(statistic)
    assert np.isnan(p_value)


def test_a_consistent_gain_is_significant() -> None:
    differences = pd.Series(np.linspace(0.01, 0.3, 30))
    _, p_value = signed_rank_test(differences, "wilcox", "two-sided")

    assert p_value < 0.001


def test_holm_multiplies_by_the_remaining_count() -> None:
    """Menor p vezes 2, o seguinte vezes 1, sem nunca descer abaixo do anterior."""
    assert list(holm(pd.Series([0.01, 0.04]))) == pytest.approx([0.02, 0.04])


# A direcao das dependencias.


def imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                modules.append(alias.name)

    return modules


def imports_from(folder: Path, prefixes: tuple[str, ...]) -> list[str]:
    """Os imports de uma pasta que comecam por algum dos prefixos."""
    found = []

    for path in sorted(folder.rglob("*.py")):
        for module in imported_modules(path):
            is_forbidden = module.startswith(prefixes)

            if is_forbidden:
                found.append(f"{path.relative_to(SOURCE)}: {module}")

    return found


def test_the_formulas_import_nothing_from_the_project() -> None:
    assert imports_from(SOURCE / "formulas", ("src",)) == []


def test_no_entity_imports_the_evaluation() -> None:
    """A avaliacao le o que as entidades gravaram, e nunca o contrario."""
    forbidden = ("src.metrics", "src.pipeline", "src.viewer")

    assert imports_from(SOURCE / "entities", forbidden) == []
