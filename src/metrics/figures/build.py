"""As figuras da monografia, desenhadas a partir das tabelas do M12 e das predicoes.

Duas figuras:

- `f1_sigma`: o F1 de cada mecanismo ao longo de sigma, mediana e intervalo
  interquartilico das 30 sementes, no holdout inteiro e so entre administradores.
- `roc`: a curva ROC dos dois modelos em sigma 0,7, 0,8 e 0,9, com o baseline como
  um ponto, porque ele so decide e nao tem escore (proposta, Tabela 9).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import roc_curve

from src.entities.dataset_generator.dataset.build import LABEL
from src.entities.models.build import score_column
from src.entities.models.parameters import MODEL_NAMES
from src.entities.policy_engine.baseline.build import MECHANISM as RULES
from src.formulas.classification import confusion, roc_auc, roc_point
from src.metrics.evaluation.build import ADMINISTRATORS, ALL_SESSIONS, MECHANISMS
from src.metrics.figures.parameters import (
    BAND_ALPHA,
    EXCLUDED_SHADE,
    GRID,
    LINE_WIDTH,
    MARKER_SIZE,
    MECHANISM_STYLE,
    ROC_FALSE_POSITIVE_LIMIT,
    RESOLUTION,
    ROC_SIGMAS,
    SURFACE,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)
from src.shared import layout
from src.shared.experiment import SIGMAS

def comma(value: float, places: int = 2) -> str:
    return f"{value:.{places}f}".replace(".", ",")


def axis_label_one_place(value: float, position: int) -> str:
    """O rotulo de um eixo com uma casa e virgula.

    O matplotlib chama com o valor e a posicao do rotulo; a posicao nao se usa.
    """
    return comma(value, 1)


def axis_label_two_places(value: float, position: int) -> str:
    """O rotulo de um eixo com duas casas e virgula, para o eixo recortado da ROC."""
    return comma(value, 2)


WITH_COMMA = FuncFormatter(axis_label_one_place)
WITH_COMMA_FINE = FuncFormatter(axis_label_two_places)


def styled_axes(axes) -> None:
    """Grade discreta, sem moldura de cima e da direita, texto em tinta neutra."""
    axes.set_facecolor(SURFACE)
    axes.grid(color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)

    for side in ("top", "right"):
        axes.spines[side].set_visible(False)

    for side in ("left", "bottom"):
        axes.spines[side].set_color(TEXT_SECONDARY)

    axes.tick_params(colors=TEXT_SECONDARY, labelcolor=TEXT_PRIMARY)


# O F1 ao longo de sigma.


def f1_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Mediana e quartis do F1, por recorte, sigma e mecanismo, sobre as 30 sementes."""
    grouped = metrics.groupby(["scope", "sigma", "mechanism"])["f1"]

    return pd.DataFrame({
        "median": grouped.median(),
        "q1": grouped.quantile(0.25),
        "q3": grouped.quantile(0.75),
    }).reset_index()


def draw_f1_panel(axes, summary: pd.DataFrame, excluded: list[float], title: str) -> None:
    has_exclusion = len(excluded) > 0

    if has_exclusion:
        axes.axvspan(min(excluded) - 0.05, max(excluded) + 0.05, color=EXCLUDED_SHADE, zorder=0)
        axes.text((min(excluded) + max(excluded)) / 2, 0.06, "excluídas\n(árvore rasa ≥ 0,95)",
                  ha="center", va="bottom", fontsize=8, color=TEXT_SECONDARY)

    for mechanism in MECHANISMS:
        style = MECHANISM_STYLE[mechanism]
        rows = summary[summary["mechanism"] == mechanism].sort_values("sigma")

        axes.fill_between(rows["sigma"], rows["q1"], rows["q3"],
                          color=style["color"], alpha=BAND_ALPHA, linewidth=0)
        axes.plot(rows["sigma"], rows["median"], color=style["color"], linewidth=LINE_WIDTH,
                  marker=style["marker"], markersize=MARKER_SIZE,
                  markeredgecolor=SURFACE, markeredgewidth=1.2, label=style["label"])

    styled_axes(axes)
    axes.set_title(title, fontsize=10.5, color=TEXT_PRIMARY, loc="left")
    axes.set_xlim(-0.05, 1.05)
    axes.set_ylim(-0.02, 1.04)
    axes.set_xticks(SIGMAS)
    axes.xaxis.set_major_formatter(WITH_COMMA)
    axes.yaxis.set_major_formatter(WITH_COMMA)
    axes.set_xlabel("σ (furtividade do atacante)", color=TEXT_PRIMARY)


def draw_f1_by_sigma(metrics: pd.DataFrame, comparison: pd.DataFrame) -> Figure:
    """Dois paineis lado a lado, mesma escala: o desfecho primario e o secundario."""
    summary = f1_summary(metrics)
    excluded = sorted(comparison.loc[comparison["excluded"], "sigma"].unique())

    figure, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True)
    figure.set_facecolor(SURFACE)

    draw_f1_panel(left, summary[summary["scope"] == ALL_SESSIONS], excluded,
                  "Holdout inteiro (desfecho primário)")
    draw_f1_panel(right, summary[summary["scope"] == ADMINISTRATORS], excluded,
                  "Só sessões de administradores (desfecho secundário)")

    left.set_ylabel("F1 (mediana e intervalo interquartílico)", color=TEXT_PRIMARY)
    handles, labels = left.get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
                  fontsize=9.5, labelcolor=TEXT_PRIMARY, bbox_to_anchor=(0.5, 1.02))
    figure.tight_layout(rect=(0, 0, 1, 0.93))

    return figure


# A curva ROC.


def pooled_predictions(
    root: Path, seeds: tuple[int, ...], sigma: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """As predicoes das sementes naquele sigma, empilhadas."""
    rules = []
    models = []

    for seed in seeds:
        directory = layout.run_directory(root, seed, sigma)
        rules.append(pd.read_csv(directory / layout.PREDICTIONS_RULES))
        models.append(pd.read_csv(directory / layout.PREDICTIONS_ML))

    return pd.concat(rules, ignore_index=True), pd.concat(models, ignore_index=True)


def baseline_point(rules: pd.DataFrame) -> tuple[float, float]:
    """O baseline no plano ROC: um ponto so, porque ele decide e nao tem escore."""
    counts = confusion(rules[LABEL], rules["predicted"])

    return roc_point(counts)


def draw_roc_panel(axes, predictions: tuple[pd.DataFrame, pd.DataFrame], sigma: float) -> None:
    rules, models = predictions

    for name in MODEL_NAMES:
        style = MECHANISM_STYLE[name]
        false_positive, true_positive, _ = roc_curve(models[LABEL], models[score_column(name)])
        area = roc_auc(models[LABEL], models[score_column(name)])
        axes.plot(false_positive, true_positive, color=style["color"], linewidth=LINE_WIDTH,
                  label=f"{style['label']} (AUC {comma(area, 3)})")

    rules_style = MECHANISM_STYLE[RULES]
    rules_false_positive, rules_true_positive = baseline_point(rules)
    axes.plot(rules_false_positive, rules_true_positive, linestyle="none",
              marker=rules_style["marker"], markersize=MARKER_SIZE + 2,
              color=rules_style["color"], markeredgecolor=SURFACE, markeredgewidth=1.5,
              label=rules_style["label"], zorder=5)

    styled_axes(axes)
    axes.set_title(f"σ {comma(sigma, 1)}", fontsize=10.5, color=TEXT_PRIMARY, loc="left")
    axes.set_xlim(0, ROC_FALSE_POSITIVE_LIMIT)
    axes.set_ylim(0, 1.02)
    axes.xaxis.set_major_formatter(WITH_COMMA_FINE)
    axes.yaxis.set_major_formatter(WITH_COMMA)
    axes.legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=TEXT_PRIMARY)


def draw_roc(root: Path, seeds: tuple[int, ...], sigmas: tuple[float, ...]) -> Figure | None:
    """Uma curva por modelo e o ponto do baseline, em cada sigma da faixa informativa.

    So desenha os sigmas da faixa que a execucao rodou; se nenhum, nao ha figura.
    """
    chosen = []

    for sigma in ROC_SIGMAS:
        was_run = sigma in sigmas

        if was_run:
            chosen.append(sigma)

    if not chosen:
        return None

    figure, panels = plt.subplots(1, len(chosen), figsize=(4 * len(chosen), 4),
                                  sharey=True, squeeze=False)
    figure.set_facecolor(SURFACE)

    for axes, sigma in zip(panels[0], chosen):
        draw_roc_panel(axes, pooled_predictions(root, seeds, sigma), sigma)
        axes.set_xlabel(f"taxa de falsos positivos (eixo até {comma(ROC_FALSE_POSITIVE_LIMIT)})",
                        color=TEXT_PRIMARY)

    panels[0][0].set_ylabel("taxa de verdadeiros positivos", color=TEXT_PRIMARY)
    figure.tight_layout()

    return figure


def save_figure(figure: Figure, directory: Path, name: str) -> list[Path]:
    """Grava a figura em PNG e em PDF vetorial, e devolve os caminhos."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = []

    for suffix in ("png", "pdf"):
        path = directory / f"{name}.{suffix}"
        figure.savefig(path, dpi=RESOLUTION, facecolor=figure.get_facecolor())
        paths.append(path)

    plt.close(figure)

    return paths
