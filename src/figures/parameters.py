"""A aparencia das figuras da monografia.

As tres cores sao os tres primeiros tons da paleta de referencia, os mesmos do
viewer, validados juntos para daltonismo. O verde-agua fica abaixo de 3:1 de
contraste sobre o fundo claro, e por isso cada mecanismo tem tambem um marcador
proprio: a identidade nunca depende so da cor, e a figura sobrevive a impressao
em preto e branco.
"""

from __future__ import annotations

MECHANISM_STYLE = {
    "rules": {"label": "Regras (baseline)", "color": "#2a78d6", "marker": "o"},
    "random_forest": {"label": "Random Forest", "color": "#eb6834", "marker": "s"},
    "xgboost": {"label": "XGBoost", "color": "#1baf7a", "marker": "^"},
}

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
EXCLUDED_SHADE = "#efeeea"

LINE_WIDTH = 2.0
MARKER_SIZE = 6.5
BAND_ALPHA = 0.14

ROC_SIGMAS = (0.7, 0.8, 0.9)
"""Os sigmas das curvas ROC: a faixa em que os mecanismos se separam. Abaixo de 0,7 as
curvas colam no canto; em 1,0 nao ha o que detectar."""

ROC_FALSE_POSITIVE_LIMIT = 0.10
"""O eixo x da curva ROC vai ate 0,10: com especificidade acima de 0,95, tudo que
distingue os mecanismos acontece ali. A AUC da legenda cobre a curva inteira."""

RESOLUTION = 200
"""Pontos por polegada do PNG. O PDF sai vetorial."""
