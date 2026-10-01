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

ROC_FALSE_POSITIVE_AXIS_LIMIT = 0.10

PNG_DPI = 200
