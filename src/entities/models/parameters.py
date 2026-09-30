"""Os numeros que governam os modelos supervisionados (M11)."""

from __future__ import annotations

MODEL_NAMES = ("random_forest", "xgboost")
"""Os dois modelos da comparacao (D-051)."""

GRIDS = {
    "random_forest": {
        "n_estimators": [200, 500],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 2, 5],
        "max_features": ["sqrt", 0.5],
        "class_weight": [None, "balanced"],
    },
    "xgboost": {
        "n_estimators": [200, 500],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "class_weight": [None, "balanced"],
    },
}
"""A grade da busca na preparatoria 902 (D-103), com o peso por classe (D-105).

72 configuracoes de Random Forest e 96 de XGBoost. `class_weight` "balanced" pesa a
classe rara por negativas / positivas do treino em uso; no XGBoost isso vira o
`scale_pos_weight`.
"""

FOLDS = 5
REPEATS = 3
"""Validacao cruzada estratificada, 5 dobras repetidas 3 vezes: 15 F1 por configuracao (D-103)."""

JOBS = 1
"""Um nucleo, no treino e na medicao do tempo (D-106)."""

SEED_CEILING = 2**31 - 1
"""Teto das sementes inteiras sorteadas para os modelos: cabe no inteiro de 32 bits do XGBoost."""
