from __future__ import annotations

MODEL_NAMES = ("random_forest", "xgboost")

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

CROSS_VALIDATION_FOLDS = 5
CROSS_VALIDATION_REPEATS = 3

CPU_CORES = 1

SEED_CEILING = 2**31 - 1
