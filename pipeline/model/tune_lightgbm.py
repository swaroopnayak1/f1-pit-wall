"""
Optuna hyperparameter search for LightGBM — MAE objective, temporal CV.

Reuses the same expanding-window walk-forward split as lightgbm.ipynb (Section 6):
fold i trains on every season strictly before the validation season and
validates on the next one. Trials are ranked by mean out-of-sample CV MAE
across folds, never by train-set error. Only train.parquet (seasons 2019-2024)
is loaded here — the 2025 test season is never read during tuning.

Usage
-----
    python pipeline/model/tune_lightgbm.py --n-trials 50
    python pipeline/model/tune_lightgbm.py --n-trials 100 --storage sqlite:///reports/lightgbm_optuna.db
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from optuna.pruners import MedianPruner
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from pipeline.feature_engineering.feature_engineering import FINAL_FEATURES, TARGET

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports"


def temporal_folds(train: pd.DataFrame):
    """Yield (fold_index, train_mask, val_mask) for an expanding-window walk-forward split over seasons."""
    years = sorted(train["year"].unique())
    for i in range(1, len(years)):
        train_years = years[:i]
        val_year = years[i]
        yield i, train["year"].isin(train_years), train["year"] == val_year


def make_objective(train_X: pd.DataFrame, train_Y: pd.Series, folds: list) -> "callable":
    def objective(trial: optuna.Trial) -> float:
        # Search space sized for a few thousand rows (not a huge dataset): shallow
        # trees and stronger regularization ranges to guard against overfitting.
        # max_depth/learning_rate floors and reg_alpha/reg_lambda/subsample/
        # colsample_bytree floors were widened after a 50-trial run showed best
        # trials pinned at the old floors (see reports/lightgbm_optuna_trials.csv),
        # and cross-checked against optuna_integration.lightgbm.LightGBMTuner's
        # official default search space (1e-8 lambda floor, 0.4 fraction floor).
        params = {
            "random_state": 42,
            "verbose": -1,
            "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
            "num_leaves": trial.suggest_int("num_leaves", 7, 63),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.003, 0.3, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.4, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        }

        fold_maes = []
        for fold_idx, train_mask, val_mask in folds:
            model = LGBMRegressor(**params)
            model.fit(train_X[train_mask], train_Y[train_mask])
            preds = model.predict(train_X[val_mask])
            fold_maes.append(mean_absolute_error(train_Y[val_mask], preds))

            # Report the running mean so far — lets MedianPruner cut a trial
            # short once its trajectory is worse than the median at the same fold.
            trial.report(sum(fold_maes) / len(fold_maes), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return sum(fold_maes) / len(fold_maes)

    return objective


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Optuna LightGBM tuning (MAE, temporal CV)")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--storage",
        default=None,
        help="Optional Optuna storage URL for a resumable study, e.g. sqlite:///reports/lightgbm_optuna.db",
    )
    parser.add_argument("--study-name", default="lightgbm_mae_temporal_cv")
    args = parser.parse_args(argv)

    train = pd.read_parquet(DATA_DIR / "train.parquet")
    train_X = train[FINAL_FEATURES]
    train_Y = train[TARGET]
    folds = list(temporal_folds(train))

    optuna.logging.set_verbosity(optuna.logging.INFO)
    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=MedianPruner(n_warmup_steps=2),
        storage=args.storage,
        load_if_exists=args.storage is not None,
    )
    study.optimize(make_objective(train_X, train_Y, folds), n_trials=args.n_trials)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(REPORTS_DIR / "lightgbm_optuna_trials.csv", index=False)

    best = {"cv_mae": study.best_value, "params": study.best_trial.params}
    with open(REPORTS_DIR / "lightgbm_best_params.json", "w") as fh:
        json.dump(best, fh, indent=2)

    print(f"Best CV MAE: {study.best_value:.4f}")
    print(f"Best params: {study.best_trial.params}")
    print(f"Trials log      -> {REPORTS_DIR / 'lightgbm_optuna_trials.csv'}")
    print(f"Best params json -> {REPORTS_DIR / 'lightgbm_best_params.json'}")


if __name__ == "__main__":
    main()
