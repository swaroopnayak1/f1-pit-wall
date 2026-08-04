# Final Evaluation — Tuned LightGBM vs. Baseline (2025 Held-Out Test Season)

## Setup

- **Tuned model**: LightGBM with Optuna-selected hyperparameters (`reports/lightgbm_best_params.json`, from `pipeline/model/tune_lightgbm.py` — 50-trial TPE search minimizing mean temporal-CV MAE), fit on all 10 `FINAL_FEATURES`, refit on the full 2019–2024 training set. Results: `reports/lightgbm_tuned_cv_results.csv`, `reports/lightgbm_tuned_test_results.csv`.
- **Baseline model**: LightGBM with default hyperparameters, fit on the 9-feature reduced set from the VIF/importance selection (`pipeline/model/lightgbm.ipynb`, Section 10c). Results: `reports/lightgbm_cv_results.csv`, `reports/lightgbm_test_results.csv`.
- **Test set**: 2025 season (479 rows) — never read during CV or the Optuna search; fully held out.
- **Metrics**: MAE (lower is better), Spearman ρ (higher is better), Macro F1 (higher is better).

## Test-set comparison (2025, held out)

| Model | MAE | Spearman ρ | Macro F1 |
|---|---|---|---|
| LightGBM baseline | 3.543 | 0.598 | 0.092 |
| LightGBM tuned | 3.345 | 0.652 | 0.058 |
| **Delta (tuned − baseline)** | **−0.198** | **+0.053** | **−0.034** |
| *RandomForest (context)* | *3.516* | *0.614* | *0.080* |

Tuning improves MAE and rank correlation (the two metrics the Optuna search was more directly aligned with — MAE was the tuning objective, and rank correlation tends to move with it) but costs Macro F1. The tuned model's predictions cluster more tightly around the mean (shallower trees, heavier `reg_alpha`/`reg_lambda`, `max_depth=3`), which lowers average position error but reduces its ability to land on exact positions, hurting the per-class F1 average. Net effect: tuning is a clear win for ranking/ordering-sensitive use, a wash-to-slight-loss for exact-position classification.

## CV-to-test inflation check

| Model | CV mean MAE | Test MAE | Gap (test − CV) | CV mean Spearman ρ | Test Spearman ρ | Gap | CV mean Macro F1 | Test Macro F1 | Gap |
|---|---|---|---|---|---|---|---|---|---|
| LightGBM baseline | 3.635 | 3.543 | **−0.092** | 0.586 | 0.598 | **+0.012** | 0.084 | 0.092 | +0.008 |
| LightGBM tuned | 3.397 | 3.345 | **−0.052** | 0.645 | 0.652 | **+0.007** | 0.069 | 0.058 | **−0.011** |

**No CV-to-test inflation observed.** For both models, MAE on the 2025 test season is *lower* than the mean CV error across the 2020–2024 validation folds, and Spearman ρ is *higher* — the opposite of the classic overfitting signature (where held-out test error exceeds CV error because the model was implicitly tuned against the CV folds). Macro F1 shows a small negative gap for the tuned model (−0.011), but Macro F1 is the noisiest of the three metrics here (rare classes among 20 finishing positions with a 479-row test set), so this isn't treated as an inflation signal.

## Where the real overfitting signal lives: train vs. CV

CV-to-test agreement doesn't mean these models generalize perfectly — it means the CV folds are a good proxy for the untouched season. The gap that does show overfitting is **train (in-sample) vs. CV (out-of-sample)**:

| Model | Train MAE | CV mean MAE | Train–CV gap |
|---|---|---|---|
| LightGBM, default params (all 10 features, diagnostic fit) | 2.108 | 3.642 | **1.534** |
| LightGBM, Optuna-tuned (all 10 features) | 3.101 | 3.397 | **0.296** |

The default-parameter model memorizes training data far more than the tuned one (unrestricted depth/leaves let it drive training MAE well below any out-of-sample fold). Optuna's search converged on strong regularization (`max_depth=3`, `reg_alpha≈0.46`, `subsample≈0.83`, `colsample_bytree≈0.71`, slow `learning_rate≈0.0084`) precisely because trials were ranked on CV MAE, not train error — and it cut the train–CV gap by roughly 80%. That tighter train/CV agreement is the real driver behind the tuned model's better held-out MAE and Spearman ρ.

## Takeaways

- Tuning delivers a genuine held-out improvement in MAE (−0.20) and Spearman ρ (+0.05), at a small cost to Macro F1 (−0.03).
- CV and test track each other closely for both models (test is, if anything, slightly *easier* than the CV folds) — there is **no evidence of CV-to-test inflation**.
- The overfitting that does exist is concentrated in the train-vs-CV gap of the untuned default model, and tuning substantially closes it (1.53 → 0.30), which is the mechanism behind the test-set gains above.
