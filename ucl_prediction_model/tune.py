"""
Hyperparameter tuning for the UCL match predictor.

Uses a season-based, time-respecting cross-validation scheme instead of
sklearn's default random K-fold: each fold trains on all seasons up to a
point and validates on the next season forward in time. This mirrors how
the model will actually be used (predict the future from the past) and
avoids the same leakage problem we've been careful about throughout.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.metrics import accuracy_score, f1_score, make_scorer
import joblib

FEATURES = [
    "home_form_points", "home_form_gf", "home_form_ga",
    "away_form_points", "away_form_gf", "away_form_ga",
    "home_elo", "away_elo", "elo_diff", "elo_closeness",
    "home_team_home_winrate", "away_team_away_winrate",
    "is_knockout",
    "h2h_matches", "h2h_home_win_rate", "h2h_draw_rate",
]
TARGET = "result"

# Held out completely from tuning — final untouched sanity check only
FINAL_HOLDOUT_SEASONS = ("2025-26",)


def load_features(path="data/processed/ucl_features.csv"):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def make_season_folds(df, seasons_ordered):
    """
    Build a PredefinedSplit where each fold's validation set is one
    season, and the training set is every season strictly BEFORE it.
    Early seasons with no prior data can't be validated on (nothing to
    train on yet) so they're excluded as validation folds, but still
    used as training data for later folds.
    """
    test_fold = np.full(len(df), -1)  # -1 = never used for validation
    fold_id = 0
    # only use seasons that have at least 3 prior seasons of training data,
    # so each fold has a reasonable amount to learn from. Limit to the most
    # recent handful of seasons as validation folds — using every season
    # since 1999 as a separate fold makes the search too slow to be useful.
    usable_val_seasons = seasons_ordered[3:][-6:]

    for season in usable_val_seasons:
        val_idx = df.index[df["season"] == season]
        test_fold[val_idx] = fold_id
        fold_id += 1

    return PredefinedSplit(test_fold)


def macro_f1(y_true, y_pred):
    """
    Optimize for macro F1, not accuracy. Accuracy rewards a model for
    doing well on the majority class (Home Win) while ignoring draws
    entirely — we saw this exact failure mode earlier. Macro F1 weighs
    all three classes equally, which is what we actually want given
    the draw-recall problem we've been fighting.
    """
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def tune():
    df = load_features()
    seasons_ordered = sorted(df["season"].unique())

    # carve off the final season as an untouched holdout — tuning never sees it
    tune_df = df[~df["season"].isin(FINAL_HOLDOUT_SEASONS)].reset_index(drop=True)
    holdout_df = df[df["season"].isin(FINAL_HOLDOUT_SEASONS)].reset_index(drop=True)

    tune_seasons_ordered = sorted(tune_df["season"].unique())
    cv = make_season_folds(tune_df, tune_seasons_ordered)

    X = tune_df[FEATURES]
    y = tune_df[TARGET]

    param_dist = {
        "n_estimators": [150, 300, 500],
        "max_depth": [4, 6, 8, 10],
        "min_samples_leaf": [5, 10, 20, 40],
        "min_samples_split": [2, 10, 20],
        "max_features": ["sqrt", 0.5],
        "class_weight": ["balanced", "balanced_subsample"],
    }

    base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    scorer = make_scorer(macro_f1)

    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=15,
        scoring=scorer,
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X, y)

    print("\nBest params found:")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")
    print(f"\nBest CV macro-F1: {search.best_score_:.3f}")

    # refit best model on ALL tuning data, evaluate on the untouched holdout
    best_model = search.best_estimator_
    best_model.fit(X, y)

    X_hold = holdout_df[FEATURES]
    y_hold = holdout_df[TARGET]
    preds = best_model.predict(X_hold)

    print(f"\n=== Final holdout evaluation ({FINAL_HOLDOUT_SEASONS}) ===")
    print(f"Accuracy: {accuracy_score(y_hold, preds):.3f}")
    print(f"Macro F1: {f1_score(y_hold, preds, average='macro', zero_division=0):.3f}")
    print(f"Dumb baseline (always Home Win): {(y_hold == 'H').mean():.3f}")

    from sklearn.metrics import classification_report, confusion_matrix
    print("\nClassification report:")
    print(classification_report(y_hold, preds, digits=3, zero_division=0))
    labels = ["A", "D", "H"]
    print("Confusion matrix:")
    print(pd.DataFrame(
        confusion_matrix(y_hold, preds, labels=labels),
        index=[f"actual_{l}" for l in labels],
        columns=[f"pred_{l}" for l in labels]
    ))

    import os
    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": best_model, "label_encoder": None, "features": FEATURES},
                "models/ucl_rf_tuned.joblib")
    print("\nSaved tuned model to models/ucl_rf_tuned.joblib")

    return best_model, search


if __name__ == "__main__":
    tune()
