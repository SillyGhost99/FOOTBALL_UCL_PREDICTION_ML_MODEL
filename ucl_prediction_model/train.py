"""
Train and evaluate the UCL match predictor.

Key rule: NEVER random-split this data. Football results are time-ordered,
and a random split would let the model "see" future matches while training
on the past, producing fake accuracy numbers. We always train on older
seasons and test on newer, unseen ones.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


FEATURES = [
    "home_form_points", "home_form_gf", "home_form_ga",
    "away_form_points", "away_form_gf", "away_form_ga",
    "home_elo", "away_elo", "elo_diff", "elo_closeness",
    "home_team_home_winrate", "away_team_away_winrate",
    "is_knockout",
    "h2h_matches", "h2h_home_win_rate", "h2h_draw_rate",
]
TARGET = "result"


def load_features(path="data/processed/ucl_features.csv"):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def time_split(df, test_seasons):
    """
    Train on every season NOT in test_seasons, test on test_seasons.
    Keeps the split honest — no shuffling, no random_state games.
    """
    train_df = df[~df["season"].isin(test_seasons)]
    test_df = df[df["season"].isin(test_seasons)]
    return train_df, test_df


def dumb_baseline_accuracy(df):
    """Always predict Home Win — the naive benchmark any real model must beat."""
    return (df[TARGET] == "H").mean()


def train_model(train_df):
    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",  # stop the model from ignoring draws
    )
    model.fit(X_train, y_train)
    return model


def train_xgb_model(train_df):
    """
    Gradient boosting tends to pick up subtler, harder-to-separate signal
    (like draws) better than Random Forest because it builds trees
    sequentially, each one correcting the previous trees' mistakes —
    rather than averaging many independent trees that each individually
    give up on the hard class.
    """
    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    # XGBoost needs numeric labels, not strings
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)  # A=0, D=1, H=2 (alphabetical)

    # class weights: inverse frequency, so the rare "D" class counts more
    class_counts = pd.Series(y_train_enc).value_counts()
    total = len(y_train_enc)
    weight_map = {cls: total / (len(class_counts) * cnt) for cls, cnt in class_counts.items()}
    sample_weights = pd.Series(y_train_enc).map(weight_map).values

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softmax",
        num_class=3,
        random_state=42,
        eval_metric="mlogloss",
    )
    model.fit(X_train, y_train_enc, sample_weight=sample_weights)
    return model, le


def evaluate(model, test_df, label_encoder=None, model_name="Model"):
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]
    preds = model.predict(X_test)

    if label_encoder is not None:
        preds = label_encoder.inverse_transform(preds)

    acc = accuracy_score(y_test, preds)
    print(f"\n=== {model_name} ===")
    print(f"Accuracy: {acc:.3f}")
    print(f"Dumb baseline (always Home Win): {dumb_baseline_accuracy(test_df):.3f}")

    print("\nClassification report:")
    print(classification_report(y_test, preds, digits=3, zero_division=0))

    print("Confusion matrix (rows=actual, cols=predicted) [A, D, H]:")
    labels = ["A", "D", "H"]
    print(pd.DataFrame(
        confusion_matrix(y_test, preds, labels=labels),
        index=[f"actual_{l}" for l in labels],
        columns=[f"pred_{l}" for l in labels]
    ))

    if hasattr(model, "feature_importances_"):
        print("\nFeature importances:")
        imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
        print(imp.round(3))

    return acc


def run(feature_path="data/processed/ucl_features.csv",
        model_out="models/ucl_model_v2.joblib",
        test_seasons=("2024-25", "2025-26")):
    df = load_features(feature_path)

    print("Seasons in dataset:", sorted(df["season"].unique()))
    print(f"Testing on: {test_seasons}")

    train_df, test_df = time_split(df, list(test_seasons))
    print(f"\nTrain rows: {len(train_df)} | Test rows: {len(test_df)}")

    # Random Forest (v1 baseline)
    rf_model = train_model(train_df)
    rf_acc = evaluate(rf_model, test_df, model_name="Random Forest")

    # XGBoost (v2 candidate)
    xgb_model, label_encoder = train_xgb_model(train_df)
    xgb_acc = evaluate(xgb_model, test_df, label_encoder=label_encoder, model_name="XGBoost")

    import os
    os.makedirs("models", exist_ok=True)

    if xgb_acc >= rf_acc:
        print(f"\nXGBoost wins ({xgb_acc:.3f} vs {rf_acc:.3f}) — saving XGBoost as the model.")
        joblib.dump({"model": xgb_model, "label_encoder": label_encoder}, model_out)
        best = xgb_model
    else:
        print(f"\nRandom Forest wins ({rf_acc:.3f} vs {xgb_acc:.3f}) — saving Random Forest as the model.")
        joblib.dump({"model": rf_model, "label_encoder": None}, model_out)
        best = rf_model

    print(f"Model saved to {model_out}")
    return best


if __name__ == "__main__":
    run()
