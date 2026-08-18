"""
Run predictions with the trained UCL model.

Usage:
    python src/predict.py "Real Madrid" "Manchester City"
    python src/predict.py "Bayern Munich" "Arsenal" --knockout

This loads the tuned model and the full match history, computes each
team's CURRENT form/Elo/h2h stats (as of the most recent match in the
dataset), and predicts the outcome of a hypothetical match between them.
"""

import argparse
import pandas as pd
import numpy as np
import joblib


FEATURES = [
    "home_form_points", "home_form_gf", "home_form_ga",
    "away_form_points", "away_form_gf", "away_form_ga",
    "home_elo", "away_elo", "elo_diff", "elo_closeness",
    "home_team_home_winrate", "away_team_away_winrate",
    "is_knockout",
    "h2h_matches", "h2h_home_win_rate", "h2h_draw_rate",
]


def load_model(path="models/ucl_rf_tuned.joblib"):
    bundle = joblib.load(path)
    return bundle["model"], bundle.get("label_encoder"), bundle.get("features", FEATURES)


def get_team_current_stats(df, team):
    """
    Pull a team's most recent computed form/Elo stats from the feature
    table — i.e. what their numbers looked like heading into their last
    known match. This is what we feed the model as "current form" for a
    hypothetical future match.
    """
    home_rows = df[df["home_team"] == team].sort_values("date")
    away_rows = df[df["away_team"] == team].sort_values("date")

    if home_rows.empty and away_rows.empty:
        return None

    last_home_date = home_rows["date"].max() if not home_rows.empty else pd.Timestamp.min
    last_away_date = away_rows["date"].max() if not away_rows.empty else pd.Timestamp.min

    if last_home_date >= last_away_date:
        row = home_rows.iloc[-1]
        return {
            "elo": row["home_elo"],
            "form_points": row["home_form_points"],
            "form_gf": row["home_form_gf"],
            "form_ga": row["home_form_ga"],
            "home_winrate": row["home_team_home_winrate"],
            "away_winrate": df[df["away_team"] == team].sort_values("date")["away_team_away_winrate"].iloc[-1]
                            if not away_rows.empty else 0.45,
        }
    else:
        row = away_rows.iloc[-1]
        return {
            "elo": row["away_elo"],
            "form_points": row["away_form_points"],
            "form_gf": row["away_form_gf"],
            "form_ga": row["away_form_ga"],
            "home_winrate": df[df["home_team"] == team].sort_values("date")["home_team_home_winrate"].iloc[-1]
                            if not home_rows.empty else 0.45,
            "away_winrate": row["away_team_away_winrate"],
        }


def get_h2h_stats(df, home_team, away_team):
    """Look up prior meetings between these two specific clubs."""
    pair_key = tuple(sorted([home_team, away_team]))
    mask = df.apply(
        lambda r: tuple(sorted([r["home_team"], r["away_team"]])) == pair_key, axis=1
    )
    past = df[mask]

    if past.empty:
        return {"matches": 0, "home_win_rate": 0.45, "draw_rate": 0.23}

    home_wins = (past["home_team"] == home_team) & (past["home_goals"] > past["away_goals"])
    away_as_home_wins = (past["away_team"] == home_team) & (past["away_goals"] > past["home_goals"])
    wins_for_home_team = home_wins.sum() + away_as_home_wins.sum()
    draws = (past["home_goals"] == past["away_goals"]).sum()

    n = len(past)
    return {
        "matches": n,
        "home_win_rate": wins_for_home_team / n,
        "draw_rate": draws / n,
    }


def build_feature_row(df, home_team, away_team, is_knockout=False):
    home_stats = get_team_current_stats(df, home_team)
    away_stats = get_team_current_stats(df, away_team)

    if home_stats is None:
        raise ValueError(f"'{home_team}' not found in match history. Check spelling/normalization.")
    if away_stats is None:
        raise ValueError(f"'{away_team}' not found in match history. Check spelling/normalization.")

    h2h = get_h2h_stats(df, home_team, away_team)

    elo_diff = home_stats["elo"] - away_stats["elo"]

    row = {
        "home_form_points": home_stats["form_points"],
        "home_form_gf": home_stats["form_gf"],
        "home_form_ga": home_stats["form_ga"],
        "away_form_points": away_stats["form_points"],
        "away_form_gf": away_stats["form_gf"],
        "away_form_ga": away_stats["form_ga"],
        "home_elo": home_stats["elo"],
        "away_elo": away_stats["elo"],
        "elo_diff": elo_diff,
        "elo_closeness": -abs(elo_diff),
        "home_team_home_winrate": home_stats["home_winrate"],
        "away_team_away_winrate": away_stats["away_winrate"],
        "is_knockout": int(is_knockout),
        "h2h_matches": h2h["matches"],
        "h2h_home_win_rate": h2h["home_win_rate"],
        "h2h_draw_rate": h2h["draw_rate"],
    }
    return pd.DataFrame([row])


def predict_match(home_team, away_team, is_knockout=False,
                   model_path="models/ucl_rf_tuned.joblib",
                   data_path="data/processed/ucl_features.csv"):
    model, label_encoder, features = load_model(model_path)
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])

    X = build_feature_row(df, home_team, away_team, is_knockout)[features]

    probs = model.predict_proba(X)[0]
    classes = model.classes_
    if label_encoder is not None:
        classes = label_encoder.inverse_transform(classes)

    result = dict(zip(classes, probs))
    # ensure consistent order regardless of model internals
    ordered = {"H": result.get("H", 0), "D": result.get("D", 0), "A": result.get("A", 0)}
    return ordered, X.iloc[0].to_dict()


def main():
    parser = argparse.ArgumentParser(description="Predict a UCL match outcome")
    parser.add_argument("home_team", help="Home team name (must match dataset naming)")
    parser.add_argument("away_team", help="Away team name (must match dataset naming)")
    parser.add_argument("--knockout", action="store_true", help="Flag if this is a knockout-stage match")
    parser.add_argument("--model", default="models/ucl_rf_tuned.joblib")
    parser.add_argument("--data", default="data/processed/ucl_features.csv")
    args = parser.parse_args()

    probs, features_used = predict_match(
        args.home_team, args.away_team, args.knockout, args.model, args.data
    )

    print(f"\n{args.home_team} (home) vs {args.away_team} (away)")
    print(f"{'Knockout leg' if args.knockout else 'League/Group phase'}\n")
    print(f"  Home Win: {probs['H']*100:5.1f}%")
    print(f"  Draw:     {probs['D']*100:5.1f}%")
    print(f"  Away Win: {probs['A']*100:5.1f}%")

    print(f"\n  {args.home_team} Elo: {features_used['home_elo']:.0f}")
    print(f"  {args.away_team} Elo: {features_used['away_elo']:.0f}")
    print(f"  Prior meetings: {int(features_used['h2h_matches'])}")


if __name__ == "__main__":
    main()
