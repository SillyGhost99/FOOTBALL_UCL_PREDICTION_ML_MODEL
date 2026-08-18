"""
Feature engineering for the UCL match predictor.

Takes the cleaned master match dataset and produces a feature table
ready for model training. Every feature is computed using only
information available BEFORE the match kicked off (no leakage).
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------

def load_matches(path="data/processed/ucl_master_dataset_clean.csv"):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    # some very early qualifying-round rows can have identical timestamps;
    # break ties deterministically by round order in the source file
    df = df.sort_values("date", kind="stable").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 2. Target variable
# ---------------------------------------------------------------------------

def add_result(df):
    conditions = [
        df["home_goals"] > df["away_goals"],
        df["home_goals"] == df["away_goals"],
        df["home_goals"] < df["away_goals"],
    ]
    df["result"] = np.select(conditions, ["H", "D", "A"], default="D")
    return df


# ---------------------------------------------------------------------------
# 3. Rolling form (last N matches, computed BEFORE each match)
# ---------------------------------------------------------------------------

def add_rolling_form(df, window=5):
    """
    For every match, attach each team's form over their last `window`
    UCL matches (any venue) — points won and goals scored/conceded.
    Uses shift(1) so a match never sees its own result.
    """
    # build a long "team-level" match log: one row per team per match
    home = df[["date", "home_team", "home_goals", "away_goals"]].copy()
    home.columns = ["date", "team", "goals_for", "goals_against"]
    home["points"] = np.select(
        [home["goals_for"] > home["goals_against"], home["goals_for"] == home["goals_against"]],
        [3, 1], default=0
    )

    away = df[["date", "away_team", "away_goals", "home_goals"]].copy()
    away.columns = ["date", "team", "goals_for", "goals_against"]
    away["points"] = np.select(
        [away["goals_for"] > away["goals_against"], away["goals_for"] == away["goals_against"]],
        [3, 1], default=0
    )

    long_log = pd.concat([home, away], ignore_index=True)
    long_log = long_log.sort_values(["team", "date"], kind="stable")

    # rolling averages computed on PRIOR matches only (shift before rolling)
    long_log["form_points"] = (
        long_log.groupby("team")["points"]
        .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    )
    long_log["form_goals_for"] = (
        long_log.groupby("team")["goals_for"]
        .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    )
    long_log["form_goals_against"] = (
        long_log.groupby("team")["goals_against"]
        .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    )

    form_lookup = long_log[["date", "team", "form_points", "form_goals_for", "form_goals_against"]]

    df = df.merge(
        form_lookup.rename(columns={
            "team": "home_team", "form_points": "home_form_points",
            "form_goals_for": "home_form_gf", "form_goals_against": "home_form_ga"
        }),
        on=["date", "home_team"], how="left"
    )
    df = df.merge(
        form_lookup.rename(columns={
            "team": "away_team", "form_points": "away_form_points",
            "form_goals_for": "away_form_gf", "form_goals_against": "away_form_ga"
        }),
        on=["date", "away_team"], how="left"
    )
    return df


# ---------------------------------------------------------------------------
# 4. Self-calculated Elo ratings
# ---------------------------------------------------------------------------

def compute_elo(df, k=30, base_rating=1500, home_advantage=60):
    """
    Standard Elo update, walked forward chronologically.
    Returns the dataframe with home_elo/away_elo = each team's rating
    BEFORE that match was played, plus a fitted ratings dict for later use.
    """
    ratings = {}
    home_elo_col = []
    away_elo_col = []

    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        r_h = ratings.get(h, base_rating)
        r_a = ratings.get(a, base_rating)

        home_elo_col.append(r_h)
        away_elo_col.append(r_a)

        # expected score with home advantage baked in
        exp_h = 1 / (1 + 10 ** (((r_a) - (r_h + home_advantage)) / 400))

        if row["home_goals"] > row["away_goals"]:
            score_h = 1.0
        elif row["home_goals"] == row["away_goals"]:
            score_h = 0.5
        else:
            score_h = 0.0

        # margin-of-victory multiplier (bigger wins move rating more)
        gd = abs(row["home_goals"] - row["away_goals"])
        mov_mult = np.log(gd + 1) + 1

        new_r_h = r_h + k * mov_mult * (score_h - exp_h)
        new_r_a = r_a + k * mov_mult * ((1 - score_h) - (1 - exp_h))

        ratings[h] = new_r_h
        ratings[a] = new_r_a

    df["home_elo"] = home_elo_col
    df["away_elo"] = away_elo_col
    df["elo_diff"] = df["home_elo"] - df["away_elo"]
    return df, ratings


# ---------------------------------------------------------------------------
# 5. Home/away split strength (career-to-date, no leakage)
# ---------------------------------------------------------------------------

def add_home_away_split(df):
    """
    Each team's historical win rate at home / away, computed only from
    matches strictly before the current one.
    """
    df["home_win"] = (df["home_goals"] > df["away_goals"]).astype(int)
    df["away_win"] = (df["away_goals"] > df["home_goals"]).astype(int)

    df["home_team_home_winrate"] = (
        df.groupby("home_team")["home_win"]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    )
    df["away_team_away_winrate"] = (
        df.groupby("away_team")["away_win"]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    )
    return df


# ---------------------------------------------------------------------------
# 5b. Head-to-head history
# ---------------------------------------------------------------------------

def add_head_to_head(df):
    """
    For every match, look back at all PRIOR meetings between these two
    exact clubs (regardless of which was home/away that time) and compute:
      - h2h_matches: how many times they've met before
      - h2h_home_win_rate: fraction of those PRIOR meetings the CURRENT
        home team won
      - h2h_draw_rate: fraction of those PRIOR meetings that were draws

    Draws often cluster between specific rival pairs (tactical familiarity,
    evenly matched squads year after year), so this can help exactly the
    class the model struggles most with.
    """
    df = df.reset_index(drop=True)
    # unordered pair key so "A vs B" and "B vs A" share one history
    df["pair_key"] = df.apply(
        lambda r: tuple(sorted([r["home_team"], r["away_team"]])), axis=1
    )

    h2h_matches = []
    h2h_home_win_rate = []
    h2h_draw_rate = []

    # running history per pair: list of (winner_team_or_None, ) outcomes
    history = {}

    for _, row in df.iterrows():
        key = row["pair_key"]
        past = history.get(key, [])

        h2h_matches.append(len(past))
        if past:
            home_wins = sum(1 for w in past if w == row["home_team"])
            draws = sum(1 for w in past if w == "DRAW")
            h2h_home_win_rate.append(home_wins / len(past))
            h2h_draw_rate.append(draws / len(past))
        else:
            h2h_home_win_rate.append(np.nan)
            h2h_draw_rate.append(np.nan)

        # record this match's outcome for future lookups
        if row["home_goals"] > row["away_goals"]:
            winner = row["home_team"]
        elif row["away_goals"] > row["home_goals"]:
            winner = row["away_team"]
        else:
            winner = "DRAW"
        history.setdefault(key, []).append(winner)

    df["h2h_matches"] = h2h_matches
    df["h2h_home_win_rate"] = h2h_home_win_rate
    df["h2h_draw_rate"] = h2h_draw_rate
    df = df.drop(columns=["pair_key"])
    return df


# ---------------------------------------------------------------------------
# 6. Run pipeline
# ---------------------------------------------------------------------------

def build_features(input_path="data/processed/ucl_master_dataset_clean.csv",
                    output_path="data/processed/ucl_features.csv"):
    df = load_matches(input_path)
    df = add_result(df)
    df = add_rolling_form(df, window=5)
    df, final_ratings = compute_elo(df)
    df = add_home_away_split(df)
    df = add_head_to_head(df)

    # closeness of the two teams' Elo ratings — draws tend to happen
    # between closely-matched teams regardless of which side is favored
    df["elo_closeness"] = -df["elo_diff"].abs()

    # knockout legs can still end level on the scoreline, but a draw
    # there carries different context than in the group/league phase
    df["is_knockout"] = (df["stage"] == "Knockout").astype(int)

    # fill early-career NaNs (a team's first ever match has no history yet)
    fill_cols = [
        "home_form_points", "home_form_gf", "home_form_ga",
        "away_form_points", "away_form_gf", "away_form_ga",
        "home_team_home_winrate", "away_team_away_winrate",
    ]
    df[fill_cols] = df[fill_cols].fillna(df[fill_cols].median())

    # h2h: no prior meetings -> neutral defaults, not zero (zero would
    # falsely say "this pair never draws" instead of "we don't know yet")
    df["h2h_home_win_rate"] = df["h2h_home_win_rate"].fillna(0.45)  # ~league avg home win rate
    df["h2h_draw_rate"] = df["h2h_draw_rate"].fillna(0.23)          # ~league avg draw rate

    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows x {len(df.columns)} columns to {output_path}")
    return df, final_ratings


if __name__ == "__main__":
    build_features()
