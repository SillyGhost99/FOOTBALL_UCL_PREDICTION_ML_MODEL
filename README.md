# UCL Predictor

A machine learning model to predict UEFA Champions League match outcomes, built on 30 years of historical match data.

#Overview

This project collects, cleans, and models Champions League match data spanning 30 seasons (~1996/97 – 2024/25) to predict match results (Home Win / Draw / Away Win) and, eventually, knockout-stage progression probabilities.

#Project Structure
ucl-predictor/
├── data/
│   ├── raw/                # Untouched downloaded CSVs (not committed — see Data section)
│   └── processed/          # Cleaned, feature-engineered datasets
├── notebooks/               # Exploration notebooks (EDA, prototyping — not production code)
├── src/
│   ├── data_loader.py       # Loads and merges raw season data
│   ├── features.py          # Feature engineering (form, Elo, rolling stats)
│   ├── train.py              # Model training and evaluation
│   └── predict.py            # Inference on new fixtures
├── models/                   # Saved trained models (.pkl / .joblib — not committed)
├── requirements.txt
├── .gitignore
└── README.md

Data

Timeframe: 30 years of UEFA Champions League matches (target range: 1996/97 season onward), covering league phase / group stage, knockout rounds, and finals.

Core fields collected per match:

Field	Description
Season	e.g. 2023-24
Date	Match date
Round	League phase, R16, QF, SF, Final
Home Team / Away Team	Club names (normalized)
Home Goals / Away Goals	Final score
Home xG / Away xG	Expected goals (underlying performance signal)
Home Possession % / Away Possession %	Match control
Venue	Stadium (neutral for finals)

Planned additional features:

Team form (rolling last-5/last-10 results, goals for/against)
Elo ratings (self-calculated)
Squad value (Transfermarkt)
Days of rest / fixture congestion
Head-to-head history
Key player availability (injuries/suspensions)

Data sources:

football-data.co.uk — free historical CSVs
Kaggle UCL datasets
FBref (xG, advanced stats — manual/rate-limited extraction only)
ClubElo — free Elo ratings.
