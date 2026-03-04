# La Liga Match Predictor

A machine learning pipeline that predicts La Liga match outcomes and total goals using historical match data from the [football-data.org](https://www.football-data.org/) API.

## What It Does

1. **Fetches** historical La Liga match results (2023/24 and 2024/25 seasons)
2. **Engineers** per-match features from rolling team stats and head-to-head records
3. **Trains** two gradient boosting models — one for outcome (H/D/A), one for total goals
4. **Predicts** all upcoming scheduled La Liga fixtures

## Quickstart

```bash
pip install pandas scikit-learn requests
python main.py
```

## Project Structure

```
Soccer-ML/
├── config.py           # API key, competition, seasons, feature window
├── data_collector.py   # Fetch & cache match data from football-data.org
├── features.py         # Feature engineering (form, H2H, home/away stats)
├── model.py            # Train & save GBM models
├── predict.py          # Load models and predict upcoming matches
├── main.py             # Orchestrates the full pipeline
├── data/               # Cached raw match JSON (auto-created)
└── models/             # Saved model files (auto-created)
```

## Features Used

| Feature | Description |
|---|---|
| `home_home_win_rate` | Home team's win rate in home games (last 5) |
| `home_home_avg_gf/ga` | Home team's avg goals scored/conceded at home |
| `away_away_win_rate` | Away team's win rate in away games (last 5) |
| `away_away_avg_gf/ga` | Away team's avg goals scored/conceded away |
| `home/away_form_pts` | Overall form points ratio (last 5 games, all venues) |
| `form_pts_diff` | Form points difference (home minus away) |
| `gf_diff / ga_diff` | Goals scored/conceded differential |
| `h2h_home_wins/draws/away_wins` | Head-to-head record (last 5 meetings) |
| `h2h_avg_goals` | Average goals in recent H2H matches |

All features are computed using only data available **before** each match (no data leakage).

## Models

- **Outcome classifier**: `GradientBoostingClassifier` — predicts H (home win), D (draw), or A (away win)
- **Goals regressor**: `GradientBoostingRegressor` — predicts total goals scored

Both trained on 80% of available matches, evaluated on the remaining 20%.

Current performance (2023–2025 data):
- Outcome accuracy: ~44%
- Goals MAE: ~1.49 goals

## Output

```
Date         Home                      Away                       Pred  Goals  Probs (H/D/A)
-----------------------------------------------------------------------------------------------
2026-03-07   Athletic Club             FC Barcelona                  A    3.0  0.70  0.15  0.15
```

- **Pred**: Predicted result from the home team's perspective (H/D/A)
- **Goals**: Predicted total goals
- **Probs**: Model confidence for each outcome (Home win / Draw / Away win)

## API Key

The default API key in `config.py` is a free-tier key. Free accounts are rate-limited to ~10 requests/minute and support a limited set of competitions. Get your own key at [football-data.org](https://www.football-data.org/).

Set it via environment variable to avoid hardcoding:

```bash
export FOOTBALL_API_KEY=your_key_here
python main.py
```