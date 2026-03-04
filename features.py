import pandas as pd
from config import FORM_WINDOW


def _team_stats_before(df: pd.DataFrame, team_id: int, before_date, as_home: bool) -> dict:
    """Compute rolling stats for a team in their home/away role before a given date."""
    if as_home:
        games = df[(df["home_id"] == team_id) & (df["date"] < before_date)].tail(FORM_WINDOW)
        gf = games["home_goals"].tolist()
        ga = games["away_goals"].tolist()
        wins = sum(1 for o in games["outcome"] if o == "H")
        draws = sum(1 for o in games["outcome"] if o == "D")
    else:
        games = df[(df["away_id"] == team_id) & (df["date"] < before_date)].tail(FORM_WINDOW)
        gf = games["away_goals"].tolist()
        ga = games["home_goals"].tolist()
        wins = sum(1 for o in games["outcome"] if o == "A")
        draws = sum(1 for o in games["outcome"] if o == "D")

    n = len(games)
    return {
        "n":        n,
        "win_rate": wins / n if n else 0,
        "draw_rate": draws / n if n else 0,
        "avg_gf":   sum(gf) / n if n else 0,
        "avg_ga":   sum(ga) / n if n else 0,
    }


def _overall_form(df: pd.DataFrame, team_id: int, before_date) -> dict:
    """Overall recent form regardless of home/away."""
    home = df[(df["home_id"] == team_id) & (df["date"] < before_date)].copy()
    home["gf"] = home["home_goals"]
    home["ga"] = home["away_goals"]
    home["win"] = home["outcome"] == "H"

    away = df[(df["away_id"] == team_id) & (df["date"] < before_date)].copy()
    away["gf"] = away["away_goals"]
    away["ga"] = away["home_goals"]
    away["win"] = away["outcome"] == "A"

    all_games = pd.concat([home, away]).sort_values("date").tail(FORM_WINDOW)
    n = len(all_games)
    return {
        "form_pts":  (all_games["win"].sum() * 3 + (all_games["outcome"] == "D").sum()) / (n * 3) if n else 0,
        "form_gf":   all_games["gf"].mean() if n else 0,
        "form_ga":   all_games["ga"].mean() if n else 0,
    }


def _h2h(df: pd.DataFrame, home_id: int, away_id: int, before_date, n: int = 5) -> dict:
    """Head-to-head stats between two teams before a given date."""
    mask = (
        ((df["home_id"] == home_id) & (df["away_id"] == away_id)) |
        ((df["home_id"] == away_id) & (df["away_id"] == home_id))
    ) & (df["date"] < before_date)
    games = df[mask].tail(n)
    total = len(games)
    if total == 0:
        return {"h2h_home_wins": 0, "h2h_draws": 0, "h2h_away_wins": 0, "h2h_avg_goals": 0}

    home_wins = sum(
        1 for _, r in games.iterrows()
        if (r["home_id"] == home_id and r["outcome"] == "H") or
           (r["away_id"] == home_id and r["outcome"] == "A")
    )
    draws = sum(1 for o in games["outcome"] if o == "D")
    return {
        "h2h_home_wins":  home_wins / total,
        "h2h_draws":      draws / total,
        "h2h_away_wins":  (total - home_wins - draws) / total,
        "h2h_avg_goals":  games["total_goals"].mean(),
    }


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each match in df, compute features using only data available before that match.
    Returns a DataFrame ready for ML training.
    """
    rows = []

    for _, match in df.iterrows():
        date      = match["date"]
        home_id   = match["home_id"]
        away_id   = match["away_id"]
        past      = df[df["date"] < date]

        if len(past) < 10:  # skip early matches with insufficient history
            continue

        h_home = _team_stats_before(past, home_id, date, as_home=True)
        a_away = _team_stats_before(past, away_id, date, as_home=False)
        h_form = _overall_form(past, home_id, date)
        a_form = _overall_form(past, away_id, date)
        h2h    = _h2h(past, home_id, away_id, date)

        row = {
            "match_id": match["match_id"],
            # Home team home stats
            "home_home_win_rate":  h_home["win_rate"],
            "home_home_draw_rate": h_home["draw_rate"],
            "home_home_avg_gf":    h_home["avg_gf"],
            "home_home_avg_ga":    h_home["avg_ga"],
            # Away team away stats
            "away_away_win_rate":  a_away["win_rate"],
            "away_away_draw_rate": a_away["draw_rate"],
            "away_away_avg_gf":    a_away["avg_gf"],
            "away_away_avg_ga":    a_away["avg_ga"],
            # Overall form
            "home_form_pts":  h_form["form_pts"],
            "home_form_gf":   h_form["form_gf"],
            "home_form_ga":   h_form["form_ga"],
            "away_form_pts":  a_form["form_pts"],
            "away_form_gf":   a_form["form_gf"],
            "away_form_ga":   a_form["form_ga"],
            # Differences (home advantage proxy)
            "form_pts_diff":  h_form["form_pts"] - a_form["form_pts"],
            "gf_diff":        h_form["form_gf"] - a_form["form_gf"],
            "ga_diff":        h_form["form_ga"] - a_form["form_ga"],
            # H2H
            **h2h,
            # Targets
            "outcome":     match["outcome"],
            "total_goals": match["total_goals"],
            "home_goals":  match["home_goals"],
            "away_goals":  match["away_goals"],
        }
        rows.append(row)

    return pd.DataFrame(rows).reset_index(drop=True)