import pandas as pd
from data_collector import fetch_upcoming_matches
from features import _team_stats_before, _overall_form, _h2h
from model import load_models, FEATURE_COLS


def predict_upcoming(historical_df: pd.DataFrame):
    """Predict outcomes and goals for upcoming La Liga matches."""
    clf, reg, le = load_models()
    upcoming = fetch_upcoming_matches()

    if not upcoming:
        print("No upcoming La Liga matches found.")
        return

    rows = []
    meta = []

    for m in upcoming:
        home_id   = m["homeTeam"]["id"]
        away_id   = m["awayTeam"]["id"]
        home_name = m["homeTeam"]["name"]
        away_name = m["awayTeam"]["name"]
        date      = pd.Timestamp(m["utcDate"][:10])

        h_home = _team_stats_before(historical_df, home_id, date, as_home=True)
        a_away = _team_stats_before(historical_df, away_id, date, as_home=False)
        h_form = _overall_form(historical_df, home_id, date)
        a_form = _overall_form(historical_df, away_id, date)
        h2h    = _h2h(historical_df, home_id, away_id, date)

        rows.append({
            "home_home_win_rate":  h_home["win_rate"],
            "home_home_draw_rate": h_home["draw_rate"],
            "home_home_avg_gf":    h_home["avg_gf"],
            "home_home_avg_ga":    h_home["avg_ga"],
            "away_away_win_rate":  a_away["win_rate"],
            "away_away_draw_rate": a_away["draw_rate"],
            "away_away_avg_gf":    a_away["avg_gf"],
            "away_away_avg_ga":    a_away["avg_ga"],
            "home_form_pts":  h_form["form_pts"],
            "home_form_gf":   h_form["form_gf"],
            "home_form_ga":   h_form["form_ga"],
            "away_form_pts":  a_form["form_pts"],
            "away_form_gf":   a_form["form_gf"],
            "away_form_ga":   a_form["form_ga"],
            "form_pts_diff":  h_form["form_pts"] - a_form["form_pts"],
            "gf_diff":        h_form["form_gf"] - a_form["form_gf"],
            "ga_diff":        h_form["form_ga"] - a_form["form_ga"],
            **h2h,
        })
        meta.append({
            "date":      m["utcDate"][:10],
            "home_team": home_name,
            "away_team": away_name,
        })

    if not rows:
        print("Could not build features for upcoming matches.")
        return

    X = pd.DataFrame(rows)[FEATURE_COLS].fillna(0)

    outcome_enc  = clf.predict(X)
    outcome_prob = clf.predict_proba(X)
    outcomes     = le.inverse_transform(outcome_enc)
    goals_pred   = reg.predict(X)

    print(f"\n{'Date':<12} {'Home':<25} {'Away':<25} {'Pred':>5} {'Goals':>6}  Probs (H/D/A)")
    print("-" * 95)
    for i, m in enumerate(meta):
        probs = outcome_prob[i]
        prob_str = "  ".join(f"{p:.2f}" for p in probs)
        print(
            f"{m['date']:<12} {m['home_team']:<25} {m['away_team']:<25} "
            f"{outcomes[i]:>5} {goals_pred[i]:>6.1f}  {prob_str}"
        )