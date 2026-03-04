import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error

from data_collector import load_all_matches, fetch_upcoming_matches
from features import build_features, _team_stats_before, _overall_form, _h2h
from model import train, load_models, FEATURE_COLS

st.set_page_config(page_title="La Liga Match Predictor", page_icon="⚽", layout="wide")

OUTCOME_COLORS = {"H": "#1a7a3e", "D": "#b8860b", "A": "#9b1c1c"}


# ── Session state ─────────────────────────────────────────────────────────────

def _init_session_state():
    defaults = {
        "pipeline_run": False,
        "historical_df": None,
        "feature_df": None,
        "predictions_df": None,
        "accuracy": None,
        "mae": None,
        "clf": None,
        "reg": None,
        "le": None,
        "filter_date_range": None,
        "filter_teams": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_session_state()


# ── Core helpers ──────────────────────────────────────────────────────────────

def get_predictions_df(historical_df, clf, reg, le):
    upcoming = fetch_upcoming_matches()
    if not upcoming:
        return pd.DataFrame()

    rows, meta = [], []
    for m in upcoming:
        home_id = m["homeTeam"]["id"]
        away_id = m["awayTeam"]["id"]
        date = pd.Timestamp(m["utcDate"][:10])

        h_home = _team_stats_before(historical_df, home_id, date, as_home=True)
        a_away = _team_stats_before(historical_df, away_id, date, as_home=False)
        h_form = _overall_form(historical_df, home_id, date)
        a_form = _overall_form(historical_df, away_id, date)
        h2h = _h2h(historical_df, home_id, away_id, date)

        rows.append({
            "home_home_win_rate": h_home["win_rate"],
            "home_home_draw_rate": h_home["draw_rate"],
            "home_home_avg_gf": h_home["avg_gf"],
            "home_home_avg_ga": h_home["avg_ga"],
            "away_away_win_rate": a_away["win_rate"],
            "away_away_draw_rate": a_away["draw_rate"],
            "away_away_avg_gf": a_away["avg_gf"],
            "away_away_avg_ga": a_away["avg_ga"],
            "home_form_pts": h_form["form_pts"],
            "home_form_gf": h_form["form_gf"],
            "home_form_ga": h_form["form_ga"],
            "away_form_pts": a_form["form_pts"],
            "away_form_gf": a_form["form_gf"],
            "away_form_ga": a_form["form_ga"],
            "form_pts_diff": h_form["form_pts"] - a_form["form_pts"],
            "gf_diff": h_form["form_gf"] - a_form["form_gf"],
            "ga_diff": h_form["form_ga"] - a_form["form_ga"],
            **h2h,
        })
        meta.append({
            "date": m["utcDate"][:10],
            "home_team": m["homeTeam"]["name"],
            "away_team": m["awayTeam"]["name"],
        })

    if not rows:
        return pd.DataFrame()

    X = pd.DataFrame(rows)[FEATURE_COLS].fillna(0)
    outcome_enc = clf.predict(X)
    outcome_prob = clf.predict_proba(X)
    outcomes = le.inverse_transform(outcome_enc)
    goals_pred = reg.predict(X)

    result = pd.DataFrame(meta)
    result["predicted_outcome"] = outcomes
    result["predicted_goals"] = goals_pred.round(1)
    for i, c in enumerate(le.classes_):
        result[f"prob_{c}"] = (outcome_prob[:, i] * 100).round(1)

    result["date"] = pd.to_datetime(result["date"])
    return result.sort_values("date").reset_index(drop=True)


def _compute_metrics(feature_df, clf, reg, le):
    X = feature_df[FEATURE_COLS].fillna(0)
    y_outcome = feature_df["outcome"]
    y_goals = feature_df["total_goals"]
    _, X_test, _, yo_test, _, yg_test = train_test_split(
        X, y_outcome, y_goals, test_size=0.2, random_state=42
    )
    yo_test_enc = le.transform(yo_test)
    accuracy = accuracy_score(yo_test_enc, clf.predict(X_test))
    mae = mean_absolute_error(yg_test, reg.predict(X_test))
    return accuracy, mae


# ── Pipeline actions ──────────────────────────────────────────────────────────

def run_pipeline():
    progress = st.sidebar.progress(0, text="Loading match data...")
    historical_df = load_all_matches()
    st.session_state["historical_df"] = historical_df

    progress.progress(30, text="Engineering features...")
    feature_df = build_features(historical_df)
    st.session_state["feature_df"] = feature_df

    progress.progress(55, text="Training models...")
    clf, reg, le = train(feature_df)
    st.session_state.update({"clf": clf, "reg": reg, "le": le})
    accuracy, mae = _compute_metrics(feature_df, clf, reg, le)
    st.session_state.update({"accuracy": accuracy, "mae": mae})

    progress.progress(80, text="Generating predictions...")
    predictions_df = get_predictions_df(historical_df, clf, reg, le)
    st.session_state["predictions_df"] = predictions_df
    st.session_state["pipeline_run"] = True
    progress.progress(100, text="Done!")
    progress.empty()


def _load_saved_models_into_state():
    with st.spinner("Loading saved models..."):
        clf, reg, le = load_models()
        historical_df = load_all_matches()
        feature_df = build_features(historical_df)
        accuracy, mae = _compute_metrics(feature_df, clf, reg, le)
        predictions_df = get_predictions_df(historical_df, clf, reg, le)
        st.session_state.update({
            "historical_df": historical_df,
            "feature_df": feature_df,
            "clf": clf, "reg": reg, "le": le,
            "accuracy": accuracy, "mae": mae,
            "predictions_df": predictions_df,
            "pipeline_run": True,
        })


# ── Render helpers ────────────────────────────────────────────────────────────

def _color_outcome_row(row):
    color = OUTCOME_COLORS.get(row["Pred"], "#ffffff")
    text = "white" if row["Pred"] in OUTCOME_COLORS else "black"
    return [f"background-color: {color}; color: {text}"] * len(row)


def _render_styled_predictions_table(df):
    display = df.copy()
    display["date"] = display["date"].dt.strftime("%Y-%m-%d")
    display = display.rename(columns={
        "date": "Date", "home_team": "Home", "away_team": "Away",
        "predicted_outcome": "Pred", "predicted_goals": "Goals",
        "prob_H": "P(Home)%", "prob_D": "P(Draw)%", "prob_A": "P(Away)%",
    })
    col_order = ["Date", "Home", "Away", "Pred", "Goals", "P(Home)%", "P(Draw)%", "P(Away)%"]
    # Only include columns that exist (prob columns depend on le.classes_)
    col_order = [c for c in col_order if c in display.columns]
    display = display[col_order]
    styled = (
        display.style
        .apply(_color_outcome_row, axis=1)
        .format({"Goals": "{:.1f}", "P(Home)%": "{:.1f}", "P(Draw)%": "{:.1f}", "P(Away)%": "{:.1f}"})
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)


def _render_probability_chart(row):
    labels = ["Home Win", "Draw", "Away Win"]
    probs = [row.get("prob_H", 0), row.get("prob_D", 0), row.get("prob_A", 0)]
    colors = ["#28a745", "#ffc107", "#dc3545"]
    label_map = {"H": "Home Win", "D": "Draw", "A": "Away Win"}

    fig = go.Figure(go.Bar(
        x=probs, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{p:.1f}%" for p in probs],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"{row['home_team']}  vs  {row['away_team']}  "
              f"({pd.Timestamp(row['date']).strftime('%b %d')}) — "
              f"Predicted: {label_map.get(row['predicted_outcome'], row['predicted_outcome'])}",
        xaxis=dict(title="Probability (%)", range=[0, 110]),
        yaxis=dict(autorange="reversed"),
        height=260,
        margin=dict(l=10, r=70, t=55, b=30),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_predictions_tab():
    st.header("Upcoming Match Predictions")
    df = st.session_state["predictions_df"]

    if df is None or df.empty:
        st.warning("No upcoming La Liga fixtures found.")
        return

    # Apply filters
    filtered = df.copy()
    date_range = st.session_state.get("filter_date_range")
    if date_range and len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        filtered = filtered[(filtered["date"] >= start) & (filtered["date"] <= end)]
    selected_teams = st.session_state.get("filter_teams", [])
    if selected_teams:
        filtered = filtered[
            filtered["home_team"].isin(selected_teams) | filtered["away_team"].isin(selected_teams)
        ]

    if filtered.empty:
        st.info("No matches match the current filters.")
        return

    st.caption(f"Showing {len(filtered)} of {len(df)} upcoming fixtures")
    _render_styled_predictions_table(filtered)

    st.divider()
    st.subheader("Probability Breakdown")
    match_labels = [
        f"{row.date.strftime('%b %d')}  {row.home_team} vs {row.away_team}"
        for row in filtered.itertuples()
    ]
    selected_label = st.selectbox("Select a match", match_labels)
    idx = match_labels.index(selected_label)
    _render_probability_chart(filtered.iloc[idx])


def _render_metrics_tab():
    st.header("Model Performance")
    acc = st.session_state["accuracy"]
    mae = st.session_state["mae"]
    feature_df = st.session_state["feature_df"]
    clf = st.session_state["clf"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Outcome Accuracy", f"{acc:.1%}", help="H/D/A accuracy on 20% held-out test set")
    c2.metric("Goals MAE", f"{mae:.2f} goals", help="Mean absolute error on total goals, test set")
    c3.metric("Training Seasons", "2023/24 + 2024/25")

    st.divider()
    st.subheader("Historical Outcome Distribution")
    counts = feature_df["outcome"].value_counts().reindex(["H", "D", "A"])
    fig = go.Figure(go.Bar(
        x=["Home Win", "Draw", "Away Win"],
        y=counts.values.tolist(),
        marker_color=["#28a745", "#ffc107", "#dc3545"],
        text=counts.values.tolist(),
        textposition="outside",
    ))
    fig.update_layout(yaxis_title="Count", height=320, margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Top Feature Importances — Outcome Model")
    importances = pd.Series(clf.feature_importances_, index=FEATURE_COLS).sort_values()
    top = importances.tail(12)
    fig2 = go.Figure(go.Bar(
        x=top.values.tolist(), y=top.index.tolist(),
        orientation="h", marker_color="#4c78a8",
    ))
    fig2.update_layout(xaxis_title="Importance", height=380, margin=dict(l=200, t=20))
    st.plotly_chart(fig2, use_container_width=True)


def _render_data_tab():
    st.header("Historical Match Data")
    hist_df = st.session_state["historical_df"]
    st.caption(f"{len(hist_df)} matches")

    seasons = sorted(hist_df["season"].unique(), reverse=True)
    selected = st.selectbox("Season", ["All"] + [str(s) for s in seasons])
    view = hist_df if selected == "All" else hist_df[hist_df["season"] == int(selected)]

    cols = ["date", "matchday", "home_team", "away_team", "home_goals", "away_goals", "outcome"]
    st.dataframe(view[cols].sort_values("date", ascending=False), use_container_width=True, hide_index=True)


def _render_welcome_screen():
    st.title("⚽ La Liga Match Predictor")
    st.markdown("""
    A machine learning dashboard trained on two seasons of La Liga data (2023/24 + 2024/25).

    **Click "Run Pipeline" in the sidebar to get started.**

    The pipeline will:
    1. Load historical match data (cached locally in `data/`)
    2. Engineer rolling form, H2H, and home/away features
    3. Train an outcome classifier (H/D/A) and a goals regressor
    4. Fetch upcoming fixtures and generate predictions

    Expected performance: **~44% outcome accuracy**, **~1.49 goals MAE**
    """)
    if Path("models/outcome_model.pkl").exists():
        st.info("Saved models detected. Use **Load Saved Models** in the sidebar to skip retraining.")


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("La Liga Predictor")
    st.caption("football-data.org  |  GradientBoosting")
    st.divider()

    st.button("Run Pipeline", on_click=run_pipeline, type="primary", use_container_width=True)

    if Path("models/outcome_model.pkl").exists() and not st.session_state["pipeline_run"]:
        if st.button("Load Saved Models", use_container_width=True):
            _load_saved_models_into_state()

    if st.session_state["pipeline_run"] and st.session_state["predictions_df"] is not None:
        df = st.session_state["predictions_df"]
        st.divider()
        st.subheader("Filters")

        min_d = df["date"].min().date()
        max_d = df["date"].max().date()
        date_range = st.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
        st.session_state["filter_date_range"] = date_range

        all_teams = sorted(set(df["home_team"].tolist() + df["away_team"].tolist()))
        selected_teams = st.multiselect("Filter by team", all_teams)
        st.session_state["filter_teams"] = selected_teams


# ── Main content ──────────────────────────────────────────────────────────────

if not st.session_state["pipeline_run"]:
    _render_welcome_screen()
else:
    tab1, tab2, tab3 = st.tabs(["Upcoming Predictions", "Model Performance", "Historical Data"])
    with tab1:
        _render_predictions_tab()
    with tab2:
        _render_metrics_tab()
    with tab3:
        _render_data_tab()