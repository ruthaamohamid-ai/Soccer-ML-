import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "home_home_win_rate", "home_home_draw_rate", "home_home_avg_gf", "home_home_avg_ga",
    "away_away_win_rate", "away_away_draw_rate", "away_away_avg_gf", "away_away_avg_ga",
    "home_form_pts", "home_form_gf", "home_form_ga",
    "away_form_pts", "away_form_gf", "away_form_ga",
    "form_pts_diff", "gf_diff", "ga_diff",
    "h2h_home_wins", "h2h_draws", "h2h_away_wins", "h2h_avg_goals",
]


def train(feature_df: pd.DataFrame):
    """Train outcome classifier + goals regressor and save both models."""
    X = feature_df[FEATURE_COLS].fillna(0)
    y_outcome = feature_df["outcome"]
    y_goals   = feature_df["total_goals"]

    X_train, X_test, yo_train, yo_test, yg_train, yg_test = train_test_split(
        X, y_outcome, y_goals, test_size=0.2, random_state=42
    )

    # --- Outcome model ---
    le = LabelEncoder()
    yo_train_enc = le.fit_transform(yo_train)
    yo_test_enc  = le.transform(yo_test)

    clf = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    clf.fit(X_train, yo_train_enc)

    yo_pred = clf.predict(X_test)
    print("\n=== Outcome Model ===")
    print(f"Accuracy: {accuracy_score(yo_test_enc, yo_pred):.3f}")
    print(classification_report(yo_test_enc, yo_pred, target_names=le.classes_))

    # --- Goals model ---
    reg = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    reg.fit(X_train, yg_train)

    yg_pred = reg.predict(X_test)
    print("\n=== Goals Model ===")
    print(f"MAE: {mean_absolute_error(yg_test, yg_pred):.3f} goals")

    # Save
    joblib.dump(clf, MODEL_DIR / "outcome_model.pkl")
    joblib.dump(reg, MODEL_DIR / "goals_model.pkl")
    joblib.dump(le,  MODEL_DIR / "label_encoder.pkl")
    print("\nModels saved to models/")
    return clf, reg, le


def load_models():
    clf = joblib.load(MODEL_DIR / "outcome_model.pkl")
    reg = joblib.load(MODEL_DIR / "goals_model.pkl")
    le  = joblib.load(MODEL_DIR / "label_encoder.pkl")
    return clf, reg, le