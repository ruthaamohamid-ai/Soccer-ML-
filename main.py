from data_collector import load_all_matches
from features import build_features
from model import train
from predict import predict_upcoming


def main():
    print("=== La Liga Match Predictor ===\n")

    # 1. Load historical data
    print("[1/4] Loading match data...")
    df = load_all_matches()

    # 2. Build features
    print("\n[2/4] Engineering features...")
    feature_df = build_features(df)
    print(f"Feature set: {len(feature_df)} matches × {feature_df.shape[1]} columns")

    # 3. Train models
    print("\n[3/4] Training models...")
    train(feature_df)

    # 4. Predict upcoming matches
    print("\n[4/4] Predicting upcoming matches...")
    predict_upcoming(df)


if __name__ == "__main__":
    main()