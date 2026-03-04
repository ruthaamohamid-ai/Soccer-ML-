import time
import requests
import pandas as pd
from pathlib import Path
from config import BASE_URL, HEADERS, COMPETITION, SEASONS

CACHE_DIR = Path("data")
CACHE_DIR.mkdir(exist_ok=True)


def _get(url: str, params: dict = None) -> dict:
    """Make a rate-limited API request."""
    response = requests.get(url, headers=HEADERS, params=params, timeout=10)
    if response.status_code == 429:
        print("Rate limited — waiting 60s...")
        time.sleep(60)
        response = requests.get(url, headers=HEADERS, params=params, timeout=10)
    response.raise_for_status()
    time.sleep(0.5)  # free tier: ~10 req/min
    return response.json()


def fetch_matches(season: int) -> list[dict]:
    """Fetch all finished La Liga matches for a given season start year."""
    cache_file = CACHE_DIR / f"matches_{COMPETITION}_{season}.json"
    if cache_file.exists():
        print(f"  Loading cached matches for {season}/{season+1}")
        import json
        return json.loads(cache_file.read_text())

    print(f"  Fetching matches for {season}/{season+1}...")
    url = f"{BASE_URL}/competitions/{COMPETITION}/matches"
    data = _get(url, {"season": season, "status": "FINISHED"})
    matches = data.get("matches", [])

    import json
    cache_file.write_text(json.dumps(matches))
    print(f"  Got {len(matches)} matches")
    return matches


def fetch_upcoming_matches() -> list[dict]:
    """Fetch scheduled La Liga matches."""
    url = f"{BASE_URL}/competitions/{COMPETITION}/matches"
    data = _get(url, {"status": "SCHEDULED"})
    return data.get("matches", [])


def matches_to_dataframe(matches: list[dict]) -> pd.DataFrame:
    """Flatten raw match JSON into a tidy DataFrame."""
    rows = []
    for m in matches:
        score = m.get("score", {})
        full = score.get("fullTime", {})
        home_goals = full.get("home")
        away_goals = full.get("away")

        if home_goals is None or away_goals is None:
            continue

        if home_goals > away_goals:
            outcome = "H"
        elif home_goals < away_goals:
            outcome = "A"
        else:
            outcome = "D"

        rows.append({
            "match_id":    m["id"],
            "date":        m["utcDate"][:10],
            "season":      m.get("season", {}).get("startDate", "")[:4],
            "matchday":    m.get("matchday"),
            "home_team":   m["homeTeam"]["name"],
            "away_team":   m["awayTeam"]["name"],
            "home_id":     m["homeTeam"]["id"],
            "away_id":     m["awayTeam"]["id"],
            "home_goals":  home_goals,
            "away_goals":  away_goals,
            "total_goals": home_goals + away_goals,
            "outcome":     outcome,
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_all_matches() -> pd.DataFrame:
    """Load and combine all configured seasons."""
    all_matches = []
    for season in SEASONS:
        raw = fetch_matches(season)
        all_matches.extend(raw)
    df = matches_to_dataframe(all_matches)
    df = df.drop_duplicates("match_id").reset_index(drop=True)
    print(f"Total matches loaded: {len(df)}")
    return df