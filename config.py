import os

API_KEY = os.environ.get("FOOTBALL_API_KEY", "4f508817f30a424880872aede7e5c0d4")
BASE_URL = "https://api.football-data.org/v4"
COMPETITION = "PD"  # La Liga

HEADERS = {"X-Auth-Token": API_KEY}

# How many past seasons to pull (current + N previous)
SEASONS = [2023, 2024]  # La Liga seasons by start year

# Feature window: last N matches for form calculation
FORM_WINDOW = 5