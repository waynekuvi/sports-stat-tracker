# Real-Time Sports Stat Tracker (Flask + Pandas + Matplotlib + Scikit-learn)

Lightweight prototype to display live football match stats (goals, possession) in the browser with a basic ML model predicting who scores next.

## Features

- Live data via Football-Data.org (optional) with automatic demo fallback
- Matplotlib live-updating plot (possession trend + goal markers)
- Logistic Regression model trained on a tiny (50-row) synthetic dataset
- Simple Flask UI, minimal dependencies, runs locally

## Quick Start

1) Install Python 3.10+ and create a virtualenv (recommended)

```bash
cd /Users/macuser/SportsStatTracker_v1_20251020
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) (Optional) Set env vars to use live API instead of demo mode:

```bash
export FOOTBALL_DATA_API_TOKEN="YOUR_TOKEN"
export MATCH_ID="YOUR_MATCH_ID"   # e.g., 416448 for a specific match
```

If not set, the app runs in demo mode with simulated stats.

3) Run the server:

```bash
python run.py
```

Open `http://localhost:5000` in your browser.

## Notes

- The model is extremely simple (Logistic Regression) and uses a synthetic dataset auto-generated at first run in `data/sample_matches.csv`. Replace it with your own historical data if available.
- Possession is not consistently available in some free API responses; this app simulates possession when missing to keep visuals informative.
- The plot auto-refreshes every ~3 seconds. Goals are marked with vertical lines (green: home, red: away).

## Folder Structure

```
app/
  app.py            # Flask app factory and endpoints
  templates/
    index.html      # UI
  static/
    app.js          # Polls API and refreshes plot
    styles.css      # Minimal styling
data/
  sample_matches.csv (auto-created on first run)
run.py              # Entry point
requirements.txt
```

## API Setup (Football-Data.org)

1) Create a free account and get an API token.
2) Export your token and a match id:

```bash
export FOOTBALL_DATA_API_TOKEN="YOUR_TOKEN"
export MATCH_ID="12345"
```

If calls fail or rate limit is reached, the app automatically uses demo mode.

## Why this works

- Innovative: Live stats + ML prediction + visualization
- Quick: Minimal stack, only free tools
- Relevant: Sports and data engineering showcase ready for a portfolio


## Deploy on Render (free)

1) Connect repository `waynekuvi/sports-stat-tracker`.
2) Build command: `pip install -r requirements.txt`
3) Start command: `gunicorn -w 2 -b 0.0.0.0:$PORT app.app:create_app()`
4) Environment variables:
   - `PYTHON_VERSION=3.11.9`  ← Render requires major.minor.patch
   - `FOOTBALL_DATA_API_TOKEN=YOUR_TOKEN`
   - `MATCH_ID=12345`


