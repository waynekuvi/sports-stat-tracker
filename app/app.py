import io
import os
import random
import threading
import time
from collections import deque
from datetime import datetime
from typing import Dict, Tuple

import matplotlib

# Use non-interactive backend for server environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402
from flask import Flask, jsonify, render_template, request, send_file  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


# In-memory time series buffer of recent snapshots
_series_lock = threading.Lock()
_timeseries = deque(maxlen=200)


def _ensure_sample_dataset(dataset_path: str) -> pd.DataFrame:
    """Create a small synthetic dataset if none exists, then load it.

    Columns:
    - time_minute (0-90)
    - possession_home (0-100)
    - shots_on_target_diff (-10..10)
    - is_home_next_goal (0/1)
    """
    if not os.path.exists(dataset_path):
        rng = np.random.default_rng(seed=42)
        rows = []
        for _ in range(50):
            minute = int(rng.integers(5, 88))
            possession_home = max(0, min(100, rng.normal(50, 15)))
            shots_diff = int(rng.integers(-6, 7))
            # Synthetic rule-of-thumb for labeling
            logits = (
                -2.0
                + 0.02 * possession_home
                + 0.12 * shots_diff
                - 0.01 * abs(45 - minute)
            )
            prob = 1 / (1 + np.exp(-logits))
            label = 1 if rng.random() < prob else 0
            rows.append(
                {
                    "time_minute": minute,
                    "possession_home": round(float(possession_home), 2),
                    "shots_on_target_diff": shots_diff,
                    "is_home_next_goal": label,
                }
            )
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        pd.DataFrame(rows).to_csv(dataset_path, index=False)
    return pd.read_csv(dataset_path)


def _train_model(df: pd.DataFrame) -> Pipeline:
    features = ["time_minute", "possession_home", "shots_on_target_diff"]
    X = df[features]
    y = df["is_home_next_goal"]
    pipeline = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=1000)),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


def _football_data_fetch(match_id: str, token: str) -> Tuple[Dict, str]:
    """Fetch live data from Football-Data.org. Returns (parsed, mode).

    Note: Possession may not be available for all matches via this API; if absent,
    we simulate possession around 50% to keep the prototype functional.
    """
    headers = {"X-Auth-Token": token}
    url = f"https://api.football-data.org/v4/matches/{match_id}"
    try:
        resp = requests.get(url, headers=headers, timeout=6)
        if resp.status_code != 200:
            return _demo_fetch(), "demo"
        data = resp.json()
        # Basic score extraction and team names
        match_obj = data.get("match", data)
        score = match_obj.get("score", {})
        full_time = score.get("fullTime", {})
        home_goals = (
            full_time.get("home")
            if full_time
            else score.get("home", 0)
        ) or 0
        away_goals = (
            full_time.get("away")
            if full_time
            else score.get("away", 0)
        ) or 0
        home_team_name = (match_obj.get("homeTeam") or {}).get("name")
        away_team_name = (match_obj.get("awayTeam") or {}).get("name")

        # Possession is not guaranteed; simulate if missing
        home_possession = None
        away_possession = None
        stats = data.get("match", {}).get("liveData", {}).get("statistics", {})
        if isinstance(stats, dict):
            # This structure is API-dependent; best-effort extraction
            poss = stats.get("possession", {})
            home_possession = poss.get("home")
            away_possession = poss.get("away")

        if home_possession is None or away_possession is None:
            # Simulate around 50 with small random walk to keep the UI lively
            home_possession = max(35, min(65, 50 + random.uniform(-8, 8)))
            away_possession = 100 - home_possession

        return (
            {
                "timestamp": datetime.utcnow().isoformat(),
                "homeGoals": int(home_goals),
                "awayGoals": int(away_goals),
                "homePossession": float(round(home_possession, 2)),
                "awayPossession": float(round(away_possession, 2)),
                "homeTeamName": home_team_name or "Home",
                "awayTeamName": away_team_name or "Away",
            },
            "live",
        )
    except Exception:
        return _demo_fetch(), "demo"


_demo_state = {
    "minute": 0,
    "homeGoals": 0,
    "awayGoals": 0,
    "homePossession": 50.0,
}


def _demo_fetch() -> Dict:
    """Generate plausible demo stats quickly for the prototype."""
    _demo_state["minute"] = min(95, _demo_state["minute"] + random.randint(1, 3))
    # Random walk for possession
    _demo_state["homePossession"] = max(
        35.0, min(65.0, _demo_state["homePossession"] + random.uniform(-3, 3))
    )
    # Small chance of a goal each tick
    if random.random() < 0.12:
        if random.random() < _demo_state["homePossession"] / 100:
            _demo_state["homeGoals"] += 1
        else:
            _demo_state["awayGoals"] += 1
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "homeGoals": _demo_state["homeGoals"],
        "awayGoals": _demo_state["awayGoals"],
        "homePossession": round(_demo_state["homePossession"], 2),
        "awayPossession": round(100 - _demo_state["homePossession"], 2),
        "homeTeamName": "Home",
        "awayTeamName": "Away",
    }


def _get_or_update_snapshot(config: Dict) -> Tuple[Dict, str]:
    token = os.environ.get("FOOTBALL_DATA_API_TOKEN")
    match_id = config.get("MATCH_ID", os.environ.get("MATCH_ID", ""))
    if token and match_id:
        snap, mode = _football_data_fetch(match_id, token)
    else:
        snap, mode = _demo_fetch(), "demo"

    with _series_lock:
        _timeseries.append(snap)
    return snap, mode


def _prepare_features_for_prediction(snap: Dict) -> np.ndarray:
    # Map current snapshot to model features
    # Approximate time using series length (since we may not have a clock)
    with _series_lock:
        minute = min(90, len(_timeseries))
    possession_home = snap.get("homePossession", 50.0)
    shots_diff = 0  # Not available live; keep zero for prototype
    return np.array([[minute, possession_home, shots_diff]], dtype=float)


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")

    # Load or build tiny training dataset and model once
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_matches.csv")
    df = _ensure_sample_dataset(os.path.abspath(data_path))
    model = _train_model(df)

    # Basic runtime config
    app.config.update({
        "MATCH_ID": os.environ.get("MATCH_ID", ""),
    })

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/stats")
    def api_stats():
        snap, mode = _get_or_update_snapshot(app.config)
        X = _prepare_features_for_prediction(snap)
        prob_home_next = float(model.predict_proba(X)[0, 1])
        payload = {
            **snap,
            "mode": mode,
            "probHomeNextGoal": round(prob_home_next, 3),
        }
        return jsonify(payload)

    @app.route("/plot.png")
    def plot_png():
        # Build a simple possession and goals chart
        with _series_lock:
            series = list(_timeseries)

        if not series:
            # Initialize with one demo point for first render
            series = [_demo_fetch()]

        xs = list(range(1, len(series) + 1))
        home_pos = [s["homePossession"] for s in series]
        away_pos = [s["awayPossession"] for s in series]
        home_goals = [s["homeGoals"] for s in series]
        away_goals = [s["awayGoals"] for s in series]

        fig, ax1 = plt.subplots(figsize=(6, 3.2), layout="constrained")
        ax1.plot(xs, home_pos, label="Home Possession %", color="#1f77b4")
        ax1.plot(xs, away_pos, label="Away Possession %", color="#ff7f0e")
        ax1.set_ylim(0, 100)
        ax1.set_xlabel("Updates")
        ax1.set_ylabel("Possession %")
        ax1.grid(True, alpha=0.25)

        # Mark goals as vertical lines
        for i in range(1, len(xs)):
            if home_goals[i] > home_goals[i - 1]:
                ax1.axvline(xs[i], color="#2ca02c", linestyle="--", alpha=0.6)
            if away_goals[i] > away_goals[i - 1]:
                ax1.axvline(xs[i], color="#d62728", linestyle=":", alpha=0.6)

        ax1.legend(loc="upper left", fontsize=8)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120)
        plt.close(fig)
        buf.seek(0)
        return send_file(buf, mimetype="image/png")

    return app



