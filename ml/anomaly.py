"""
ml/anomaly.py — Anomaly detection & clustering for disaster events
Uses K-Means (risk profiling) + Isolation Forest (anomaly detection)
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import json
import os

# ─── Risk cluster labels (assigned after training) ───────────────────────────
CLUSTER_LABELS = {
    0: {"name": "Low Risk",      "color": "🟢", "description": "Moderate magnitude, deep, low-population"},
    1: {"name": "Medium Risk",   "color": "🟡", "description": "Moderate magnitude, mid-depth, some population"},
    2: {"name": "High Risk",     "color": "🟠", "description": "High magnitude or shallow, dense population"},
    3: {"name": "Critical Risk", "color": "🔴", "description": "Very high magnitude, shallow, urban area"},
}

_kmeans = None
_iso_forest = None
_scaler = None


def _extract_features(event: dict) -> list[float]:
    """
    Extract numeric feature vector from an event dict.
    Features: [magnitude_norm, depth_inv, lat_abs, lon_abs, year_recency]
    All normalized to [0,1] during training.
    """
    mag   = float(event.get("magnitude") or 5.0)
    depth = float(event.get("depth_km") or 30.0)
    lat   = abs(float(event.get("latitude") or 0.0))
    lon   = abs(float(event.get("longitude") or 0.0))
    year  = float(event.get("year") or 2020)

    return [mag, 1.0 / max(depth, 1.0), lat / 90.0, lon / 180.0, (year - 1900) / 125.0]


def _train_models():
    """Train K-Means and Isolation Forest on historical data."""
    global _kmeans, _iso_forest, _scaler

    # Use historical sample data for training
    sample_events = [
        {"magnitude": 9.1, "depth_km": 30, "latitude": 38.3, "longitude": 142.4, "year": 2011},
        {"magnitude": 7.0, "depth_km": 13, "latitude": 18.5, "longitude": -72.5, "year": 2010},
        {"magnitude": 9.1, "depth_km": 30, "latitude": 3.3,  "longitude": 95.9,  "year": 2004},
        {"magnitude": 5.0, "depth_km": 10, "latitude": 30.0, "longitude": 120.0, "year": 2020},
        {"magnitude": 7.8, "depth_km": 18, "latitude": 37.2, "longitude": 37.0,  "year": 2023},
        {"magnitude": 6.0, "depth_km": 60, "latitude": 45.0, "longitude": 15.0,  "year": 2020},
        {"magnitude": 4.5, "depth_km": 100,"latitude": 20.0, "longitude": -100.0,"year": 2022},
        {"magnitude": 8.5, "depth_km": 20, "latitude": -8.0, "longitude": 115.0, "year": 2005},
        {"magnitude": 7.9, "depth_km": 19, "latitude": 31.0, "longitude": 103.4, "year": 2008},
        {"magnitude": 6.9, "depth_km": 16, "latitude": 34.6, "longitude": 135.0, "year": 1995},
        {"magnitude": 5.5, "depth_km": 80, "latitude": 50.0, "longitude": 10.0,  "year": 2019},
        {"magnitude": 3.5, "depth_km": 15, "latitude": 33.0, "longitude": -117.0,"year": 2023},
    ]

    X = np.array([_extract_features(e) for e in sample_events], dtype=float)

    _scaler = MinMaxScaler()
    X_scaled = _scaler.fit_transform(X)

    _kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    _kmeans.fit(X_scaled)

    _iso_forest = IsolationForest(
        n_estimators=100, contamination=0.1, random_state=42
    )
    _iso_forest.fit(X_scaled)


def analyze_event(event: dict) -> dict:
    """
    Analyze a single disaster event.
    Returns: {cluster_id, cluster_name, risk_level, is_anomaly, anomaly_score, description}
    """
    global _kmeans, _iso_forest, _scaler

    if _kmeans is None:
        _train_models()

    features = np.array([_extract_features(event)], dtype=float)
    features_scaled = _scaler.transform(features)

    cluster_id    = int(_kmeans.predict(features_scaled)[0])
    anomaly_score = float(_iso_forest.score_samples(features_scaled)[0])
    is_anomaly    = anomaly_score < -0.1

    cluster_info = CLUSTER_LABELS.get(cluster_id, CLUSTER_LABELS[0])

    return {
        "cluster_id":    cluster_id,
        "cluster_name":  cluster_info["name"],
        "risk_color":    cluster_info["color"],
        "description":   cluster_info["description"],
        "is_anomaly":    is_anomaly,
        "anomaly_score": round(anomaly_score, 3),
        "risk_summary": (
            f"{cluster_info['color']} {cluster_info['name']}"
            + (" ⚠️ ANOMALY DETECTED" if is_anomaly else "")
        ),
    }


def format_analysis(event_title: str, analysis: dict) -> str:
    """Format ML analysis result as a readable string for the LLM."""
    lines = [
        f"\n🤖 ML Analysis — {event_title}:",
        f"  Risk Profile: {analysis['risk_color']} {analysis['cluster_name']}",
        f"  Cluster Info: {analysis['description']}",
        f"  Anomaly Score: {analysis['anomaly_score']} "
        f"({'⚠️ ANOMALOUS' if analysis['is_anomaly'] else '✅ Normal pattern'})",
    ]
    return "\n".join(lines)
