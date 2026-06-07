"""
tools/usgs_tool.py — Fetch earthquake data from USGS real-time API
"""
import requests
from datetime import datetime, timedelta, timezone
from tenacity import retry, stop_after_attempt, wait_exponential
from config import USGS_API
from ml.anomaly import analyze_event

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=4))
def get_earthquakes(
    min_magnitude: float = 4.5,
    days_back: int = 1,
    region: str = "",
) -> str:
    """Fetch recent earthquake data from USGS, with ML-based risk profiling.

    Use when the user asks about earthquakes, seismic activity, or tsunami
    risk from specific quakes. Returns magnitude, location, depth,
    coordinates, a basic tsunami-risk assessment, and a machine-learning
    risk-cluster + anomaly-detection profile (K-Means + Isolation Forest)
    for each quake.

    Args:
        min_magnitude: Minimum magnitude to include (e.g. 5.0 for
            significant quakes, 6.5 for major ones).
        days_back: How many days back to search, 1-30. Default 1 (recent).
        region: Optional region/place name to filter by (e.g. "Japan",
            "Turkey", "California"). Leave empty for global results.
    """
    end_time   = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=max(1, min(30, days_back)))

    params = {
        "format":    "geojson",
        "starttime": start_time.strftime("%Y-%m-%dT%H:%M:%S"),
        "endtime":   end_time.strftime("%Y-%m-%dT%H:%M:%S"),
        "minmagnitude": min_magnitude,
        "orderby":   "magnitude",
        "limit":     20,
    }

    try:
        resp = requests.get(USGS_API, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        return f"[USGS ERROR] Could not fetch earthquakes: {e}"

    features = data.get("features", [])
    if not features:
        return f"[USGS] No earthquakes ≥ M{min_magnitude} in the past {days_back} day(s)."

    # Optional region filter (simple substring match on place)
    if region:
        features = [
            f for f in features
            if region.lower() in f["properties"].get("place", "").lower()
        ]
        if not features:
            return f"[USGS] No earthquakes found near '{region}' matching criteria."

    total = data.get("metadata", {}).get("count", len(features))
    lines = [f"USGS Earthquakes - {len(features)} shown (of {total} total >= M{min_magnitude}):\n"]

    for feat in features[:10]:
        props = feat["properties"]
        coords = feat["geometry"]["coordinates"]  # [lon, lat, depth]
        mag    = props.get("mag", "?")
        place  = props.get("place", "Unknown location")
        depth  = coords[2] if len(coords) > 2 else "?"
        lat    = coords[1]
        lon    = coords[0]
        time_ms = props.get("time", 0)
        dt     = datetime.fromtimestamp(time_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        # Tsunami warning assessment
        tsunami_risk = ""
        if isinstance(mag, (int, float)) and isinstance(depth, (int, float)):
            if mag >= 7.0 and depth < 70:
                tsunami_risk = " [WARNING] TSUNAMI RISK - shallow major quake"
            elif mag >= 6.0 and depth < 30:
                tsunami_risk = " [ALERT] Monitor for tsunami"

        lines.append(
            f"[ALERT] M{mag} - {place} | Depth: {depth}km | "
            f"Lat: {lat:.2f}, Lon: {lon:.2f} | {dt}{tsunami_risk}"
        )

        # ML-based risk clustering & anomaly detection (K-Means + Isolation Forest)
        if isinstance(mag, (int, float)) and isinstance(depth, (int, float)):
            try:
                analysis = analyze_event({
                    "magnitude": mag, "depth_km": depth,
                    "latitude": lat, "longitude": lon,
                    "year": dt[:4],
                })
                lines.append(
                    f"   ML risk profile: {analysis['risk_summary']} "
                    f"(anomaly score: {analysis['anomaly_score']})"
                )
            except Exception:
                pass

    return "\n".join(lines)
