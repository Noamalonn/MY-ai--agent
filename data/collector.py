"""
data/collector.py — Data collection pipeline
Fetches real historical data from USGS and saves to CSV + JSON
Run once before first deployment: python data/collector.py
"""
import requests
import pandas as pd
import json
import os
from datetime import datetime, timedelta, timezone

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def fetch_usgs_historical(years_back: int = 5, min_mag: float = 5.0) -> pd.DataFrame:
    """Fetch historical earthquakes from USGS for the past N years."""
    print(f"Fetching USGS earthquakes M≥{min_mag} for past {years_back} years...")

    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=365 * years_back)

    params = {
        "format":        "geojson",
        "starttime":     start.strftime("%Y-%m-%d"),
        "endtime":       end.strftime("%Y-%m-%d"),
        "minmagnitude":  min_mag,
        "orderby":       "magnitude",
        "limit":         1000,
    }

    resp = requests.get(
        "https://earthquake.usgs.gov/fdsnws/event/1/query",
        params=params, timeout=30
    )
    resp.raise_for_status()
    data = resp.json()

    records = []
    for feat in data.get("features", []):
        p     = feat["properties"]
        coords = feat["geometry"]["coordinates"]
        records.append({
            "id":        feat["id"],
            "title":     p.get("title", ""),
            "magnitude": p.get("mag"),
            "place":     p.get("place", ""),
            "depth_km":  coords[2] if len(coords) > 2 else None,
            "latitude":  coords[1],
            "longitude": coords[0],
            "datetime":  datetime.fromtimestamp(
                p["time"] / 1000, tz=timezone.utc
            ).isoformat(),
            "tsunami":   p.get("tsunami", 0),
            "type":      "EQ",
        })

    df = pd.DataFrame(records)
    print(f"  ✓ Fetched {len(df)} earthquake records")
    return df


def fetch_nasa_eonet_historical(days_back: int = 365) -> pd.DataFrame:
    """Fetch historical events from NASA EONET."""
    print(f"Fetching NASA EONET events for past {days_back} days...")

    params = {"days": days_back, "limit": 500}
    resp   = requests.get("https://eonet.gsfc.nasa.gov/api/v3/events",
                          params=params, timeout=30)
    resp.raise_for_status()
    data   = resp.json()

    records = []
    for ev in data.get("events", []):
        cats   = ev.get("categories", [{}])
        geo    = ev.get("geometry", [{}])
        latest = geo[-1] if geo else {}
        coords = latest.get("coordinates", [0, 0])
        records.append({
            "id":        ev.get("id", ""),
            "title":     ev.get("title", ""),
            "magnitude": None,
            "place":     ev.get("title", ""),
            "latitude":  coords[1] if len(coords) > 1 else None,
            "longitude": coords[0] if len(coords) > 0 else None,
            "depth_km":  None,
            "datetime":  latest.get("date", ""),
            "type":      cats[0].get("id", "OTHER") if cats else "OTHER",
        })

    df = pd.DataFrame(records)
    print(f"  ✓ Fetched {len(df)} NASA EONET records")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize the dataset."""
    df = df.copy()

    # Drop rows with no location
    df = df.dropna(subset=["latitude", "longitude"])

    # Fill missing magnitude with median
    if "magnitude" in df.columns:
        median_mag = df["magnitude"].median()
        df["magnitude"] = df["magnitude"].fillna(median_mag)

    # Fill missing depth
    if "depth_km" in df.columns:
        median_depth = df["depth_km"].median()
        df["depth_km"] = df["depth_km"].fillna(median_depth)

    # Add derived features
    df["lat_abs"]       = df["latitude"].abs()
    df["depth_shallow"] = (df.get("depth_km", 30) < 70).astype(int)
    df["high_mag"]      = (df.get("magnitude", 5) >= 6.5).astype(int)
    df["risk_flag"]     = df["depth_shallow"] & df["high_mag"]

    # Deduplication
    df = df.drop_duplicates(subset=["latitude", "longitude", "datetime"])

    df = df.reset_index(drop=True)
    print(f"  ✓ After preprocessing: {len(df)} records")
    return df


def save(df: pd.DataFrame, name: str = "events"):
    """Save dataset as CSV and JSON."""
    csv_path  = os.path.join(OUTPUT_DIR, f"{name}.csv")
    json_path = os.path.join(OUTPUT_DIR, f"{name}.json")

    df.to_csv(csv_path, index=False, encoding="utf-8")

    # JSON for FAISS/RAG use
    records = df.to_dict(orient="records")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2, default=str)

    print(f"  ✓ Saved: {csv_path} ({len(df)} rows)")
    print(f"  ✓ Saved: {json_path}")
    return csv_path, json_path


if __name__ == "__main__":
    print("=" * 50)
    print("DisasterGuard — Data Collection Pipeline")
    print("=" * 50)

    # Fetch
    eq_df    = fetch_usgs_historical(years_back=3, min_mag=5.0)
    nasa_df  = fetch_nasa_eonet_historical(days_back=365)

    # Combine
    combined = pd.concat([eq_df, nasa_df], ignore_index=True)

    # Preprocess
    clean_df = preprocess(combined)

    # Save
    save(clean_df, name="events")

    print("\n✅ Data collection complete!")
    print(f"   Total records: {len(clean_df)}")
    print(f"   Output: data/events.csv + data/events.json")
