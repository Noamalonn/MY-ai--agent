"""
tools/nasa_tool.py — Fetch natural events from NASA EONET API
"""
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from config import NASA_EONET_API

CATEGORY_ICONS = {
    "wildfires":          "[FIRE]",
    "severeStorms":       "[STORM]",
    "volcanoes":          "[VOLCANO]",
    "seaLakeIce":         "[ICE]",
    "snowIce":            "[SNOW]",
    "dustHaze":           "[DUST]",
    "floods":             "[FLOOD]",
    "landslides":         "[LANDSLIDE]",
    "earthquakes":        "[EQ]",
    "temperatureExtremes":"[TEMP]",
}

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=4))
def get_nasa_events(category: str = "ALL", days: int = 7) -> str:
    """Fetch active natural-event data from NASA EONET.

    NASA EONET (Earth Observatory Natural Event Tracker) is great for
    wildfires, volcanic eruptions, severe storms, floods, landslides and
    dust/haze events. Use when the user asks about wildfires, active
    volcanoes, or extreme weather.

    Args:
        category: Event category — one of "wildfires", "severeStorms",
            "volcanoes", "floods", "landslides", "dustHaze", or "ALL"
            for every active category.
        days: How many days back to look, default 7, max 30.
    """
    params = {"status": "open", "days": min(30, max(1, days)), "limit": 30}
    if category != "ALL":
        params["category"] = category

    try:
        resp = requests.get(NASA_EONET_API, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        return f"[NASA EONET ERROR] {e}"

    events = data.get("events", [])
    if not events:
        return f"[NASA EONET] No active events found for category '{category}'."

    lines = [f"NASA EONET - {len(events)} active events:\n"]
    for ev in events[:12]:
        title  = ev.get("title", "Unknown")
        cats   = [c.get("id", "") for c in ev.get("categories", [])]
        icon   = CATEGORY_ICONS.get(cats[0], "[EVENT]") if cats else "[EVENT]"
        geo    = ev.get("geometry", [])
        coords = geo[-1].get("coordinates", []) if geo else []
        date   = geo[-1].get("date", "")[:10] if geo else "?"

        coord_str = ""
        if coords and len(coords) >= 2:
            coord_str = f" | Lat: {coords[1]:.1f}, Lon: {coords[0]:.1f}"

        lines.append(f"{icon} {title} | Date: {date}{coord_str}")

    return "\n".join(lines)
