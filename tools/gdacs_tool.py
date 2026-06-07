"""
tools/gdacs_tool.py — Fetch current global disaster events from GDACS
"""
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from config import GDACS_API

EVENT_TYPE_MAP = {
    "EQ": "Earthquake",
    "FL": "Flood",
    "TC": "Tropical Cyclone",
    "WF": "Wildfire",
    "VO": "Volcano",
    "DR": "Drought",
}

ALERT_COLORS = {"Red": "[RED]", "Orange": "[ORANGE]", "Green": "[GREEN]", "": "[?]"}

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=4))
def get_gdacs_events(event_type: str = "ALL", alert_level: str = "ALL") -> str:
    """Fetch current active global disaster events from GDACS.

    GDACS (Global Disaster Alert and Coordination System) tracks ongoing
    floods, cyclones, earthquakes, wildfires, volcanoes and droughts
    worldwide. Use this when the user asks about ongoing disasters or
    general "what's happening right now" questions. Returns up to 10
    of the most recent/severe active events.

    Args:
        event_type: Filter by event type — one of "EQ" (earthquake),
            "FL" (flood), "TC" (tropical cyclone), "WF" (wildfire),
            "VO" (volcano), "DR" (drought), or "ALL" for no filter.
        alert_level: Filter by severity — "Red" (most severe), "Orange",
            "Green", or "ALL" for no filter.
    """
    # GDACS expects semicolon-separated lists, not "ALL" — map "ALL" to every value.
    all_event_types  = "EQ;TC;FL;VO;DR;WF"
    all_alert_levels = "Green;Orange;Red"

    params = {
        "eventlist":  all_event_types if event_type == "ALL" else event_type,
        "alertlevel": all_alert_levels if alert_level == "ALL" else alert_level,
    }

    try:
        resp = requests.get(GDACS_API, params=params, timeout=10)
        resp.raise_for_status()
        if not resp.content:
            return "[GDACS] No active events found matching your criteria."
        data = resp.json()
    except requests.RequestException as e:
        return f"[GDACS ERROR] Could not fetch events: {e}"
    except ValueError as e:
        return f"[GDACS ERROR] Unexpected response format: {e}"

    features = data.get("features", [])
    if not features:
        return "[GDACS] No active events found matching your criteria."

    lines = [f"GDACS - {len(features)} active events (showing top 10):\n"]
    for feat in features[:10]:
        props = feat.get("properties", {})
        etype  = props.get("eventtype", "")
        name   = props.get("name", "Unknown")
        alert  = props.get("alertlevel", "")
        date   = props.get("fromdate", "")[:10]
        country = props.get("country", "")
        severity = props.get("severitydata", {}).get("severity", "N/A")

        icon = ALERT_COLORS.get(alert, "⚪")
        etype_full = EVENT_TYPE_MAP.get(etype, etype)

        lines.append(
            f"{icon} [{etype_full}] {name} | Country: {country} | "
            f"Alert: {alert} | Severity: {severity} | Date: {date}"
        )

    return "\n".join(lines)
