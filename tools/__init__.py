"""tools package — plain Python functions exposed to Gemini as callable tools."""
from tools.gdacs_tool  import get_gdacs_events
from tools.usgs_tool   import get_earthquakes
from tools.nasa_tool   import get_nasa_events
from tools.search_tool import search_historical_events
from .israel_tool import get_israel_fire_and_hazard_alerts

# Registry: name → callable (used for manual lookups / tests)
TOOL_REGISTRY = {
    "get_gdacs_events":          get_gdacs_events,
    "get_earthquakes":           get_earthquakes,
    "get_nasa_events":           get_nasa_events,
    "search_historical_events":  search_historical_events,
    "get_israel_fire_and_hazard_alerts" 
}


# Gemini consumes plain Python callables directly as tools.
ALL_TOOLS = list(TOOL_REGISTRY.values())
