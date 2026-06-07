"""
tools/search_tool.py — RAG: semantic-ish search over historical disaster events
Uses TF-IDF + cosine similarity (scikit-learn). No external embedding API needed —
keeps the agent fully self-contained on Gemini for reasoning while doing local
retrieval over the historical events corpus.
"""
import json
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import EVENTS_DB_PATH, TOP_K_SIMILAR

# Lazy-loaded globals
_vectorizer = None
_matrix = None
_events_db = None


def _event_text(e: dict) -> str:
    return (
        f"{e['title']}. {e.get('description', '')} "
        f"Country: {e.get('country', '')}. Type: {e.get('type', '')}. Year: {e.get('year', '')}."
    )


def _build_sample_events() -> list[dict]:
    """Historical disaster records used as the default RAG corpus."""
    return [
        {"id": 1,  "title": "2011 Tōhoku Earthquake and Tsunami", "type": "EQ+Tsunami",
         "magnitude": 9.1, "country": "Japan", "deaths": 15897, "year": 2011,
         "description": "Massive undersea megathrust earthquake off Japan coast triggering devastating tsunami waves up to 40m, causing Fukushima nuclear disaster."},
        {"id": 2,  "title": "2010 Haiti Earthquake", "type": "EQ",
         "magnitude": 7.0, "country": "Haiti", "deaths": 316000, "year": 2010,
         "description": "Catastrophic earthquake near Port-au-Prince, one of the deadliest in recorded history."},
        {"id": 3,  "title": "2004 Indian Ocean Tsunami", "type": "Tsunami",
         "magnitude": 9.1, "country": "Indonesia/Thailand/India", "deaths": 227898, "year": 2004,
         "description": "Undersea megathrust earthquake off Sumatra triggering series of devastating tsunamis across Indian Ocean."},
        {"id": 4,  "title": "2005 Hurricane Katrina", "type": "TC",
         "magnitude": None, "country": "USA", "deaths": 1833, "year": 2005,
         "description": "Category 5 hurricane devastating New Orleans and Gulf Coast, massive flooding from levee failures."},
        {"id": 5,  "title": "2013 Typhoon Haiyan (Yolanda)", "type": "TC",
         "magnitude": None, "country": "Philippines", "deaths": 6300, "year": 2013,
         "description": "One of the strongest tropical cyclones ever recorded, devastating Tacloban City."},
        {"id": 6,  "title": "2020 Australian Bushfires (Black Summer)", "type": "WF",
         "magnitude": None, "country": "Australia", "deaths": 33, "year": 2020,
         "description": "Unprecedented wildfire season burning over 18 million hectares, destroying 3,000+ homes."},
        {"id": 7,  "title": "1999 Marmara Earthquake", "type": "EQ",
         "magnitude": 7.6, "country": "Turkey", "deaths": 17127, "year": 1999,
         "description": "Major earthquake striking northwestern Turkey near Istanbul industrial zone."},
        {"id": 8,  "title": "2023 Turkey-Syria Earthquake", "type": "EQ",
         "magnitude": 7.8, "country": "Turkey/Syria", "deaths": 56000, "year": 2023,
         "description": "Devastating double earthquake sequence striking southeastern Turkey and northern Syria in February 2023."},
        {"id": 9,  "title": "2022 Pakistan Floods", "type": "FL",
         "magnitude": None, "country": "Pakistan", "deaths": 1739, "year": 2022,
         "description": "Exceptional monsoon flooding covering one-third of Pakistan, displacing 33 million people."},
        {"id": 10, "title": "2019 Amazon Wildfires", "type": "WF",
         "magnitude": None, "country": "Brazil", "deaths": 0, "year": 2019,
         "description": "Record-breaking wildfire season in Amazon basin with 72,843 fires recorded, massive deforestation."},
        {"id": 11, "title": "2021 Haiti Earthquake", "type": "EQ",
         "magnitude": 7.2, "country": "Haiti", "deaths": 2248, "year": 2021,
         "description": "Major earthquake in southern Haiti causing widespread destruction in already fragile country."},
        {"id": 12, "title": "2017 Hurricane Maria", "type": "TC",
         "magnitude": None, "country": "Puerto Rico", "deaths": 2975, "year": 2017,
         "description": "Category 5 hurricane causing catastrophic damage to Puerto Rico, destroying power grid."},
        {"id": 13, "title": "1995 Kobe Earthquake", "type": "EQ",
         "magnitude": 6.9, "country": "Japan", "deaths": 6434, "year": 1995,
         "description": "Great Hanshin earthquake striking Kobe urban area causing massive structural damage."},
        {"id": 14, "title": "2008 Sichuan Earthquake", "type": "EQ",
         "magnitude": 7.9, "country": "China", "deaths": 87587, "year": 2008,
         "description": "Catastrophic earthquake in Sichuan province causing widespread building collapses including schools."},
        {"id": 15, "title": "2000 Mozambique Floods", "type": "FL",
         "magnitude": None, "country": "Mozambique", "deaths": 800, "year": 2000,
         "description": "Severe flooding across Mozambique following Cyclone Eline, displacing 650,000 people."},
        {"id": 16, "title": "2010 Eyjafjallajokull Eruption", "type": "VO",
         "magnitude": None, "country": "Iceland", "deaths": 0, "year": 2010,
         "description": "Volcanic eruption causing massive ash cloud disrupting European air travel for weeks."},
        {"id": 17, "title": "1991 Mount Pinatubo Eruption", "type": "VO",
         "magnitude": None, "country": "Philippines", "deaths": 847, "year": 1991,
         "description": "Second-largest volcanic eruption of 20th century, ejecting 10 billion tons of magma."},
        {"id": 18, "title": "2016 Ecuador Earthquake", "type": "EQ",
         "magnitude": 7.8, "country": "Ecuador", "deaths": 654, "year": 2016,
         "description": "Major earthquake striking coastal Ecuador, strongest since 1979."},
        {"id": 19, "title": "2021 Germany Floods", "type": "FL",
         "magnitude": None, "country": "Germany/Belgium", "deaths": 243, "year": 2021,
         "description": "Extreme flash flooding in western Germany and Belgium, deadliest European flood disaster in decades."},
        {"id": 20, "title": "2019 Cyclone Idai", "type": "TC",
         "magnitude": None, "country": "Mozambique/Zimbabwe/Malawi", "deaths": 1302, "year": 2019,
         "description": "One of the worst tropical cyclones ever recorded in the Southern Hemisphere, devastating Beira city."},
    ]


def _load_corpus():
    """Lazy-load the events DB and fit the TF-IDF vectorizer over it."""
    global _vectorizer, _matrix, _events_db

    if _events_db is not None:
        return

    if os.path.exists(EVENTS_DB_PATH):
        with open(EVENTS_DB_PATH, "r", encoding="utf-8") as f:
            _events_db = json.load(f)
    else:
        _events_db = _build_sample_events()

    texts = [_event_text(e) for e in _events_db]
    _vectorizer = TfidfVectorizer(stop_words="english")
    _matrix = _vectorizer.fit_transform(texts)


def search_historical_events(query: str, top_k: int = TOP_K_SIMILAR) -> str:
    """Search the historical disaster database for events similar to the query.

    Use this when the user asks about past disasters, historical comparisons,
    precedents, or "has this happened before" type questions. Performs a
    TF-IDF + cosine-similarity search over a corpus of major historical
    natural disasters and returns the most similar matches with key facts
    (type, country, year, death toll, description).

    Args:
        query: Natural language description of what to search for
            (e.g. "floods similar to Pakistan 2022" or "major tsunamis").
        top_k: Number of results to return (default 5).
    """
    _load_corpus()

    q_vec = _vectorizer.transform([query])
    sims = cosine_similarity(q_vec, _matrix)[0]

    k = max(1, min(top_k, len(_events_db)))
    top_indices = np.argsort(sims)[::-1][:k]

    lines = [f"Historical Disaster Database — top {k} similar events for '{query}':\n"]
    for rank, idx in enumerate(top_indices, 1):
        ev = _events_db[idx]
        score = round(float(sims[idx]) * 100, 1)
        deaths = f"{ev['deaths']:,}" if ev.get("deaths") else "N/A"
        lines.append(
            f"{rank}. [{score}% match] {ev['title']} ({ev.get('year', '?')}) "
            f"| Type: {ev.get('type', '?')} | Country: {ev.get('country', '?')} "
            f"| Deaths: {deaths}"
        )
        lines.append(f"   → {ev.get('description', '')}")

    return "\n".join(lines)
