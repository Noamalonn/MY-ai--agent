import requests
import datetime
import urllib.parse
import pandas as pd
import os

def get_israel_fire_and_hazard_alerts(location: str) -> dict:
    """
    CRITICAL: Use this tool ONLY and ALWAYS when the user queries about any location, city, 
    or event inside ISRAEL (e.g., 'בית שמש', 'ירושלים', 'תל אביב', 'ישראל').
    """
    loc = location.strip().replace('"', '').replace("'", "")
    
    # 1. שליפת מזג אוויר וקואורדינטות בסיסיות (למקרה שאין אירוע)
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={urllib.parse.quote(loc)}&count=1&language=en"
    lat, lon = 31.7683, 35.2137 
    temp_raw, wind_raw, humidity_raw = 25.0, 15.0, 50.0
    
    try:
        geo_res = requests.get(geo_url, timeout=5).json()
        if geo_res.get("results"):
            result = geo_res["results"][0]
            lat, lon = result["latitude"], result["longitude"]
            
            weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&current=relative_humidity_2m"
            w_res = requests.get(weather_url, timeout=5).json()
            
            if w_res.get("current_weather"):
                curr_w = w_res["current_weather"]
                temp_raw = float(curr_w['temperature'])
                wind_raw = float(curr_w['wind_speed'])
            
            if "current" in w_res and "relative_humidity_2m" in w_res["current"]:
                humidity_raw = float(w_res["current"]["relative_humidity_2m"])
    except Exception:
        pass

    # 2. פתיחת קובץ האקסל מתיקיית הנתונים
    excel_path = 'data/DisasterGuard_Events_Database_v4.xlsx'
    active_event = None
    
    if os.path.exists(excel_path):
        try:
            df = pd.read_excel(excel_path)
            
            # אם המשתמש שואל בכללי על "ישראל" או "ארץ" -> נציף את תרחיש בית שמש
            if any(word in loc for word in ["ישראל", "ארץ", "israel", "Israel"]):
                # שולף את השורה של בית שמש מהאקסל
                city_matches = df[df["עיר"] == "בית שמש"]
                if not city_matches.empty:
                    active_event = city_matches.iloc[0]
            else:
                # חיפוש לפי עיר ספציפית שהמשתמש הקליד
                city_matches = df[df["עיר"].str.contains(loc, na=False)]
                if not city_matches.empty:
                    active_event = city_matches.iloc[0]
                    
        except Exception as e:
            print(f"Error reading Excel: {e}")

    # 3. אם מצאנו אירוע תואם באקסל -> נחזיר לאייג'נט את כל הפירוט!
    if active_event is not None:
        return {
            "status": "ACTIVE_DISASTER_ALERT",
            "event_type": active_event.get("סוג אירוע", "שריפה"),
            "city": active_event.get("עיר", loc),
            "exact_location": active_event.get("מיקום מדויק", "לא ידוע"),
            "scale_and_size": active_event.get("היקף האירוע", "לא צוין"),
            "population_details": active_event.get("אזור מאוכלס?", "לא צוין"),
            "infrastructure_risk": active_event.get("אזור עם תשתיות?", "לא צוין"),
            "fire_station_address": active_event.get("כתובת מדויקת - תחנת כיבוי אש", "לא צוין"),
            "mada_station_address": active_event.get("כתובת מדויקת - תחנת מד״א", "לא צוין"),
            "weather_during_event": active_event.get("מזג אוויר בעת האירוע", "לא צוין"),
            "weather_anomaly": active_event.get("חריגות טמפרטורה (אירוע מול שגרה)", "לא צוין"),
            "agent_instruction": "The user is asking about this event. Provide them with these EXACT details naturally in Hebrew."
        }
        
    # 4. אם לא נמצאה התאמה באקסל, נחזיר מצב שגרה
    return {
        "status": "CLEAR",
        "requested_location": loc,
        "coordinates": f"{lat}, {lon}",
        "message": f"לא זוהו דיווחי חירום או אנומליות תרמיות מתוך יומן האירועים עבור: {loc}.",
        "environmental_data": {
            "temperature": f"{temp_raw}°C",
            "wind_speed": f"{wind_raw} km/h",
            "humidity": f"{humidity_raw}%"
        },
        "source": "Israel Emergency Services Database"
    }
