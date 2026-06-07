# 🌍 DisasterGuard Agent

סוכן AI לניהול משברים ואסונות טבע בזמן אמת — מבוסס Gemini API (`google-genai`), עם קריאת כלים אוטומטית (function calling), חיפוש סמנטי היסטורי (TF-IDF), וניתוח ML (K-Means + Isolation Forest) לרעידות אדמה.

## ארכיטקטורה

```
disaster_agent/
├── app.py           # שירות אינטרנט Flask — ממשק צ'אט + JSON API (לפריסה ב-Render)
├── main.py          # ממשק שורת פקודה (CLI) — להרצה/בדיקה מקומית
├── agent.py         # ליבת הסוכן — Gemini chat + function calling אוטומטי
├── config.py        # הגדרות מרכזיות
├── requirements.txt
├── render.yaml      # פריסה אוטומטית ב-Render (Web Service)
├── .env.example     # תבנית משתני סביבה
├── tools/
│   ├── gdacs_tool.py   # GDACS — אסונות גלובליים פעילים
│   ├── usgs_tool.py    # USGS — רעידות אדמה + ניתוח ML (K-Means/Isolation Forest)
│   ├── nasa_tool.py    # NASA EONET — שריפות/סופות/וולקנים
│   └── search_tool.py  # RAG — חיפוש היסטורי (TF-IDF + cosine similarity)
├── ml/
│   └── anomaly.py      # K-Means clustering + Isolation Forest anomaly detection
└── data/
    └── events_db.json  # מאגר 20 אירועי אסון היסטוריים
```

---

## הפעלה מקומית

### 1. התקן תלויות

```bash
pip install -r requirements.txt
```

### 2. הגדר מפתח API

```bash
cp .env.example .env
# ערוך את .env ומלא:
# GEMINI_API_KEY=your-gemini-api-key
# GEMINI_MODEL=gemini-2.5-flash   (אופציונלי, זו ברירת המחדל)
```

מפתח Gemini API ניתן לקבל בחינם דרך [Google AI Studio](https://aistudio.google.com/apikey).

### 3. הפעל

**ממשק צ'אט בדפדפן (כמו בפריסה ב-Render):**
```bash
python app.py
# פתח http://127.0.0.1:5000
```

**ממשק שורת פקודה:**
```bash
python main.py
```

---

## פריסה ב-Render (חינמי)

### 1. העלה ל-GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USER/disaster-agent.git
git push -u origin main
```

### 2. צור Web Service ב-Render

1. כנס ל-[dashboard.render.com/web/new](https://dashboard.render.com/web/new?onboarding=active)
2. בחר **"Build and deploy from a Git repository"** וחבר את ה-repo שלך מ-GitHub
3. Render יזהה אוטומטית את `render.yaml` (Blueprint) ויגדיר הכול לבד:
   - Runtime: Python 3.11.9
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT`
   - Plan: Free

   (אם Render לא מזהה את ה-Blueprint אוטומטית, הגדר את אלה ידנית באותו מסך.)

### 3. הגדר Environment Variable

במסך יצירת השירות (או אחר כך תחת **Environment**):
- `GEMINI_API_KEY` = מפתח ה-Gemini API שלך — **זהו המשתנה היחיד שחובה להגדיר ידנית**
- `GEMINI_MODEL` ו-`PYTHON_VERSION` כבר מוגדרים אוטומטית דרך `render.yaml`

### 4. Deploy ולחיצה ✅

תוך כמה דקות הסוכן יהיה זמין בכתובת `https://disasterguard-agent.onrender.com` (או שם דומה שRender ייתן).

### 5. בדיקה שהכול עובד

- **בריאות השירות:** גלוש ל-`https://<your-app>.onrender.com/healthz` — אמור להחזיר `{"status": "ok"}`
- **ממשק צ'אט:** גלוש לכתובת הראשית `/` ושאל שאלה, למשל:
  - "What disasters are happening right now around the world?"
  - "האם יש סיכון לצונאמי מרעידות האדמה האחרונות ביפן?"

  הסוכן ישלוף נתונים בזמן אמת מ-GDACS/USGS/NASA EONET, יצרף ניתוח ML, ויענה באותה שפה שבה נשאל, עם ציון מקורות ורמת ביטחון.

> שירותי Free tier ב-Render "נרדמים" אחרי כ-15 דקות חוסר פעילות — הבקשה הראשונה אחרי תקופת שינה עשויה לקחת כ-30-60 שניות עד שהשירות מתעורר.

---

## שאלות לדוגמה לסוכן

- "What disasters are happening right now?"
- "Is there tsunami risk from recent earthquakes in Japan?"
- "Show me major earthquakes in the past 3 days"
- "Tell me about historical floods similar to Pakistan 2022"
- "מה האסונות הפעילים כרגע בעולם?"
- "ספר לי על שיטפונות היסטוריים דומים לשיטפונות בפקיסטן 2022"

---

## מקורות נתונים וטכנולוגיות

| מקור / רכיב | תפקיד | טכנולוגיה |
|------|------|-----|
| [GDACS](https://www.gdacs.org) | אסונות גלובליים פעילים בזמן אמת | REST API |
| [USGS](https://earthquake.usgs.gov) | רעידות אדמה בזמן אמת | REST API |
| [NASA EONET](https://eonet.gsfc.nasa.gov) | שריפות/סופות/וולקנים פעילים | REST API |
| Historical DB | 20 אירועי אסון היסטוריים — חיפוש סמנטי | TF-IDF + cosine similarity (scikit-learn) |
| ניתוח סיכון רעידות אדמה | אשכול סיכון + זיהוי חריגות | K-Means + Isolation Forest (scikit-learn) |
| מנוע השפה (LLM) | הבנת שאלות, קריאת כלים אוטומטית, ניסוח תשובות | Gemini API (`google-genai`, `gemini-2.5-flash`) |
