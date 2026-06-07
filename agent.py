"""
agent.py — DisasterGuard Agent Core
ReAct-style agent built on the Gemini API (google-genai SDK) with native
automatic function calling: the model decides which tools to call (live
disaster feeds, historical RAG search, ML risk analysis), the SDK executes
them, and the final natural-language answer is returned.
"""
import logging
import time

from google import genai
from google.genai import types
from google.genai.errors import ServerError, ClientError

from config import GEMINI_API_KEY, GEMINI_MODEL, MAX_TOKENS
from tools import ALL_TOOLS

logger = logging.getLogger(__name__)

# ─── System Prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are DisasterGuard, an expert AI agent specializing in natural disaster monitoring, crisis management, and emergency response.

Your capabilities (via tools):
- Real-time disaster data from GDACS, USGS, and NASA EONET
- Semantic search over a historical database of major disasters (RAG)
- ML-based risk clustering and anomaly detection for earthquakes

Your behavior rules:
1. ALWAYS call at least one tool before answering factual questions about current events.
2. ALWAYS cite your data sources (e.g., "According to USGS data...").
3. Provide advanced, deeply analyzed insights, not just raw data.
4. If asked about tsunami risk, check earthquake depth AND magnitude (shallow + M>=7 = high risk).
5. ALWAYS respond in the SAME LANGUAGE the user writes in (Hebrew -> Hebrew, English -> English).
6. Be thorough, structured, and professional. Use emojis sparingly for readability.
7. If a query is outside disaster/emergency scope, politely redirect:
   "I specialize in disaster monitoring. For [topic], I recommend [alternative]."
8. Include a confidence level when making assessments: [HIGH] / [MEDIUM] / [LOW] concern."""

CRITICAL EVALUATION RULES (Project Rubric Requirements):
- Anomaly Detection: You must actively look for statistical or meteorological anomalies in the tool outputs (e.g., spikes in wind speed, extreme magnitude, unusual clustering). Explicitly state if an anomaly is detected.
- Historical Similarity & Profile Matching (RAG): When analyzing an event, you must cross-reference it with historical data via RAG search to identify patterns or matching risk profiles from past disasters.
- Customized Response Recommendation System: You must formulate a structured emergency action plan (e.g., evacuation zones, resource allocation, rescue protocols) tailored to the specific disaster type and geographical impact.
- NLP Media Verification: When analyzing user reports or external text-based news, evaluate the credibility of the text against physical sensor data from your live tools to identify potential fake news or contradictions.

Response format for disaster reports (Adhere strictly to this):
[Location]: ...
[Event]: ...
[Data Summary]: ...
[Anomaly & Pattern Detection]: (Identify unusual metrics or data fluctuations)
[Historical Similarity Search (RAG)]: (Compare this event with past disaster profiles)
[Emergency Crisis Management Recommendations]: (Provide actionable rescue, resource allocation, and evacuation protocols)
[Media Reliability Assessment]: (Evaluate text/report credibility vs. sensor data if applicable)
[Assessment Level]: [HIGH] / [MEDIUM] / [LOW] concern

if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY is not set — the agent will not be able to reach Gemini.")


class DisasterGuardAgent:
    """
    Function-calling agent for disaster monitoring, backed by Gemini.
    Maintains one chat session per session_id (in-memory, not persisted across restarts).
    """

    def __init__(self):
        self._client = genai.Client(api_key=GEMINI_API_KEY)
        self._chat_config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=ALL_TOOLS,
            max_output_tokens=MAX_TOKENS,
            temperature=0.3,
        )
        self._chats: dict[str, "genai.chats.Chat"] = {}

    def _get_chat(self, session_id: str):
        if session_id not in self._chats:
            self._chats[session_id] = self._client.chats.create(
                model=GEMINI_MODEL,
                config=self._chat_config,
            )
        return self._chats[session_id]

    def clear_history(self, session_id: str):
        self._chats.pop(session_id, None)

    def run(self, user_message: str, session_id: str = "default") -> str:
        """Process a user message and return the agent's final text response."""
        chat = self._get_chat(session_id)
        retry_delays = (2, 5)  # seconds — Gemini's "high demand" 503s are usually brief
        for attempt, delay in enumerate((*retry_delays, None)):
            try:
                response = chat.send_message(user_message)
                return response.text
            except ServerError as e:
                if delay is not None and e.code == 503:
                    logger.warning(f"Gemini overloaded (503) for session {session_id}, retrying in {delay}s...")
                    time.sleep(delay)
                    continue
                logger.exception(f"Agent run failed for session {session_id}: {e}")
                self._chats.pop(session_id, None)
                return "DisasterGuard is temporarily unavailable (Gemini is under high demand). Please try again in a moment."
            except ClientError as e:
                logger.exception(f"Agent run failed for session {session_id}: {e}")
                self._chats.pop(session_id, None)
                if e.code == 429:
                    return "You've reached the AI's daily usage limit. Please try again later (quota resets every 24 hours)."
                return "DisasterGuard is temporarily unavailable. Please try again."
            except Exception as e:
                logger.exception(f"Agent run failed for session {session_id}: {e}")
                # Drop the broken chat session so the next message starts fresh.
                self._chats.pop(session_id, None)
                return "DisasterGuard is temporarily unavailable. Please try again."
