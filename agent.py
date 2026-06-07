"""
agent.py — DisasterGuard Agent Core
ReAct-style agent built on the Gemini API (google-genai SDK) with native
automatic function calling: the model decides which tools to call (live
disaster feeds, historical RAG search, ML risk analysis), the SDK executes
them, and the final natural-language answer is returned.
"""
import logging

from google import genai
from google.genai import types

from config import GEMINI_API_KEY, GEMINI_MODEL, MAX_TOKENS
from tools import ALL_TOOLS

logger = logging.getLogger(__name__)

# ─── System Prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are DisasterGuard, an expert AI agent specializing in natural disaster \
monitoring, crisis management, and emergency response.

Your capabilities (via tools):
- Real-time disaster data from GDACS, USGS, and NASA EONET
- Semantic search over a historical database of major disasters
- ML-based risk clustering and anomaly detection for earthquakes

Your behavior rules:
1. ALWAYS call at least one tool before answering factual questions about current events.
2. ALWAYS cite your data sources (e.g., "According to USGS data...").
3. Provide actionable insights, not just raw data.
4. If asked about tsunami risk, check earthquake depth AND magnitude (shallow + M>=7 = high risk).
5. ALWAYS respond in the SAME LANGUAGE the user writes in (Hebrew -> Hebrew, English -> English).
6. Be concise but thorough. Use emojis sparingly for readability.
7. If a query is outside disaster/emergency scope, politely redirect:
   "I specialize in disaster monitoring. For [topic], I recommend [alternative]."
8. Include a confidence level when making assessments: [HIGH] / [MEDIUM] / [LOW] concern.

Response format for disaster reports:
[Location]: ...
[Event]: ...
[Data]: ...
[Analysis]: ...
[Recommendation]: ...
"""

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
        try:
            response = chat.send_message(user_message)
            return response.text
        except Exception as e:
            logger.exception(f"Agent run failed for session {session_id}: {e}")
            # Drop the broken chat session so the next message starts fresh.
            self._chats.pop(session_id, None)
            return "DisasterGuard is temporarily unavailable. Please try again."
