"""
main.py — DisasterGuard Agent: CLI Entry Point
Run: python main.py
"""
import logging
from agent import DisasterGuardAgent

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

WELCOME_MSG = """
===========================================
DisasterGuard Agent - CLI Interface
===========================================

I'm an AI system specialized in natural disaster monitoring and crisis management.

What I can do:
- Real-time global disaster alerts (GDACS)
- Live earthquake data (USGS)
- Active wildfires, volcanoes & storms (NASA EONET)
- Historical disaster database (semantic search)
- ML-based risk assessment

Example questions:
- "What disasters are happening right now?"
- "Is there tsunami risk from recent earthquakes?"
- "Tell me about historical floods similar to Pakistan 2022"
- "What are the active wildfires today?"

Type 'exit' or 'quit' to stop.
===========================================
"""


def main():
    agent = DisasterGuardAgent()
    session_id = "cli_session"
    
    print(WELCOME_MSG)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\nGoodbye!")
                break
                
            if user_input.lower() == 'clear':
                agent.clear_history(session_id)
                print("Conversation history cleared.")
                continue
            
            response = agent.run(user_input, session_id=session_id)
            print(f"\nAgent: {response}")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
