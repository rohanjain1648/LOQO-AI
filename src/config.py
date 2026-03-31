"""
Configuration module — Gemini LLM + LangFuse observability setup.

Provides both standard and retry-aware LLM constructors.
Progressive temperature reduction on retries makes the model more
deterministic as we narrow down to specific fixes.
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langfuse.langchain import CallbackHandler

load_dotenv()

# ── LangFuse Observability (mandatory) ──
if os.getenv("LANGFUSE_BASE_URL") and not os.getenv("LANGFUSE_HOST"):
    os.environ["LANGFUSE_HOST"] = os.environ["LANGFUSE_BASE_URL"]

langfuse_handler = CallbackHandler()


# ── Retry temperature schedule (more precise on each retry) ──
RETRY_TEMPERATURE_SCHEDULE = {
    0: 0.7,   # First attempt — creative
    1: 0.5,   # Retry 1 — balanced
    2: 0.4,   # Retry 2 — focused
    3: 0.3,   # Retry 3 — precise
    4: 0.2,   # Retry 4 — very precise
    5: 0.1,   # Retry 5 — near-deterministic
}

# ── Per-agent retry budget ──
MAX_RETRIES_PER_AGENT = {
    "editor": 5,
    "visual": 5,
    "headline": 5,
}

# ── Global cycle safety cap ──
MAX_GLOBAL_CYCLES = 8


def get_llm(temperature: float = 0.7) -> ChatGoogleGenerativeAI:
    """
    Returns a configured Gemini 2.0 Flash instance.
    
    Args:
        temperature: 0.7 for creative tasks (narration, visuals),
                     0.2 for analytical tasks (QA scoring).
    """
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=temperature,
    )


def get_llm_for_retry(agent_name: str, retry_count: int) -> ChatGoogleGenerativeAI:
    """
    Returns a Gemini instance with progressively lower temperature for retries.
    
    Each retry reduces temperature to make the model more deterministic
    and focused on specific fixes rather than creative regeneration.
    
    Args:
        agent_name: "editor", "visual", or "headline"
        retry_count: How many times this specific agent has been retried
    """
    temp = RETRY_TEMPERATURE_SCHEDULE.get(retry_count, 0.1)
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=temp,
    )
