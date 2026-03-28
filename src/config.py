"""
Configuration module — Gemini LLM + LangFuse observability setup.
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langfuse.langchain import CallbackHandler

load_dotenv()

# ── LangFuse Observability (mandatory) ──
langfuse_handler = CallbackHandler(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
)


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
