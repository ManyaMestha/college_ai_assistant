import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


# --------------------------------------------------
# 1. Load API key and settings from the .env file
# --------------------------------------------------

# The .env file (project root) should contain:
#   GOOGLE_API_KEY=your-gemini-api-key
# Optionally:
#   GEMINI_MODEL=gemini-3.8-flash

load_dotenv()

DEFAULT_MODEL = "gemini-3.8-flash"


# --------------------------------------------------
# 2. Shared Gemini model
# --------------------------------------------------

@lru_cache(maxsize=None)
def get_llm(temperature=0.3):
    """
    Return the Gemini chat model shared by
    the chatbot and the study planner.

    The model is created only when first needed,
    so importing this file does not require an API key.

    Args:
        temperature: Higher values give more varied answers.

    Returns:
        A LangChain ChatGoogleGenerativeAI model.
    """

    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        raise RuntimeError(
            "Gemini API key not found. Add GOOGLE_API_KEY "
            "to the .env file in the project root."
        )

    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", DEFAULT_MODEL),
        temperature=temperature,
        # Fail fast instead of retrying for minutes when Gemini is busy
        max_retries=2,
        timeout=60
    )
