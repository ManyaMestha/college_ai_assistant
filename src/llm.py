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
#   GEMINI_FALLBACK_MODELS=gemini-3.6-flash,gemini-3.5-flash

load_dotenv()

DEFAULT_MODEL = "gemini-3.8-flash"

# The free tier allows about 20 requests per day for EACH model,
# so when one model hits its limit (or is busy) the next one is used.
DEFAULT_FALLBACK_MODELS = [
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite"
]


# --------------------------------------------------
# 2. Create one Gemini model
# --------------------------------------------------

def create_model(model_name, temperature):

    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        # Fail fast so the next fallback model is tried quickly
        max_retries=1,
        timeout=60
    )


# --------------------------------------------------
# 3. Shared Gemini model with fallbacks
# --------------------------------------------------

@lru_cache(maxsize=None)
def get_llm(temperature=0.3):
    """
    Return the Gemini chat model shared by
    the chatbot and the study planner.

    If the main model fails (daily limit reached or busy),
    the fallback models are tried one by one.

    The model is created only when first needed,
    so importing this file does not require an API key.

    Args:
        temperature: Higher values give more varied answers.

    Returns:
        A LangChain chat model. Use it with .invoke(...).
    """

    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        raise RuntimeError(
            "Gemini API key not found. Add GOOGLE_API_KEY "
            "to the .env file in the project root."
        )

    main_model = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)

    if os.getenv("GEMINI_FALLBACK_MODELS"):
        fallback_models = [
            name.strip()
            for name in os.getenv("GEMINI_FALLBACK_MODELS").split(",")
            if name.strip()
        ]

    else:
        fallback_models = DEFAULT_FALLBACK_MODELS

    fallback_models = [
        name for name in fallback_models
        if name != main_model
    ]

    return create_model(main_model, temperature).with_fallbacks([
        create_model(name, temperature)
        for name in fallback_models
    ])
