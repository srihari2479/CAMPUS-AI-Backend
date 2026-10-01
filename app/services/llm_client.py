import httpx
from groq import Groq
import cohere
from app.core.config import settings

# Initialize Groq client if key is present
groq_client = None
if settings.GROQ_API_KEY:
    groq_client = Groq(api_key=settings.GROQ_API_KEY)
else:
    print("Warning: GROQ_API_KEY is not configured.")

# Initialize Cohere client if key is present
cohere_client = None
if settings.COHERE_API_KEY:
    cohere_client = cohere.Client(api_key=settings.COHERE_API_KEY)
else:
    print("Warning: COHERE_API_KEY is not configured.")

async def call_openrouter(
    messages: list, 
    model: str = "openai/gpt-4o-mini",
    temperature: float = 0.3,
    max_tokens: int = 1000
) -> str:
    """
    Asynchronous helper to call LLMs via OpenRouter with fallback support.
    Provides support for Claude, GPT, or Gemini models via a single client.
    """
    if not settings.OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is not configured.")

    headers = {
        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/srihari2479/TaskManager",
        "X-Title": "College RAG Assistant"
    }

    models_to_try = [
        model,
        "openai/gpt-4o-mini",
        "meta-llama/llama-3.3-70b-instruct",
        "google/gemini-2.0-flash-001"
    ]
    # Deduplicate while preserving order
    models_to_try = list(dict.fromkeys(models_to_try))

    last_error = None
    async with httpx.AsyncClient() as client:
        for current_model in models_to_try:
            payload = {
                "model": current_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            try:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=8.0
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
            except httpx.HTTPStatusError as e:
                print(f"OpenRouter HTTP Error on model {current_model}: {e.response.status_code} - {e.response.text}")
                last_error = e
            except Exception as e:
                print(f"OpenRouter connection error on model {current_model}: {e}")
                last_error = e

    if last_error:
        raise last_error
    raise RuntimeError("Failed to obtain response from OpenRouter.")
