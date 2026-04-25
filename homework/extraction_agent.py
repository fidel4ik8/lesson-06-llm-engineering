import openai
import requests
import json
import os
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Create homework/.env from .env.example "
        "or export the variable in your shell."
    )

# gpt-4o-mini pricing (per 1M tokens, USD)
OPENAI_PRICE_INPUT = 0.150 / 1_000_000
OPENAI_PRICE_OUTPUT = 0.600 / 1_000_000


def call_ollama(prompt: str):
    """Returns (text, input_tokens, output_tokens)."""
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        },
        timeout=300,
    )
    data = response.json()
    return (
        data["response"],
        data.get("prompt_eval_count", 0),
        data.get("eval_count", 0),
    )


def call_openai(prompt: str):
    """Returns (text, input_tokens, output_tokens)."""
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a meeting transcription parser. "
                    "Extract tasks, decisions, and a summary from meeting text. "
                    "Return ONLY valid JSON, no markdown, no extra text."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    return (
        response.choices[0].message.content,
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
    )


def build_prompt(text: str) -> str:
    return f"""Прочитай цей текст зустрічі та витягни:
1. summary (одне речення українською)
2. tasks (список завдань з owner, task, deadline)
3. decisions (рішення, які було прийнято)

Текст:
{text}

Поверни ТІЛЬКИ JSON, нічого більше:
{{
  "summary": "...",
  "tasks": [
    {{"owner": "...", "task": "...", "deadline": "..."}}
  ],
  "decisions": ["..."]
}}"""


def extract_meeting_data(text: str, provider: str = "ollama"):
    """Returns (data_or_none, metrics_dict). metrics always populated."""
    prompt = build_prompt(text)
    start = time.perf_counter()
    if provider == "ollama":
        response_text, in_tok, out_tok = call_ollama(prompt)
    elif provider == "openai":
        response_text, in_tok, out_tok = call_openai(prompt)
    else:
        raise ValueError(f"Unknown provider: {provider}")
    latency = time.perf_counter() - start

    if provider == "openai":
        cost = in_tok * OPENAI_PRICE_INPUT + out_tok * OPENAI_PRICE_OUTPUT
    else:
        cost = 0.0

    metrics = {
        "provider": provider,
        "latency_sec": round(latency, 3),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "total_tokens": in_tok + out_tok,
        "cost_usd": round(cost, 6),
        "json_valid": False,
        "raw_response": response_text,
    }

    try:
        data = json.loads(response_text)
        metrics["json_valid"] = True
        return data, metrics
    except json.JSONDecodeError:
        return None, metrics


if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            meeting_text = f.read()
    else:
        meeting_text = "Анна: тестова зустріч. Іван робить UI до 5 травня."

    print(f"🤖 Testing Ollama ({OLLAMA_MODEL})...")
    res, m = extract_meeting_data(meeting_text, provider="ollama")
    print(f"  latency={m['latency_sec']}s tokens={m['total_tokens']} valid={m['json_valid']}")
    print(json.dumps(res, indent=2, ensure_ascii=False))

    print(f"\n☁️  Testing OpenAI ({OPENAI_MODEL})...")
    res, m = extract_meeting_data(meeting_text, provider="openai")
    print(f"  latency={m['latency_sec']}s tokens={m['total_tokens']} cost=${m['cost_usd']} valid={m['json_valid']}")
    print(json.dumps(res, indent=2, ensure_ascii=False))
