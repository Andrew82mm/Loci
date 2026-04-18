import requests
import json
from llm_memory.config import OPENROUTER_BASE_URL, get_openrouter_key
from llm_memory.colors import log_llm

_FREE_FALLBACK = "meta-llama/llama-3-8b-instruct:free"


class LLMClient:
    def __init__(self) -> None:
        pass

    def generate(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        fallback_model: str = _FREE_FALLBACK,
    ) -> str:
        result = self._call(model, system_prompt, user_prompt, temperature)
        if result.startswith("Error:") and model != fallback_model:
            log_llm(f"Модель {model} недоступна, пробую fallback: {fallback_model}")
            result = self._call(fallback_model, system_prompt, user_prompt, temperature)
        return result

    def _call(self, model: str, system_prompt: str, user_prompt: str, temperature: float) -> str:
        headers = {
            "Authorization": f"Bearer {get_openrouter_key()}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            "temperature": temperature,
        }
        try:
            resp = requests.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers=headers,
                data=json.dumps(payload),
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                raise ValueError(data["error"].get("message", str(data["error"])))
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            log_llm(f"Model: {model} | {e}")
            return f"Error: {e}"


llm_client = LLMClient()
