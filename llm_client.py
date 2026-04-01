# llm_client.py
import requests
import json
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL
from colors import log_llm

# Модели, для которых OpenRouter требует жёсткий :free-суффикс
_FREE_FALLBACK = "meta-llama/llama-3-8b-instruct:free"

class LLMClient:
    def __init__(self):
        self.headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
        }

    def generate(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        fallback_model: str = _FREE_FALLBACK,
    ) -> str:
        """Генерирует ответ. При ошибке пробует fallback_model."""
        result = self._call(model, system_prompt, user_prompt, temperature)
        if result.startswith("Error:") and model != fallback_model:
            log_llm(f"Модель {model} недоступна, пробую fallback: {fallback_model}")
            result = self._call(fallback_model, system_prompt, user_prompt, temperature)
        return result

    def _call(self, model: str, system_prompt: str, user_prompt: str, temperature: float) -> str:
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
                headers=self.headers,
                data=json.dumps(payload),
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            # Иногда OpenRouter возвращает ошибку внутри 200-ответа
            if "error" in data:
                raise ValueError(data["error"].get("message", str(data["error"])))
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            log_llm(f"Model: {model} | {e}")
            return f"Error: {e}"

llm_client = LLMClient()
