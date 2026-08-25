# -*- coding: utf-8 -*-
import json
import urllib.request
from typing import Optional
from config import config

class OllamaClient:
    """
    Клиент для локальной LLM через Ollama API.
    """
    def __init__(self, base_url: str = config.ollama_url, model: str = config.ollama_model):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def is_available(self) -> bool:
        """Проверяет доступность сервера Ollama."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=2) as res:
                return res.status == 200
        except Exception:
            return False

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Отправляет запрос к локальной модели и возвращает сгенерированный ответ.
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": config.llm_temperature
            }
        }
        if system_prompt:
            payload["system"] = system_prompt

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        
        try:
            with urllib.request.urlopen(req, timeout=90) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                return res_data.get("response", "").strip()
        except Exception as e:
            return f"[Ошибка генерации LLM]: {e}"
