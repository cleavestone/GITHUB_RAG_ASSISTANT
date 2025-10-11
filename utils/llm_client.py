# src/utils/llm_client.py

import os
import requests
from dotenv import load_dotenv

load_dotenv()

class LLMClient:
    """
    Centralized LLM client for making requests to Groq (or other LLM providers).
    Allows easy switching of model, temperature, etc.
    """

    def __init__(self, api_key=None, model="llama-3.3-70b-versatile", temperature=0.7, max_tokens=300):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Missing GROQ API key")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """
        Send a chat request to the LLM and return the text response.
        """
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        response = requests.post(
            self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=data,
            timeout=30,
        )
        response.raise_for_status()

        return response.json()["choices"][0]["message"]["content"].strip()
