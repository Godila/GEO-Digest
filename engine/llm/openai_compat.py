"""OpenAI-Compatible LLM Provider (for Reviewer Agent)."""
from __future__ import annotations
import json, time, urllib.error, urllib.request
from engine.llm.base import LLMProvider

class OpenAICompatProvider(LLMProvider):
    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    
    def __init__(self, api_key="", base_url="", model="gpt-4o", timeout=180, retries=3, **kwargs):
        super().__init__(model, **kwargs)
        self.api_key = api_key
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self.timeout = timeout
        self.retries = retries
    
    def _headers(self):
        return {"Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"}
    
    def complete(self, prompt, system_prompt="", temperature=0.3, max_tokens=4096):
        url = f"{self.base_url}/chat/completions"
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": prompt})
        payload = {"model": self.model, "messages": msgs,
                   "temperature": temperature, "max_tokens": max_tokens}
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, headers=self._headers(), method="POST")
        last_err = None
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    result = json.loads(resp.read().decode())
                return result["choices"][0]["message"]["content"]
            except Exception as e:
                last_err = e
                if attempt < self.retries:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"OpenAI API error: {last_err}")
    
    def health_check(self):
        try:
            self.complete("ping", max_tokens=5); return True
        except Exception:
            return False
