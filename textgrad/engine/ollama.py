import os

import platformdirs

import ollama

from .base import CachedEngine, EngineLM


class ChatOllama(EngineLM, CachedEngine):
    def __init__(
        self,
        model_string="llama3.1",
    ) -> None:
        root = platformdirs.user_cache_dir("textgrad")
        cache_path = os.path.join(root, f"cache_ollama_{model_string}.db")
        EngineLM.__init__(self, model_string=model_string)
        CachedEngine.__init__(self, cache_path=cache_path)

    def __call__(self, input_text: str, prompt: str) -> str:
        response = ollama.chat(
            model=self.model_string,
            messages=[
                ollama.Message(role="assistant", content=prompt),
                ollama.Message(role="user", content=input_text),
            ],
        )
        return response["message"]["content"]
