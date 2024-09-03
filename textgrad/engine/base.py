import hashlib
from abc import ABC, abstractmethod
from typing import Any

import diskcache as dc


class EngineLM(ABC):
    system_prompt: str = "You are a helpful, creative, and smart assistant."
    model_string: str

    @abstractmethod
    def generate(self, prompt, system_prompt=None, **kwargs):
        pass

    def __call__(self, *args, **kwargs) -> Any:
        pass


class CachedEngine:
    def __init__(self, cache_path: str) -> None:
        super().__init__()
        self.cache_path = cache_path
        self.cache = dc.Cache(cache_path)

    def _hash_prompt(self, prompt: str) -> str:
        return hashlib.sha256(f"{prompt}".encode()).hexdigest()

    def _check_cache(self, prompt: str) -> str:
        if prompt in self.cache:
            return self.cache[prompt]
        return None

    def _save_cache(self, prompt: str, response: str) -> None:
        self.cache[prompt] = response

    def __getstate__(self):
        # Remove the cache from the state before pickling
        state = self.__dict__.copy()
        del state["cache"]
        return state

    def __setstate__(self, state):
        # Restore the cache after unpickling
        self.__dict__.update(state)
        self.cache = dc.Cache(self.cache_path)
