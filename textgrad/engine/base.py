import hashlib
from abc import ABC
from typing import Any

import diskcache as dc


class EngineLM(ABC):
    def __init__(
        self, system_prompt: str | None = None, model_string: str | None = None
    ) -> None:
        self.system_prompt = (
            system_prompt
            if system_prompt is not None
            else "You are a helpful, creative, and smart assistant."
        )
        self.__model_string: str | None = model_string

    @property
    def model_string(self) -> str:
        assert self.__model_string is not None
        return self.__model_string

    # @abstractmethod
    # def generate(self, prompt, system_prompt=None, **kwargs) -> Any:
    #     pass

    def __call__(self, input_text: str, prompt: str, **kwargs) -> Any:
        pass


class CachedEngine:
    def __init__(self, cache_path: str) -> None:
        super().__init__()
        self.cache_path = cache_path
        self.cache = dc.Cache(cache_path)

    def _hash_prompt(self, prompt: str) -> str:
        return hashlib.sha256(f"{prompt}".encode()).hexdigest()

    def _check_cache(self, prompt: str) -> str | None:
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
