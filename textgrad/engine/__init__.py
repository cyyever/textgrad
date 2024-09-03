from .base import EngineLM

__ENGINE_NAME_SHORTCUTS__ = {
    "opus": "claude-3-opus-20240229",
    "haiku": "claude-3-haiku-20240307",
    "sonnet": "claude-3-sonnet-20240229",
    "sonnet-3.5": "claude-3-5-sonnet-20240620",
    "together-llama-3-70b": "together-meta-llama/Llama-3-70b-chat-hf",
    "vllm-llama-3-8b": "vllm-meta-llama/Meta-Llama-3-8B-Instruct",
}

# Any better way to do this?
__MULTIMODAL_ENGINES__ = [
    "gpt-4-turbo",
    "gpt-4o",
    "claude-3-5-sonnet-20240620",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    "gpt-4-turbo-2024-04-09",
]


def _check_if_multimodal(engine_name: str):
    return any(name == engine_name for name in __MULTIMODAL_ENGINES__)


def validate_multimodal_engine(engine):
    if not _check_if_multimodal(engine.model_string):
        raise ValueError(
            f"The engine provided is not multimodal. Please provide a multimodal engine, one of the following: {__MULTIMODAL_ENGINES__}"
        )


def get_engine(engine_name: str, **kwargs) -> EngineLM:
    engine_name = __ENGINE_NAME_SHORTCUTS__.get(engine_name, engine_name)
    if engine_name.startswith("ollama"):
        from .ollama import ChatOllama

        return ChatOllama(model_string=engine_name.split("-")[-1], **kwargs)

    if (
        "seed" in kwargs
        and "gpt-4" not in engine_name
        and "gpt-3.5" not in engine_name
        and "gpt-35" not in engine_name
    ):
        raise ValueError(
            f"Seed is currently supported only for OpenAI engines, not {engine_name}"
        )

    if engine_name.startswith("azure"):
        from .openai import AzureChatOpenAI

        # remove engine_name "azure-" prefix
        engine_name = engine_name[6:]
        return AzureChatOpenAI(model_string=engine_name, **kwargs)
    if ("gpt-4" in engine_name) or ("gpt-3.5" in engine_name):
        from .openai import ChatOpenAI

        return ChatOpenAI(
            model_string=engine_name,
            is_multimodal=_check_if_multimodal(engine_name),
            **kwargs,
        )
    if "claude" in engine_name:
        from .anthropic import ChatAnthropic

        return ChatAnthropic(
            model_string=engine_name,
            is_multimodal=_check_if_multimodal(engine_name),
            **kwargs,
        )
    if "gemini" in engine_name:
        from .gemini import ChatGemini

        return ChatGemini(model_string=engine_name, **kwargs)
    if "together" in engine_name:
        from .together import ChatTogether

        engine_name = engine_name.replace("together-", "")
        return ChatTogether(model_string=engine_name, **kwargs)
    if engine_name in ["command-r-plus", "command-r", "command", "command-light"]:
        from .cohere import ChatCohere

        return ChatCohere(model_string=engine_name, **kwargs)
    if engine_name.startswith("ollama"):
        from .openai import OLLAMA_BASE_URL, ChatOpenAI

        model_string = engine_name.replace("ollama-", "")
        return ChatOpenAI(model_string=model_string, base_url=OLLAMA_BASE_URL, **kwargs)
    if "vllm" in engine_name:
        from .vllm import ChatVLLM

        engine_name = engine_name.replace("vllm-", "")
        return ChatVLLM(model_string=engine_name, **kwargs)
    if "groq" in engine_name:
        from .groq import ChatGroq

        engine_name = engine_name.replace("groq-", "")
        return ChatGroq(model_string=engine_name, **kwargs)
    raise ValueError(f"Engine {engine_name} not supported")
