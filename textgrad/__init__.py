import json
import logging
import os
from datetime import datetime

from .autograd import aggregate, sum
from .config import SingletonBackwardEngine, set_backward_engine
from .engine import EngineLM, get_engine
from .loss import TextLoss
from .model import BlackboxLLM
from .optimizer import TGD, TextualGradientDescent
from .variable import Variable


class CustomJsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        super(CustomJsonFormatter, self).format(record)
        output = {k: str(v) for k, v in record.__dict__.items()}
        return json.dumps(output, indent=4)


cf = CustomJsonFormatter()
os.makedirs("./logs/", exist_ok=True)
sh = logging.FileHandler(f"./logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jsonl")
sh.setFormatter(cf)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(sh)


singleton_backward_engine = SingletonBackwardEngine()

__all__ = [
    "Variable",
    "TextLoss",
    "BlackboxLLM",
    "EngineLM",
    "get_engine",
    "TextualGradientDescent",
    "TGD",
    "set_backward_engine",
    "SingletonBackwardEngine",
    "sum",
    "aggregate",
]
