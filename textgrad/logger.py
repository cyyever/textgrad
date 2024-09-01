import json
import logging
import os
from datetime import datetime


class CustomJsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        super().format(record)
        output = {k: str(v) for k, v in record.__dict__.items()}
        return json.dumps(output, indent=4)


cf = CustomJsonFormatter()
os.makedirs("./logs/", exist_ok=True)
sh = logging.FileHandler(f"./logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jsonl")
sh.setFormatter(cf)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(sh)
