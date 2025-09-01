import os
import logging
import re
from typing import List

# Define the Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
LOGGER = logging.getLogger("Preprocessing")

def getCustomLogger(name : str) -> logging.Logger:
    return logging.getLogger(name)

# Data Location Configuration
DATA_ROOT = "./imageCaptioning/rsicdDataset"
OUTPUT_DIRECTORY = "./imageCaptioning/outputs"
CHECKPOINT_PATH = "./imageCaptioning/checkpoints"
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

# Define the common tokenize to be used across the project
TOKENIZER = re.compile(r"\w+([’']\w+)?|[.,!?;:-]")
def tokenize(text : str) -> List[str]:
    return [m.group(0).lower() for m in TOKENIZER.finditer(text)]

# Device Details for the Torch Modules and Training
DEVICE = "cpu"

