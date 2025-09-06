import os
import logging
import re
import psutil
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
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, "rsicdDataset")
OUTPUT_DIRECTORY = os.path.join(PROJECT_ROOT, "outputs")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints")

# Define the common tokenize to be used across the project
def tokenize(text : str) -> List[str]:
    return re.findall(r'\w+', text.lower())

# Device and Utility Details for the Torch Modules and Training
DEVICE = "cpu"

def getMemoryMB():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)