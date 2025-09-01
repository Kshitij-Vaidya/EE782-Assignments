from typing import Dict, Any
import torch
import random
import numpy as np 
import os

from imageCaptioning.config import getCustomLogger

LOGGER = getCustomLogger("Utilities")

def getSeed(seed: int = 42) -> None:
    """
    Ensure reproducibility across runs
    """
    LOGGER.info(f"Setting random seed = {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def saveCheckpoint(state : Dict[str, Any],
                   filename : str) -> None:
    """
    Save the model/optimizer to a file
    """
    LOGGER.info(f"Saving checkpoint to {filename}")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)

def loadCheckpoint(filename : str,
                   device: str = "cpu") -> Dict[str, Any]:
    """
    Load the model/optimizer
    """
    LOGGER.info(f"Loading checkpoint from {filename}")
    return torch.load(filename,
                      map_location=device)