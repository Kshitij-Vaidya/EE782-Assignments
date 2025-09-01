import torch.nn as nn
from imageCaptioning.config import getCustomLogger

LOGGER = getCustomLogger("Loss Function")

def getLossFunction(paddingIndex: int) -> nn.Module:
    """
    Return the Cross Entropy Loss ignoring the Padding Tokens
    """
    LOGGER.info(f"Initialised CrossEntropyLoss with ignore_index = {paddingIndex}")
    return nn.CrossEntropyLoss(ignore_index=paddingIndex)