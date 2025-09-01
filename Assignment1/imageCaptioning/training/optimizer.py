import torch.nn as nn
import torch.optim as optim 
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, StepLR

from imageCaptioning.config import getCustomLogger

LOGGER = getCustomLogger("Optimizer")

def buildOptimizer(model: nn.Module,
                   lrCNN: float = 1e-4,
                   lrDecoder: float = 2e-4,
                   lrTransformer: float = 2e-5) -> Optimizer:
    """
    Build the Adam Optimizer with separate CNN Encoder, LSYM/Transformer Decoder
    """
    LOGGER.info(f"Building Adam Optimizer (CNN LR = {lrCNN}, Decoder LR = {lrDecoder}, Transformer LR = {lrTransformer})")
    parameters = []

    if hasattr(model, "encoder"):
        parameters.append({"params" : model.encoder.parameters(),
                           "lr" : lrCNN})
    if hasattr(model, "decoder"):
        parameters.append({"params" : model.decoder.parameters(),
                           "lr" : lrDecoder})
    if hasattr(model, "transformer"):
        parameters.append({"params" : model.transformer.parameters(),
                           "lr" : lrTransformer})
    
    optimizer = optim.Adam(params=parameters, betas=(0.9, 0.999))
    return optimizer

def buildScheduler(optimizer: Optimizer,
                   stepSize: int = 5,
                   gamma: float = 0.5) -> _LRScheduler:
    """
    StepLR Scheduler: reduce the LR every 'stepSize' epochs by gamma
    """
    LOGGER.info(f"Using StepLR Scheduler: Step Size = {stepSize}, Gamma = {gamma}")
    return StepLR(optimizer=optimizer,
                  step_size=stepSize,
                  gamma=gamma)