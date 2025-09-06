import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from imageCaptioning.training.lossFunction import getLossFunction
from imageCaptioning.training.optimizer import buildOptimizer, buildScheduler
from imageCaptioning.training.utils import saveCheckpoint
from imageCaptioning.config import getCustomLogger, DEVICE, CHECKPOINT_PATH

LOGGER = getCustomLogger("Train")

def trainEpoch(model: nn.Module,
               dataloader: DataLoader,
               criterion: nn.Module,
               optimizer: torch.optim.Optimizer,
               epoch: int,
               gradClip: float = 5.0) -> float:
    """
    Train model for a single epoch
    """
    model.train()
    totalLoss = 0.0

    for _,(images, captions) in enumerate(tqdm(dataloader,
                                               desc=f"Epoch {epoch} [Train]")):
        images, captions = images.to(DEVICE), captions.to(DEVICE)

        outputs : torch.Tensor = model(images, captions[:, :-1])
        loss: torch.Tensor = criterion(outputs.reshape(-1, outputs.size(-1)),
                                       captions[:, 1:].reshape(-1))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradClip)
        optimizer.step()

        totalLoss += loss.item()
    
    return totalLoss / len(dataloader)


def validate(model: nn.Module,
             dataloader: DataLoader,
             criterion: nn.Module,
             epoch: int) -> float:
    """
    Validate the model on the Validation Set
    """
    model.eval()
    totalLoss = 0.0

    with torch.no_grad():
        for images, captions in tqdm(dataloader,
                                     desc=f"Epoch {epoch} [Valid]"):
            images, captions = images.to(DEVICE), captions.to(DEVICE)
            outputs: torch.Tensor = model(images, captions[:, :-1])
            loss: torch.Tensor = criterion(outputs.reshape(-1, outputs.size(-1)),
                                           captions[:, 1:].reshape(-1))
            totalLoss += loss.item()
    return totalLoss / len(dataloader)


def trainModel(model: nn.Module,
               trainLoader: DataLoader,
               valLoader: DataLoader,
               paddingIndex: int,
               checkpointPath: str,
               epochs: int = 50,
               lrCNN: float = 1e-4,
               lrDecoder: float = 2e-4,
               lrTransformer: float = 2e-5) -> None:
    """
    Main Training Loop
    """
    criterion = getLossFunction(paddingIndex=paddingIndex)
    optimizer = buildOptimizer(model, lrCNN, lrDecoder, lrTransformer)
    scheduler = buildScheduler(optimizer)

    bestValidationLoss = float("inf")

    for epoch in range(1, epochs + 1):
        trainingLoss = trainEpoch(model, trainLoader, criterion, optimizer, epoch)
        validationLoss = validate(model, valLoader, criterion, epoch)

        LOGGER.info(f"Epoch {epoch}: Train Loss = {trainingLoss:.4f}, Validation Loss = {validationLoss:.4f}")

        scheduler.step()

        if validationLoss < bestValidationLoss:
            LOGGER.info(f"Validation Loss improved from {bestValidationLoss:.4f} to {validationLoss:.4f}")
            bestValidationLoss = validationLoss
            saveCheckpoint({
                "Epoch" : epoch,
                "modelState" : model.state_dict(),
                "optimizerState" : optimizer.state_dict(),
                "validationLoss" : validationLoss
            }, os.path.join(CHECKPOINT_PATH, checkpointPath))