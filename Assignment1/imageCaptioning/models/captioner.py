import torch
import torch.nn as nn
from imageCaptioning.models.encoder import CNNEncoder
from imageCaptioning.models.lstmDecoder import LSTMDecoder
from imageCaptioning.models.transformerDecoder import TransformerDecoder
from imageCaptioning.config import getCustomLogger

LOGGER = getCustomLogger("Captioner")

class Captioner(nn.Module):
    """
    Wrapper for the Encoder + Decoder Framework
    """
    def __init__(self, vocabSize: int,
                 modelType: str = "lstm",
                 encoderName: str = "resnet18",
                 finetune: bool = True) -> None:
        super().__init__()

        self.encoder = CNNEncoder(modelName=encoderName,
                                  pretrained=True,
                                  finetune=finetune,
                                  outputDim=512)
        
        if modelType == "lstm":
            self.decoder = LSTMDecoder(vocabSize=vocabSize)
        elif modelType == "transformer":
            self.decoder = TransformerDecoder(vocabSize=vocabSize)
        else:
            raise ValueError(f"Unknown Decoder Type : {modelType}")
        
        LOGGER.info(f"Initialized Captioner with Encoder = {encoderName} and Decoder = {modelType}")
    
    def forward(self, image: torch.Tensor,
                captions: torch.Tensor) -> torch.Tensor:
        features = self.encoder(image)
        return self.decoder(features, captions)
        