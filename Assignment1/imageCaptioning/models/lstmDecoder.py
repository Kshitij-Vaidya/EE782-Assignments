import torch
import torch.nn as nn
from typing import List
from imageCaptioning.config import getCustomLogger, DEVICE

LOGGER = getCustomLogger("LSTM Decoder")

class LSTMDecoder(nn.Module):
    '''
    LSTM Caption Decoder
    Projects CNN feature to init hidden feature
    '''
    def __init__(self,
                 vocabSize: int,
                 embedDimension: int = 3,
                 hiddenDimension: int = 512,
                 numLayers: int = 3,
                 paddingIndex: int = 0) -> None:
        super().__init__()
        self.numLayers = numLayers
        self.embedding = nn.Embedding(vocabSize, 
                                      embedDimension, 
                                      padding_idx=paddingIndex)
        self.LSTM = nn.LSTM(embedDimension, 
                            hiddenDimension,
                            numLayers,
                            batch_first=True)
        self.fc = nn.Linear(hiddenDimension, vocabSize)

        self.initH = nn.Linear(hiddenDimension, hiddenDimension)
        self.initC = nn.Linear(hiddenDimension, hiddenDimension)

        LOGGER.info(f"Initialised LSTM Decoder with Vocabulary={vocabSize}, "
                    f"Embedding Dimension = {embedDimension}, Hidden Dimensions = {hiddenDimension}, Layers = {numLayers}")
    
    def forward(self, features: torch.Tensor,
                captions: torch.Tensor) -> torch.Tensor:
        """
        Teacher Forcing Mode
        Arguments:
            features: (B, hiddenDimension) : initialised hidden tokens
            captions: (B, L): input sequence of tokens
        """
        embeddings = self.embedding(captions)
        H0 = torch.tanh(self.initH(features)).unsqueeze(0).repeat(self.numLayers, 1, 1) # (numLayers, B, H)
        C0 = torch.tanh(self.initC(features)).unsqueeze(0).repeat(self.numLayers, 1, 1) # (numLayers, B, H)

        outputs, _ = self.LSTM(embeddings, (H0, C0))
        return self.fc(outputs) # Output Size : (B, L, vocabSize)

    def generate(self, features: torch.Tensor,
                 maxLength: int = 24,
                 BOSIndex: int = 1,
                 EOSIndex: int = 2) -> torch.Tensor:
        """
        Greedy Decoding
        """
        H = torch.tanh(self.initH(features)).unsqueeze(0).repeat(self.numLayers, 1, 1)
        C = torch.tanh(self.initH(features)).unsqueeze(0).repeat(self.numLayers, 1, 1)
        inputs = torch.Tensor([BOSIndex],
                              device=features.device).unsqueeze(0)
        embeddings = self.embedding(inputs)
        outputs = []

        for _ in range(maxLength):
            output, (H, C) = self.LSTM(embeddings, (H, C))
            logits = self.fc(output[:, -1, :]) # Last token 
            predicted = torch.argmax(logits, dim=-1)
            outputs.append(predicted.item())
            if predicted.item() == EOSIndex:
                break
            embeddings = self.embedding(predicted.unsqueeze(0))
        
        return outputs
    
    def generateBatch(self, features: torch.Tensor,
                      maxLength: int = 24,
                      BOSIndex: int = 1,
                      EOSIndex: int = 2) -> List[List[int]]:
        batchSize = features.size(0)
        H = torch.tanh(self.initH(features)).unsqueeze(0)
        C = torch.tanh(self.initC(features)).unsqueeze(0)
        inputs = torch.full((batchSize, 1), BOSIndex,
                            dtype=torch.long,
                            device=DEVICE)
        outputs = [[] for _ in range(batchSize)]
        finished = torch.zeros(batchSize, dtype=torch.bool,
                               device=DEVICE)
        
        for _ in range(maxLength):
            embeddings = self.embedding(inputs)
            output, (H, C) = self.LSTM(embeddings, (H, C))
            logits = self.fc(output[:, -1, :])
            predicted = torch.argmax(logits, dim=-1)

            for i in range(batchSize):
                if not finished[i]:
                    outputs[i].append(predicted[i].item())
                    if predicted[i].item() == EOSIndex:
                        finished[i] = True
            
            if finished.all():
                break

            inputs = predicted.unsqueeze(1)
        
        return outputs


