import torch
import torch.nn as nn
from typing import List

from imageCaptioning.config import getCustomLogger, DEVICE

LOGGER = getCustomLogger("Transformer Decoder")

class TransformerDecoder(nn.Module):
    def __init__(self,
                 vocabSize: int,
                 dModel: int = 256,
                 numLayers: int = 2,
                 numHeads: int = 2,
                 ffDim: int = 1024,
                 dropout: float = 0.2,
                 paddingIndex: int = 0,
                 encoderDim: int = 512) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocabSize, dModel, padding_idx=paddingIndex)
        self.positionEncoder = nn.Parameter(torch.zeros(1, 32, dModel)) # Learnable Positional Encoder
        decoderLayer = nn.TransformerDecoderLayer(dModel, numHeads, ffDim, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoderLayer, numLayers)
        self.fc = nn.Linear(dModel, vocabSize)
        # Image Projection : Convert feature into memory tokens
        self.imageProjection = nn.Linear(encoderDim, dModel)

        LOGGER.info(f"Initialised Transformer Decoder with Vocab = {vocabSize}, "
                    f"d_model = {dModel}, Layers = {numLayers}, Heads = {numHeads}")
    
    def forward(self, features: torch.Tensor,
                captions: torch.Tensor) -> torch.Tensor:
        _, L = captions.shape
        embeddings = self.embedding(captions) + self.positionEncoder[:, :L, :]
        # Project the features onto a (B, 1, D) Memory
        memory = self.imageProjection(features).unsqueeze(1)
        # Causal Mask for the Decoder
        mask = torch.triu(torch.ones(L, L), diagonal=1).bool().to(captions.device)
        output = self.decoder(tgt=embeddings,
                              memory=memory,
                              tgt_mask=mask)
        return self.fc(output)
    
    def generateBatch(self, features : torch.Tensor,
                      maxLength : int = 24,
                      BOSIndex : int = 1,
                      EOSIndex : int = 2) -> List[List[int]]:
        batchSize = features.size(0)
        generated = torch.full((batchSize, 1), 
                               BOSIndex, dtype=torch.long)
        finished = torch.zeros(batchSize, dtype=torch.bool,
                               device=DEVICE)
        predictions : List[List[int]] = [[] for _ in range(batchSize)]

        for _ in range(maxLength):
            # Embeddings of Dimension : (B, seqLen, dModel)
            embeddings = self.embedding(generated) + self.positionEncoder[:, :generated.size(1), :]
            memory = self.imageProjection(features).unsqueeze(1)
            mask = torch.triu(torch.ones(generated.size(1), generated.size(1), device=DEVICE), diagonal=1).bool()
            output = self.decoder(tgt=embeddings,
                                  memory=memory,
                                  tgt_mask=mask)
            logits = self.fc(output[:, -1, :])
            nextTokens = torch.argmax(logits, dim=1)

            for i in range(batchSize):
                if not finished[i]:
                    predictions[i].append(nextTokens[i].item())
                    if nextTokens[i].item() == EOSIndex:
                        finished[i] = True
            
            if finished.all():
                break

            generated = torch.cat([generated, nextTokens.unsqueeze(1)], dim=1)
        
        return predictions