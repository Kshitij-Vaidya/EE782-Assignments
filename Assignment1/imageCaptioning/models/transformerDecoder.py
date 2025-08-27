import torch
import torch.nn as nn
from config import getCustomLogger

LOGGER = getCustomLogger("Transformer Decoder")

class TransformerDecoder(nn.Module):
    def __init__(self,
                 vocabSize: int,
                 dModel: int = 512,
                 numLayers: int = 3,
                 numHeads: int = 8,
                 ffDim: int = 2048,
                 dropout: float = 0.1,
                 paddingIndex: int = 0) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocabSize, dModel, padding_idx=paddingIndex)
        self.positionEncoder = nn.Parameter(torch.zeros(1, 100, dModel)) # Learnable Positional Encoder
        decoderLayer = nn.TransformerDecoderLayer(dModel, numHeads, ffDim, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoderLayer, vocabSize)
        self.fc = nn.Linear(dModel, vocabSize)
        # Image Projection : Convert feature into memory tokens
        self.imageProjection = nn.Linear(dModel, dModel)

        LOGGER.info(f"Initialised Transformer Decoder with Vocab = {vocabSize}, "
                    f"d_model = {dModel}, Layers = {numLayers}, Heads = {numHeads}")
    
    def forward(self, features: torch.Tensor,
                captions: torch.Tensor) -> torch.Tensor:
        B, L = captions.shape
        embeddings = self.embedding(captions) + self.positionEncoder[:, :L, :]
        # Project the features onto a (B, 1, D) Memory
        memory = self.imageProjection(features).unsqueeze(1)
        # Causal Mask for the Decoder
        mask = torch.triu(torch.ones(L, L), diagonal=1).bool().to(captions.device)
        output = self.decoder(tgt=embeddings,
                              memory=memory,
                              tgt_mask=mask)
        return self.fc(output)