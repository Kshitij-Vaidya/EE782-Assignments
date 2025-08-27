import torch
import torch.nn as nn
import torchvision.models as models
from config import getCustomLogger

LOGGER = getCustomLogger("Encoder")

class CNNEncoder(nn.Module):
    '''
    CNN Encoder supporting ResNet-18 and MobileNet
    Resolves  classifier head, applies global average pooling
    '''
    def __init__(self, modelName: str = 'resnet18',
                 pretrained: bool = True,
                 finetune: bool = False,
                 outputDim: int = 512) -> None:
        super().__init__()

        if modelName == 'resnet18':
            baseModel = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            modules = list(baseModel.children())[:-2]
            self.CNN = nn.Sequential(*modules)
            featureDim = baseModel.fc.in_features
        
        elif modelName == "mobilenet":
            baseModel = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
            self.CNN = baseModel.features
            featureDim = baseModel.last_channel
        
        else:
            raise ValueError(f"Unsupported Model Name: {modelName}")
        
        # Global Average Pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # Projection into a Common Dimension
        self.projection = nn.Linear(featureDim, outputDim)
        # Define the finetuning policy
        for param in self.CNN.parameters():
            param.requires_grad = finetune
        LOGGER.info(f"Initialized CNN Encoder with {modelName},"
                    f"Finetune = {finetune}, Output Dimensions = {outputDim}")
    

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        features = self.CNN(x)
        pooledOutput = self.pool(features).squeeze(-1).squeeze(-1)
        return self.projection(pooledOutput)