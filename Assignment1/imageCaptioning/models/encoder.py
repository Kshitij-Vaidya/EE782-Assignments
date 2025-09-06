import os
import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Dict
from imageCaptioning.config import getCustomLogger, DEVICE, OUTPUT_DIRECTORY
from imageCaptioning.data.dataset import RSICDDataset

LOGGER = getCustomLogger("Encoder")

class CNNEncoder(nn.Module):
    '''
    CNN Encoder supporting ResNet-18 and MobileNet
    Resolves  classifier head, applies global average pooling
    '''
    def __init__(self, modelName: str = 'resnet18',
                 pretrained: bool = True,
                 finetune: bool = True,
                 numLayers: int = 2,
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
        parameters = list(self.CNN.children())
        for layer in parameters[-numLayers:]:
            for param in layer.parameters():
                param.requires_grad = finetune
        
        self.activations = None
        self.gradients = None
        self.lastConv = None
        for m in self.CNN.modules():
            if isinstance(m, nn.Conv2d):
                self.lastConv = m
        
        if self.lastConv is not None:
            self.lastConv.register_forward_hook(self.saveActivation)
            self.lastConv.register_full_backward_hook(self.saveGradient)

        LOGGER.info(f"Initialized CNN Encoder with {modelName},"
                    f"Finetune = {finetune}, Output Dimensions = {outputDim}")
    

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        features = self.CNN(x)
        pooledOutput = self.pool(features).squeeze(-1).squeeze(-1)
        return self.projection(pooledOutput)
    
    def cacheFeatures(self, imagePaths : List[str],
                      dataset: RSICDDataset,
                      batchSize: int = 32,
                      savePath: str = "featureCache.pt") -> None:
        self.eval()
        featureDict = {}
        with torch.no_grad():
            for i in range(0, len(imagePaths), batchSize):
                batchPaths = imagePaths[i : i + batchSize]
                batchImages = [dataset.loadImage(path).to(DEVICE)
                               for path in batchPaths]
                batchTensors = torch.stack(batchImages)
                batchFeatures = self.forward(batchTensors).cpu()

                for imagePath, features in zip(batchPaths, batchFeatures):
                    featureDict[imagePath] = features
        savePath = os.path.join(OUTPUT_DIRECTORY, savePath)
        torch.save(featureDict, savePath)
        LOGGER.info(f"Cached features for {len(featureDict)} images to {savePath}")
    
    @staticmethod
    def loadCachedFeatures(loadPath) -> Dict:
        """
        Load the cached features from a .pt file
        """
        return torch.load(loadPath)
    
    def saveActivation(self, module : nn.Module,
                       input : torch.Tensor,
                       output : torch.Tensor):
        self.activations = output.detach()
    
    def saveGradient(self, module : nn.Module,
                     inputGrad : torch.Tensor,
                     outputGrad : torch.Tensor):
        self.gradients = outputGrad[0].detach()
    
    def getGradCAM(self):
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        CAM = (weights * self.activations).sum(dim=1, keepdim=True)
        CAM = torch.relu(CAM)
        CAM = CAM.squeeze().cpu().numpy()
        CAM = (CAM - CAM.min()) / (CAM.max() - CAM.min() + 1e-8)
        return CAM
