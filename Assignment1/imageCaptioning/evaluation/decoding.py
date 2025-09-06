import torch
from typing import List
from PIL import Image
import os
from torchvision import transforms
import json
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

from imageCaptioning.models.captioner import Captioner
from imageCaptioning.data.vocabulary import Vocabulary
from imageCaptioning.data.preprocess import getTransforms
from imageCaptioning.config import (DEVICE, DATA_ROOT, 
                                    getCustomLogger, 
                                    OUTPUT_DIRECTORY, CHECKPOINT_PATH)

LOGGER = getCustomLogger("Decoding Test")

@torch.no_grad()
def greedyDecodeLSTM(model : Captioner,
                     images : torch.Tensor,
                     maxLength : int = 24,
                     BOSIndex : int = 1,
                     EOSIndex : int = 2) -> List[List[int]]:
    """
    Greedy decode for LSTM Decoder on a batch of images
    Arguments:
        model : Image captioning model with encoder and LSTM decoder
        images : Tensor of shape (batchSize, C, H, W)
        maxLength : Maximum Length of the generated captions
        BOSIndex : BOS Token Index
        EOSIndex : EOS Token Index
    """
    features: torch.Tensor = model.encoder(images)
    return model.decoder.generateBatch(features=features,
                                       maxLength=maxLength,
                                       BOSIndex=BOSIndex,
                                       EOSIndex=EOSIndex)


@torch.no_grad()
def greedyDecodeTransformer(model : Captioner,
                            images : torch.Tensor,
                            maxLength : int = 24,
                            BOSIndex : int = 0,
                            EOSIndex : int = 1) -> List[List[int]]:
    """
    Greedy decode for LSTM Decoder on a batch of images
    Arguments:
        model : Image captioning model with encoder and LSTM decoder
        images : Tensor of shape (batchSize, C, H, W)
        maxLength : Maximum Length of the generated captions
        BOSIndex : BOS Token Index
        EOSIndex : EOS Token Index
    Returns:
        List of Tokens Lists
    """
    features : torch.Tensor = model.encoder(images)
    return model.decoder.generateBatch(features=features,
                                       maxLength=maxLength,
                                       EOSIndex=EOSIndex,
                                       BOSIndex=BOSIndex)

def gradCAM(model : Captioner,
            imageTensor : torch.Tensor,
            captionTensor : torch.Tensor,
            index : int,
            EOSIndex : int = 2):
    model.eval()
    imageTensor = imageTensor.unsqueeze(0)
    captionTensor = captionTensor.unsqueeze(0)
    # Forward Pass
    output = model(imageTensor, captionTensor)
    EOSLogit = output[0, -1, EOSIndex]
    # Zero gradients and Backward on the EOS Logit
    model.zero_grad()
    EOSLogit.backward(retain_graph=True)
    CAM = model.encoder.getGradCAM()
    imageNumpy = imageTensor.squeeze().permute(1, 2, 0).cpu().numpy()
    imageNumpy = (imageNumpy - imageNumpy.min()) / (imageNumpy.max() - imageNumpy.min())
    CAMResized = cv2.resize(CAM, (imageNumpy.shape[1], imageNumpy.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * CAMResized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(np.uint8(255 * imageNumpy), 0.5, heatmap, 0.5, 0)

    plt.figure(figsize=(8, 5))
    plt.imshow(overlay)
    plt.axis('off')
    plt.title("Grad-CAM Overlay")
    plt.savefig(os.path.join(OUTPUT_DIRECTORY, f"gradCAM{index}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decoding Script using Checkpointed Model")
    parser.add_argument("--model-type", type=str, default="transformer", choices=["lstm", "transformer"])
    parser.add_argument("--encoder-name", type=str, default="resnet18", choices=["resnet18", "mobilenet"])
    args = parser.parse_args()
    preprocessTransforms: transforms = getTransforms()

    testImageDir = os.path.join(DATA_ROOT, "images", "test")
    imageFiles = [os.path.join(testImageDir, file) 
                  for file in os.listdir(testImageDir)
                  if file.lower().endswith((".jpg", ".png", ".jpeg"))]

    images = []
    for imagePath in imageFiles:
        image = Image.open(imagePath).convert("RGB")
        image = preprocessTransforms(image)
        images.append(image)
    imageTensor = torch.stack(images).to(DEVICE)

    # Load the Checkpoint Model
    checkpoint = torch.load(os.path.join(CHECKPOINT_PATH, f"model_{args.model_type}_{args.encoder_name}.pt"),
                            map_location=DEVICE)
    vocabPath = os.path.join(OUTPUT_DIRECTORY, "vocab.json")
    LOGGER.info(f"Loaded Vocaulary from path {vocabPath}")
    vocabulary = Vocabulary.load(vocabPath) 
    vocabSize = len(vocabulary)
    model = Captioner(vocabSize=vocabSize,
                      modelType=args.model_type,
                      encoderName=args.encoder_name)
    model.load_state_dict(checkpoint["modelState"], strict=False)
    model = model.to(DEVICE)
    model.eval()

    # Decode Captions
    decoded = greedyDecodeLSTM(model,
                               imageTensor)
    LOGGER.info("Decoded Captions (Token Indices)")
    outputPath = os.path.join(OUTPUT_DIRECTORY, f"{args.model_type}_{args.encoder_name}_DecodedTest.json")
    with open(outputPath, "w") as file:
        json.dump({
            "imageFiles" : imageFiles,
            "decodedTokens" : decoded
        }, file, indent=2)
    LOGGER.info(f"Saved decoded token indices to {outputPath}")

    # for i in range(9):
    #     testImage = imageTensor[i]
    #     testCaption = torch.tensor(decoded[i])
    #     gradCAM(model,
    #             testImage,
    #             testCaption, i)
    
    if args.model_type == "lstm":
        allDeltas = []
        for i in range(len(imageTensor)):
            testImage = imageTensor[i]
            features = model.encoder(testImage.unsqueeze(0))
            testCaption = torch.tensor(decoded[i])
            deltas = model.decoder.tokenOcclusion(features, testCaption)
            allDeltas.append(deltas)
        LOGGER.info(f"Token Occlusion Deltas Collected")
        # Pad Deltas to the same length for plotting
        maxLen = max(len(delta) for delta in allDeltas)
        allDeltas = [delta + [np.nan] * (maxLen - len(delta)) for delta in allDeltas]
        allDeltas = np.array(allDeltas)
        meanDeltas = np.nanmean(allDeltas, axis = 0)
        stdDeltas = np.nanstd(allDeltas, axis = 0)
        plt.figure(figsize=(10, 6))
        plt.errorbar(range(1, maxLen + 1), meanDeltas, yerr=stdDeltas, fmt='-o', capsize=4)
        plt.xlabel("Token Position in Caption")
        plt.ylabel("Mean Occlusion Delta (± std)")
        plt.title("Token Occlusion Deltas Across Test Samples")
        plt.tight_layout()
        plotPath = os.path.join(OUTPUT_DIRECTORY, f"{args.model_type}_{args.encoder_name}_occlusionDeltas.png")
        plt.savefig(plotPath)
        plt.close()
        LOGGER.info(f"Saved occlusion delta plot to {plotPath}")
    
    if args.model_type == "transformer":
        testImage = imageTensor[0]
        testCaption = torch.tensor(decoded[0])
        output = model(testImage.unsqueeze(0),
                       testCaption.unsqueeze(0))
        attentionMap = model.decoder.getLastAttentionMap()
        if attentionMap is not None:
            plt.imshow(attentionMap.mean(dim = 0), cmap="viridis")
            plt.title("Average Last-Layer Attention Layer")
            plt.colorbar()
            plt.savefig(os.path.join(OUTPUT_DIRECTORY, f"{args.model_type}_{args.encoder_name}_AttentionMap.png"))