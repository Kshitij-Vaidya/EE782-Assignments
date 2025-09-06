import torch
from typing import List
from PIL import Image
import os
from torchvision import transforms
import json
import argparse

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
    model.load_state_dict(checkpoint["modelState"])
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