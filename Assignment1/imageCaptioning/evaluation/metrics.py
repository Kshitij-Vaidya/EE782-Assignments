import json
import os
from typing import List, Dict, Union
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import argparse

from imageCaptioning.config import getCustomLogger, DATA_ROOT, OUTPUT_DIRECTORY

LOGGER = getCustomLogger("Metrics")

def loadJson(path : str) -> Union[Dict, List]:
    with open(path, "r") as file:
        return json.load(file)
    
def computeBLEU4(predictions : list[str],
                 references: List[List[str]]) -> float:
    smoothing = SmoothingFunction().method4
    scores = [
        sentence_bleu(reference, prediction.split(), 
                      weights=(0.25, 0.25, 0.25, 0.25),
                      smoothing_function=smoothing)
        for prediction, reference in zip(predictions, references)
    ]
    return sum(scores) / len(scores)

def computeMeteor(predictions : List[str],
                  references : List[List[List[str]]]) -> float:
    scores = [
        meteor_score(reference, 
                     prediction.split())
        for prediction, reference in zip(predictions, references)
    ]
    return sum(scores) / len(scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, default="transformer", choices=["lstm", "transformer"])
    args = parser.parse_args()

    decodedCaptionPath = os.path.join(OUTPUT_DIRECTORY, f"{args.model_type}DecodedTestCaptions.json")
    testAnnotationPath = os.path.join(DATA_ROOT, "testAnnotations.json")

    decodedCaptions = loadJson(path=decodedCaptionPath)
    testAnnotations = loadJson(path=testAnnotationPath)

    # Build Mapping from Filename to List of Reference Captions
    referenceMap = {
        item["filename"] : item["captions"]
        for item in testAnnotations
    }
    # Prepare the references in the same order as the generated captions
    references = []
    missing = 0
    for file in decodedCaptions["imageFiles"]:
        filename = os.path.basename(file)
        if filename in referenceMap:
            references.append([reference.split() for reference in referenceMap[filename]])
        else:
            references.append([[""]])
            missing += 1
    
    if missing > 0:
        LOGGER.warning(f"{missing} files from generated captions not found in references")
    else:
        LOGGER.info(f"No Missing Files. Can evaluate BLEU4 and METEOR metrics")
    
    predictions = [caption for caption in decodedCaptions["captions"]]

    # For the BLEU-4 and METEOR, use all references for each image
    BLEU4 = computeBLEU4(predictions=predictions,
                         references=[reference for reference in references])
    METEOR = computeMeteor(predictions=predictions,
                           references=[reference for reference in references])
    
    LOGGER.info(f"BLEU-4 : {BLEU4:.4f}")
    LOGGER.info(f"METEOR : {METEOR:.4f}")
    