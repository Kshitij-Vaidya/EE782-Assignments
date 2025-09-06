import json
import os
from typing import List, Dict, Union, Tuple
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import argparse
import numpy as np 
import matplotlib.pyplot as plt 

from imageCaptioning.config import getCustomLogger, DATA_ROOT, OUTPUT_DIRECTORY, tokenize

LOGGER = getCustomLogger("Metrics")

def loadJson(path : str) -> Union[Dict, List]:
    with open(path, "r") as file:
        return json.load(file)
    
def computeBLEU4(predictions : list[str],
                 references: List[List[str]]) -> Tuple[List[float], float]:
    smoothing = SmoothingFunction().method4
    scores = [
        sentence_bleu(reference, prediction.split(), 
                      weights=(0.25, 0.25, 0.25, 0.25),
                      smoothing_function=smoothing)
        for prediction, reference in zip(predictions, references)
    ]
    return scores, sum(scores) / len(scores)

def computeMeteor(predictions : List[str],
                  references : List[List[List[str]]]) -> Tuple[List[float], float]:
    scores = [
        meteor_score(reference, 
                     prediction.split())
        for prediction, reference in zip(predictions, references)
    ]
    return scores, sum(scores) / len(scores)

def captionStats(predictions : List[str]) -> Tuple[List[int], List[int]]:
    lengths, degenerate = [], []
    for prediction in predictions:
        tokens = tokenize(prediction)
        length = len(tokens)
        lengths.append(length)
        degen = any(tokens[i] == tokens[i + 1] == tokens[i + 2]
                        for i in range(len(tokens) - 2))
        degenerate.append(degen)
    return lengths, degenerate

def selectExamples(filePaths : List[str],
                   captions : List[str],
                   BLEUScores : List[float],
                   model : str,
                   encoder : str,
                   numCaptions : int = 10) -> None:
    captionMap = list(zip(filePaths, captions, BLEUScores))
    sortedPairs = sorted(captionMap, key = lambda x : x[2])
    failures = sortedPairs[:numCaptions]
    successes = sortedPairs[-numCaptions:]

    data = []
    outputPath = os.path.join(OUTPUT_DIRECTORY, f"{model}_{encoder}_captions.json")
    for fp, cap, bleu in successes:
        data.append({"filepath": fp, "caption": cap, "bleu": bleu, "type": "success"})
    for fp, cap, bleu in failures:
        data.append({"filepath": fp, "caption": cap, "bleu": bleu, "type": "failure"})
    with open(outputPath, "w") as file:
        json.dump(data, file, indent=2)
    LOGGER.info(f"Saved Best and Worst Examples at {outputPath}")

def getErrorSlices(gtCaptions : List[List[str]],
                   filepaths : List[str],
                   model : str,
                   encoder : str):
    slices = {"short" : [],
              "long" : [],
              "runway" : [],
              "harbour" : [],
              "farmland" : []}
    for index, captions in enumerate(gtCaptions):
        joined = " ".join(captions).lower()
        lengths = [len(caption.split()) for caption in captions]
        if any(l <= 8 for l in lengths):
            slices["short"].append(filepaths[index])
        if any(l >= 16 for l in lengths):
            slices["long"].append(filepaths[index])
        if "runway" in joined:
            slices["runway"].append(filepaths[index])
        if "harbor" in joined:
            slices["harbour"].append(filepaths[index])
        if "farmland" in joined:
            slices["farmland"].append(filepaths[index])
    path = os.path.join(OUTPUT_DIRECTORY, f"{model}_{encoder}_slices.json")
    with open(path, "w") as file:
        json.dump(slices, file, indent=2)
    LOGGER.info(f"Saved Slices at {path}")
    return slices

def plotSliceBLEUDeltas(sliceDict : dict,
                        BLEUMap : dict,
                        model : str,
                        encoder : str) -> None:
    sliceNames, sliceBleus = [], []
    for name, files in sliceDict.items():
        scores = [BLEUMap.get(path, 0.0) for path in files]
        if scores:
            sliceNames.append(name)
            sliceBleus.append(np.mean(scores))
    plt.figure(figsize=(8, 5))
    plt.bar(sliceNames, sliceBleus)
    plt.ylabel("Mean Per-Slice BLEU-4")
    plt.title(f"Per-slice BLEU-4 Scores ({model}, {encoder})")
    plt.tight_layout()
    outputPath = os.path.join(OUTPUT_DIRECTORY, f"{model}_{encoder}_slice_bleu.png")
    plt.savefig(outputPath)
    plt.close()
    LOGGER.info(f"Saved per-slice BLEU-4 plot to {outputPath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, default="transformer", choices=["lstm", "transformer"])
    parser.add_argument("--encoder-name", type=str, default="resnet18", choices=["resnet18", "mobilenet"])
    args = parser.parse_args()

    decodedCaptionPath = os.path.join(OUTPUT_DIRECTORY, f"{args.model_type}_{args.encoder_name}_DecodedTestCaptions.json")
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
    allBLEU, BLEU4 = computeBLEU4(predictions=predictions,
                         references=[reference for reference in references])
    allMETEOR, METEOR = computeMeteor(predictions=predictions,
                           references=[reference for reference in references])
    lengths, degenerate = captionStats(predictions)
    degenPct = 100 * sum(degenerate) / len(degenerate)
    selectExamples(decodedCaptions["imageFiles"],
                   predictions,
                   allBLEU,
                   args.model_type,
                   args.encoder_name)
    gtCaptions = []
    for file in decodedCaptions["imageFiles"]:
        filename = os.path.basename(file)
        gtCaptions.append(referenceMap.get(filename, [""]))
    slices = getErrorSlices(gtCaptions,
                   decodedCaptions["imageFiles"],
                   args.model_type,
                   args.encoder_name)
    
    bleuMap = {
        path : bleu for path, bleu in zip(decodedCaptions["imageFiles"], allBLEU)
    }
    plotSliceBLEUDeltas(slices,
                        bleuMap,
                        args.model_type,
                        args.encoder_name)
    
    LOGGER.info(f"BLEU-4 : {BLEU4:.4f}")
    LOGGER.info(f"METEOR : {METEOR:.4f}")
    LOGGER.info(f"Mean Caption Lengths : {np.mean(lengths):.2f} StdDev {np.std(lengths):.2f}")
    LOGGER.info(f"Degenerate Repitition (%) : {degenPct:.2f}")
    