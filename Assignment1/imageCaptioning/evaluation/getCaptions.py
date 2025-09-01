import json
import os
from typing import List, Tuple, Dict
from imageCaptioning.config import (DATA_ROOT, getCustomLogger,
                                    OUTPUT_DIRECTORY)

LOGGER = getCustomLogger("Caption Decoding")

def loadVocabulary(vocabPath : str) -> Tuple[List[str], int]:
    """
    Load the vocabulary from the json file
    """
    with open(vocabPath, "r") as file:
        vocabulary : Dict = json.load(file)
    # Need the Token to Word Mapping (ITOS)
    ITOSDict = vocabulary["ITOS"]
    EOSIndex = vocabulary.get("EOSIndex", 2)
    LOGGER.info(f"Loaded Vocabulary from {vocabPath}")
    return ITOSDict, EOSIndex

def tokenToCaption(tokens : List[int],
                   ITOS : Dict[int, str],
                   EOSIndex : int) -> str:
    words = []
    for index in tokens:
        if index == EOSIndex:
            break
        words.append(ITOS[str(index)])
    return " ".join(words)


if __name__ == "__main__":
    decodedJsonPath = os.path.join(OUTPUT_DIRECTORY, "decodedTest.json")
    vocabJsonPath = os.path.join(OUTPUT_DIRECTORY, "vocab.json")
    outputCaptionPath = os.path.join(OUTPUT_DIRECTORY, "decodedTestCaptions.json")

    # Load the decoded tokens
    with open(decodedJsonPath, "r") as file:
        data = json.load(file)
    imageFiles = data["imageFiles"]
    decodedTokens = data["decodedTokens"]

    # Load Vocabulary
    ITOS, EOSIndex = loadVocabulary(vocabPath=vocabJsonPath)

    # Convert the tokens into captions
    captions = []
    for tokens in decodedTokens:
        captions.append(tokenToCaption(tokens=tokens,
                                       ITOS=ITOS,
                                       EOSIndex=EOSIndex))
    
    # Save the captions
    with open(outputCaptionPath, "w") as file:
        json.dump({
            "imageFiles" : imageFiles,
            "captions" : captions
        }, file, indent=2)
    LOGGER.info(f"Saved the decoded captions to {outputCaptionPath}")


