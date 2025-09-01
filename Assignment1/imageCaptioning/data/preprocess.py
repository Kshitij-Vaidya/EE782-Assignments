import os
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
from torchvision import transforms

from imageCaptioning.data.dataset import RSICDDataset
from imageCaptioning.data.vocabulary import Vocabulary
from imageCaptioning.config import DATA_ROOT, LOGGER, OUTPUT_DIRECTORY



def getTransforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Normalise according to the ImageNet Statistics
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def buildVocabFromTrain(minFrequency: int = 5,
                        maxSize: int = 10000) -> Vocabulary:
    trainDataset = RSICDDataset(root=DATA_ROOT, split="train")
    counter = Counter()
    LOGGER.debug(f"Sample annotation : {trainDataset.annotations[0]['captions']}")
    for annotation in trainDataset.annotations:
        for caption in annotation["captions"]:
            counter.update(caption.lower().split())
    
    for i, (word, freq) in enumerate(counter.most_common(20)):
        LOGGER.debug(f"Word {i}: {word}, freq={freq}")
    LOGGER.debug(f"Total Unique tokens in counter: {len(counter)}")
    
    vocabulary = Vocabulary(counter=counter,
                            minFrequency=minFrequency,
                            maxSize=maxSize)
    vocabularyPath = os.path.join(OUTPUT_DIRECTORY, "vocab.json")
    vocabulary.save(vocabularyPath)
    LOGGER.info(f"Saved vocabulary to {vocabularyPath} (Size = {len(vocabulary)})")
    return vocabulary

def computeStatistics(vocabulary: Vocabulary,
                      split: str,
                      maxLength: int = 24) -> None:
    dataset = RSICDDataset(root=DATA_ROOT,
                           split=split,
                           vocab=vocabulary,
                           transform=getTransforms(),
                           maxLength=maxLength)
    lengths = []
    OOVCount, totalCount = 0, 0

    for annotation in dataset.annotations:
        for caption in annotation["captions"]:
            _, rawLength = vocabulary.encode(caption, maxLength=maxLength)
            lengths.append(rawLength)
            totalCount += len(caption.split())
            OOVCount += sum(1 for word in caption.split() if word not in vocabulary.STOI)
    
    coverage = 100 * (1 - OOVCount / totalCount)
    LOGGER.info(f"[{split}] Vocabulary Coverage : {coverage:.2f}")

    # Histogram Plot
    plt.hist(lengths, bins=20)
    plt.title(f"{split} Caption Length Distribution")
    plt.savefig(os.path.join(OUTPUT_DIRECTORY, f"{split}LengthsHistogram.png"))
    plt.close()

    return {
        "split" : split,
        "coverage" : coverage,
        "averageLength" : np.mean(lengths),
        "standardDeviation" : np.std(lengths),
    }


if __name__ == "__main__":
    LOGGER.info("Building Vocabulary from Training Annotations")
    vocabulary = buildVocabFromTrain()
    LOGGER.info("Computing training and validation statistics")
    trainingStatistics = computeStatistics(vocabulary=vocabulary, split="train")
    validationStatistics = computeStatistics(vocabulary=vocabulary, split="valid")
    data = pd.DataFrame([trainingStatistics, validationStatistics])
    data.to_csv(os.path.join(OUTPUT_DIRECTORY, "tokenStatisticsTrainingValidation.csv"), index=False)
    LOGGER.info("Saved token statistics CSV")
