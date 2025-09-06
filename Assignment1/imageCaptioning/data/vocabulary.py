from typing import Dict, List, Tuple
import json
from collections import Counter

from imageCaptioning.config import LOGGER


class Vocabulary:
    """
    Word Level Vocabulary for image captioning
    Handles token-to-index and index-to-token mappings
    """
    def __init__(self, counter : Counter,
                 minFrequency: int = 1,
                 maxSize: int = 10000) -> None:
        self.frequencies: Counter = counter
        self.minFrequency: int = minFrequency
        self.maxSize: int = maxSize

        # Special Tokens
        self.padToken: str = "<pad>"
        self.bosToken: str = "<bos>"
        self.eosToken: str = "<eos>"

        # Define the string to int and int to string mappings
        self.STOI: Dict[str, int] = {}
        self.ITOS: Dict[int, str] = {}

        # Build the Vocabulary on Initialization
        self._buildVocabulary()
    
    def _buildVocabulary(self) -> None:
        tokens = [self.padToken, self.bosToken, self.eosToken]
        mostCommon = [word for word, count in self.frequencies.items() if count >= self.minFrequency]
        mostCommon = mostCommon[: self.maxSize - len(tokens)]
        tokens.extend(mostCommon)

        self.STOI = {token : index for index, token in enumerate(tokens)}
        self.ITOS = {index : token for token, index in self.STOI.items()}

        LOGGER.info(
            f"Built Vocabulary: Size={len(self)}, "
            f"Minimum Frequency={self.minFrequency}, "
            f"Maximum Size={self.maxSize}, "
            f"Unique Tokens={len(self.frequencies)}"
        )
        
    def __len__(self) -> int:
        return len(self.STOI)

    def encode(self, text : str, maxLength: int = 24) -> Tuple[List[int], int]:
        """
        Convert the caption string into a list of token IDs with BOS/EOS and padding
        """
        tokens = text.lower().split()
        rawTokenLength = len(tokens)
        tokenIds = [self.STOI.get(self.bosToken)]
        tokenIds += [self.STOI.get(token, self.STOI.get(self.padToken)) for token in tokens]
        tokenIds.append(self.STOI.get(self.eosToken))
        # Padding / Truncation
        if (len(tokenIds)) < maxLength:
            tokenIds.extend([self.STOI.get(self.padToken)] * (maxLength - len(tokenIds)))
        else:
            tokenIds = tokenIds[:maxLength]
        return tokenIds, rawTokenLength
    
    def decode(self, tokenIds: List[int]) -> str:
        """
        Convert list of token IDs to the caption of strings stopping at EOS
        """
        captionWords = []
        for index in tokenIds:
            token = self.ITOS.get(index, self.padToken)
            if (token == self.eosToken):
                break
            if token not in {self.padToken, self.bosToken}:
                captionWords.append(token)
        return " ".join(captionWords)

    def save(self, path : str) -> None:
        object = {
            "STOI" : self.STOI,
            "ITOS" : self.ITOS,
            "Pad" : self.STOI.get(self.padToken, 0),
            "BOS" : self.STOI.get(self.bosToken, 1),
            "EOS" : self.STOI.get(self.eosToken, 2),
            "Size" : len(self),
            "MinFrequency" : self.minFrequency,
            "MaxSize" : self.maxSize
        }
        with open(path, "w") as file:
            json.dump(obj=object, fp=file, indent=2)
    
    @classmethod
    def load(cls, path : str) -> "Vocabulary":
        with open(path, "r") as file:
            object: Dict[str, int] = json.load(file)
        vocabulary = cls.__new__(cls)
        vocabulary.minFrequency = object.get("MinFrequency", 1)
        vocabulary.maxSize = object.get("MaxSize", 10000)
        vocabulary.padToken = "<pad>"
        vocabulary.eosToken = "<eos>"
        vocabulary.bosToken = "<bos>"
        vocabulary.STOI = object.get("STOI")
        vocabulary.ITOS = object.get("ITOS")
        vocabulary.frequencies = Counter()
        return vocabulary

    


