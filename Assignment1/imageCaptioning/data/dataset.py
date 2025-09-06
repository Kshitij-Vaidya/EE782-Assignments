import os
import json
from typing import Optional, List
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from imageCaptioning.data.vocabulary import Vocabulary

class RSICDDataset(Dataset):
    def __init__(self, root : str, 
                 split : str,
                 vocab: Optional["Vocabulary"] = None,
                 transform: Optional[transforms.Compose] = None,
                 maxLength : int = 24):
        """
        Arguments:
            1. root (str): dataset root containing the images/{split}/ and annotations.json file
            2. split (str): train | test | valid
            3. vocab (Vocabulary): tokenizer/vocab object only needed for caption encoding
            4. transform : torchvision transforms
            5. maxLength (int): max caption length
        """
        self.split = split
        self.root = root
        self.vocabulary = vocab
        self.maxLength = maxLength
        self.transform = transform

        annotationPath = os.path.join(root, f"{self.split}Annotations.json")
        with open(annotationPath, "r") as file:
            self.annotations = json.load(file)
        
        self.imageDirectory = os.path.join(root, "images", self.split)
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        annotation = self.annotations[index]
        imagePath = os.path.join(self.imageDirectory, annotation["filename"])
        image = Image.open(imagePath).convert("RGB")

        if self.transform:
            image = self.transform(image)
        # Use the first caption
        caption = annotation["captions"][0]
        if self.vocabulary:
            tokenIds, _ = self.vocabulary.encode(caption, maxLength = self.maxLength)
            tokens = torch.tensor(tokenIds, dtype = torch.long)
        else:
            tokens = caption
        return image, tokens
    
    def getImagePaths(self) -> List[str]:
        """
        Returns list of all image file paths in the dataset
        """
        return [os.path.join(self.imageDirectory, annotation["filename"])
                for annotation in self.annotations]
    
    def loadImage(self, imagePath: Path) -> Image:
        """
        Loads and transforms an image from the given Path object
        """
        image = Image.open(str(imagePath)).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image
