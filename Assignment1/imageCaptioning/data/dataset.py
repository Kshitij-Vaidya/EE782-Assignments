import os
import json
from typing import Optional, Callable
from data.vocabulary import Vocabulary
from PIL import Image
from torch.utils.data import Dataset
from torch import Tensor
from torchvision import transforms

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
            tokens = self.vocabulary.encode(caption, max_len = self.maxLength)
        else:
            tokens = caption
        return image, tokens