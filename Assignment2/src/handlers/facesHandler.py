import face_recognition
import pickle
import os
import logging
from pathlib import Path
from typing import Dict, Any

LOGGER = logging.getLogger(__name__)

def enrollTrustedFaces(dirPath : Path,
                       encodingFilePath : Path) -> None:
    """
    Scans the trusted faces directory and saves their embeddings
    """
    LOGGER.info("Starting Face Enrollment .. ")
    knownEncodings = []
    knownNames = []

    for filename in os.listdir(dirPath):
        if filename.endswith((".jpg", ".png")):
            name = os.path.splitext(filename)[0]
            imagePath = os.path.join(dirPath, filename)

            try:
                image = face_recognition.load_image_file(imagePath)
                encoding = face_recognition.face_encodings(image)[0]
                knownEncodings.append(encoding)
                knownNames.append(name)
                LOGGER.info(f"Enrolled {name}")
            except IndexError:
                LOGGER.warning(f"No face found in {filename}. Skipping")
    
    with open(encodingFilePath, 'wb') as file:
        pickle.dump({
            "encodings" : knownEncodings,
            "names" : knownNames
        }, file)
    LOGGER.info(f"Enrollment Complete. Saving data to {encodingFilePath}")


def loadEnrolledFaces(encodingPath : Path) -> Dict[str, Any]:
    """
    Loads the trusted face encodings from the file
    """
    with open(encodingPath, "rb") as file:
        encodings = pickle.load(file)
    return encodings