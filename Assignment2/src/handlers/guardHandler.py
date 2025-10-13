import cv2
import face_recognition
import logging
import os
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GRPC_TRACE", "")
import numpy as np

from typing import Dict, Any, List
from google.generativeai import GenerativeModel

from src.handlers.audioHandler import speak, listenForAudio
import src.utils as utils

LOGGER = logging.getLogger(__name__)

def handleIntruder(llmModel : GenerativeModel) -> None:
    """
    Manages the escalating conversation with an unrecognised face
    """
    def _willLeave(response : str) -> bool:
        if not response:
            return False
        
        response = response.lower()
        leaveKeywords = ["leave", "i'll leave", "i will leave", "i'm leaving", "i am leaving",
                          "going to leave", "going to go", "i'll go", "i will go", "i'll be going",
                          "ok", "okay", "sure", "alright", "yes", "yeah", "fine", "sorry"]
        refuseKeywords = ["no", "not", "won't", "will not", "refuse", "stay", "stay here", "i won't"]

        if any(k in response for k in leaveKeywords) and not any(k in response for k in refuseKeywords):
            return True
        return False
    # Level 1 
    speak("Intruder detected. Who are you? Please state your purpose")
    # Listen for the response 
    response : str = listenForAudio()

    if _willLeave(response):
        speak("Thank you. Please leave now")
        LOGGER.info("Intruder agreed to leave at Level 1")
        return

    # Level 2 : Escalation using LLM
    promptLevel2 = ("You are a room security AI. An unknown person is in the room",
                    f"Your first warning was : `Who are you?`. They responded {response}",
                    "Now, be sterner. Tell them that this is a private area and that they must leave immediately")
    llmResponse = llmModel.generate_content(promptLevel2).text
    speak(llmResponse)

    response2 : str = listenForAudio()
    if _willLeave(response2):
        speak("Thank you. Please leave now.")
        LOGGER.info("Intruder agreed to leave at level 2.")
        return

    #Level 3 Escalation
    speak("This is your final warning. The owner has been notified. Leave now.")


def monitorRoom(knownFaceData : Dict[str, Any],
                llmModel : GenerativeModel,
                tolerance : float = 0.7) -> None:
    """
    Activates the webcam and performs facial recognition
    """
    videoCapture = cv2.VideoCapture(0)
    MAX_DISTANCE = 0.45

    while utils.GUARD_MODE_ACTIVE:
        ret, frame = videoCapture.read()
        if not ret:
            break
        # Find the faces in the current frame
        faceLocations = face_recognition.face_locations(frame)
        faceEncodings = face_recognition.face_encodings(frame, faceLocations)

        for faceEncoding in faceEncodings:
            matches : List[bool] = face_recognition.compare_faces(knownFaceData["encodings"],
                                                                  faceEncoding,
                                                                  tolerance=tolerance)
            name = "Unknown"

            distances = face_recognition.face_distance(knownFaceData["encodings"], faceEncoding)
            minIndex = int(np.argmin(distances))
            minDistance = float(distances[minIndex])

            confidence = max(0.0, minDistance / MAX_DISTANCE)
            confidence = min(1.0, confidence)

            if minDistance <= tolerance:
                LOGGER.info(f"Known face with distance {minDistance:.4f} and confidence {confidence:.4f}")
                firstMatch = matches.index(True)
                name = knownFaceData["names"][firstMatch]
                LOGGER.info(f"Trusted Person : {name}")
            else:
                LOGGER.info(f"Intruder detected with confidence {confidence:.4f}")
                # Handle the intruder
                handleIntruder(llmModel)
                return
                
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    videoCapture.release()
    cv2.destroyAllWindows()