import cv2
import face_recognition
import logging

from typing import Dict, Any, List
from google.generativeai import GenerativeModel

from src.handlers.audioHandler import speak, listenForAudio
import src.utils as utils

LOGGER = logging.getLogger(__name__)

def handleIntruder(llmModel : GenerativeModel) -> None:
    """
    Manages the escalating conversation with an unrecognised face
    """
    # Level 1 
    speak("Intruder detected. Who are you? Please state your purpose")
    # Listen for the response 
    response : str = listenForAudio()

    # Level 2 : Escalation using LLM
    promptLevel2 = ("You are a room security AI. An unknown person is in the room",
                    f"Your first warning was : `Who are you?`. They responded {response}",
                    "Now, be sterner. Tell them that this is a private area and that they must leave immediately")
    llmResponse = llmModel.generate_content(promptLevel2).text
    speak(llmResponse)

    #Level 3 Escalation
    speak("This is your final warning. The owner has been notified. Leave now.")


def monitorRoom(knownFaceData : Dict[str, Any],
                llmModel : GenerativeModel,
                tolerance : float = 0.5) -> None:
    """
    Activates the webcam and performs facial recognition
    """
    videoCapture = cv2.VideoCapture(0)

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

            if True in matches:
                firstMatch = matches.index(True)
                name = knownFaceData["names"][firstMatch]
                LOGGER.info(f"Trusted Person : {name}")
            else:
                # Handle the intruder
                handleIntruder(llmModel)
                return
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    videoCapture.release()
    cv2.destroyAllWindows()