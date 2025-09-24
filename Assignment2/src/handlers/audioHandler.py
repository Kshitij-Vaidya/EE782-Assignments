import logging
import os
import speech_recognition as sr
from gtts import gTTS
from playsound import playsound
from pathlib import Path

LOGGER = logging.getLogger(__name__)

def speak(text : str,
          tempFile : Path):
    """
    Converts the text to speech and plays it.
    """
    LOGGER.info(f"AI Guard : {text}")
    try:
        textToSpeech = gTTS(text=text,
                            lang='en')
        textToSpeech.save(tempFile)
        playsound(tempFile)
        os.remove(tempFile)
    except Exception as e:
        LOGGER.error(f"TTS Error \n {e}")


def listenForAudio() -> str:
    """
    Listens for audio from the microphone and transcribes it.
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        LOGGER.info("Listening ...")
        recognizer.pause_threshold = 1
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio,
                                           language='en-in').lower()
        LOGGER.info(f"You said : {text}")
        return text
    except sr.UnknownValueError:
        LOGGER.info("Could not understand audio")
        return ""
    except sr.RequestError as e:
        LOGGER.info(f"Could not request results :\n {e}")
        return ""