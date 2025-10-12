import logging
import os

from pathlib import Path
from google.generativeai import GenerativeModel
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

from src.handlers.facesHandler import enrollTrustedFaces, loadEnrolledFaces
from src.handlers.audioHandler import listenForAudio, speak
from src.handlers.guardHandler import monitorRoom
from src.utils import setupLogging, loadConfig
import src.utils as utils

LOGGER = logging.getLogger(__name__)

APIKEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("MODEL")

def main():
    """
    Main function to run the guard agent
    """
    global GUARD_MODE_ACTIVE
    setupLogging()
    # Ensure that the API Key is present and loaded
    if not APIKEY:
        LOGGER.error("API Key is not present. Add to the `.env` file")
    genai.configure(api_key=APIKEY)
    llmModel = GenerativeModel(MODEL)
    # Load the configuration 
    config = loadConfig()
    encodingFilePath = Path(config['paths']['encodingFilePath'])
    dirPath = Path(config['paths']['facesDir'])
    if not encodingFilePath.exists():
        # Calculate and store encodings only if the pickle file does not exist
        enrollTrustedFaces(dirPath, encodingFilePath)
    
    # Load the encodings of the trusted faces
    knownFaces = loadEnrolledFaces(encodingFilePath)
    commands = config['commands']

    # Start the agent
    speak("AI Guard Agent. Speak the activation command to begin.")

    while True:
        command = listenForAudio()
        if not command:
            continue

        if any(actCom in command for actCom in commands['activationCommand']) and not utils.GUARD_MODE_ACTIVE:
            utils.GUARD_MODE_ACTIVE = True
            LOGGER.info("Guard Mode Active. Monitoring the Room.")
            speak("Guard Mode Active. Monitoring the Room.")
            # TODO : Modify the logic to ensure continuous monitoring of the room 
            # And listening for audio commands if needed    
            monitorRoom(knownFaces, llmModel)

            if not utils.GUARD_MODE_ACTIVE:
                LOGGER.info("Guard Mode Deactivated.")
                speak("Guard Mode Deactivated")
        
        elif any(deactCom in command for deactCom in commands['deactivationCommand']) and utils.GUARD_MODE_ACTIVE:
            utils.GUARD_MODE_ACTIVE = False
            LOGGER.info("Guard Mode deactivated after receiving command")
        
        elif 'stop' in command:
            LOGGER.info("Shutting down.")
            speak("Shutting Down.")
            break


if __name__ == '__main__':
    main()



