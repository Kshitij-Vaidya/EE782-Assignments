import logging
import datetime
import yaml
from typing import Dict, Any
from pathlib import Path

LOGGER = logging.getLogger(__name__)

# Define the global guard mode variable state that will be used across the module
GUARD_MODE_ACTIVE : bool = False

def setupLogging(logDir : str = "logs",
                 logLevel : int = logging.INFO) -> None:
    """
    Configures the logger to output a file and the console
    
    This function sets up the root logger and creates a new, timestamped log file
    It is designed to be called once at the beginning of the main script
    
    Args:
        logDir (str): The directory where the log file will be saved
        logLevel (int): The minimum logging level to capture
    """
    logPath = Path(logDir)
    logPath.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logFilename = logPath / f"{timestamp}.log"
    logFormat = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    logger = logging.getLogger()
    logger.setLevel(logLevel)

    # Clear previous handlers if any
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Console Handler
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormat)
    logger.addHandler(consoleHandler)

    # File Handler 
    fileHandler = logging.FileHandler(logFilename)
    fileHandler.setFormatter(logFormat)
    logger.addHandler(fileHandler)

    LOGGER.info(f"Logging configured successfully. Outputting to console and {logFilename}")


def loadConfig(configPath : str = "config.yaml") -> Dict[str, Any]:
    """
    Loads the YAML configuration file for the project
    
    Args:
        configPath (str): The path to the config.yaml file
    
    Returns:
        Dict[str, Any]: A dictionary containing the project configuration
    """
    try:
        with open(configPath, "r") as file:
            config = yaml.safe_load(file)

        LOGGER.info(f"Successfully loaded the configuration from {configPath}")
        return config
    except FileNotFoundError:
        LOGGER.error(f"Configuration file not found at {configPath}")
        raise