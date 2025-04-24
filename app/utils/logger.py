# logger.py
import logging
import sys

# --- Configuration ---
LOG_LEVEL = logging.INFO  # Or logging.DEBUG, logging.WARNING, etc.
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
LOGGER_NAME = 'nlp_processor' # Or another relevant name for your application

# --- Setup ---
def _setup_logger() -> logging.Logger:
    """
    Internal function to configure and return the base logger instance.
    This should ideally be called only once.
    """
    logging.basicConfig(
        level=LOG_LEVEL,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        stream=sys.stdout # Log to standard output
        # filename='app.log', # Uncomment to log to a file
        # filemode='a'        # Append mode for file logging
    )
    # Get the specific logger
    logger = logging.getLogger(LOGGER_NAME)
    # logger.setLevel(LOG_LEVEL) # Level set by basicConfig is usually sufficient
    return logger

# --- Get the logger instance ---
# This ensures setup runs only once when the module is imported
_logger = _setup_logger()

# --- Logging Functions ---
def log_info(message: str):
    """Logs a message with level INFO."""
    _logger.info(message)

def log_warning(message: str):
    """Logs a message with level WARNING."""
    _logger.warning(message)

def log_error(message: str, exc_info: bool = False):
    """
    Logs a message with level ERROR.

    Args:
        message: The error message string.
        exc_info: If True, exception information is added to the log message.
                  Defaults to False. Set to True inside except blocks.
    """
    _logger.error(message, exc_info=exc_info)

def log_debug(message: str):
    """Logs a message with level DEBUG."""
    _logger.debug(message)