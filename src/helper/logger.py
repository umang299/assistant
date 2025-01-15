import os
import sys
import logging

cwd = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(cwd)


def get_logger(name, log_file_path=None, log_level=logging.INFO):
    """
    Creates and returns a logger object that logs messages to the console and optionally to a file.
    
    Args:
        name (str): Name of the logger.
        log_file_path (str or None): Path to the log file. If None, 
                                     logs are only written to the console.
        log_level (int): Logging level (default: logging.INFO).
    
    Returns:
        logging.Logger: Configured logger object.
    """
    # Create a logger object
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Prevent duplicate logs if the logger is already configured
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    # Create a formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Attach the formatter to the console handler
    console_handler.setFormatter(formatter)

    # Add the console handler to the logger
    logger.addHandler(console_handler)

    # If a log file path is provided, add a file handler
    if log_file_path:
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
