import logging
from logging.handlers import TimedRotatingFileHandler
import os

def get_logger(name: str, log_dir: str = "logs", level=logging.INFO):
    """
    Creates a logger with both console and file output.
    
    Args:
        name (str): Name of the logger (e.g., __name__).
        log_dir (str): Directory where log files are stored.
        level: Logging level (default INFO).
    
    Returns:
        logging.Logger: Configured logger instance.
    """

    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Log file path
    log_file = os.path.join(log_dir, f"{name}.log")

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if logger already exists
    if logger.hasHandlers():
        logger.handlers.clear()

    # Format: timestamp - level - logger name - message
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (rotates daily, keeps 7 days)
    file_handler = TimedRotatingFileHandler(log_file, when="midnight", backupCount=7, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
