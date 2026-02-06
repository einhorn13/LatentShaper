# core/logger.py

import logging
import sys

class ColoredFormatter(logging.Formatter):
    """
    Custom formatter to add colors to console output.
    """
    
    # ANSI Colors
    GREY = "\x1b[38;20m"
    GREEN = "\x1b[32;20m"
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"
    
    # Format: [TIME] [LEVEL] Message
    FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s"

    FORMATS = {
        logging.DEBUG: GREY + FORMAT + RESET,
        logging.INFO: GREEN + FORMAT + RESET,
        logging.WARNING: YELLOW + FORMAT + RESET,
        logging.ERROR: RED + FORMAT + RESET,
        logging.CRITICAL: BOLD_RED + FORMAT + RESET
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%H:%M:%S")
        return formatter.format(record)

class Logger:
    """
    Static wrapper around standard logging.
    Ensures thread-safety and consistent formatting.
    """
    _instance = None

    @staticmethod
    def setup():
        if Logger._instance: return
        
        # Renamed logger to LatentShaper
        logger = logging.getLogger("LatentShaper")
        logger.setLevel(logging.INFO)
        
        # Console Handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(ColoredFormatter())
        
        logger.addHandler(ch)
        Logger._instance = logger

    @staticmethod
    def info(msg: str):
        if not Logger._instance: Logger.setup()
        Logger._instance.info(msg)

    @staticmethod
    def warning(msg: str):
        if not Logger._instance: Logger.setup()
        Logger._instance.warning(msg)

    @staticmethod
    def error(msg: str):
        if not Logger._instance: Logger.setup()
        Logger._instance.error(msg)

# Initialize on import
Logger.setup()