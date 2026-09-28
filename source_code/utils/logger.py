"""
Storyteller - Automated Text-to-Video Synthesis Platform
Logging Utility

Provides centralized, structured logging for all modules in the Storyteller pipeline.
Adheres to PEP-8 standards and replaces raw print calls with informative, level-aware logs.
"""

import logging
import sys
from typing import Optional


def setup_logger(
    name: str = "Storyteller",
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Configures and returns a logger instance with standardized formatting.

    Args:
        name: Logger hierarchy name (defaults to 'Storyteller').
        level: Minimum logging level threshold (e.g., logging.INFO, logging.DEBUG).
        log_file: Optional file path to persist log output.

    Returns:
        logging.Logger: Fully configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if setup_logger is called repeatedly
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Standard console handler (stdout)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Optional file handler
        if log_file:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
