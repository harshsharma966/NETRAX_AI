import logging
import os
import sys
from pathlib import Path


def setup_logger(name: str = "entry_system", log_dir: str = "logs", level: str = "INFO") -> logging.Logger:
    """Configure and return a structured logger with console and file handlers."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / f"{name}.log"

    logger = logging.getLogger(name)
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s:%(funcName)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Console Handler
        c_handler = logging.StreamHandler(sys.stdout)
        c_handler.setFormatter(formatter)
        logger.addHandler(c_handler)

        # File Handler
        f_handler = logging.FileHandler(log_file, encoding="utf-8")
        f_handler.setFormatter(formatter)
        logger.addHandler(f_handler)

    return logger
