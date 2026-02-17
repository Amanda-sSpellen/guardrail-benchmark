"""
logging_config.py: Structured logging configuration for benchmark experiments.

Provides centralized logging setup that outputs to both console and a rotating
log file for audit trails and debugging.
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from datetime import datetime


def setup_logger(
    name: str = "guardrail_benchmark",
    log_dir: str = "logs",
    log_level: int = logging.INFO
) -> logging.Logger:
    """
    Configure and return a logger with both console and file handlers.
    
    This function sets up structured logging that writes to:
    1. Console (stdout) - for real-time feedback during experiments
    2. Rotating file - for persistent audit trail and debugging
    
    Args:
        name: Logger name (default: "guardrail_benchmark")
        log_dir: Directory to store log files (default: "logs")
        log_level: Logging level (default: logging.INFO)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Prevent duplicate handlers if logger is already configured
    if logger.handlers:
        return logger
    
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    console_formatter = logging.Formatter(
        fmt="[%(levelname)s] %(message)s"
    )
    
    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    
    # File handler with rotation (max 5 files, 10MB each)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"benchmark_{timestamp}.log"
    
    file_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(detailed_formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    logger.info(f"Logger initialized. Log file: {log_file}")
    
    return logger
