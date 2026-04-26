"""
Centralized logging configuration for the Music Recommender project.

Entry points (main.py, app.py) call setup_logging() once at startup so all
log output uses the same format and level. Library modules (recommender.py)
should just do:

    from loggings import get_logger
    logger = get_logger(__name__)

…and call logger.info / logger.warning / logger.error as needed. They will
silently no-op until an entry point configures logging.
"""
import logging

DEFAULT_LEVEL = logging.INFO
DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

_configured = False


def setup_logging(level: int = DEFAULT_LEVEL, fmt: str = DEFAULT_FORMAT) -> None:
    """Configure root logging once per process. Safe to call repeatedly."""
    global _configured
    if _configured:
        return
    logging.basicConfig(level=level, format=fmt)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Thin wrapper around logging.getLogger so callers don't import stdlib directly."""
    return logging.getLogger(name)
