import logging
import os
import sys


ADAPTER_NO_ACTIVE_WARNING = "There are adapters available but none are activated for the forward pass"


class _DropAdapterNoActiveWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            # Some versions include a trailing period, and some logging setups may
            # add extra whitespace; match by substring to be robust.
            return ADAPTER_NO_ACTIVE_WARNING not in record.getMessage()
        except Exception:
            # If formatting fails, don't hide anything.
            return True


def setup_logging(log_path: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("r2reviewer")
    if logger.handlers:
        return logger

    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_path:
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def suppress_adapter_no_active_warning() -> None:
    """
    The adapters library may emit a misleading warning even when adapters are active.
    Filter only that specific message to avoid false alarms.
    """
    flt = _DropAdapterNoActiveWarningFilter()

    # Primary source of the message (adapters/model_mixin.py).
    logger = logging.getLogger("adapters.model_mixin")
    if not getattr(logger, "_r2reviewer_drop_no_active_installed", False):
        logger.addFilter(flt)
        logger._r2reviewer_drop_no_active_installed = True

    # Be defensive: depending on logging configuration, the record may be handled
    # by ancestor/root loggers.
    root = logging.getLogger()
    if not getattr(root, "_r2reviewer_drop_no_active_installed", False):
        root.addFilter(flt)
        root._r2reviewer_drop_no_active_installed = True
