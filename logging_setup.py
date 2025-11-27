import logging
from pathlib import Path
from datetime import datetime

from config import LOG_DIR


def setup_logging() -> tuple[logging.Logger, Path]:
    """
    Настраивает логирование в файл и консоль.
    Каждый запуск — отдельный файл logfile_YYYYMMDD_HHMMSS.log.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"logfile_{ts}.log"

    logger = logging.getLogger("nbki_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    logger.propagate = False
    logger.info("Логирование инициализировано. Файл: %s", log_path)

    return logger, log_path
