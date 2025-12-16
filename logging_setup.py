import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
import threading
from threading import Lock
from typing import Dict, Any, List, Optional

from config import LOG_DIR

CSV_FIELDNAMES = [
    "date",
    "time",
    "log_level",
    "stage",
    "telegram_user_id",
    "telegram_username",
    "request_id",
    "duration_seconds",
    "model",
    "api_key_id",
    "file_name",
    "message",
]

STATS_FIELDNAMES = [
    "date",
    "time",
    "telegram_user_id",
    "telegram_username",
    "request_id",
    "pdf_filename",
    "pages_original",
    "pages_after_crop",
    "processed_successfully",
    "model",
    "api_key_id",
    "llm_calls_count",
    "total_request_chars",
    "total_response_chars",
    "max_latency_seconds",
]

STATS_PATH = LOG_DIR / "stats.csv"


def _clean_stats_value(value: object) -> str:
    s = "" if value is None else str(value)
    s = s.replace("\n", "\\n").replace("\r", "\\r")
    s = s.replace(";", ",")
    return s


@dataclass
class FileStats:
    """
    Аккумулятор сводной телеметрии по одному PDF-файлу.
    """

    start_dt: datetime
    telegram_user_id: str
    telegram_username: str
    request_id: str
    pdf_filename: str
    pages_original: int | None = None
    pages_after_crop: int | None = None
    processed_successfully: str = "error"
    models: set[str] = field(default_factory=set)
    api_keys: set[str] = field(default_factory=set)
    llm_calls_count: int = 0
    total_request_chars: int = 0
    total_response_chars: int = 0
    max_latency_seconds: float = 0.0
    _lock: Lock = field(default_factory=Lock, repr=False, compare=False)

    def set_pages_original(self, value: int) -> None:
        with self._lock:
            self.pages_original = value

    def set_pages_after_crop(self, value: int) -> None:
        with self._lock:
            self.pages_after_crop = value

    def register_llm_call(
        self,
        model: str,
        api_key_id: str,
        request_text: str,
        response_text: str,
        latency_seconds: float,
    ) -> None:
        with self._lock:
            if model:
                self.models.add(model)
            if api_key_id:
                self.api_keys.add(api_key_id)
            self.llm_calls_count += 1
            self.total_request_chars += len(request_text or "")
            self.total_response_chars += len(response_text or "")
            try:
                lat = float(latency_seconds)
            except (TypeError, ValueError):
                lat = 0.0
            if lat > self.max_latency_seconds:
                self.max_latency_seconds = lat

    def mark_success(self) -> None:
        with self._lock:
            self.processed_successfully = "success"

    def mark_error(self) -> None:
        with self._lock:
            self.processed_successfully = "error"

    def to_row(self) -> Dict[str, Any]:
        with self._lock:
            date_str = self.start_dt.strftime("%Y-%m-%d")
            time_str = self.start_dt.strftime("%H:%M:%S")
            model_str = ",".join(sorted(self.models)) if self.models else "N/A"
            api_key_str = ",".join(sorted(self.api_keys)) if self.api_keys else "N/A"
            pages_original = "" if self.pages_original is None else str(self.pages_original)
            pages_after_crop = "" if self.pages_after_crop is None else str(self.pages_after_crop)
            max_latency = (
                f"{self.max_latency_seconds:.3f}" if self.max_latency_seconds else "0.0"
            )
            return {
                "date": date_str,
                "time": time_str,
                "telegram_user_id": str(self.telegram_user_id),
                "telegram_username": self.telegram_username or "N/A",
                "request_id": self.request_id,
                "pdf_filename": self.pdf_filename,
                "pages_original": pages_original,
                "pages_after_crop": pages_after_crop,
                "processed_successfully": self.processed_successfully,
                "model": model_str,
                "api_key_id": api_key_str,
                "llm_calls_count": str(self.llm_calls_count),
                "total_request_chars": str(self.total_request_chars),
                "total_response_chars": str(self.total_response_chars),
                "max_latency_seconds": max_latency,
            }


def ensure_stats_file_exists() -> Path:
    """
    Готовит stats.csv: создаёт файл с заголовком, если его ещё нет,
    и открывает в режиме append (без очистки данных).
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not STATS_PATH.exists() or STATS_PATH.stat().st_size == 0:
        with STATS_PATH.open("w", encoding="utf-8") as f:
            f.write(";".join(STATS_FIELDNAMES) + "\n")
    # Открытие в append-режиме по требованию задания.
    with STATS_PATH.open("a", encoding="utf-8"):
        pass
    return STATS_PATH


def append_stats_row(row: Dict[str, Any]) -> None:
    """
    Добавляет одну строку сводной телеметрии в stats.csv.
    Ошибки записи не должны ломать основной пайплайн.
    """
    try:
        path = ensure_stats_file_exists()
        with path.open("a", encoding="utf-8") as f:
            values: List[str] = [
                _clean_stats_value(row.get(name, "")) for name in STATS_FIELDNAMES
            ]
            f.write(";".join(values) + "\n")
    except Exception:
        logging.getLogger("nbki_pipeline").error(
            "Не удалось записать строку в stats.csv", exc_info=True
        )


class CsvFormatter(logging.Formatter):
    """
    Форматтер, который пишет структурированный CSV с фиксированными колонками.
    Без трассировок исключений в поле message.
    """

    def format(self, record: logging.LogRecord) -> str:
        dt = datetime.fromtimestamp(record.created)
        date_str = dt.strftime("%Y-%m-%d")
        time_str = dt.strftime("%H:%M:%S")

        def get_attr(name: str, default):
            return getattr(record, name, default)

        def clean(value: object) -> str:
            s = "" if value is None else str(value)
            s = s.replace("\n", "\\n").replace("\r", "\\r")
            s = s.replace(";", ",")
            return s

        stage = get_attr("stage", "general")
        telegram_user_id = get_attr("telegram_user_id", "N/A")
        telegram_username = get_attr("telegram_username", "N/A")
        request_id = get_attr("request_id", "N/A")
        duration = get_attr("duration_seconds", 0)
        model = get_attr("model", "N/A")
        api_key_id = get_attr("api_key_id", "N/A")
        file_name = get_attr("file_name", "N/A")

        # Краткое сообщение без traceback.
        msg = record.getMessage()
        record.message = msg

        row = {
            "date": date_str,
            "time": time_str,
            "log_level": record.levelname,
            "stage": stage,
            "telegram_user_id": telegram_user_id,
            "telegram_username": telegram_username,
            "request_id": request_id,
            "duration_seconds": duration,
            "model": model,
            "api_key_id": api_key_id,
            "file_name": file_name,
            "message": msg,
        }

        values = [clean(row.get(name, "")) for name in CSV_FIELDNAMES]
        return ";".join(values)


class QuietConsoleFormatter(logging.Formatter):
    """
    Форматтер для консоли: выводит только краткое сообщение без traceback.
    """

    def format(self, record: logging.LogRecord) -> str:
        exc_info = record.exc_info
        record.exc_info = None
        try:
            return super().format(record)
        finally:
            record.exc_info = exc_info


class ErrorsFormatter(logging.Formatter):
    """
    Форматтер для подробного лога ошибок в errors.txt.
    Содержит дату/время, request_id, telegram_* и traceback (если есть).
    """

    def format(self, record: logging.LogRecord) -> str:
        dt = datetime.fromtimestamp(record.created)
        date_str = dt.strftime("%Y-%m-%d")
        time_str = dt.strftime("%H:%M:%S")

        def get_attr(name: str, default):
            return getattr(record, name, default)

        stage = get_attr("stage", "general")
        telegram_user_id = get_attr("telegram_user_id", "N/A")
        telegram_username = get_attr("telegram_username", "N/A")
        request_id = get_attr("request_id", "N/A")
        file_name = get_attr("file_name", "N/A")

        message = record.getMessage()
        record.message = message

        exc_text = ""
        if record.exc_info:
            exc_text = self.formatException(record.exc_info)

        lines = [
            "===== ERROR START =====",
            f"date: {date_str}",
            f"time: {time_str}",
            f"log_level: {record.levelname}",
            f"stage: {stage}",
            f"telegram_user_id: {telegram_user_id}",
            f"telegram_username: {telegram_username}",
            f"request_id: {request_id}",
            f"file_name: {file_name}",
            f"message: {message}",
        ]
        if exc_text:
            lines.append("traceback:")
            lines.append(exc_text)
        lines.append("===== ERROR END =====")
        return "\n".join(lines)


def setup_logging() -> tuple[logging.Logger, Path]:
    """
    Настраивает логирование в CSV, errors.txt и консоль.
    Каждый запуск — отдельный файл logfile_YYYYMMDD_HHMMSS.csv.
    errors.txt — общий, пополняемый.
    stats.csv — общий, пополняемый (append).
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"logfile_{ts}.csv"

    # Создаём файл с заголовком колонок.
    with log_path.open("w", encoding="utf-8") as f:
        f.write(";".join(CSV_FIELDNAMES) + "\n")

    # Готовим stats.csv (append, без очистки старых данных).
    ensure_stats_file_exists()

    logger = logging.getLogger("nbki_pipeline")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # CSV-файл
    csv_formatter = CsvFormatter("%(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(csv_formatter)
    logger.addHandler(file_handler)

    # Подробный лог ошибок
    errors_path = LOG_DIR / "errors.txt"
    errors_handler = logging.FileHandler(errors_path, encoding="utf-8", mode="a")
    errors_handler.setLevel(logging.ERROR)
    errors_handler.setFormatter(ErrorsFormatter("%(message)s"))
    logger.addHandler(errors_handler)

    # Консоль без traceback
    stream_formatter = QuietConsoleFormatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)

    logger.propagate = False

    logger.info(
        "Логирование инициализировано. CSV-файл: %s",
        log_path,
        extra={
            "stage": "logging_init",
            "telegram_user_id": "N/A",
            "telegram_username": "N/A",
            "request_id": "N/A",
            "duration_seconds": 0,
            "model": "N/A",
            "api_key_id": "N/A",
            "file_name": "N/A",
        },
    )

    return logger, log_path

LLM_IO_LOCK = threading.Lock()


def log_llm_call(
    llm_io_path: Path,
    *,
    request_id: str,
    telegram_user_id: str,
    telegram_username: str,
    pdf_filename: str,
    model: str,
    api_key_id: str,
    request_text: str,
    response_text: str,
    latency_seconds: Optional[float],
    status: str,
    error_type: Optional[str] = None,
    error_code: Optional[str] = None,
) -> None:
    """
    Записывает один вызов ЛЛМ в llm_io_...txt в человекочитаемом виде.
    """
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    request_text = request_text or ""
    response_text = response_text or ""

    request_chars = len(request_text)
    response_chars = len(response_text)
    latency_str = "N/A" if latency_seconds is None else f"{latency_seconds:.3f}"

    llm_io_path.parent.mkdir(parents=True, exist_ok=True)

    with LLM_IO_LOCK:
        with llm_io_path.open("a", encoding="utf-8") as f:
            f.write("===== LLM CALL START =====\n")
            f.write("[META]\n")
            f.write(f"date: {date_str}\n")
            f.write(f"time: {time_str}\n")
            f.write(f"request_id: {request_id}\n")
            f.write(f"telegram_user_id: {telegram_user_id}\n")
            f.write(f"telegram_username: {telegram_username}\n")
            f.write(f"pdf_filename: {pdf_filename}\n")
            f.write(f"model: {model}\n")
            f.write(f"api_key_id: {api_key_id}\n")
            f.write(f"request_chars: {request_chars}\n")
            f.write(f"response_chars: {response_chars}\n")
            f.write(f"latency_seconds: {latency_str}\n")
            f.write(f"status: {status}\n")
            if error_type:
                f.write(f"error_type: {error_type}\n")
            if error_code:
                f.write(f"error_code: {error_code}\n")

            f.write("\n[REQUEST]\n")
            if request_text:
                f.write(request_text)
                if not request_text.endswith("\n"):
                    f.write("\n")
            else:
                f.write("N/A\n")

            f.write("\n[RESPONSE]\n")
            if response_text:
                f.write(response_text)
                if not response_text.endswith("\n"):
                    f.write("\n")
            else:
                f.write("N/A\n")

            f.write("===== LLM CALL END =====\n\n")
