import os
import re
from pathlib import Path

from dotenv import load_dotenv

# Загружаем переменные окружения из .env
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
LOG_DIR = DATA_DIR / "logfiles"

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
FREE_API_KEY = os.getenv("FREE_API_KEY", "").strip()
PAID_API_KEY = os.getenv("PAID_API_KEY", "").strip()


def _parse_admin_user_ids(raw: str) -> set[int]:
    ids: set[int] = set()
    if not raw:
        return ids
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        if p.isdigit():
            try:
                ids.add(int(p))
            except Exception:
                continue
    return ids


ADMIN_USER_IDS = _parse_admin_user_ids(os.getenv("ADMIN_USER_IDS", "").strip())


def is_admin(user_id: int) -> bool:
    return int(user_id) in ADMIN_USER_IDS


def ensure_directories() -> None:
    """
    Создаёт базовые директории для данных и логов.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def make_request_dir(user_id: int, original_filename: str, request_ts: str) -> Path:
    """
    Создаёт директорию для конкретного запроса пользователя:
    data/<telegram_id>/<request_ts>_<base_filename>/
    """
    base_name = Path(original_filename).stem
    user_dir = DATA_DIR / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)

    request_dir_name = f"{request_ts}_{base_name}"
    request_dir = user_dir / request_dir_name
    request_dir.mkdir(parents=True, exist_ok=True)
    return request_dir


def make_test_request_dir(user_id: int, test_name: str, request_ts: str) -> Path:
    """
    Создаёт директорию для тестового запроса (команда /test):
    data/<telegram_id>/<request_ts>_TEST_<name>/
    """
    name_raw = (test_name or "").strip()
    if not name_raw:
        name_raw = "manual"

    safe = re.sub(r"[^A-Za-z0-9_-]", "_", name_raw)
    safe = safe.strip("_")
    if not safe:
        safe = "manual"
    safe = safe[:60]

    user_dir = DATA_DIR / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)

    request_dir_name = f"{request_ts}_TEST_{safe}"
    request_dir = user_dir / request_dir_name
    request_dir.mkdir(parents=True, exist_ok=True)
    return request_dir


def make_llm_io_path(request_dir: Path, original_filename: str, request_ts: str) -> Path:
    """
    Возвращает путь к файлу llm_io_<name_of_the_sent_pdf_file>_<date_time>.txt
    внутри директории запроса.
    """
    base_name = Path(original_filename).name
    llm_io_name = f"llm_io_{base_name}_{request_ts}.txt"
    return request_dir / llm_io_name

