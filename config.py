import os
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


def ensure_directories() -> None:
    """
    Создаёт базовые директории для данных и логов.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def make_request_dir(user_id: int, original_filename: str, request_ts: str) -> Path:
    """
    Создаёт директорию для конкретного запроса пользователя:
    data/<telegram_id>/<base_filename>_<request_ts>/
    """
    base_name = Path(original_filename).stem
    user_dir = DATA_DIR / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)

    request_dir_name = f"{base_name}_{request_ts}"
    request_dir = user_dir / request_dir_name
    request_dir.mkdir(parents=True, exist_ok=True)
    return request_dir
