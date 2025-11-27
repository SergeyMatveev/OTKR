from __future__ import annotations

from config import ensure_directories
from logging_setup import setup_logging
from bot import create_application


def main() -> None:
    ensure_directories()
    logger, log_path = setup_logging()
    logger.info("Старт системы обработки отчётов НБКИ. Лог: %s", log_path)

    application = create_application(logger)
    logger.info("Telegram-бот запущен. Ожидаю сообщения...")
    application.run_polling()


if __name__ == "__main__":
    main()
