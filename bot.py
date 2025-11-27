from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from logging import Logger

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from config import BOT_TOKEN, make_request_dir
from pipeline import process_nbki_pdf


INTRO_TEXT = "Я готов обработать отчёт НБКИ, пришлите PDF."
NOT_PDF_TEXT = "Это не PDF, пришлите PDF и я обработаю."


def _now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(INTRO_TEXT)


async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(INTRO_TEXT)


async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # На любое текстовое сообщение — одна и та же подсказка
    await update.message.reply_text(INTRO_TEXT)


async def document_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Обработка входящего документа:
      - если PDF — запускаем пайплайн,
      - если нет — отвечаем, что нужен PDF.
    """
    logger: Logger = context.application.bot_data["logger"]

    message = update.message
    if message is None or message.document is None:
        return

    doc = message.document
    file_name = doc.file_name or f"report_{doc.file_unique_id}.pdf"
    mime = doc.mime_type or ""

    if not (mime == "application/pdf" or file_name.lower().endswith(".pdf")):
        await message.reply_text(NOT_PDF_TEXT)
        logger.info(
            "Пользователь %s прислал не-PDF (%s, mime=%s).",
            update.effective_user.id if update.effective_user else "unknown",
            file_name,
            mime,
        )
        return

    user_id = update.effective_user.id if update.effective_user else 0
    req_ts = _now_ts()
    request_dir = make_request_dir(user_id=user_id, original_filename=file_name, request_ts=req_ts)

    logger.info(
        "Получен PDF от пользователя %s: %s. Директория запроса: %s",
        user_id,
        file_name,
        request_dir,
    )

    await message.reply_text("PDF получен, начинаю обработку, подождите...")

    # Сохраняем исходный файл в директорию запроса
    local_pdf_path = request_dir / file_name
    file = await context.bot.get_file(doc.file_id)
    await file.download_to_drive(custom_path=str(local_pdf_path))

    logger.info("Исходный PDF сохранён в %s", local_pdf_path)

    # Запускаем тяжёлый пайплайн в отдельном потоке
    loop = asyncio.get_running_loop()

    try:
        result_csv_path: Path = await loop.run_in_executor(
            None,
            process_nbki_pdf,
            local_pdf_path,
            request_dir,
            req_ts,
            logger,
        )
    except Exception as e:
        logger.exception("Ошибка при обработке PDF для пользователя %s: %s", user_id, e)
        await message.reply_text(
            "Произошла ошибка при обработке отчёта НБКИ. Попробуйте ещё раз позже "
            "или отправьте сообщение разработчику: @Sergey_robots."
        )
        return

    # Отправляем результат пользователю
    try:
        with result_csv_path.open("rb") as f:
            await message.reply_document(
                document=f,
                filename=result_csv_path.name,
                caption="Готовый CSV с результатами анализа отчёта НБКИ.",
            )
        logger.info(
            "Результирующий CSV отправлен пользователю %s: %s",
            user_id,
            result_csv_path,
        )
    except Exception as e:
        logger.exception(
            "Не удалось отправить CSV пользователю %s: %s", user_id, e
        )
        await message.reply_text(
            "Обработка завершена, но возникла ошибка при отправке файла. "
            "Свяжитесь с @Sergey_robots и сообщите о проблеме."
        )


def create_application(logger: Logger) -> Application:
    """
    Создаёт и настраивает Telegram-приложение (python-telegram-bot 20.8).
    """
    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN не задан в .env, запуск бота невозможен.")

    application = Application.builder().token(BOT_TOKEN).build()
    application.bot_data["logger"] = logger

    application.add_handler(CommandHandler("start", start_handler))
    application.add_handler(CommandHandler("help", help_handler))
    application.add_handler(MessageHandler(filters.Document.ALL, document_handler))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))

    return application
