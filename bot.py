from __future__ import annotations

import asyncio
import time
from datetime import datetime
from pathlib import Path
from logging import Logger

from telegram import Update, InputFile
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from config import BOT_TOKEN, make_request_dir, is_admin, make_test_request_dir
from pipeline import process_nbki_pdf, process_nbki_md
from logging_setup import FileStats, append_stats_row
from report_builder import build_credit_report_from_csv
from commands import download_logs_handler


INTRO_TEXT = "Я готов обработать отчёт НБКИ, пришлите PDF."
NOT_PDF_TEXT = "Это не PDF, пришлите PDF и я обработаю."


def _now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _get_user_context(update: Update) -> dict:
    """
    Возвращает словарь с телеграм-идентификаторами для логирования.
    """
    user = update.effective_user
    if user is None:
        return {
            "telegram_user_id": "N/A",
            "telegram_username": "N/A",
        }
    return {
        "telegram_user_id": user.id,
        "telegram_username": user.username or "N/A",
    }


async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger: Logger = context.application.bot_data["logger"]
    user_ctx = _get_user_context(update)

    logger.info(
        "Получена команда /start.",
        extra={
            "stage": "start_command",
            **user_ctx,
            "request_id": "N/A",
            "duration_seconds": 0,
            "model": "N/A",
            "api_key_id": "N/A",
            "file_name": "N/A",
        },
    )

    if update.message:
        await update.message.reply_text(INTRO_TEXT)


async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger: Logger = context.application.bot_data["logger"]
    user_ctx = _get_user_context(update)

    logger.info(
        "Получена команда /help.",
        extra={
            "stage": "help_command",
            **user_ctx,
            "request_id": "N/A",
            "duration_seconds": 0,
            "model": "N/A",
            "api_key_id": "N/A",
            "file_name": "N/A",
        },
    )

    if update.message:
        await update.message.reply_text(INTRO_TEXT)


async def test_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger: Logger = context.application.bot_data["logger"]
    user_ctx = _get_user_context(update)
    message = update.message

    user_id = update.effective_user.id if update.effective_user else 0

    if not is_admin(user_id):
        logger.warning(
            "Попытка команды /test без прав.",
            extra={
                "stage": "test_denied",
                **user_ctx,
                "request_id": "N/A",
                "duration_seconds": 0,
                "model": "N/A",
                "api_key_id": "N/A",
                "file_name": "N/A",
            },
        )
        if message:
            await message.reply_text("Недостаточно прав.")
        return

    test_name = "manual"
    if context.args:
        test_name = " ".join(context.args).strip() or "manual"

    context.user_data["awaiting_test_md"] = True
    context.user_data["test_name"] = test_name

    logger.info(
        "Команда /test принята. Ожидаю .md.",
        extra={
            "stage": "test_command",
            **user_ctx,
            "request_id": "N/A",
            "duration_seconds": 0,
            "model": "N/A",
            "api_key_id": "N/A",
            "file_name": "N/A",
        },
    )

    if message:
        await message.reply_text("Пришлите .md файлом (документом) для теста.")


async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Обработка любого текстового сообщения: отправляем подсказку и логируем текст.
    """
    logger: Logger = context.application.bot_data["logger"]
    user_ctx = _get_user_context(update)
    message = update.message
    text_preview = ""
    if message and message.text:
        text_preview = message.text[:200]

    logger.info(
        "Получено текстовое сообщение: %s",
        text_preview,
        extra={
            "stage": "incoming_text",
            **user_ctx,
            "request_id": "N/A",
            "duration_seconds": 0,
            "model": "N/A",
            "api_key_id": "N/A",
            "file_name": "N/A",
        },
    )

    if message:
        await message.reply_text(INTRO_TEXT)


async def document_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Обработка входящего документа:
      - если PDF — запускаем пайплайн,
      - если нет — отвечаем, что нужен PDF.
    """
    logger: Logger = context.application.bot_data["logger"]

    message = update.message
    user_ctx = _get_user_context(update)

    if message is None or message.document is None:
        logger.warning(
            "Сообщение в document_handler без документа.",
            extra={
                "stage": "document_handler",
                **user_ctx,
                "request_id": "N/A",
                "duration_seconds": 0,
                "model": "N/A",
                "api_key_id": "N/A",
                "file_name": "N/A",
            },
        )
        return

    doc = message.document
    file_name = doc.file_name or f"report_{doc.file_unique_id}.pdf"
    mime = doc.mime_type or ""

    logger.info(
        "Получен документ от пользователя: имя=%s, mime=%s.",
        file_name,
        mime,
        extra={
            "stage": "incoming_document",
            **user_ctx,
            "request_id": "N/A",
            "duration_seconds": 0,
            "model": "N/A",
            "api_key_id": "N/A",
            "file_name": file_name,
        },
    )

    user_id = update.effective_user.id if update.effective_user else 0

    awaiting_test_md = bool(context.user_data.get("awaiting_test_md", False))
    if awaiting_test_md:
        if not is_admin(user_id):
            logger.warning(
                "Тестовый режим document_handler для не-админа.",
                extra={
                    "stage": "test_denied",
                    **user_ctx,
                    "request_id": "N/A",
                    "duration_seconds": 0,
                    "model": "N/A",
                    "api_key_id": "N/A",
                    "file_name": file_name,
                },
            )
            context.user_data.pop("awaiting_test_md", None)
            context.user_data.pop("test_name", None)
            await message.reply_text("Недостаточно прав.")
            return

        is_md = file_name.lower().endswith(".md") or mime in (
            "text/plain",
            "text/markdown",
            "text/x-markdown",
        )

        if not is_md:
            await message.reply_text("Это не .md, пришлите .md файлом (документом).")
            logger.warning(
                "В тестовом режиме прислан не-MD (%s, mime=%s).",
                file_name,
                mime,
                extra={
                    "stage": "test_not_md",
                    **user_ctx,
                    "request_id": "N/A",
                    "duration_seconds": 0,
                    "model": "N/A",
                    "api_key_id": "N/A",
                    "file_name": file_name,
                },
            )
            return

        test_name = str(context.user_data.get("test_name") or "manual")
        req_ts = _now_ts()
        request_dir = make_test_request_dir(user_id=user_id, test_name=test_name, request_ts=req_ts)
        request_id = request_dir.name

        start_dt = datetime.strptime(req_ts, "%Y%m%d_%H%M%S")
        file_stats = FileStats(
            start_dt=start_dt,
            telegram_user_id=str(user_ctx["telegram_user_id"]),
            telegram_username=user_ctx["telegram_username"],
            request_id=request_id,
            pdf_filename=file_name,
        )

        logger.info(
            "MD получен от пользователя %s: %s. Директория теста: %s",
            user_id,
            file_name,
            request_dir,
            extra={
                "stage": "test_file_received",
                **user_ctx,
                "request_id": request_id,
                "duration_seconds": 0,
                "model": "N/A",
                "api_key_id": "N/A",
                "file_name": file_name,
            },
        )

        await message.reply_text("MD получен, начинаю тест, подождите...")

        overall_start = time.perf_counter()
        local_md_path = request_dir / file_name
        file = await context.bot.get_file(doc.file_id)
        await file.download_to_drive(custom_path=str(local_md_path))

        logger.info(
            "Исходный MD сохранён в %s",
            local_md_path,
            extra={
                "stage": "test_file_saved",
                **user_ctx,
                "request_id": request_id,
                "duration_seconds": 0,
                "model": "N/A",
                "api_key_id": "N/A",
                "file_name": file_name,
            },
        )

        loop = asyncio.get_running_loop()

        processing_error = False
        result_csv_path: Path | None = None

        try:
            logger.info(
                "Запуск тестового MD-пайплайна NBKI для файла %s.",
                local_md_path,
                extra={
                    "stage": "test_processing_start",
                    **user_ctx,
                    "request_id": request_id,
                    "duration_seconds": 0,
                    "model": "N/A",
                    "api_key_id": "N/A",
                    "file_name": file_name,
                },
            )

            result_csv_path = await loop.run_in_executor(
                None,
                process_nbki_md,
                local_md_path,
                file_name,
                request_dir,
                req_ts,
                logger,
                file_stats,
            )
            file_stats.mark_success()
        except Exception as e:
            processing_error = True
            duration = time.perf_counter() - overall_start
            logger.error(
                "Ошибка при тестовой обработке MD для пользователя %s: %s",
                user_id,
                e,
                exc_info=True,
                extra={
                    "stage": "test_processing_error",
                    **user_ctx,
                    "request_id": request_id,
                    "duration_seconds": round(duration, 3),
                    "model": "N/A",
                    "api_key_id": "N/A",
                    "file_name": file_name,
                },
            )
            file_stats.mark_error()
            await message.reply_text(
                "Произошла ошибка при тестовой обработке .md. "
                "Попробуйте ещё раз позже или отправьте сообщение разработчику: @Sergey_robots."
            )
        finally:
            append_stats_row(file_stats.to_row())
            context.user_data.pop("awaiting_test_md", None)
            context.user_data.pop("test_name", None)

        if processing_error or result_csv_path is None:
            return

        try:
            report_messages, xlsx_path = build_credit_report_from_csv(
                result_csv_path,
                logger,
                telegram_user_id=str(user_ctx["telegram_user_id"]),
                telegram_username=user_ctx["telegram_username"],
                request_id=request_id,
                file_name=file_name,
            )
        except Exception as e:
            total_duration = time.perf_counter() - overall_start
            logger.error(
                "Не удалось построить текстовый отчёт по CSV %s: %s",
                result_csv_path,
                e,
                exc_info=True,
                extra={
                    "stage": "test_report_build_error",
                    **user_ctx,
                    "request_id": request_id,
                    "duration_seconds": round(total_duration, 3),
                    "model": "N/A",
                    "api_key_id": "N/A",
                    "file_name": file_name,
                },
            )
            await message.reply_text(
                "Тестовая обработка завершена, но возникла ошибка при подготовке текстового отчёта. "
                "Свяжитесь с @Sergey_robots и сообщите о проблеме."
            )
            return

        total_duration = time.perf_counter() - overall_start

        if not report_messages:
            logger.info(
                "Тестовый отчёт сформирован: открытые кредиты по правилам отбора отсутствуют.",
                extra={
                    "stage": "test_processing_done",
                    **user_ctx,
                    "request_id": request_id,
                    "duration_seconds": round(total_duration, 3),
                    "model": "N/A",
                    "api_key_id": "N/A",
                    "file_name": file_name,
                },
            )
            await message.reply_text(
                "Тестовая обработка завершена. Открытые кредиты по заданным правилам не найдены."
            )
            try:
                with open(xlsx_path, "rb") as f:
                    await message.reply_document(document=InputFile(f, filename=xlsx_path.name))
            except Exception as e:
                logger.error(
                    "Не удалось отправить xlsx %s: %s",
                    xlsx_path,
                    e,
                    exc_info=True,
                    extra={
                        "stage": "test_xlsx_send_error",
                        **user_ctx,
                        "request_id": request_id,
                        "duration_seconds": round(total_duration, 3),
                        "model": "N/A",
                        "api_key_id": "N/A",
                        "file_name": file_name,
                    },
                )
            return

        for msg_text in report_messages:
            await message.reply_text(msg_text)

        try:
            with open(xlsx_path, "rb") as f:
                await message.reply_document(document=InputFile(f, filename=xlsx_path.name))
        except Exception as e:
            logger.error(
                "Не удалось отправить xlsx %s: %s",
                xlsx_path,
                e,
                exc_info=True,
                extra={
                    "stage": "test_xlsx_send_error",
                    **user_ctx,
                    "request_id": request_id,
                    "duration_seconds": round(total_duration, 3),
                    "model": "N/A",
                    "api_key_id": "N/A",
                    "file_name": file_name,
                },
            )

        logger.info(
            "Тестовый текстовый отчёт отправлен пользователю %s: сообщений=%d.",
            user_id,
            len(report_messages),
            extra={
                "stage": "test_processing_done",
                **user_ctx,
                "request_id": request_id,
                "duration_seconds": round(total_duration, 3),
                "model": "N/A",
                "api_key_id": "N/A",
                "file_name": file_name,
            },
        )
        return

    if not (mime == "application/pdf" or file_name.lower().endswith(".pdf")):
        await message.reply_text(NOT_PDF_TEXT)
        logger.warning(
            "Пользователь прислал не-PDF (%s, mime=%s).",
            file_name,
            mime,
            extra={
                "stage": "not_pdf",
                **user_ctx,
                "request_id": "N/A",
                "duration_seconds": 0,
                "model": "N/A",
                "api_key_id": "N/A",
                "file_name": file_name,
            },
        )
        return

    req_ts = _now_ts()
    request_dir = make_request_dir(user_id=user_id, original_filename=file_name, request_ts=req_ts)
    request_id = request_dir.name

    start_dt = datetime.strptime(req_ts, "%Y%m%d_%H%M%S")
    file_stats = FileStats(
        start_dt=start_dt,
        telegram_user_id=str(user_ctx["telegram_user_id"]),
        telegram_username=user_ctx["telegram_username"],
        request_id=request_id,
        pdf_filename=file_name,
    )

    logger.info(
        "PDF получен от пользователя %s: %s. Директория запроса: %s",
        user_id,
        file_name,
        request_dir,
        extra={
            "stage": "file_received",
            **user_ctx,
            "request_id": request_id,
            "duration_seconds": 0,
            "model": "N/A",
            "api_key_id": "N/A",
            "file_name": file_name,
        },
    )

    await message.reply_text("PDF получен, начинаю обработку, подождите...")

    # Сохраняем исходный файл в директорию запроса
    overall_start = time.perf_counter()
    local_pdf_path = request_dir / file_name
    file = await context.bot.get_file(doc.file_id)
    await file.download_to_drive(custom_path=str(local_pdf_path))

    logger.info(
        "Исходный PDF сохранён в %s",
        local_pdf_path,
        extra={
            "stage": "file_saved",
            **user_ctx,
            "request_id": request_id,
            "duration_seconds": 0,
            "model": "N/A",
            "api_key_id": "N/A",
            "file_name": file_name,
        },
    )

    # Запускаем тяжёлый пайплайн в отдельном потоке
    loop = asyncio.get_running_loop()

    processing_error = False
    result_csv_path: Path | None = None

    try:
        logger.info(
            "Запуск пайплайна обработки NBKI для файла %s.",
            local_pdf_path,
            extra={
                "stage": "processing_start",
                **user_ctx,
                "request_id": request_id,
                "duration_seconds": 0,
                "model": "N/A",
                "api_key_id": "N/A",
                "file_name": file_name,
            },
        )

        result_csv_path = await loop.run_in_executor(
            None,
            process_nbki_pdf,
            local_pdf_path,
            request_dir,
            req_ts,
            logger,
            file_stats,
        )
        file_stats.mark_success()
    except Exception as e:
        processing_error = True
        duration = time.perf_counter() - overall_start
        logger.error(
            "Ошибка при обработке PDF для пользователя %s: %s",
            user_id,
            e,
            exc_info=True,
            extra={
                "stage": "processing_error",
                **user_ctx,
                "request_id": request_id,
                "duration_seconds": round(duration, 3),
                "model": "N/A",
                "api_key_id": "N/A",
                "file_name": file_name,
            },
        )
        file_stats.mark_error()
        await message.reply_text(
            "Произошла ошибка при обработке отчёта НБКИ. Попробуйте ещё раз позже "
            "или отправьте сообщение разработчику: @Sergey_robots."
        )
    finally:
        append_stats_row(file_stats.to_row())

    if processing_error or result_csv_path is None:
        return

    # Формируем текстовый отчёт и отправляем пользователю
    try:
        report_messages, xlsx_path = build_credit_report_from_csv(
            result_csv_path,
            logger,
            telegram_user_id=str(user_ctx["telegram_user_id"]),
            telegram_username=user_ctx["telegram_username"],
            request_id=request_id,
            file_name=file_name,
        )
    except Exception as e:
        total_duration = time.perf_counter() - overall_start
        logger.error(
            "Не удалось построить текстовый отчёт по CSV %s: %s",
            result_csv_path,
            e,
            exc_info=True,
            extra={
                "stage": "report_build_error",
                **user_ctx,
                "request_id": request_id,
                "duration_seconds": round(total_duration, 3),
                "model": "N/A",
                "api_key_id": "N/A",
                "file_name": file_name,
            },
        )
        await message.reply_text(
            "Обработка завершена, но возникла ошибка при подготовке текстового отчёта. "
            "Свяжитесь с @Sergey_robots и сообщите о проблеме."
        )
        return

    total_duration = time.perf_counter() - overall_start

    if not report_messages:
        logger.info(
            "Текстовый отчёт сформирован: открытые кредиты по правилам отбора отсутствуют.",
            extra={
                "stage": "processing_done",
                **user_ctx,
                "request_id": request_id,
                "duration_seconds": round(total_duration, 3),
                "model": "N/A",
                "api_key_id": "N/A",
                "file_name": file_name,
            },
        )
        await message.reply_text(
            "Обработка завершена. Открытые кредиты по заданным правилам не найдены."
        )
        try:
            with open(xlsx_path, "rb") as f:
                await message.reply_document(document=InputFile(f, filename=xlsx_path.name))
        except Exception as e:
            logger.error(
                "Не удалось отправить xlsx %s: %s",
                xlsx_path,
                e,
                exc_info=True,
                extra={
                    "stage": "xlsx_send_error",
                    **user_ctx,
                    "request_id": request_id,
                    "duration_seconds": round(total_duration, 3),
                    "model": "N/A",
                    "api_key_id": "N/A",
                    "file_name": file_name,
                },
            )
        return

    for msg_text in report_messages:
        await message.reply_text(msg_text)

    try:
        with open(xlsx_path, "rb") as f:
            await message.reply_document(document=InputFile(f, filename=xlsx_path.name))
    except Exception as e:
        logger.error(
            "Не удалось отправить xlsx %s: %s",
            xlsx_path,
            e,
            exc_info=True,
            extra={
                "stage": "xlsx_send_error",
                **user_ctx,
                "request_id": request_id,
                "duration_seconds": round(total_duration, 3),
                "model": "N/A",
                "api_key_id": "N/A",
                "file_name": file_name,
            },
        )

    logger.info(
        "Текстовый отчёт по открытым кредитам отправлен пользователю %s: сообщений=%d.",
        user_id,
        len(report_messages),
        extra={
            "stage": "processing_done",
            **user_ctx,
            "request_id": request_id,
            "duration_seconds": round(total_duration, 3),
            "model": "N/A",
            "api_key_id": "N/A",
            "file_name": file_name,
        },
    )


def create_application(logger: Logger) -> Application:
    """
    Создаёт и настраивает Telegram-приложение (python-telegram-bot 21.1.1).
    """
    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN не задан в .env, запуск бота невозможен.")

    application = Application.builder().token(BOT_TOKEN).build()
    application.bot_data["logger"] = logger

    application.add_handler(CommandHandler("start", start_handler))
    application.add_handler(CommandHandler("help", help_handler))
    application.add_handler(CommandHandler("test", test_handler))
    application.add_handler(CommandHandler("download_logs", download_logs_handler))
    application.add_handler(MessageHandler(filters.Document.ALL, document_handler))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))

    return application

