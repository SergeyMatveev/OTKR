from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict
import json
import time

from logging import Logger
from pypdf import PdfReader, PdfWriter
from mistralai import Mistral

from config import FREE_API_KEY, make_llm_io_path
from logging_setup import log_llm_call
from logging_setup import FileStats


def search_phrase_multi(
    pdf_path: Path,
    phrase: str,
    attempts: int,
    logger: Optional[Logger] = None,
    *,
    telegram_user_id: str = "N/A",
    telegram_username: str = "N/A",
    request_id: str = "N/A",
) -> Optional[int]:
    """
    Ищет фразу по всему PDF несколько раз.
    Каждый раз PDF перечитывается и текст страниц вытаскивается заново.
    Возвращает индекс страницы (0-based) или None.
    """
    for attempt in range(1, attempts + 1):
        if logger:
            logger.info(
                'Поиск фразы "%s", попытка %d из %d...',
                phrase,
                attempt,
                attempts,
                extra={
                    "stage": "pdf_phrase_search",
                    "telegram_user_id": telegram_user_id,
                    "telegram_username": telegram_username,
                    "request_id": request_id,
                    "duration_seconds": 0,
                    "model": "N/A",
                    "api_key_id": "N/A",
                    "file_name": pdf_path.name,
                },
            )
        reader = PdfReader(str(pdf_path))
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if phrase in text:
                if logger:
                    logger.info(
                        'Фраза "%s" найдена на странице %d',
                        phrase,
                        i + 1,
                        extra={
                            "stage": "pdf_phrase_found",
                            "telegram_user_id": telegram_user_id,
                            "telegram_username": telegram_username,
                            "request_id": request_id,
                            "duration_seconds": 0,
                            "model": "N/A",
                            "api_key_id": "N/A",
                            "file_name": pdf_path.name,
                        },
                    )
                return i
    if logger:
        logger.warning(
            'Фраза "%s" не найдена после %d попыток.',
            phrase,
            attempts,
            extra={
                "stage": "pdf_phrase_not_found",
                "telegram_user_id": telegram_user_id,
                "telegram_username": telegram_username,
                "request_id": request_id,
                "duration_seconds": 0,
                "model": "N/A",
                "api_key_id": "N/A",
                "file_name": pdf_path.name,
            },
        )
    return None


def run_pdf_crop_and_ocr(
    pdf_path: Path,
    request_dir: Path,
    request_ts: str,
    logger: Logger,
    file_stats: FileStats | None = None,
) -> Dict[str, Path]:
    """
    Шаг 1:
      - Обрезает PDF до страницы с фразой
        "Расшифровка основных событий" или "Информационная часть" (включительно).
      - Запускает OCR в Mistral (mistral-ocr-latest).
      - Сохраняет обрезанный PDF, JSON и MD в request_dir.

    Возвращает словарь с путями:
      {
        "cropped_pdf": Path,
        "ocr_json": Path,
        "ocr_md": Path | None,
        "original_pdf": Path,
      }
    """
    if not FREE_API_KEY:
        raise RuntimeError("FREE_API_KEY не задан в .env, невозможен OCR Mistral.")

    pdf_path = pdf_path.resolve()
    request_dir = request_dir.resolve()
    base_name = pdf_path.stem

    telegram_user_id = request_dir.parent.name
    telegram_username = "N/A"
    request_id = request_dir.name

    llm_io_path = make_llm_io_path(request_dir, pdf_path.name, request_ts)

    total_start = time.perf_counter()

    logger.info(
        "Шаг 1: обрезка PDF и OCR для файла %s",
        pdf_path,
        extra={
            "stage": "pdf_crop_ocr_start",
            "telegram_user_id": telegram_user_id,
            "telegram_username": telegram_username,
            "request_id": request_id,
            "duration_seconds": 0,
            "model": "N/A",
            "api_key_id": "N/A",
            "file_name": pdf_path.name,
        },
    )

    phrase_events = "Расшифровка основных событий"
    phrase_info = "Информационная часть"

    # Поиск фраз: сначала "Расшифровка основных событий", затем "Информационная часть"
    search_start = time.perf_counter()
    page_index = search_phrase_multi(
        pdf_path,
        phrase_events,
        attempts=3,
        logger=logger,
        telegram_user_id=telegram_user_id,
        telegram_username=telegram_username,
        request_id=request_id,
    )
    found_phrase = None

    if page_index is None:
        page_index = search_phrase_multi(
            pdf_path,
            phrase_info,
            attempts=3,
            logger=logger,
            telegram_user_id=telegram_user_id,
            telegram_username=telegram_username,
            request_id=request_id,
        )
        if page_index is not None:
            found_phrase = phrase_info
    else:
        found_phrase = phrase_events

    search_duration = time.perf_counter() - search_start
    logger.info(
        "Поиск целевых фраз завершён, длительность %.3f с.",
        search_duration,
        extra={
            "stage": "pdf_phrase_search_done",
            "telegram_user_id": telegram_user_id,
            "telegram_username": telegram_username,
            "request_id": request_id,
            "duration_seconds": round(search_duration, 3),
            "model": "N/A",
            "api_key_id": "N/A",
            "file_name": pdf_path.name,
        },
    )

    if page_index is None:
        error_msg = (
            f'ОШИБКА: в файле "{pdf_path.name}" ожидается фраза '
            f'"{phrase_events}" или "{phrase_info}", но они не были найдены '
            f'даже после повторных попыток. Процесс прерван. '
            f'Проверьте файл или обратитесь к разработчику Сергею в Telegram: @Sergey_robots.'
        )
        logger.error(
            error_msg,
            extra={
                "stage": "pdf_phrase_missing_error",
                "telegram_user_id": telegram_user_id,
                "telegram_username": telegram_username,
                "request_id": request_id,
                "duration_seconds": round(time.perf_counter() - total_start, 3),
                "model": "N/A",
                "api_key_id": "N/A",
                "file_name": pdf_path.name,
            },
        )
        raise RuntimeError(error_msg)

    reader = PdfReader(str(pdf_path))
    num_pages = len(reader.pages)
    if file_stats is not None:
        file_stats.set_pages_original(num_pages)
    logger.info(
        "Всего страниц в PDF: %d, найденная фраза на странице (0-based): %d",
        num_pages,
        page_index,
        extra={
            "stage": "pdf_pages_info",
            "telegram_user_id": telegram_user_id,
            "telegram_username": telegram_username,
            "request_id": request_id,
            "duration_seconds": 0,
            "model": "N/A",
            "api_key_id": "N/A",
            "file_name": pdf_path.name,
        },
    )

    # Включаем страницу, где найдена фраза
    end_index = page_index

    if end_index < 0:
        error_msg = (
            f'ОШИБКА: вычисленный диапазон страниц для файла "{pdf_path.name}" пустой. '
            f'Процесс прерван. Обратитесь к разработчику Сергею: @Sergey_robots.'
        )
        logger.error(
            error_msg,
            extra={
                "stage": "pdf_page_range_error",
                "telegram_user_id": telegram_user_id,
                "telegram_username": telegram_username,
                "request_id": request_id,
                "duration_seconds": round(time.perf_counter() - total_start, 3),
                "model": "N/A",
                "api_key_id": "N/A",
                "file_name": pdf_path.name,
            },
        )
        raise RuntimeError(error_msg)

    crop_start = time.perf_counter()
    writer = PdfWriter()
    for page_idx in range(0, end_index + 1):
        writer.add_page(reader.pages[page_idx])

    start_page_num = 1
    end_page_num = end_index + 1
    cropped_pdf_name = f"{base_name}_({start_page_num}-{end_page_num}).pdf"
    cropped_pdf_path = request_dir / cropped_pdf_name

    with cropped_pdf_path.open("wb") as out_f:
        writer.write(out_f)

    if file_stats is not None:
        file_stats.set_pages_after_crop(end_page_num)

    crop_duration = time.perf_counter() - crop_start
    logger.info(
        "Обрезанный PDF сохранён: %s (страницы %d-%d, по фразе '%s')",
        cropped_pdf_path,
        start_page_num,
        end_page_num,
        found_phrase,
        extra={
            "stage": "pdf_cropped",
            "telegram_user_id": telegram_user_id,
            "telegram_username": telegram_username,
            "request_id": request_id,
            "duration_seconds": round(crop_duration, 3),
            "model": "N/A",
            "api_key_id": "N/A",
            "file_name": cropped_pdf_path.name,
        },
    )

    # OCR в Mistral
    ocr_start = time.perf_counter()
    logger.info(
        "Шаг 1: отправляем обрезанный PDF в Mistral OCR (mistral-ocr-latest).",
        extra={
            "stage": "ocr_request",
            "telegram_user_id": telegram_user_id,
            "telegram_username": telegram_username,
            "request_id": request_id,
            "duration_seconds": 0,
            "model": "mistral-ocr-latest",
            "api_key_id": "FREE_API_KEY",
            "file_name": cropped_pdf_path.name,
        },
    )

    client = Mistral(api_key=FREE_API_KEY)

    ocr_request_text = (
        "OCR NBKI PDF report.\n"
        f"Original PDF: {pdf_path.name}\n"
        f"Cropped PDF: {cropped_pdf_path.name}\n"
        "Model: mistral-ocr-latest"
    )

    try:
        with cropped_pdf_path.open("rb") as f:
            uploaded_pdf = client.files.upload(
                file={"file_name": cropped_pdf_path.name, "content": f},
                purpose="ocr",
            )

        signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id).url

        resp = client.ocr.process(
            model="mistral-ocr-latest",
            document={"type": "document_url", "document_url": signed_url},
            include_image_base64=False,
        )
    except Exception as e:
        ocr_duration = time.perf_counter() - ocr_start
        logger.error(
            "Исключение во время OCR Mistral для файла %s: %s",
            cropped_pdf_path,
            e,
            exc_info=True,
            extra={
                "stage": "ocr_error",
                "telegram_user_id": telegram_user_id,
                "telegram_username": telegram_username,
                "request_id": request_id,
                "duration_seconds": round(ocr_duration, 3),
                "model": "mistral-ocr-latest",
                "api_key_id": "FREE_API_KEY",
                "file_name": cropped_pdf_path.name,
            },
        )
        try:
            log_llm_call(
                llm_io_path,
                request_id=request_id,
                telegram_user_id=str(telegram_user_id),
                telegram_username=telegram_username,
                pdf_filename=pdf_path.name,
                model="mistral-ocr-latest",
                api_key_id="FREE_API_KEY",
                request_text=ocr_request_text,
                response_text=str(e),
                latency_seconds=ocr_duration,
                status="error",
                error_type=type(e).__name__,
                error_code=None,
            )
            if file_stats is not None:
                file_stats.register_llm_call(
                    model="mistral-ocr-latest",
                    api_key_id="FREE_API_KEY",
                    request_text=ocr_request_text,
                    response_text=str(e),
                    latency_seconds=ocr_duration,
                )
        except Exception:
            pass
        raise

    ocr_duration = time.perf_counter() - ocr_start

    logger.info(
        "OCR Mistral завершён за %.3f с.",
        ocr_duration,
        extra={
            "stage": "ocr_response",
            "telegram_user_id": telegram_user_id,
            "telegram_username": telegram_username,
            "request_id": request_id,
            "duration_seconds": round(ocr_duration, 3),
            "model": "mistral-ocr-latest",
            "api_key_id": "FREE_API_KEY",
            "file_name": cropped_pdf_path.name,
        },
    )

    ocr_base = cropped_pdf_path.stem
    ocr_json_path = request_dir / f"{ocr_base}_ocr.json"

    try:
        data = resp if isinstance(resp, dict) else resp.model_dump()
    except AttributeError:
        data = getattr(resp, "__dict__", {"response": str(resp)})

    # Логируем полный ответ OCR в llm_io_...txt
    try:
        ocr_response_text = json.dumps(data, ensure_ascii=False, indent=2)
        log_llm_call(
            llm_io_path,
            request_id=request_id,
            telegram_user_id=str(telegram_user_id),
            telegram_username=telegram_username,
            pdf_filename=pdf_path.name,
            model="mistral-ocr-latest",
            api_key_id="FREE_API_KEY",
            request_text=ocr_request_text,
            response_text=ocr_response_text,
            latency_seconds=ocr_duration,
            status="success",
            error_type=None,
            error_code=None,
        )
        if file_stats is not None:
            file_stats.register_llm_call(
                model="mistral-ocr-latest",
                api_key_id="FREE_API_KEY",
                request_text=ocr_request_text,
                response_text=ocr_response_text,
                latency_seconds=ocr_duration,
            )
    except Exception:
        pass

    with ocr_json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    pages = data.get("pages") if isinstance(data, dict) else None
    ocr_md_path: Optional[Path] = None
    if pages:
        ocr_md_path = request_dir / f"{ocr_base}_ocr.md"
        with ocr_md_path.open("w", encoding="utf-8") as f:
            f.write("\n\n".join(p.get("markdown", "") for p in pages))
        logger.info(
            "OCR: сохранены файлы %s, %s",
            ocr_json_path,
            ocr_md_path,
            extra={
                "stage": "ocr_files_saved",
                "telegram_user_id": telegram_user_id,
                "telegram_username": telegram_username,
                "request_id": request_id,
                "duration_seconds": 0,
                "model": "mistral-ocr-latest",
                "api_key_id": "FREE_API_KEY",
                "file_name": cropped_pdf_path.name,
            },
        )
    else:
        logger.warning(
            "OCR: сохранён только JSON (нет pages->markdown): %s",
            ocr_json_path,
            extra={
                "stage": "ocr_no_markdown",
                "telegram_user_id": telegram_user_id,
                "telegram_username": telegram_username,
                "request_id": request_id,
                "duration_seconds": 0,
                "model": "mistral-ocr-latest",
                "api_key_id": "FREE_API_KEY",
                "file_name": cropped_pdf_path.name,
            },
        )

    total_duration = time.perf_counter() - total_start
    logger.info(
        "Шаг 1 (обрезка + OCR) завершён за %.3f с.",
        total_duration,
        extra={
            "stage": "pdf_crop_ocr_done",
            "telegram_user_id": telegram_user_id,
            "telegram_username": telegram_username,
            "request_id": request_id,
            "duration_seconds": round(total_duration, 3),
            "model": "mistral-ocr-latest",
            "api_key_id": "FREE_API_KEY",
            "file_name": cropped_pdf_path.name,
        },
    )

    return {
        "original_pdf": pdf_path,
        "cropped_pdf": cropped_pdf_path,
        "ocr_json": ocr_json_path,
        "ocr_md": ocr_md_path,
    }

