from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict
import json

from logging import Logger
from pypdf import PdfReader, PdfWriter
from mistralai import Mistral

from config import FREE_API_KEY


def search_phrase_multi(pdf_path: Path, phrase: str, attempts: int, logger: Optional[Logger] = None) -> Optional[int]:
    """
    Ищет фразу по всему PDF несколько раз.
    Каждый раз PDF перечитывается и текст страниц вытаскивается заново.
    Возвращает индекс страницы (0-based) или None.
    """
    for attempt in range(1, attempts + 1):
        if logger:
            logger.info('Поиск фразы "%s", попытка %d из %d...', phrase, attempt, attempts)
        reader = PdfReader(str(pdf_path))
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if phrase in text:
                if logger:
                    logger.info('Фраза "%s" найдена на странице %d', phrase, i + 1)
                return i
    if logger:
        logger.warning('Фраза "%s" не найдена после %d попыток.', phrase, attempts)
    return None


def run_pdf_crop_and_ocr(pdf_path: Path, request_dir: Path, logger: Logger) -> Dict[str, Path]:
    """
    Шаг 1:
      - Обрезает PDF до страницы перед фразой
        "Расшифровка основных событий" или "Информационная часть".
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

    logger.info("Шаг 1: обрезка PDF и OCR для файла %s", pdf_path)

    phrase_events = "Расшифровка основных событий"
    phrase_info = "Информационная часть"

    # Поиск фраз: сначала "Расшифровка основных событий", затем "Информационная часть"
    page_index = search_phrase_multi(pdf_path, phrase_events, attempts=3, logger=logger)
    found_phrase = None

    if page_index is None:
        page_index = search_phrase_multi(pdf_path, phrase_info, attempts=3, logger=logger)
        if page_index is not None:
            found_phrase = phrase_info
    else:
        found_phrase = phrase_events

    if page_index is None:
        error_msg = (
            f'ОШИБКА: в файле "{pdf_path.name}" ожидается фраза '
            f'"{phrase_events}" или "{phrase_info}", но они не были найдены '
            f'даже после повторных попыток. Процесс прерван. '
            f'Проверьте файл или обратитесь к разработчику Сергею в Telegram: @Sergey_robots.'
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    reader = PdfReader(str(pdf_path))
    num_pages = len(reader.pages)
    logger.info("Всего страниц в PDF: %d, найденная фраза на странице (0-based): %d", num_pages, page_index)

    if page_index == 0:
        end_index = 0
    else:
        end_index = page_index - 1

    if end_index < 0:
        error_msg = (
            f'ОШИБКА: вычисленный диапазон страниц для файла "{pdf_path.name}" пустой. '
            f'Процесс прерван. Обратитесь к разработчику Сергею: @Sergey_robots.'
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    writer = PdfWriter()
    for page_idx in range(0, end_index + 1):
        writer.add_page(reader.pages[page_idx])

    start_page_num = 1
    end_page_num = end_index + 1
    cropped_pdf_name = f"{base_name}_({start_page_num}-{end_page_num}).pdf"
    cropped_pdf_path = request_dir / cropped_pdf_name

    with cropped_pdf_path.open("wb") as out_f:
        writer.write(out_f)

    logger.info(
        "Обрезанный PDF сохранён: %s (страницы %d-%d, по фразе '%s')",
        cropped_pdf_path,
        start_page_num,
        end_page_num,
        found_phrase,
    )

    # OCR в Mistral
    logger.info("Шаг 1: отправляем обрезанный PDF в Mistral OCR (mistral-ocr-latest).")
    client = Mistral(api_key=FREE_API_KEY)

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

    ocr_base = cropped_pdf_path.stem
    ocr_json_path = request_dir / f"{ocr_base}_ocr.json"

    try:
        data = resp if isinstance(resp, dict) else resp.model_dump()
    except AttributeError:
        data = getattr(resp, "__dict__", {"response": str(resp)})

    with ocr_json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    pages = data.get("pages") if isinstance(data, dict) else None
    ocr_md_path: Optional[Path] = None
    if pages:
        ocr_md_path = request_dir / f"{ocr_base}_ocr.md"
        with ocr_md_path.open("w", encoding="utf-8") as f:
            f.write("\n\n".join(p.get("markdown", "") for p in pages))
        logger.info("OCR: сохранены файлы %s, %s", ocr_json_path, ocr_md_path)
    else:
        logger.warning("OCR: сохранён только JSON (нет pages->markdown): %s", ocr_json_path)

    return {
        "original_pdf": pdf_path,
        "cropped_pdf": cropped_pdf_path,
        "ocr_json": ocr_json_path,
        "ocr_md": ocr_md_path,
    }
