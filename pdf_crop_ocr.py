from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict
import time
import re

from logging import Logger
from pypdf import PdfReader, PdfWriter
from mistralai import Mistral

from config import FREE_API_KEY, PAID_API_KEY, make_llm_io_path
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


def _count_pdf_pages(pdf_path: Path) -> int:
    reader = PdfReader(str(pdf_path))
    return len(reader.pages)


def _split_pdf_into_parts(pdf_path: Path, out_dir: Path, chunk_size: int = 500) -> list[Path]:
    """
    Делит PDF на части "примерно по 500 страниц" так, чтобы:
      - не было повторов страниц,
      - последняя часть была "хвостом" (все оставшиеся страницы),
      - размер любой части не превышал 1000 страниц.

    Алгоритм:
      - пока оставшихся страниц > 1000 — режем по chunk_size (500),
      - когда осталось <= 1000 — делаем последнюю часть как весь остаток.
    """
    reader = PdfReader(str(pdf_path))
    total = len(reader.pages)

    parts: list[Path] = []
    start = 0
    part_idx = 0

    while (total - start) > 1000:
        end = min(start + chunk_size, total)
        writer = PdfWriter()
        for i in range(start, end):
            writer.add_page(reader.pages[i])

        part_idx += 1
        start1 = start + 1
        end1 = end
        part_name = f"{pdf_path.stem}_part{part_idx:03d}_{start1}-{end1}.pdf"
        part_path = out_dir / part_name
        with part_path.open("wb") as out_f:
            writer.write(out_f)
        parts.append(part_path)
        start = end

    # Последняя часть — весь остаток (<= 1000 страниц)
    if start < total:
        end = total
        writer = PdfWriter()
        for i in range(start, end):
            writer.add_page(reader.pages[i])

        part_idx += 1
        start1 = start + 1
        end1 = end
        part_name = f"{pdf_path.stem}_part{part_idx:03d}_{start1}-{end1}.pdf"
        part_path = out_dir / part_name
        with part_path.open("wb") as out_f:
            writer.write(out_f)
        parts.append(part_path)

    # Лёгкая проверка: сумма страниц частей == total
    total_parts_pages = 0
    for p in parts:
        total_parts_pages += _count_pdf_pages(p)
    if total_parts_pages != total:
        raise RuntimeError(
            f"ОШИБКА: сумма страниц частей ({total_parts_pages}) != страниц исходного PDF ({total})."
        )

    return parts


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
      - Запускает OCR в Mistral (mistral-ocr-2512).
      - Сохраняет обрезанный PDF и MD в request_dir.
      - Если обрезанный PDF > 1000 страниц — делит его на части (~500 страниц),
        делает OCR по частям строго последовательно, сохраняет *_partXXX_*_ocr.md,
        и склеивает их в один итоговый *_ocr.md.

    Возвращает словарь с путями:
      {
        "cropped_pdf": Path,
        "ocr_json": None,
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

    client_free = Mistral(api_key=FREE_API_KEY)
    client_paid = Mistral(api_key=PAID_API_KEY) if PAID_API_KEY else None

    def _mistral_ocr_one_pdf_to_md(pdf_for_ocr_path: Path) -> Path:
        ocr_request_text = (
            "OCR NBKI PDF report.\n"
            f"Original PDF: {pdf_path.name}\n"
            f"PDF for OCR: {pdf_for_ocr_path.name}\n"
            "Model: mistral-ocr-2512"
        )

        def _try_get_status_code(exc: Exception) -> Optional[int]:
            for attr in ("status_code", "status", "http_status", "code"):
                v = getattr(exc, attr, None)
                if isinstance(v, int):
                    return v
                if isinstance(v, str) and v.isdigit():
                    try:
                        return int(v)
                    except Exception:
                        pass
            resp = getattr(exc, "response", None)
            if resp is not None:
                sc = getattr(resp, "status_code", None)
                if isinstance(sc, int):
                    return sc
            msg = str(exc)
            if "429" in msg:
                return 429
            return None

        ocr_last_exc: Optional[Exception] = None
        use_paid = False
        paid_tried = False
        force_free_next_attempt: bool = False

        for attempt in range(1, 4):
            client_to_use = client_free
            api_key_id = "FREE_API_KEY"

            if force_free_next_attempt:
                force_free_next_attempt = False
            else:
                if use_paid and client_paid is not None:
                    client_to_use = client_paid
                    api_key_id = "PAID_API_KEY"

            ocr_start = time.perf_counter()
            logger.info(
                "Шаг 1: отправляем PDF в Mistral OCR (mistral-ocr-2512): %s (попытка %d/3, ключ=%s)",
                pdf_for_ocr_path.name,
                attempt,
                api_key_id,
                extra={
                    "stage": "ocr_request",
                    "telegram_user_id": telegram_user_id,
                    "telegram_username": telegram_username,
                    "request_id": request_id,
                    "duration_seconds": 0,
                    "model": "mistral-ocr-2512",
                    "api_key_id": api_key_id,
                    "file_name": pdf_for_ocr_path.name,
                },
            )

            try:
                with pdf_for_ocr_path.open("rb") as f:
                    uploaded_pdf = client_to_use.files.upload(
                        file={"file_name": pdf_for_ocr_path.name, "content": f},
                        purpose="ocr",
                    )

                signed_url = client_to_use.files.get_signed_url(file_id=uploaded_pdf.id).url

                resp = client_to_use.ocr.process(
                    model="mistral-ocr-2512",
                    document={"type": "document_url", "document_url": signed_url},
                    include_image_base64=False,
                )
            except Exception as e:
                ocr_last_exc = e
                ocr_duration = time.perf_counter() - ocr_start
                status_code = _try_get_status_code(e)
                err_code = str(status_code) if status_code is not None else None

                logger.error(
                    "Исключение во время OCR Mistral для файла %s: %s",
                    pdf_for_ocr_path,
                    e,
                    exc_info=True,
                    extra={
                        "stage": "ocr_error",
                        "telegram_user_id": telegram_user_id,
                        "telegram_username": telegram_username,
                        "request_id": request_id,
                        "duration_seconds": round(ocr_duration, 3),
                        "model": "mistral-ocr-2512",
                        "api_key_id": api_key_id,
                        "file_name": pdf_for_ocr_path.name,
                    },
                )
                try:
                    log_llm_call(
                        llm_io_path,
                        request_id=request_id,
                        telegram_user_id=str(telegram_user_id),
                        telegram_username=telegram_username,
                        pdf_filename=pdf_path.name,
                        model="mistral-ocr-2512",
                        api_key_id=api_key_id,
                        request_text=ocr_request_text,
                        response_text=str(e),
                        latency_seconds=ocr_duration,
                        status="error",
                        error_type=type(e).__name__,
                        error_code=err_code,
                    )
                    if file_stats is not None:
                        file_stats.register_llm_call(
                            model="mistral-ocr-2512",
                            api_key_id=api_key_id,
                            request_text=ocr_request_text,
                            response_text=str(e),
                            latency_seconds=ocr_duration,
                        )
                except Exception:
                    pass

                if status_code == 429:
                    time.sleep(30)
                    force_free_next_attempt = True
                    continue

                if (client_paid is not None) and (not paid_tried):
                    use_paid = True
                    paid_tried = True
                    continue

                raise

            ocr_duration = time.perf_counter() - ocr_start
            logger.info(
                "OCR Mistral завершён за %.3f с. (%s)",
                ocr_duration,
                pdf_for_ocr_path.name,
                extra={
                    "stage": "ocr_response",
                    "telegram_user_id": telegram_user_id,
                    "telegram_username": telegram_username,
                    "request_id": request_id,
                    "duration_seconds": round(ocr_duration, 3),
                    "model": "mistral-ocr-2512",
                    "api_key_id": api_key_id,
                    "file_name": pdf_for_ocr_path.name,
                },
            )

            try:
                data = resp if isinstance(resp, dict) else resp.model_dump()
            except AttributeError:
                data = getattr(resp, "__dict__", {"response": str(resp)})

            pages = data.get("pages") if isinstance(data, dict) else None
            if not pages:
                raise RuntimeError(
                    f"OCR Mistral не вернул pages->markdown для файла {pdf_for_ocr_path.name}."
                )

            stem = pdf_for_ocr_path.stem
            start_page_for_this_pdf = 1

            m = re.search(r"_part\d{3}_(\d+)-(\d+)$", stem)
            if m:
                try:
                    start_page_for_this_pdf = int(m.group(1))
                except Exception:
                    start_page_for_this_pdf = 1
            else:
                m2 = re.search(r"_\((\d+)-(\d+)\)$", stem)
                if m2:
                    try:
                        start_page_for_this_pdf = int(m2.group(1))
                    except Exception:
                        start_page_for_this_pdf = 1

            ocr_md_path = request_dir / f"{pdf_for_ocr_path.stem}_ocr.md"
            md_parts = []
            for i, p in enumerate(pages):
                page_markdown = p.get("markdown", "") or ""
                marker = f"@@@PAGE: {start_page_for_this_pdf + i}@@@\n"
                md_parts.append(marker + page_markdown)

            md_text = "\n\n".join(md_parts)
            with ocr_md_path.open("w", encoding="utf-8") as f:
                f.write(md_text)

            try:
                log_llm_call(
                    llm_io_path,
                    request_id=request_id,
                    telegram_user_id=str(telegram_user_id),
                    telegram_username=telegram_username,
                    pdf_filename=pdf_path.name,
                    model="mistral-ocr-2512",
                    api_key_id=api_key_id,
                    request_text=ocr_request_text,
                    response_text=f"saved_md={ocr_md_path.name}; chars={len(md_text)}",
                    latency_seconds=ocr_duration,
                    status="success",
                    error_type=None,
                    error_code=None,
                )
                if file_stats is not None:
                    file_stats.register_llm_call(
                        model="mistral-ocr-2512",
                        api_key_id=api_key_id,
                        request_text=ocr_request_text,
                        response_text=f"saved_md={ocr_md_path.name}; chars={len(md_text)}",
                        latency_seconds=ocr_duration,
                    )
            except Exception:
                pass

            logger.info(
                "OCR: сохранён MD: %s",
                ocr_md_path,
                extra={
                    "stage": "ocr_files_saved",
                    "telegram_user_id": telegram_user_id,
                    "telegram_username": telegram_username,
                    "request_id": request_id,
                    "duration_seconds": 0,
                    "model": "mistral-ocr-2512",
                    "api_key_id": api_key_id,
                    "file_name": pdf_for_ocr_path.name,
                },
            )
            return ocr_md_path

        if ocr_last_exc is not None:
            raise ocr_last_exc
        raise RuntimeError(f"OCR Mistral не завершился успешно для файла {pdf_for_ocr_path.name}.")

    pages_after_crop = _count_pdf_pages(cropped_pdf_path)

    if pages_after_crop <= 1000:
        ocr_md_path = _mistral_ocr_one_pdf_to_md(cropped_pdf_path)

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
                "model": "mistral-ocr-2512",
                "api_key_id": "FREE_API_KEY",
                "file_name": cropped_pdf_path.name,
            },
        )

        return {
            "original_pdf": pdf_path,
            "cropped_pdf": cropped_pdf_path,
            "ocr_json": None,
            "ocr_md": ocr_md_path,
        }

    logger.warning(
        "Обрезанный PDF содержит %d страниц (>1000). Включаю разбиение на части (~500) и последовательный OCR.",
        pages_after_crop,
        extra={
            "stage": "pdf_split_start",
            "telegram_user_id": telegram_user_id,
            "telegram_username": telegram_username,
            "request_id": request_id,
            "duration_seconds": 0,
            "model": "N/A",
            "api_key_id": "N/A",
            "file_name": cropped_pdf_path.name,
        },
    )

    parts = _split_pdf_into_parts(cropped_pdf_path, request_dir, chunk_size=500)

    ranges_for_log = []
    for p in parts:
        # имя вида ..._partXXX_A-B.pdf
        stem = p.stem
        dash_pos = stem.rfind("_")
        rng = stem[dash_pos + 1 :] if dash_pos != -1 else ""
        ranges_for_log.append(rng)

    logger.info(
        "PDF разбит на %d частей. Диапазоны страниц: %s",
        len(parts),
        ", ".join(ranges_for_log),
        extra={
            "stage": "pdf_split_done",
            "telegram_user_id": telegram_user_id,
            "telegram_username": telegram_username,
            "request_id": request_id,
            "duration_seconds": 0,
            "model": "N/A",
            "api_key_id": "N/A",
            "file_name": cropped_pdf_path.name,
        },
    )

    part_md_paths: list[Path] = []
    for part_pdf_path in parts:
        part_md_paths.append(_mistral_ocr_one_pdf_to_md(part_pdf_path))

    final_ocr_md_path = request_dir / f"{cropped_pdf_path.stem}_ocr.md"
    with final_ocr_md_path.open("w", encoding="utf-8") as out_f:
        for i, md_p in enumerate(part_md_paths):
            with md_p.open("r", encoding="utf-8") as in_f:
                txt = in_f.read()
            if i > 0:
                out_f.write("\n\n")
            out_f.write(txt)

    # Удаляем PDF-части только после успешного OCR и склейки
    for part_pdf_path in parts:
        try:
            part_pdf_path.unlink()
        except Exception:
            pass

    total_duration = time.perf_counter() - total_start
    logger.info(
        "Шаг 1 (обрезка + OCR по частям) завершён за %.3f с. Итоговый MD: %s",
        total_duration,
        final_ocr_md_path,
        extra={
            "stage": "pdf_crop_ocr_done",
            "telegram_user_id": telegram_user_id,
            "telegram_username": telegram_username,
            "request_id": request_id,
            "duration_seconds": round(total_duration, 3),
            "model": "mistral-ocr-2512",
            "api_key_id": "FREE_API_KEY",
            "file_name": cropped_pdf_path.name,
        },
    )

    return {
        "original_pdf": pdf_path,
        "cropped_pdf": cropped_pdf_path,
        "ocr_json": None,
        "ocr_md": final_ocr_md_path,
    }
