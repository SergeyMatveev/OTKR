from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Dict, Optional, List, Any

from logging import Logger

import pandas as pd


def run_markdown_to_chunks(
    md_path: Path,
    request_dir: Path,
    original_base: str,
    request_ts: str,
    logger: Logger,
) -> Dict[str, Optional[Path]]:
    """
    Шаг 2:
      - Читает MD (результат OCR),
      - Делает глобальную правку «ОТВЕТСТВЕННОСТЬЮ»,
      - Вырезает корпус с первого заголовка "N. Название - Договор займа/...",
      - Делит корпус на «сырые» чанки по заголовкам,
      - Валидирует покрытие корпуса,
      - Фильтрует валидные чанки (с «УИд договора»), при этом текст невалидных
        чанков без УИд договора приклеивается к предыдущему валидному блоку,
      - Сохраняет артефакты в request_dir:

        <original_base>_corpus_<ts>.txt
        <original_base>_raw_chunks_<ts>.csv
        <original_base>_chunks_<ts>.csv
        <original_base>_chunks_<ts>.txt
        <original_base>_invalid_chunks_<ts>.txt (если есть)

    Возвращает словарь с путями.
    """
    md_path = md_path.resolve()
    request_dir = request_dir.resolve()

    telegram_user_id = request_dir.parent.name
    request_id = request_dir.name

    total_start = time.perf_counter()

    logger.info(
        "Шаг 2: разбор Markdown и построение чанков из файла %s",
        md_path,
        extra={
            "stage": "md_chunker_start",
            "telegram_user_id": telegram_user_id,
            "telegram_username": "N/A",
            "request_id": request_id,
            "duration_seconds": 0,
            "model": "N/A",
            "api_key_id": "N/A",
            "file_name": md_path.name,
        },
    )

    with md_path.open("r", encoding="utf-8-sig") as f:
        full_text = f.read()

    # Глобальная нормализация «ОТВЕТСТВЕННОСТЬЮ»
    _ooo_tail_fix = re.compile(r"(?iu)ОТВЕТСТВЕННОСТ[ЬЪB6]?\s*[-–—]?\s*[ЮYУ]")
    _full_ooo_phrase = re.compile(
        r"(?iu)ОБЩЕСТВО\s+С\s+ОГРАНИЧЕННОЙ\s+ОТВЕТСТВЕННОСТ[ЬЪB6]?\s*[-–—]?\s*[ЮYУ]"
    )

    def fix_ooo_in_text(s: str) -> str:
        s = _ooo_tail_fix.sub("ОТВЕТСТВЕННОСТЬЮ", s)
        s = _full_ooo_phrase.sub("ОБЩЕСТВО С ОГРАНИЧЕННОЙ ОТВЕТСТВЕННОСТЬЮ", s)
        return s

    full_text = fix_ooo_in_text(full_text)

    # Дополнительная обрезка OCR-текста по целевой фразе (если есть)
    phrase_events = "Расшифровка основных событий"
    phrase_info = "Информационная часть"

    len_before = len(full_text)
    found_phrase: Optional[str] = None

    idx = full_text.find(phrase_events)
    if idx != -1:
        found_phrase = phrase_events
    else:
        idx = full_text.find(phrase_info)
        if idx != -1:
            found_phrase = phrase_info

    if found_phrase is not None:
        cut_start = idx + len(found_phrase)
        full_text = full_text[:cut_start]
        len_after = len(full_text)

        logger.info(
            "Обрезан OCR-текст по фразе '%s': длина до=%d, после=%d.",
            found_phrase,
            len_before,
            len_after,
            extra={
                "stage": "md_ocr_tail_trim",
                "telegram_user_id": telegram_user_id,
                "telegram_username": "N/A",
                "request_id": request_id,
                "duration_seconds": 0,
                "model": "N/A",
                "api_key_id": "N/A",
                "file_name": md_path.name,
            },
        )
    else:
        logger.info(
            "Целевая фраза для обрезки OCR-текста не найдена, обрезка не выполнялась.",
            extra={
                "stage": "md_ocr_tail_trim",
                "telegram_user_id": telegram_user_id,
                "telegram_username": "N/A",
                "request_id": request_id,
                "duration_seconds": 0,
                "model": "N/A",
                "api_key_id": "N/A",
                "file_name": md_path.name,
            },
        )

    # Обрезка корпуса: с первого заголовка договора до конца файла
    KEYWORD_PATTERN = (
        r"(?:Договор\s+займа(?:\s*\(кредита\))?|Микрокредит|Микрозайм|Микрозаем)"
    )

    START_HEADING_RE = re.compile(
        rf"(?im)^[#\s>]*\s*\d{{1,3}}\.\s*.+?\s[-–—]+\s*{KEYWORD_PATTERN}\b.*$"
    )
    start_match = START_HEADING_RE.search(full_text)
    if not start_match:
        msg = "Не найден старт корпуса: 'N. Название - Договор займа/...'."
        logger.error(
            msg,
            extra={
                "stage": "md_chunker_start_not_found",
                "telegram_user_id": telegram_user_id,
                "telegram_username": "N/A",
                "request_id": request_id,
                "duration_seconds": 0,
                "model": "N/A",
                "api_key_id": "N/A",
                "file_name": md_path.name,
            },
        )
        raise ValueError(msg)

    start_pos = start_match.start()
    marker_pos = full_text.rfind("@@@PAGE:", 0, start_pos)
    start_idx = marker_pos if marker_pos != -1 else start_pos

    corpus = full_text[start_idx:]
    corpus_path = request_dir / f"{original_base}_corpus_{request_ts}.txt"
    with corpus_path.open("w", encoding="utf-8") as f:
        f.write(corpus)

    logger.info(
        "[Шаг 2] Корпус вырезан: длина=%d, старт=%d. Файл: %s",
        len(corpus),
        start_idx,
        corpus_path,
        extra={
            "stage": "md_corpus_built",
            "telegram_user_id": telegram_user_id,
            "telegram_username": "N/A",
            "request_id": request_id,
            "duration_seconds": 0,
            "model": "N/A",
            "api_key_id": "N/A",
            "file_name": corpus_path.name,
        },
    )

    PAGE_MARKER_RE = re.compile(r"@@@PAGE:\s*(\d+)@@@")
    page_markers: List[tuple[int, int]] = []
    for mm in PAGE_MARKER_RE.finditer(corpus):
        try:
            page_markers.append((mm.start(), int(mm.group(1))))
        except Exception:
            pass

    def get_page_for_pos(pos: int) -> Optional[int]:
        last_page: Optional[int] = None
        for mpos, pnum in page_markers:
            if mpos <= pos:
                last_page = pnum
            else:
                break
        return last_page

    # Разбивка на «сырые» чанки
    CHUNK_HEADING_RE = re.compile(
        rf"(?im)^[#\s>]*\s*(?:вкп\s*)?(?:\d{{1,3}}\.\s*)?.+?\s[-–—]+\s*{KEYWORD_PATTERN}\b.*$"
    )
    raw_matches = list(CHUNK_HEADING_RE.finditer(corpus))
    if not raw_matches:
        msg = "В корпусе не найдено ни одного заголовка кредита по ожидаемому шаблону."
        logger.error(
            msg,
            extra={
                "stage": "md_chunker_no_headings",
                "telegram_user_id": telegram_user_id,
                "telegram_username": "N/A",
                "request_id": request_id,
                "duration_seconds": 0,
                "model": "N/A",
                "api_key_id": "N/A",
                "file_name": md_path.name,
            },
        )
        raise ValueError(msg)

    def clean_heading_line(h: str) -> str:
        return fix_ooo_in_text(h.strip())

    raw_chunks: List[Dict[str, Any]] = []
    first_start = raw_matches[0].start()
    if first_start > 0:
        pre_text = fix_ooo_in_text(corpus[:first_start])
        raw_chunks.append(
            {
                "raw_id": 0,
                "start_idx": 0,
                "end_idx": first_start,
                "length": first_start,
                "heading": "",
                "text": pre_text,
                "is_preamble": True,
            }
        )

    for i, m in enumerate(raw_matches):
        s = m.start()
        e = raw_matches[i + 1].start() if i + 1 < len(raw_matches) else len(corpus)
        text_i = fix_ooo_in_text(corpus[s:e])
        heading_line = clean_heading_line(m.group(0)) or ""
        page = get_page_for_pos(s)
        raw_chunks.append(
            {
                "raw_id": i + 1,
                "start_idx": s,
                "end_idx": e,
                "length": e - s,
                "heading": heading_line,
                "text": text_i,
                "page": page,
            }
        )

    # Валидация покрытия корпуса
    def first_diff(a: str, b: str, ctx: int = 20):
        n = min(len(a), len(b))
        for i in range(n):
            if a[i] != b[i]:
                a_ctx = a[max(0, i - ctx) : i + ctx].replace("\n", "\\n")
                b_ctx = b[max(0, i - ctx) : i + ctx].replace("\n", "\\n")
                return i, a_ctx, b_ctx
        if len(a) != len(b):
            return n, f"a_end_len={len(a)}", f"b_end_len={len(b)}"
        return None

    reconstructed_raw = "".join(c["text"] for c in raw_chunks)
    sum_len_raw = sum(c["length"] for c in raw_chunks)
    diff_raw = first_diff(corpus, reconstructed_raw)
    if diff_raw is None and sum_len_raw == len(corpus):
        logger.info(
            "[Шаг 2] Покрытие корпуса сырыми чанками 1-в-1. Чанков=%d, сумма длин=%d.",
            len(raw_chunks),
            sum_len_raw,
            extra={
                "stage": "md_raw_chunks_coverage_ok",
                "telegram_user_id": telegram_user_id,
                "telegram_username": "N/A",
                "request_id": request_id,
                "duration_seconds": 0,
                "model": "N/A",
                "api_key_id": "N/A",
                "file_name": corpus_path.name,
            },
        )
    else:
        logger.error(
            "[Шаг 2] Сырые чанки не покрывают корпус 1-в-1. Сумма длин=%d, длина корпуса=%d.",
            sum_len_raw,
            len(corpus),
            extra={
                "stage": "md_raw_chunks_coverage_error",
                "telegram_user_id": telegram_user_id,
                "telegram_username": "N/A",
                "request_id": request_id,
                "duration_seconds": 0,
                "model": "N/A",
                "api_key_id": "N/A",
                "file_name": corpus_path.name,
            },
        )
        if diff_raw:
            pos, a_ctx, b_ctx = diff_raw
            logger.error(
                "Первая разница на позиции %d\nКорпус: %s\nСклейка: %s",
                pos,
                a_ctx,
                b_ctx,
                extra={
                    "stage": "md_raw_chunks_diff",
                    "telegram_user_id": telegram_user_id,
                    "telegram_username": "N/A",
                    "request_id": request_id,
                    "duration_seconds": 0,
                    "model": "N/A",
                    "api_key_id": "N/A",
                    "file_name": corpus_path.name,
                },
            )
        raise AssertionError("Потеря/искажение символов на этапе сырых чанков.")

    # Фильтр валидных чанков по «УИд договора»
    UID_RE = re.compile(r"(?i)уид\s*договора")
    NUM_PATTERNS = [
        r"^[#\s>]*\s*(?:вкп\s*)?(?P<num>\d{1,3})\s*[\.\)]",
        r"^[#\s>]*\s*№\s*(?P<num>\d{1,3})\b",
    ]

    def extract_heading_number(heading: str):
        for pat in NUM_PATTERNS:
            m = re.search(pat, heading, flags=re.IGNORECASE)
            if m:
                try:
                    return int(m.group("num"))
                except ValueError:
                    pass
        return None

    # Отдельно: исходный список с флагом has_uid, затем агрегируем
    raw_with_flags: List[Dict[str, Any]] = []
    invalid_chunks: List[Dict[str, Any]] = []

    for rc in raw_chunks:
        if rc.get("is_preamble"):
            continue
        has_uid = UID_RE.search(rc["text"]) is not None
        hnum = extract_heading_number(rc["heading"] or "")
        item = {"heading_num": hnum, **rc, "has_uid": has_uid}
        raw_with_flags.append(item)
        if not has_uid:
            invalid_chunks.append(item)

    # Лог статистики по исходным сырым чанкам
    valid_count = sum(1 for c in raw_with_flags if c["has_uid"])
    invalid_count = len(raw_with_flags) - valid_count

    logger.info(
        "[Шаг 2] Всего сырых чанков: %d; валидных (с 'УИд договора'): %d; без УИд: %d.",
        len(raw_with_flags),
        valid_count,
        invalid_count,
        extra={
            "stage": "md_chunks_stats",
            "telegram_user_id": telegram_user_id,
            "telegram_username": "N/A",
            "request_id": request_id,
            "duration_seconds": 0,
            "model": "N/A",
            "api_key_id": "N/A",
            "file_name": corpus_path.name,
        },
    )
    if invalid_chunks:
        logger.warning(
            "[Шаг 2] Есть чанки без 'УИд договора' — их текст будет приклеен к предыдущим валидным блокам.",
            extra={
                "stage": "md_chunks_invalid_present",
                "telegram_user_id": telegram_user_id,
                "telegram_username": "N/A",
                "request_id": request_id,
                "duration_seconds": 0,
                "model": "N/A",
                "api_key_id": "N/A",
                "file_name": corpus_path.name,
            },
        )
        for bad in invalid_chunks:
            logger.warning(
                "  - raw_id=%s, heading='%.120s'",
                bad["raw_id"],
                (bad["heading"] or "")[:120],
                extra={
                    "stage": "md_chunks_invalid_item",
                    "telegram_user_id": telegram_user_id,
                    "telegram_username": "N/A",
                    "request_id": request_id,
                    "duration_seconds": 0,
                    "model": "N/A",
                    "api_key_id": "N/A",
                    "file_name": corpus_path.name,
                },
            )

    # Сохранение сырых чанков 1-в-1 по корпусу (для отладки и проверки)
    df_raw = pd.DataFrame(
        [
            {
                "raw_id": c["raw_id"],
                "start_idx": c["start_idx"],
                "end_idx": c["end_idx"],
                "length": c["length"],
                "heading": c["heading"],
                "text": c["text"],
            }
            for c in raw_with_flags
        ]
    )
    df_raw["has_uid"] = [c["has_uid"] for c in raw_with_flags]
    df_raw["heading_num"] = [c["heading_num"] for c in raw_with_flags]

    raw_csv_path = request_dir / f"{original_base}_raw_chunks_{request_ts}.csv"
    df_raw.to_csv(raw_csv_path, index=False, encoding="utf-8")
    logger.info(
        "[Шаг 2] Файл сырых чанков: %s",
        raw_csv_path,
        extra={
            "stage": "md_raw_chunks_saved",
            "telegram_user_id": telegram_user_id,
            "telegram_username": "N/A",
            "request_id": request_id,
            "duration_seconds": 0,
            "model": "N/A",
            "api_key_id": "N/A",
            "file_name": raw_csv_path.name,
        },
    )

    # Формирование итоговых чанков: текст невалидных приклеивается к предыдущему валидному
    merged_chunks: List[Dict[str, Any]] = []
    current_valid: Optional[Dict[str, Any]] = None

    for c in raw_with_flags:
        if c["has_uid"]:
            # Закрываем предыдущий валидный, если был
            if current_valid is not None:
                merged_chunks.append(current_valid)
            # Начинаем новый валидный блок
            current_valid = {
                "heading_num": c["heading_num"],
                "start_idx": c["start_idx"],
                "end_idx": c["end_idx"],
                "length": c["length"],
                "heading": c["heading"],
                "text": c["text"],
                "page": c.get("page"),
            }
        else:
            # Невалидный блок: приклеиваем к предыдущему валидному, если он есть
            if current_valid is not None:
                # Склеиваем текст
                current_valid["text"] += c["text"]
                # Расширяем диапазон индексов и длину
                current_valid["end_idx"] = c["end_idx"]
                current_valid["length"] = current_valid["end_idx"] - current_valid["start_idx"]
            else:
                # Если невалидный идёт до первого валидного — считаем его "пропавшим",
                # но текст уже учтён в raw_chunks и покрытие корпуса не ломается.
                invalid_chunks.append(c)

    # Добавляем последний валидный
    if current_valid is not None:
        merged_chunks.append(current_valid)

    out_rows: List[Dict[str, Any]] = []
    for c in merged_chunks:
        out_rows.append(
            {
                "chunk_id": len(out_rows) + 1,
                "heading_num": c["heading_num"],
                "start_idx": c["start_idx"],
                "end_idx": c["end_idx"],
                "length": c["length"],
                "heading": c["heading"],
                "text": c["text"],
                "page": c.get("page"),
            }
        )

    df = pd.DataFrame(
        out_rows,
        columns=[
            "chunk_id",
            "heading_num",
            "start_idx",
            "end_idx",
            "length",
            "heading",
            "text",
            "page",
        ],
    )
    chunks_csv_path = request_dir / f"{original_base}_chunks_{request_ts}.csv"
    df.to_csv(chunks_csv_path, index=False, encoding="utf-8")
    logger.info(
        "[Шаг 2] Файл валидных чанков (CSV): %s",
        chunks_csv_path,
        extra={
            "stage": "md_chunks_csv_saved",
            "telegram_user_id": telegram_user_id,
            "telegram_username": "N/A",
            "request_id": request_id,
            "duration_seconds": 0,
            "model": "N/A",
            "api_key_id": "N/A",
            "file_name": chunks_csv_path.name,
        },
    )

    chunks_txt_path = request_dir / f"{original_base}_chunks_{request_ts}.txt"
    with chunks_txt_path.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(f"===== CHUNK {row['chunk_id']}: {row['heading']} =====\n")
            f.write(row["text"])
            if not row["text"].endswith("\n"):
                f.write("\n")
            f.write("\n")
    logger.info(
        "[Шаг 2] Файл валидных чанков (TXT): %s",
        chunks_txt_path,
        extra={
            "stage": "md_chunks_txt_saved",
            "telegram_user_id": telegram_user_id,
            "telegram_username": "N/A",
            "request_id": request_id,
            "duration_seconds": 0,
            "model": "N/A",
            "api_key_id": "N/A",
            "file_name": chunks_txt_path.name,
        },
    )

    invalid_txt_path: Optional[Path] = None
    if invalid_chunks:
        invalid_txt_path = request_dir / f"{original_base}_invalid_chunks_{request_ts}.txt"
        with invalid_txt_path.open("w", encoding="utf-8") as f:
            for c in invalid_chunks:
                f.write(
                    f"===== INVALID raw_id {c['raw_id']}: {c['heading']} =====\n"
                )
                f.write(c["text"])
                if not c["text"].endswith("\n"):
                    f.write("\n")
                f.write("\n")
        logger.info(
            "[Шаг 2] Файл с невалидными чанками: %s",
            invalid_txt_path,
            extra={
                "stage": "md_invalid_chunks_txt_saved",
                "telegram_user_id": telegram_user_id,
                "telegram_username": "N/A",
                "request_id": request_id,
                "duration_seconds": 0,
                "model": "N/A",
                "api_key_id": "N/A",
                "file_name": invalid_txt_path.name,
            },
        )

    total_duration = time.perf_counter() - total_start
    logger.info(
        "Шаг 2 (Markdown -> чанки) завершён за %.3f с.",
        total_duration,
        extra={
            "stage": "md_chunker_done",
            "telegram_user_id": telegram_user_id,
            "telegram_username": "N/A",
            "request_id": request_id,
            "duration_seconds": round(total_duration, 3),
            "model": "N/A",
            "api_key_id": "N/A",
            "file_name": md_path.name,
        },
    )

    return {
        "corpus": corpus_path,
        "raw_chunks": raw_csv_path,
        "chunks_csv": chunks_csv_path,
        "chunks_txt": chunks_txt_path,
        "invalid_chunks": invalid_txt_path,
    }

