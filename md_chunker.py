from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional

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
      - Фильтрует валидные чанки (с «УИд договора»),
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

    logger.info("Шаг 2: разбор Markdown и построение чанков из файла %s", md_path)

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

    # Обрезка корпуса: с первого заголовка договора до конца файла
    KEYWORD_PATTERN = (
        r"(?:Договор\s+займа(?:\s*\(кредита\))?|Микрокредит|Микрозайм|Микрозаем)"
    )

    START_HEADING_RE = re.compile(
        rf"(?im)^[#\s>]*\s*\d{{1,3}}\.\s*.+?\s[-–—]+\s*{KEYWORD_PATTERN}\b.*$"
    )
    start_match = START_HEADING_RE.search(full_text)
    if not start_match:
        raise ValueError("Не найден старт корпуса: 'N. Название - Договор займа/...'.")
    start_idx = start_match.start()

    corpus = full_text[start_idx:]
    corpus_path = request_dir / f"{original_base}_corpus_{request_ts}.txt"
    with corpus_path.open("w", encoding="utf-8") as f:
        f.write(corpus)

    logger.info(
        "[Шаг 2] Корпус вырезан: длина=%d, старт=%d. Файл: %s",
        len(corpus),
        start_idx,
        corpus_path,
    )

    # Разбивка на «сырые» чанки
    CHUNK_HEADING_RE = re.compile(
        rf"(?im)^[#\s>]*\s*(?:вкп\s*)?(?:\d{{1,3}}\.\s*)?.+?\s[-–—]+\s*{KEYWORD_PATTERN}\b.*$"
    )
    raw_matches = list(CHUNK_HEADING_RE.finditer(corpus))
    if not raw_matches:
        raise ValueError(
            "В корпусе не найдено ни одного заголовка кредита по ожидаемому шаблону."
        )

    def clean_heading_line(h: str) -> str:
        return fix_ooo_in_text(h.strip())

    raw_chunks = []
    for i, m in enumerate(raw_matches):
        s = m.start()
        e = raw_matches[i + 1].start() if i + 1 < len(raw_matches) else len(corpus)
        text_i = fix_ooo_in_text(corpus[s:e])
        heading_line = clean_heading_line(m.group(0)) or ""
        raw_chunks.append(
            {
                "raw_id": i + 1,
                "start_idx": s,
                "end_idx": e,
                "length": e - s,
                "heading": heading_line,
                "text": text_i,
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
        )
    else:
        logger.error(
            "[Шаг 2] Сырые чанки не покрывают корпус 1-в-1. Сумма длин=%d, длина корпуса=%d.",
            sum_len_raw,
            len(corpus),
        )
        if diff_raw:
            pos, a_ctx, b_ctx = diff_raw
            logger.error(
                "Первая разница на позиции %d\nКорпус: %s\nСклейка: %s",
                pos,
                a_ctx,
                b_ctx,
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

    valid_chunks, invalid_chunks = [], []
    for rc in raw_chunks:
        has_uid = UID_RE.search(rc["text"]) is not None
        hnum = extract_heading_number(rc["heading"] or "")
        item = {"heading_num": hnum, **rc, "has_uid": has_uid}
        (valid_chunks if has_uid else invalid_chunks).append(item)

    logger.info(
        "[Шаг 2] Всего сырых чанков: %d; валидных: %d; пропущено: %d.",
        len(raw_chunks),
        len(valid_chunks),
        len(invalid_chunks),
    )
    if invalid_chunks:
        logger.warning("[Шаг 2] Есть чанки без 'УИд договора' — пропущены.")
        for bad in invalid_chunks:
            logger.warning(
                "  - raw_id=%s, heading='%.120s'",
                bad["raw_id"],
                (bad["heading"] or "")[:120],
            )

    # Сохранение артефактов
    df_raw = pd.DataFrame(raw_chunks)
    df_raw["has_uid"] = [UID_RE.search(t) is not None for t in df_raw["text"]]
    df_raw["heading_num"] = [
        extract_heading_number(h or "") for h in df_raw["heading"]
    ]

    raw_csv_path = request_dir / f"{original_base}_raw_chunks_{request_ts}.csv"
    df_raw.to_csv(raw_csv_path, index=False, encoding="utf-8")
    logger.info("[Шаг 2] Файл сырых чанков: %s", raw_csv_path)

    out_rows = []
    for c in valid_chunks:
        out_rows.append(
            {
                "chunk_id": len(out_rows) + 1,
                "heading_num": c["heading_num"],
                "start_idx": c["start_idx"],
                "end_idx": c["end_idx"],
                "length": c["length"],
                "heading": c["heading"],
                "text": c["text"],
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
        ],
    )
    chunks_csv_path = request_dir / f"{original_base}_chunks_{request_ts}.csv"
    df.to_csv(chunks_csv_path, index=False, encoding="utf-8")
    logger.info("[Шаг 2] Файл валидных чанков (CSV): %s", chunks_csv_path)

    chunks_txt_path = request_dir / f"{original_base}_chunks_{request_ts}.txt"
    with chunks_txt_path.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(f"===== CHUNK {row['chunk_id']}: {row['heading']} =====\n")
            f.write(row["text"])
            if not row["text"].endswith("\n"):
                f.write("\n")
            f.write("\n")
    logger.info("[Шаг 2] Файл валидных чанков (TXT): %s", chunks_txt_path)

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
        logger.info("[Шаг 2] Файл с невалидными чанками: %s", invalid_txt_path)

    return {
        "corpus": corpus_path,
        "raw_chunks": raw_csv_path,
        "chunks_csv": chunks_csv_path,
        "chunks_txt": chunks_txt_path,
        "invalid_chunks": invalid_txt_path,
    }
