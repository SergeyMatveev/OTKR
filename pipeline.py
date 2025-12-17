from __future__ import annotations

import time
from pathlib import Path
from logging import Logger

from md_chunker import run_markdown_to_chunks
from llm_pipeline import run_llm_pipeline
from logging_setup import FileStats


def process_nbki_md(
    md_path: Path,
    original_name: str,
    request_dir: Path,
    request_ts: str,
    logger: Logger,
    file_stats: FileStats | None = None,
) -> Path:
    """
    Запускает "хвост" боевого пайплайна для готового MD (как будто OCR уже был):
      2) Разбор Markdown и построение чанков.
      3) LLM-пайплайн для извлечения полей и формирования итогового CSV.

    Все промежуточные и итоговые файлы сохраняются в request_dir.
    Возвращает путь к итоговому CSV.
    """
    md_path = md_path.resolve()
    request_dir = request_dir.resolve()

    original_base = Path(original_name).stem
    telegram_user_id = request_dir.parent.name
    telegram_username = "N/A"
    request_id = request_dir.name

    total_start = time.perf_counter()

    logger.info(
        "Запуск MD-пайплайна NBKI для файла %s в директории %s",
        md_path,
        request_dir,
        extra={
            "stage": "pipeline_md_start",
            "telegram_user_id": telegram_user_id,
            "telegram_username": telegram_username,
            "request_id": request_id,
            "duration_seconds": 0,
            "model": "N/A",
            "api_key_id": "N/A",
            "file_name": original_name,
        },
    )

    step2_paths = run_markdown_to_chunks(
        md_path=md_path,
        request_dir=request_dir,
        original_base=original_base,
        request_ts=request_ts,
        logger=logger,
    )

    chunks_csv_path = step2_paths["chunks_csv"]
    result_csv_path = run_llm_pipeline(
        chunks_csv_path=chunks_csv_path,
        original_pdf_name=original_name,
        request_dir=request_dir,
        request_ts=request_ts,
        logger=logger,
        file_stats=file_stats,
    )

    total_duration = time.perf_counter() - total_start
    logger.info(
        "MD-пайплайн NBKI успешно завершён. Итоговый CSV: %s (длительность %.3f с.)",
        result_csv_path,
        total_duration,
        extra={
            "stage": "pipeline_md_done",
            "telegram_user_id": telegram_user_id,
            "telegram_username": telegram_username,
            "request_id": request_id,
            "duration_seconds": round(total_duration, 3),
            "model": "N/A",
            "api_key_id": "N/A",
            "file_name": original_name,
        },
    )

    return result_csv_path


def process_nbki_pdf(
    pdf_path: Path,
    request_dir: Path,
    request_ts: str,
    logger: Logger,
    file_stats: FileStats | None = None,
) -> Path:
    """
    Запускает полный пайплайн обработки отчёта НБКИ для одного PDF:
      1) Обрезка и OCR (Mistral OCR).
      2) Разбор Markdown и построение чанков.
      3) LLM-пайплайн для извлечения полей и формирования итогового CSV.

    Все промежуточные и итоговые файлы сохраняются в request_dir.
    Возвращает путь к итоговому CSV.
    """
    from pdf_crop_ocr import run_pdf_crop_and_ocr

    pdf_path = pdf_path.resolve()
    request_dir = request_dir.resolve()

    original_base = pdf_path.stem
    telegram_user_id = request_dir.parent.name
    telegram_username = "N/A"
    request_id = request_dir.name

    total_start = time.perf_counter()

    logger.info(
        "Запуск полного пайплайна NBKI для файла %s в директории %s",
        pdf_path,
        request_dir,
        extra={
            "stage": "pipeline_start",
            "telegram_user_id": telegram_user_id,
            "telegram_username": telegram_username,
            "request_id": request_id,
            "duration_seconds": 0,
            "model": "N/A",
            "api_key_id": "N/A",
            "file_name": pdf_path.name,
        },
    )

    step1_paths = run_pdf_crop_and_ocr(pdf_path, request_dir, request_ts, logger, file_stats=file_stats)
    md_path = step1_paths["ocr_md"] or step1_paths["ocr_json"]
    if md_path is None:
        raise RuntimeError("run_pdf_crop_and_ocr не вернул путь к OCR-файлу.")

    result_csv_path = process_nbki_md(
        md_path=md_path,
        original_name=pdf_path.name,
        request_dir=request_dir,
        request_ts=request_ts,
        logger=logger,
        file_stats=file_stats,
    )

    total_duration = time.perf_counter() - total_start
    logger.info(
        "Полный пайплайн NBKI успешно завершён для файла %s. Итоговый CSV: %s (длительность %.3f с.)",
        pdf_path,
        result_csv_path,
        total_duration,
        extra={
            "stage": "pipeline_done",
            "telegram_user_id": telegram_user_id,
            "telegram_username": telegram_username,
            "request_id": request_id,
            "duration_seconds": round(total_duration, 3),
            "model": "N/A",
            "api_key_id": "N/A",
            "file_name": pdf_path.name,
        },
    )

    return result_csv_path



