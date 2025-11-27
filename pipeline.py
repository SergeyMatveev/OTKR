from __future__ import annotations

from pathlib import Path
from logging import Logger

from pdf_crop_ocr import run_pdf_crop_and_ocr
from md_chunker import run_markdown_to_chunks
from llm_pipeline import run_llm_pipeline


def process_nbki_pdf(
    pdf_path: Path,
    request_dir: Path,
    request_ts: str,
    logger: Logger,
) -> Path:
    """
    Полный пайплайн обработки одного PDF:
      PDF -> обрезка -> OCR -> MD -> корпус/чанки -> LLM -> CSV.

    Все артефакты и исходный файл должны находится в request_dir.
    Возвращает путь к итоговому CSV.
    """
    pdf_path = pdf_path.resolve()
    request_dir = request_dir.resolve()
    original_base = pdf_path.stem

    logger.info(
        "Запуск полного пайплайна NBKI для файла %s в директории %s",
        pdf_path,
        request_dir,
    )

    # Шаг 1: обрезка PDF и OCR
    step1 = run_pdf_crop_and_ocr(pdf_path=pdf_path, request_dir=request_dir, logger=logger)
    ocr_md_path = step1.get("ocr_md")
    if ocr_md_path is None:
        raise RuntimeError("OCR не вернул Markdown (pages->markdown отсутствует). Нечего парсить.")

    # Шаг 2: MD -> корпус -> чанки
    step2 = run_markdown_to_chunks(
        md_path=ocr_md_path,
        request_dir=request_dir,
        original_base=original_base,
        request_ts=request_ts,
        logger=logger,
    )
    chunks_csv_path = step2["chunks_csv"]

    # Шаг 3: LLM-пайплайн
    result_csv_path = run_llm_pipeline(
        chunks_csv_path=chunks_csv_path,
        original_pdf_name=pdf_path.name,
        request_dir=request_dir,
        request_ts=request_ts,
        logger=logger,
    )

    logger.info(
        "Полный пайплайн NBKI успешно завершён. Итоговый CSV: %s",
        result_csv_path,
    )
    return result_csv_path
