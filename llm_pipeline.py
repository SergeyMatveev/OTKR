from __future__ import annotations

import json
import random
import time
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List

import requests
import pandas as pd

from logging import Logger

from config import FREE_API_KEY, PAID_API_KEY, GPT_API_KEY, make_llm_io_path
from logging_setup import FileStats

from llm_prompts import SYSTEM_PROMPT_STAGE2, PROMPT_STAGE5, SYSTEM_PROMPT_INN_FALLBACK, SYSTEM_PROMPT_STAGE4
from llm_support import (
    PHRASE_SVEDENIYA_ISPOLN,
    PHRASE_SOURCE_CREDIT_HISTORY,
    PHRASE_SROCHNAYA_ZADOLZH,
    PHRASE_POKUPATEL_BLOCK_START,
    PHRASE_POKUPATEL_BLOCK_END,
    OUTPUT_COLUMNS,
    MistralChatClient,
    nd_normalize,
    extract_header_line,
    extract_short_title,
    slice_until_phrase,
    slice_from_phrase_to_end,
    slice_500_before_phrase,
    slice_between,
    contains_10_digits_sequence,
    extract_inn_10_digits,
    parse_stage2_response,
    parse_stage4_response,
    is_valid_debt_value,
    extract_urgent_debt,
    parse_stage5_response,
    openai_responses_call,
)


# ---------- основной LLM-пайплайн ----------

SMALL_TASKS_FULL_PARALLEL = 12
MAX_LLM_PARALLEL_WORKERS = 10


def _calc_max_workers(task_count: int) -> int:
    if task_count <= 0:
        return 1
    if task_count <= SMALL_TASKS_FULL_PARALLEL:
        return task_count
    return MAX_LLM_PARALLEL_WORKERS


def run_llm_pipeline(
    chunks_csv_path: Path,
    original_pdf_name: str,
    request_dir: Path,
    request_ts: str,
    logger: Logger,
    file_stats: Optional[FileStats] = None,
) -> Path:
    """
    Этап 3:
      - Загружает chunks.csv,
      - Прогоняет все блоки через три этапа Mistral,
      - Сохраняет финальный CSV с результатами в:
        data/<telegram_id>/<base>_<request_ts>/<base>_result_<request_ts>.csv

    Возвращает путь к итоговому CSV.
    """
    if not FREE_API_KEY:
        raise RuntimeError("FREE_API_KEY не задан в .env, невозможны запросы к Mistral.")

    chunks_csv_path = chunks_csv_path.resolve()
    request_dir = request_dir.resolve()

    telegram_user_id = request_dir.parent.name
    telegram_username = "N/A"
    request_id = request_dir.name

    llm_io_path = make_llm_io_path(request_dir, original_pdf_name, request_ts)

    total_start = time.perf_counter()

    try:
        chunks_df = pd.read_csv(chunks_csv_path, encoding="utf-8-sig")
    except Exception:
        chunks_df = pd.read_csv(chunks_csv_path, encoding="utf-8")

    if "text" not in chunks_df.columns:
        msg = "В файле chunks.csv отсутствует колонка 'text'."
        logger.error(
            msg,
            extra={
                "stage": "llm_pipeline_init_error",
                "telegram_user_id": telegram_user_id,
                "telegram_username": telegram_username,
                "request_id": request_id,
                "duration_seconds": 0,
                "model": "N/A",
                "api_key_id": "N/A",
                "file_name": original_pdf_name,
            },
        )
        raise ValueError(msg)

    logger.info(
        "LLM-пайплайн: загружено %d блок(ов) из %s.",
        len(chunks_df),
        chunks_csv_path,
        extra={
            "stage": "llm_pipeline_start",
            "telegram_user_id": telegram_user_id,
            "telegram_username": telegram_username,
            "request_id": request_id,
            "duration_seconds": 0,
            "model": "N/A",
            "api_key_id": "N/A",
            "file_name": original_pdf_name,
        },
    )

    # Базовые поля
    base_rows = []
    for _, row in chunks_df.iterrows():
        block_text = str(row.get("text", "") or "")
        title = extract_header_line(block_text)
        short_name = extract_short_title(title)

        page_raw = row.get("page", None)
        page_val = "Н/Д"
        if page_raw is not None and not pd.isna(page_raw):
            try:
                page_val_int = int(float(str(page_raw).strip()))
                page_val = str(page_val_int) if page_val_int > 0 else "Н/Д"
            except Exception:
                page_val = "Н/Д"

        base_rows.append(
            {
                "Номер": str(len(base_rows) + 1),
                "Заголовок блока": title,
                "Короткое название": short_name,
                "ИНН": "Н/Д",
                "Прекращение обязательства": "Н/Д",
                "Дата сделки": "Н/Д",
                "Сумма и валюта": "Н/Д",
                "Сумма задолженности": "Н/Д",
                "УИд договора": "Не найдено",
                "Номер договора": "Н/Д",
                "Приобретатель прав кредитора": "Н/Д",
                "ИНН приобретателя прав кредитора": "Н/Д",
                "Страница": page_val,
            }
        )
    main_df = pd.DataFrame(base_rows, columns=OUTPUT_COLUMNS)
    logger.info(
        "Этап 1 LLM: базовые поля сформированы.",
        extra={
            "stage": "llm_stage1_done",
            "telegram_user_id": telegram_user_id,
            "telegram_username": telegram_username,
            "request_id": request_id,
            "duration_seconds": 0,
            "model": "N/A",
            "api_key_id": "N/A",
            "file_name": original_pdf_name,
        },
    )

    # Клиент Mistral с переключением FREE -> PAID
    base_log_extra = {
        "telegram_user_id": telegram_user_id,
        "telegram_username": telegram_username,
        "request_id": request_id,
        "file_name": original_pdf_name,
        "llm_io_path": llm_io_path,
    }
    chat_client = MistralChatClient(
        FREE_API_KEY,
        PAID_API_KEY,
        logger,
        base_log_extra=base_log_extra,
        stats=file_stats,
    )

    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Этап 2: 4 поля
    logger.info(
        "Этап 2 LLM: извлечение полей 'Прекращение обязательства', 'Дата сделки', 'Сумма и валюта', 'УИд договора'.",
        extra={
            "stage": "llm_stage2_start",
            "telegram_user_id": telegram_user_id,
            "telegram_username": telegram_username,
            "request_id": request_id,
            "duration_seconds": 0,
            "model": "gpt-5.1",
            "api_key_id": "GPT_API_KEY",
            "file_name": original_pdf_name,
        },
    )
    stage2_start = time.perf_counter()

    if not GPT_API_KEY:
        raise RuntimeError("GPT_API_KEY не задан в .env, невозможны запросы к OpenAI (Stage2).")

    def _call_stage2(payload: str) -> str:
        resp_text, ok, _status_code, error_type, _error_code = openai_responses_call(
            api_key=GPT_API_KEY,
            model="gpt-5.1",
            instructions=SYSTEM_PROMPT_STAGE2,
            user_content=payload,
            service_tier="flex",
            timeout_seconds=10,
            llm_io_path=llm_io_path,
            base_log_extra=base_log_extra,
            stats=file_stats,
        )

        if (not ok) and (error_type == "Timeout"):
            resp_text2, ok2, _status_code2, _error_type2, _error_code2 = openai_responses_call(
                api_key=GPT_API_KEY,
                model="gpt-5.1",
                instructions=SYSTEM_PROMPT_STAGE2,
                user_content=payload,
                service_tier=None,
                timeout_seconds=30,
                llm_io_path=llm_io_path,
                base_log_extra=base_log_extra,
                stats=file_stats,
            )
            if ok2:
                return resp_text2
            return ""

        if ok:
            return resp_text
        return ""

    idxs_stage2: List[int] = list(chunks_df.index)
    if idxs_stage2:
        futures_map = {}
        with ThreadPoolExecutor(max_workers=_calc_max_workers(len(idxs_stage2))) as ex:
            for idx in idxs_stage2:
                number = main_df.at[idx, "Номер"]
                title = main_df.at[idx, "Заголовок блока"]
                full_block = str(chunks_df.at[idx, "text"] or "")
                part_for_stage2 = slice_until_phrase(full_block, PHRASE_SVEDENIYA_ISPOLN)

                futures_map[
                    ex.submit(
                        _call_stage2,
                        part_for_stage2,
                    )
                ] = idx

            for f in as_completed(futures_map):
                idx = futures_map[f]
                try:
                    resp_text = f.result()
                except Exception as e:
                    logger.error(
                        "[stage2] Ошибка будущего результата для строки %d: %s",
                        idx,
                        e,
                        exc_info=True,
                        extra={
                            "stage": "llm_stage2_future_error",
                            "telegram_user_id": telegram_user_id,
                            "telegram_username": telegram_username,
                            "request_id": request_id,
                            "duration_seconds": 0,
                            "model": "gpt-5.1",
                            "api_key_id": "GPT_API_KEY",
                            "file_name": original_pdf_name,
                        },
                    )
                    resp_text = ""
                parsed = parse_stage2_response(resp_text)
                main_df.at[idx, "Прекращение обязательства"] = parsed[
                    "Прекращение обязательства"
                ]
                main_df.at[idx, "Дата сделки"] = parsed["Дата сделки"]
                main_df.at[idx, "Сумма и валюта"] = parsed["Сумма и валюта"]
                main_df.at[idx, "УИд договора"] = parsed["УИд договора"]
                main_df.at[idx, "Номер договора"] = parsed["Номер договора"]

    stage2_duration = time.perf_counter() - stage2_start
    need_more = main_df["Прекращение обязательства"] == "Н/Д"
    logger.info(
        "Этап 2 LLM: завершён за %.3f с. В Этапы 3–5 пойдут %d строк(и).",
        stage2_duration,
        need_more.sum(),
        extra={
            "stage": "llm_stage2_done",
            "telegram_user_id": telegram_user_id,
            "telegram_username": telegram_username,
            "request_id": request_id,
            "duration_seconds": round(stage2_duration, 3),
            "model": "gpt-5.1",
            "api_key_id": "N/A",
            "file_name": original_pdf_name,
        },
    )

    # Этап 3: ИНН (10 цифр) локально + fallback через LLM
    logger.info(
        "Этап 3: извлекаем ИНН (10 цифр) локально, затем делаем fallback через LLM только для Н/Д.",
        extra={
            "stage": "llm_stage3_start",
            "telegram_user_id": telegram_user_id,
            "telegram_username": telegram_username,
            "request_id": request_id,
            "duration_seconds": 0,
            "model": "mistral-small-latest",
            "api_key_id": "FREE_API_KEY",
            "file_name": original_pdf_name,
        },
    )
    stage3_start = time.perf_counter()
    # 3.1. Локальный поиск ИНН по regex (как было)
    for idx in main_df.index[need_more]:
        full_block = str(chunks_df.at[idx, "text"] or "")
        slice_inn = slice_from_phrase_to_end(full_block, PHRASE_SOURCE_CREDIT_HISTORY)
        main_df.at[idx, "ИНН"] = extract_inn_10_digits(slice_inn)

    # 3.2. LLM-fallback только для тех, у кого осталось Н/Д
    inn_nd_mask = (main_df["ИНН"] == "Н/Д") & need_more
    idxs_inn_llm: List[int] = list(main_df.index[inn_nd_mask])

    if idxs_inn_llm:
        logger.info(
            "Этап 3: LLM-fallback для ИНН будет выполнен для %d строк.",
            len(idxs_inn_llm),
            extra={
                "stage": "llm_stage3_fallback_start",
                "telegram_user_id": telegram_user_id,
                "telegram_username": telegram_username,
                "request_id": request_id,
                "duration_seconds": 0,
                "model": "mistral-small-latest",
                "api_key_id": "FREE_API_KEY",
                "file_name": original_pdf_name,
            },
        )
        futures_map_inn = {}
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=_calc_max_workers(len(idxs_inn_llm))) as ex:
            for idx in idxs_inn_llm:
                number = main_df.at[idx, "Номер"]
                title = main_df.at[idx, "Заголовок блока"]
                full_block = str(chunks_df.at[idx, "text"] or "")
                futures_map_inn[
                    ex.submit(
                        chat_client.chat,
                        "mistral-small-latest",
                        SYSTEM_PROMPT_INN_FALLBACK,
                        full_block,
                        "stage3_inn_fallback",
                        idx,
                        number,
                        title,
                    )
                ] = idx

            for f in as_completed(futures_map_inn):
                idx = futures_map_inn[f]
                try:
                    resp_text = f.result().strip()
                except Exception as e:
                    logger.error(
                        "[stage3_inn_fallback] Ошибка будущего результата для строки %d: %s",
                        idx,
                        e,
                        exc_info=True,
                        extra={
                            "stage": "llm_stage3_fallback_future_error",
                            "telegram_user_id": telegram_user_id,
                            "telegram_username": telegram_username,
                            "request_id": request_id,
                            "duration_seconds": 0,
                            "model": "mistral-small-latest",
                            "api_key_id": "N/A",
                            "file_name": original_pdf_name,
                        },
                    )
                    resp_text = ""
                # Парсим ответ: либо 10 цифр, либо Н/Д
                m = re.fullmatch(r"\d{10}", resp_text)
                if m:
                    main_df.at[idx, "ИНН"] = m.group(0)
                else:
                    # На всякий случай нормализуем Н/Д, но поле оставляем "Н/Д", если не 10 цифр
                    main_df.at[idx, "ИНН"] = "Н/Д"

    for idx in main_df.index[~need_more]:
        for col in [
            "ИНН",
            "Сумма задолженности",
            "Приобретатель прав кредитора",
            "ИНН приобретателя прав кредитора",
        ]:
            main_df.at[idx, col] = nd_normalize(main_df.at[idx, col])

    stage3_duration = time.perf_counter() - stage3_start
    logger.info(
        "Этап 3 (локальный ИНН + LLM-fallback) завершён за %.3f с.",
        stage3_duration,
        extra={
            "stage": "llm_stage3_done",
            "telegram_user_id": telegram_user_id,
            "telegram_username": telegram_username,
            "request_id": request_id,
            "duration_seconds": round(stage3_duration, 3),
            "model": "mistral-small-latest",
            "api_key_id": "FREE_API_KEY",
            "file_name": original_pdf_name,
        },
    )

    # Этап 4: срочная задолженность (GPT-5.1 flex -> fallback)
    logger.info(
        "Этап 4: расчёт 'Сумма задолженности' (GPT-5.1 flex -> fallback).",
        extra={
            "stage": "llm_stage4_start",
            "telegram_user_id": telegram_user_id,
            "telegram_username": telegram_username,
            "request_id": request_id,
            "duration_seconds": 0,
            "model": "N/A",
            "api_key_id": "N/A",
            "file_name": original_pdf_name,
        },
    )
    stage4_start = time.perf_counter()

    if not GPT_API_KEY:
        raise RuntimeError("GPT_API_KEY не задан в .env, невозможны запросы к OpenAI (Stage4).")

    idxs_stage4: List[int] = list(main_df.index[need_more])
    stage4_payloads: Dict[int, str] = {}

    for idx in idxs_stage4:
        full_block = str(chunks_df.at[idx, "text"] or "")
        slice_500 = slice_500_before_phrase(full_block, PHRASE_SROCHNAYA_ZADOLZH)
        if not slice_500:
            main_df.at[idx, "Сумма задолженности"] = "Н/Д"
            continue
        stage4_payloads[idx] = slice_500

    def _extract_first_number_or_nd(s: str) -> str:
        if not s:
            return "Н/Д"
        m = re.search(r"\d[\d \u00A0\u202F]*(?:[.,]\d{2})?", s)
        if not m:
            return "Н/Д"
        num = m.group(0)
        num = num.replace("\u00A0", " ").replace("\u202F", " ").strip()
        return num or "Н/Д"

    def _call_stage4_model(model_name: str, payload: str) -> str:
        resp_text, ok, status_code, error_type, _error_code = openai_responses_call(
            api_key=GPT_API_KEY,
            model=model_name,
            instructions=SYSTEM_PROMPT_STAGE4,
            user_content=payload,
            service_tier="flex",
            timeout_seconds=15,
            llm_io_path=llm_io_path,
            base_log_extra=base_log_extra,
            stats=file_stats,
        )

        if (not ok) and ((status_code == 429) or (error_type == "Timeout")):
            resp_text2, ok2, _status_code2, _error_type2, _error_code2 = openai_responses_call(
                api_key=GPT_API_KEY,
                model=model_name,
                instructions=SYSTEM_PROMPT_STAGE4,
                user_content=payload,
                service_tier=None,
                timeout_seconds=30,
                llm_io_path=llm_io_path,
                base_log_extra=base_log_extra,
                stats=file_stats,
            )
            if ok2:
                resp_text = resp_text2
            else:
                resp_text = ""

        value_raw = parse_stage4_response(resp_text)
        num_or_nd = _extract_first_number_or_nd(value_raw)
        value = nd_normalize(num_or_nd)
        if is_valid_debt_value(value):
            return value
        return "Н/Д"

    def _stage4_worker(idx: int, payload: str) -> str:
        value_51 = _call_stage4_model("gpt-5.1", payload)
        if value_51 != "Н/Д":
            return value_51

        value_52 = _call_stage4_model("gpt-5.2", payload)
        if value_52 != "Н/Д":
            return value_52

        return "Н/Д"

    if stage4_payloads:
        futures_map_stage4: Dict[object, int] = {}
        from concurrent.futures import ThreadPoolExecutor, as_completed

        max_workers_stage4 = min(10, _calc_max_workers(len(stage4_payloads)))
        with ThreadPoolExecutor(max_workers=max_workers_stage4) as ex:
            for idx, payload in stage4_payloads.items():
                futures_map_stage4[ex.submit(_stage4_worker, idx, payload)] = idx

            for f in as_completed(futures_map_stage4):
                idx = futures_map_stage4[f]
                try:
                    value = f.result()
                except Exception:
                    value = "Н/Д"
                main_df.at[idx, "Сумма задолженности"] = value

    stage4_duration = time.perf_counter() - stage4_start
    logger.info(
        "Этап 4: расчёт 'Сумма задолженности' завершён за %.3f с.",
        stage4_duration,
        extra={
            "stage": "llm_stage4_done",
            "telegram_user_id": telegram_user_id,
            "telegram_username": telegram_username,
            "request_id": request_id,
            "duration_seconds": round(stage4_duration, 3),
            "model": "N/A",
            "api_key_id": "N/A",
            "file_name": original_pdf_name,
        },
    )

    # Этап 5: приобретатель прав кредитора
    logger.info(
        "Этап 5 LLM: параллельные запросы к ЛЛМ для 'Приобретатель прав кредитора' и 'ИНН приобретателя прав кредитора'.",
        extra={
            "stage": "llm_stage5_start",
            "telegram_user_id": telegram_user_id,
            "telegram_username": telegram_username,
            "request_id": request_id,
            "duration_seconds": 0,
            "model": "mistral-small-latest",
            "api_key_id": "N/A",
            "file_name": original_pdf_name,
        },
    )
    stage5_start = time.perf_counter()
    idxs_stage5: List[int] = []
    user_payloads_stage5: Dict[int, str] = {}
    context_stage5: Dict[int, str] = {}

    for idx in main_df.index[need_more]:
        full_block = str(chunks_df.at[idx, "text"] or "")
        between = slice_between(
            full_block, PHRASE_POKUPATEL_BLOCK_START, PHRASE_POKUPATEL_BLOCK_END
        )

        if not between or not contains_10_digits_sequence(between):
            main_df.at[idx, "Приобретатель прав кредитора"] = "Н/Д"
            main_df.at[idx, "ИНН приобретателя прав кредитора"] = "Н/Д"
            continue

        idxs_stage5.append(idx)
        context_stage5[idx] = between
        user_payloads_stage5[idx] = (
            f"{PROMPT_STAGE5}\n\nТЕКСТ ДЛЯ АНАЛИЗА:\n\n{between}"
        )

    if idxs_stage5:
        futures_map = {}
        with ThreadPoolExecutor(max_workers=_calc_max_workers(len(idxs_stage5))) as ex:
            for idx in idxs_stage5:
                number = main_df.at[idx, "Номер"]
                title = main_df.at[idx, "Заголовок блока"]
                futures_map[
                    ex.submit(
                        chat_client.chat,
                        "mistral-small-latest",
                        "",
                        user_payloads_stage5[idx],
                        "stage5",
                        idx,
                        number,
                        title,
                    )
                ] = idx

            for f in as_completed(futures_map):
                idx = futures_map[f]
                try:
                    resp_text = f.result()
                except Exception as e:
                    logger.error(
                        "[stage5] Ошибка будущего результата для строки %d: %s",
                        idx,
                        e,
                        exc_info=True,
                        extra={
                            "stage": "llm_stage5_future_error",
                            "telegram_user_id": telegram_user_id,
                            "telegram_username": telegram_username,
                            "request_id": request_id,
                            "duration_seconds": 0,
                            "model": "mistral-small-latest",
                            "api_key_id": "N/A",
                            "file_name": original_pdf_name,
                        },
                    )
                    resp_text = ""
                name, inn = parse_stage5_response(
                    resp_text, context_text=context_stage5.get(idx, "")
                )
                main_df.at[idx, "Приобретатель прав кредитора"] = nd_normalize(name)
                main_df.at[idx, "ИНН приобретателя прав кредитора"] = nd_normalize(inn)

    stage5_duration = time.perf_counter() - stage5_start
    logger.info(
        "Этап 5 LLM: завершён за %.3f с.",
        stage5_duration,
        extra={
            "stage": "llm_stage5_done",
            "telegram_user_id": telegram_user_id,
            "telegram_username": telegram_username,
            "request_id": request_id,
            "duration_seconds": round(stage5_duration, 3),
            "model": "mistral-small-latest",
            "api_key_id": "N/A",
            "file_name": original_pdf_name,
        },
    )

    for col in OUTPUT_COLUMNS:
        if col == "УИд договора":
            main_df[col] = main_df[col].fillna("Не найдено")
        else:
            main_df[col] = main_df[col].fillna("Н/Д").apply(nd_normalize)

    base_pdf = Path(original_pdf_name).stem
    result_csv_name = f"{base_pdf}_result_{request_ts}.csv"
    result_csv_path = request_dir / result_csv_name
    main_df.to_csv(result_csv_path, index=False, encoding="utf-8-sig")

    total_duration = time.perf_counter() - total_start
    logger.info(
        "LLM-пайплайн завершён. Итоговый CSV: %s (длительность %.3f с.)",
        result_csv_path,
        total_duration,
        extra={
            "stage": "llm_pipeline_done",
            "telegram_user_id": telegram_user_id,
            "telegram_username": telegram_username,
            "request_id": request_id,
            "duration_seconds": round(total_duration, 3),
            "model": "mistral-small-latest",
            "api_key_id": "N/A",
            "file_name": original_pdf_name,
        },
    )

    return result_csv_path

