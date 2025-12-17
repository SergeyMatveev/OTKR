from __future__ import annotations

import json
import random
import time
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List

import requests

from logging import Logger

from logging_setup import log_llm_call
from logging_setup import FileStats

# Константы/фразы границ
PHRASE_SVEDENIYA_ISPOLN = "Сведения об исполнении обязательства"
PHRASE_SOURCE_CREDIT_HISTORY = "Сведения об источнике формирования кредитной истории"
PHRASE_SROCHNAYA_ZADOLZH = "Срочная задолженность"
PHRASE_POKUPATEL_BLOCK_START = (
    "Сведения о приобретателе прав кредитора и обслуживающей организации"
)
PHRASE_POKUPATEL_BLOCK_END = "Сведения о прекращении передачи сведений"

MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

OUTPUT_COLUMNS = [
    "Номер",
    "Заголовок блока",
    "Короткое название",
    "ИНН",
    "Прекращение обязательства",
    "Дата сделки",
    "Сумма и валюта",
    "Сумма задолженности",
    "УИд договора",
    "Приобретатель прав кредитора",
    "ИНН приобретателя прав кредитора",
]


# ---------- утилиты для нормализации/парсинга ----------

def nd_normalize(val: Optional[str]) -> str:
    if not val:
        return "Н/Д"
    v = str(val).strip()
    v = v.replace("H/Д", "Н/Д").replace("Н/д", "Н/Д").replace("н/д", "Н/Д")
    return "Н/Д" if v.upper() == "Н/Д" else v


def ensure_allowed_prekr(val: str) -> str:
    v = nd_normalize(val)
    allowed = {"Н/Д", "Надлежащее исполнение обязательства"}
    return v if v in allowed else "Н/Д"


def validate_uid(uid: str) -> str:
    if not uid:
        return "Не найдено"
    candidate = uid.strip()
    pattern = (
        r"^([0-9a-f]{8})-([0-9a-f]{4})-([0-9a-f]{4})-([0-9a-f]{4})-([0-9a-f]{12})-([0-9a-z]{1})$"
    )
    import re as _re

    return candidate if _re.match(pattern, candidate) else "Не найдено"


def extract_header_line(block_text: str) -> str:
    for raw in block_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        import re as _re

        line = _re.sub(r"^\s*#{1,6}\s*", "", line)
        return line
    return ""


def extract_short_title(full_title: str) -> str:
    import re as _re

    if not full_title:
        return "Н/Д"
    m = _re.search(r"\d{1,3}\.\s*(.+?)\s*-\s+", full_title)
    if not m:
        return "Н/Д"
    title = m.group(1).strip()
    return title or "Н/Д"


def slice_until_phrase(text: str, phrase: str) -> str:
    idx = text.find(phrase)
    return text if idx == -1 else text[:idx]


def slice_from_phrase_to_end(text: str, phrase: str) -> str:
    idx = text.find(phrase)
    return "" if idx == -1 else text[idx:]


def slice_500_before_phrase(text: str, phrase: str) -> str:
    idx = text.find(phrase)
    if idx == -1:
        return ""
    start = max(0, idx - 500)
    return text[start:idx]


def slice_between(text: str, start_phrase: str, end_phrase: str) -> str:
    s = text.find(start_phrase)
    e = text.find(end_phrase)
    if s == -1 or e == -1 or e <= s:
        return ""
    return text[s:e]


def contains_10_digits_sequence(text: str) -> bool:
    if not text:
        return False
    import re as _re

    return _re.search(r"(?<!\d)\d{10}(?!\d)", text) is not None


def extract_inn_10_digits(text: str) -> str:
    if not text:
        return "Н/Д"
    import re as _re

    m = _re.search(r"(?<!\d)(\d{10})(?!\d)", text)
    return m.group(1) if m else "Н/Д"


def parse_stage2_response(raw: str) -> Dict[str, str]:
    result = {
        "Прекращение обязательства": "Н/Д",
        "Дата сделки": "Н/Д",
        "Сумма и валюта": "Н/Д",
        "УИд договора": "Не найдено",
    }
    if not raw:
        return result
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    mapping = {
        "прекращение обязательства": "Прекращение обязательства",
        "дата сделки": "Дата сделки",
        "сумма и валюта": "Сумма и валюта",
        "уид договора": "УИд договора",
    }
    for ln in lines:
        if ":" not in ln:
            continue
        k, v = ln.split(":", 1)
        key_norm = k.strip().lower()
        val = v.strip()
        if key_norm in mapping:
            result[mapping[key_norm]] = val
    result["Прекращение обязательства"] = ensure_allowed_prekr(
        result["Прекращение обязательства"]
    )
    result["УИд договора"] = validate_uid(result["УИд договора"])
    for k in ("Дата сделки", "Сумма и валюта"):
        if not result[k] or not result[k].strip():
            result[k] = "Н/Д"
    return result


def parse_stage4_response(raw: str) -> str:
    if not raw:
        return "Н/Д"
    for ln in raw.splitlines():
        ln = ln.strip()
        if ":" in ln:
            k, v = ln.split(":", 1)
            if k.strip().lower() == "срочная задолженность":
                return v.strip() or "Н/Д"
    return "Н/Д"


def is_valid_debt_value(val: str) -> bool:
    """
    Проверяет, что значение суммы задолженности является числом
    (цифры, пробелы, опционально запятая/точка и 2 знака после неё).
    """
    if not val:
        return False
    v = nd_normalize(val)
    if v == "Н/Д":
        return False
    v_compact = v.replace(" ", "")
    return re.fullmatch(r"\d+(?:[.,]\d{2})?", v_compact) is not None


def _normalize_urgent_debt_table_rows(text: str) -> str:
    """
    Нормализует таблицу для extract_urgent_debt():
    склеивает перенос строки внутри одной логической строки Markdown-таблицы, когда:
      - текущая строка начинается с '|' и содержит дату dd-mm-yyyy / dd.mm.yyyy,
      - в текущей строке нет ни одного денежного значения вида ...,\d{2},
      - следующая строка НЕ содержит дату, но содержит '|' и денежные значения.
    """
    if not text:
        return text

    date_any_re = re.compile(r"\b\d{2}[.-]\d{2}[.-]\d{4}\b")
    money_any_re = re.compile(r"\b\d[\d \u00A0\u202F]*,\d{2}\b")

    lines = text.splitlines()
    out_lines: List[str] = []

    i = 0
    while i < len(lines):
        ln = lines[i]
        if (ln or "").lstrip().startswith("|") and date_any_re.search(ln or ""):
            if money_any_re.search(ln or "") is None and i + 1 < len(lines):
                nxt = lines[i + 1]
                if (
                    date_any_re.search(nxt or "") is None
                    and "|" in (nxt or "")
                    and money_any_re.search(nxt or "") is not None
                ):
                    left = (ln or "").rstrip()
                    right = (nxt or "").strip()

                    right2 = right.lstrip()
                    if right2.startswith("|"):
                        right2 = right2.lstrip("|").lstrip()

                    if not left.rstrip().endswith("|"):
                        left = left.rstrip() + " |"

                    out_lines.append(left + " " + right2)
                    i += 2
                    continue

        out_lines.append(ln)
        i += 1

    return "\n".join(out_lines)


def extract_urgent_debt(text: str) -> str:
    """
    Принимает многосрочный текст с таблицей, находит строку с самой поздней датой
    (формат dd-mm-yyyy или dd.mm.yyyy) и извлекает сумму срочной задолженности
    из третьей колонки слева (1 — дата, 2 — слово Да/Нет/Н/Д, 3 — сумма).
    Парсит ТОЛЬКО строки Markdown-таблицы (строки, начинающиеся с '|' после пробелов).
    Если по колонкам не получается, возвращает "Срочная задолженность: Н/Д"
    (чтобы сработал существующий LLM-fallback).
    """
    if not text:
        return "Срочная задолженность: Н/Д"

    text = _normalize_urgent_debt_table_rows(text)

    date_full_re = re.compile(r"^\d{2}[.-]\d{2}[.-]\d{4}$")

    # Строгая проверка "денежного" числа в 3-й колонке:
    # - обязательно запятая и ровно 2 цифры после неё
    # - до запятой минимум 4 цифры (пробелы/nbsp как разделители тысяч допустимы)
    money_re = re.compile(r"^\s*(\d[\d \u00A0\u202F]*)\s*,\s*(\d{2})\s*$")

    def _is_md_separator_line(line: str) -> bool:
        s = (line or "").strip()
        if "|" not in s:
            return False
        tmp = s.replace("|", "").strip()
        if not tmp:
            return False
        for ch in tmp:
            if ch not in "-: ":
                return False
        return True

    def _is_valid_money_cell(cell: str) -> bool:
        if not cell:
            return False
        s = str(cell).strip()
        m = money_re.fullmatch(s)
        if not m:
            return False
        int_part = m.group(1) or ""
        int_digits = re.sub(r"[ \u00A0\u202F]", "", int_part)
        if len(int_digits) < 4:
            return False
        return int_digits.isdigit()

    # 1) Препроцессинг: убираем строки-разделители Markdown таблиц
    kept_lines: List[str] = []
    for ln in text.splitlines():
        if _is_md_separator_line(ln):
            continue
        kept_lines.append(ln)

    # 2) Парсим ТОЛЬКО строки markdown-таблицы (начинаются с '|', имеют >= 3 колонки)
    parsed_rows: List[tuple[datetime, str]] = []
    for ln in kept_lines:
        if not (ln or "").lstrip().startswith("|"):
            continue
        if (ln or "").count("|") < 4:
            continue

        raw_parts = (ln or "").strip().split("|")
        if raw_parts and raw_parts[0].strip() == "":
            raw_parts = raw_parts[1:]
        if raw_parts and raw_parts[-1].strip() == "":
            raw_parts = raw_parts[:-1]

        cells = [p.strip() for p in raw_parts]
        if len(cells) < 3:
            continue

        date_cell = cells[0]
        if not date_full_re.fullmatch(date_cell):
            continue

        try:
            dt = datetime.strptime(date_cell.replace(".", "-"), "%d-%m-%Y")
        except ValueError:
            continue

        col3 = cells[2]
        if not _is_valid_money_cell(col3):
            continue

        parsed_rows.append((dt, col3.strip()))

    if not parsed_rows:
        return "Срочная задолженность: Н/Д"

    _latest_dt, latest_value = max(parsed_rows, key=lambda t: t[0])
    if latest_value:
        return f"Срочная задолженность: {latest_value}"

    return "Срочная задолженность: Н/Д"


def smart_guess_org_name(text: str) -> str:
    import re as _re

    if not text:
        return "Н/Д"
    m = _re.search(r'\b(АО|ООО|ПАО|ОАО|ЗАО)\s*[«"]([^»"]+)[»"]', text, flags=_re.IGNORECASE)
    if m:
        prefix = m.group(1).upper()
        name = m.group(2).strip()
        return f"{prefix} «{name}»"
    candidates = []
    for line in text.splitlines():
        L = line.strip()
        if not L:
            continue
        if _re.search(
            r"\b(АО|ООО|ПАО|ОАО|ЗАО|БАНК|КРЕДИТ|МФК|МФО)\b", L, flags=_re.IGNORECASE
        ):
            if not _re.search(r"\bИНН\b|\bРег\.?номер\b", L, flags=_re.IGNORECASE):
                candidates.append(L)
    if candidates:
        candidates.sort(key=lambda s: len(s), reverse=True)
        return candidates[0]
    nonnum = [
        l.strip()
        for l in text.splitlines()
        if l.strip() and not _re.fullmatch(r"[0-9\W_]+", l.strip(), flags=_re.UNICODE)
    ]
    if nonnum:
        nonnum.sort(key=lambda s: len(s), reverse=True)
        return nonnum[0]
    return "Н/Д"


def parse_stage5_response(raw: str, context_text: Optional[str] = None) -> tuple[str, str]:
    import re as _re

    name = "Н/Д"
    inn = "Н/Д"
    if not raw:
        return name, inn

    got_prefix_name = False
    got_prefix_inn = False
    for ln in raw.splitlines():
        ln = ln.strip()
        if ":" not in ln:
            continue
        k, v = ln.split(":", 1)
        key = k.strip().lower()
        val = v.strip()
        if key == "приобретатель прав кредитора":
            name = val or "Н/Д"
            got_prefix_name = True
        elif key == "инн приобретателя прав кредитора":
            m = _re.search(r"(?<!\d)(\d{10})(?!\d)", val)
            inn = m.group(1) if m else "Н/Д"
            got_prefix_inn = True

    if got_prefix_name and got_prefix_inn:
        return name, inn

    m = _re.search(r"(?<!\d)(\d{10})(?!\d)", raw)
    if m:
        inn = m.group(1)

    name_guess = smart_guess_org_name(raw)
    if name_guess == "Н/Д" and context_text:
        name_guess = smart_guess_org_name(context_text)
    if name == "Н/Д":
        name = name_guess

    return name, inn


# ---------- Mistral-клиент с переключением FREE -> PAID и LLM-логами ----------

class MistralChatClient:
    """
    Клиент для Mistral Chat Completions с переключением FREE_API_KEY -> PAID_API_KEY
    при ошибках сервера (429, 5xx) и расширенным логированием + llm_io_...txt.
    """

    def __init__(
        self,
        free_key: str,
        paid_key: str,
        logger: Logger,
        base_log_extra: Optional[Dict[str, object]] = None,
        stats: Optional[FileStats] = None,
    ):
        self.free_key = free_key.strip() if free_key else ""
        self.paid_key = paid_key.strip() if paid_key else ""
        self.current_key = self.free_key
        self.current_label = "FREE"
        self.switched = False
        self.logger = logger
        self.base_log_extra = base_log_extra or {}
        self.stats = stats

        if not self.free_key:
            raise RuntimeError("FREE_API_KEY не задан в .env, невозможен вызов Mistral Chat.")

        if not self.paid_key:
            self.logger.warning(
                "PAID_API_KEY не задан. Переключение на платный ключ будет невозможно.",
                extra={
                    **self.base_log_extra,
                    "stage": "llm_client_init",
                    "duration_seconds": 0,
                    "model": "N/A",
                    "api_key_id": "FREE_API_KEY",
                },
            )

        self.logger.info(
            "Инициализация MistralChatClient: текущий ключ=%s (FREE_API_KEY).",
            self.current_label,
            extra={
                **self.base_log_extra,
                "stage": "llm_client_init",
                "duration_seconds": 0,
                "model": "N/A",
                "api_key_id": "FREE_API_KEY",
            },
        )

    def _switch_to_paid(self, reason: str) -> None:
        if self.switched:
            return
        if not self.paid_key:
            self.logger.error(
                "Хотели переключиться на PAID_API_KEY, но он не задан. Остаёмся на FREE_API_KEY. Причина: %s",
                reason,
                extra={
                    **self.base_log_extra,
                    "stage": "llm_switch_failed",
                    "duration_seconds": 0,
                    "model": "N/A",
                    "api_key_id": "FREE_API_KEY",
                },
            )
            return
        self.current_key = self.paid_key
        self.current_label = "PAID"
        self.switched = True
        self.logger.warning(
            "Переключаюсь на PAID_API_KEY из-за ошибок сервера Mistral. Причина: %s",
            reason,
            extra={
                **self.base_log_extra,
                "stage": "llm_switch_to_paid",
                "duration_seconds": 0,
                "model": "N/A",
                "api_key_id": "PAID_API_KEY",
            },
        )

    def _log_llm_io(
        self,
        *,
        llm_io_path: Optional[Path],
        model_name: str,
        api_key_label: str,
        request_text: str,
        response_text: str,
        latency_seconds: float,
        status: str,
        error_type: Optional[str] = None,
        error_code: Optional[str] = None,
    ) -> None:
        """
        Хелпер для записи одного вызова ЛЛМ в llm_io_...txt и обновления сводной телеметрии.
        """
        base = self.base_log_extra or {}
        path_obj = llm_io_path
        if path_obj is None:
            tmp = base.get("llm_io_path")
            if isinstance(tmp, str):
                path_obj = Path(tmp)
            elif isinstance(tmp, Path):
                path_obj = tmp

        if path_obj is not None:
            try:
                request_id = str(base.get("request_id", "N/A"))
                telegram_user_id = str(base.get("telegram_user_id", "N/A"))
                telegram_username = str(base.get("telegram_username", "N/A"))
                pdf_filename = str(base.get("file_name", "N/A"))

                log_llm_call(
                    path_obj,
                    request_id=request_id,
                    telegram_user_id=telegram_user_id,
                    telegram_username=telegram_username,
                    pdf_filename=pdf_filename,
                    model=model_name,
                    api_key_id=f"{api_key_label}_API_KEY",
                    request_text=request_text,
                    response_text=response_text,
                    latency_seconds=latency_seconds,
                    status=status,
                    error_type=error_type,
                    error_code=error_code,
                )
            except Exception:
                # Логирование не должно ломать основной пайплайн.
                pass

        if self.stats is not None:
            try:
                self.stats.register_llm_call(
                    model=model_name,
                    api_key_id=f"{api_key_label}_API_KEY",
                    request_text=request_text,
                    response_text=response_text,
                    latency_seconds=latency_seconds,
                )
            except Exception:
                # Телеметрия не должна ломать пайплайн.
                pass

    def chat(
        self,
        model_name: str,
        system_prompt: str,
        user_content: str,
        stage: str,
        row_index: int,
        number: str,
        title: str,
        max_retries: int = 3,
        timeout: int = 60,
    ) -> str:
        """
        Отправляет один запрос в Mistral Chat, при ошибках сервера
        (429, 5xx) один раз переключается на PAID_API_KEY и продолжает.
        Логирует отправку запроса, получение ответа и ошибки.
        Также пишет детали каждого вызова в llm_io_...txt.
        """
        headers_base = {"Content-Type": "application/json"}
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt or ""},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 256,
        }

        attempt = 0
        last_err = None

        base_extra = self.base_log_extra or {}
        llm_io_path = base_extra.get("llm_io_path")
        if isinstance(llm_io_path, str):
            llm_io_path = Path(llm_io_path)

        request_text_for_io = (
            "[SYSTEM PROMPT]\n"
            f"{system_prompt or ''}\n\n"
            "[USER CONTENT]\n"
            f"{user_content or ''}"
        )

        while attempt <= max_retries:
            attempt += 1
            key_label = self.current_label
            headers = {
                **headers_base,
                "Authorization": f"Bearer {self.current_key}",
            }

            send_start = time.perf_counter()

            self.logger.info(
                "[%s] Отправлен запрос к Mistral (попытка %d/%d, ключ=%s, модель=%s, row=%d).",
                stage,
                attempt,
                max_retries,
                key_label,
                model_name,
                row_index,
                extra={
                    **self.base_log_extra,
                    "stage": f"{stage}_llm_request",
                    "duration_seconds": 0,
                    "model": model_name,
                    "api_key_id": f"{key_label}_API_KEY",
                },
            )

            try:
                resp = requests.post(
                    MISTRAL_API_URL,
                    headers=headers,
                    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    timeout=timeout,
                )
            except Exception as e:  # сетевые/прочие ошибки
                call_duration = time.perf_counter() - send_start
                last_err = str(e)
                self.logger.error(
                    "[%s] Исключение при запросе к Mistral для строки %d: %s",
                    stage,
                    row_index,
                    last_err,
                    exc_info=True,
                    extra={
                        **self.base_log_extra,
                        "stage": f"{stage}_llm_error",
                        "duration_seconds": round(call_duration, 3),
                        "model": model_name,
                        "api_key_id": f"{key_label}_API_KEY",
                    },
                )
                self._log_llm_io(
                    llm_io_path=llm_io_path,
                    model_name=model_name,
                    api_key_label=key_label,
                    request_text=request_text_for_io,
                    response_text=str(e),
                    latency_seconds=call_duration,
                    status="error",
                    error_type=type(e).__name__,
                    error_code=None,
                )
                # считаем это тоже серверной проблемой: пробуем переключиться
                if not self.switched and self.paid_key:
                    self._switch_to_paid(f"network/exception: {last_err}")
                    attempt = 0  # начать попытки заново с платным
                    continue
                # если уже переключены — просто делаем бэкофф
                time.sleep(min(10, 2 * attempt) + random.uniform(0.0, 0.5))
                continue

            call_duration = time.perf_counter() - send_start

            if resp.status_code == 200:
                try:
                    data = resp.json()
                except Exception as e:
                    last_err = str(e)
                    self.logger.error(
                        "[%s] Ошибка парсинга JSON от Mistral для строки %d: %s",
                        stage,
                        row_index,
                        last_err,
                        exc_info=True,
                        extra={
                            **self.base_log_extra,
                            "stage": f"{stage}_llm_error",
                            "duration_seconds": round(call_duration, 3),
                            "model": model_name,
                            "api_key_id": f"{key_label}_API_KEY",
                        },
                    )
                    self._log_llm_io(
                        llm_io_path=llm_io_path,
                        model_name=model_name,
                        api_key_label=key_label,
                        request_text=request_text_for_io,
                        response_text=f"JSON parse error: {last_err}. Raw body: {resp.text[:1000]}",
                        latency_seconds=call_duration,
                        status="error",
                        error_type=type(e).__name__,
                        error_code=None,
                    )
                    continue
                text = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                ).strip()
                self.logger.info(
                    "[%s] Успешный ответ от Mistral (ключ=%s, row=%d).",
                    stage,
                    key_label,
                    row_index,
                    extra={
                        **self.base_log_extra,
                        "stage": f"{stage}_llm_response",
                        "duration_seconds": round(call_duration, 3),
                        "model": model_name,
                        "api_key_id": f"{key_label}_API_KEY",
                    },
                )
                # Подробные тексты в CSV не пишем, только в llm_io_...txt
                self._log_llm_io(
                    llm_io_path=llm_io_path,
                    model_name=model_name,
                    api_key_label=key_label,
                    request_text=request_text_for_io,
                    response_text=text,
                    latency_seconds=call_duration,
                    status="success",
                    error_type=None,
                    error_code=None,
                )
                return text

            # Ошибки сервера
            if resp.status_code in (429, 500, 502, 503, 504):
                body = resp.text[:1000]
                last_err = f"HTTP {resp.status_code}: {body}"
                self.logger.error(
                    "[%s] Ошибка сервера Mistral для строки %d: %s",
                    stage,
                    row_index,
                    last_err,
                    extra={
                        **self.base_log_extra,
                        "stage": f"{stage}_llm_error",
                        "duration_seconds": round(call_duration, 3),
                        "model": model_name,
                        "api_key_id": f"{key_label}_API_KEY",
                    },
                )
                self._log_llm_io(
                    llm_io_path=llm_io_path,
                    model_name=model_name,
                    api_key_label=key_label,
                    request_text=request_text_for_io,
                    response_text=body,
                    latency_seconds=call_duration,
                    status="error",
                    error_type="HTTPError",
                    error_code=str(resp.status_code),
                )
                if not self.switched and self.paid_key:
                    self._switch_to_paid(last_err)
                    attempt = 0  # начать попытки заново на платном ключе
                    continue
                # Уже на платном или переключение невозможно — просто бэкофф и повтор
                sleep_s = min(10, 2 * attempt) + random.uniform(0.0, 0.5)
                time.sleep(sleep_s)
                continue

            # Прочие 4xx считаем фатальными (не переключаемся)
            body = resp.text[:1000]
            last_err = f"HTTP {resp.status_code}: {body}"
            self.logger.error(
                "[%s] Фатальная ошибка Mistral для строки %d (ключ=%s): %s",
                stage,
                row_index,
                key_label,
                last_err,
                extra={
                    **self.base_log_extra,
                    "stage": f"{stage}_llm_error",
                    "duration_seconds": round(call_duration, 3),
                    "model": model_name,
                    "api_key_id": f"{key_label}_API_KEY",
                },
            )
            self._log_llm_io(
                llm_io_path=llm_io_path,
                model_name=model_name,
                api_key_label=key_label,
                request_text=request_text_for_io,
                response_text=body,
                latency_seconds=call_duration,
                status="error",
                error_type="HTTPError",
                error_code=str(resp.status_code),
            )
            break

        self.logger.error(
            "[%s] Не удалось получить ответ от Mistral для строки %d после повторов. Последняя ошибка: %s",
            stage,
            row_index,
            last_err,
            extra={
                **self.base_log_extra,
                "stage": f"{stage}_llm_error",
                "duration_seconds": 0,
                "model": model_name,
                "api_key_id": f"{self.current_label}_API_KEY",
            },
        )
        return ""

