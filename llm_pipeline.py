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

from config import FREE_API_KEY, PAID_API_KEY, make_llm_io_path
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

# Системные промпты (без изменений)
SYSTEM_PROMPT_STAGE2 = """
Вход: один кредитный блок в Markdown.

Ответ: РОВНО 4 строки без кода, таблиц и любой Markdown-разметки, без «#», «##» и т.п.

Текст не изменяй, кроме склейки переносов слов/строк. Данные бери только из блока, ничего не придумывай.

Если значение поля не найдено — ставь ровно «Н/Д» (кириллица, заглавные Н и Д, слэш /). Любые OCR-варианты («H/Д», «Н/д», «н/д» и т.п.) приводи к «Н/Д».

УИд договора:
— Ищи метки «УИд договора», «УИД договора», «Уид договора».
— Кандидат нормализуй: убери пробелы и переводы строк, все тире замени на «-», латиницу сделай строчной.
— Допустимые OCR-подстановки: O↔0, I↔1, l↔1, S↔5, Z↔2, B↔8.
— В первых 5 группах разрешены только [0-9a-f], кириллица запрещена.
— Формат обязателен: 8-4-4-4-12-1 (пример: c456294b-dcd0-11aa-81b3-efa2ccd7b24f-7).
— Можно восстановить только пропущенные дефисы по схеме 8-4-4-4-12-1, символы не придумывай.
— Если валидного идентификатора нет — пиши «Не найдено».

Поля:

Прекращение обязательства — только «Н/Д» или «Надлежащее исполнение обязательства».

Дата сделки — как в тексте.

Сумма и валюта — сумма как в тексте, в ответе валюта обязательно «RUB».

УИд договора — см. правила выше.

Формат ответа — строго 4 строки, в этом порядке, без лишних строк и комментариев:

Прекращение обязательства: …
Дата сделки: …
Сумма и валюта: …
УИд договора: … 
""".strip()

PROMPT_STAGE5 = """
Ты получаешь ОДИН кредитный блок в формате Markdown (MD).
ТВОЯ ЗАДАЧА — вернуть РОВНО 2 СТРОКИ С ЖЁСТКИМИ ПРЕФИКСАМИ (без кода, без таблиц, без Markdown-разметки):

Приобретатель прав кредитора: ...
ИНН приобретателя прав кредитора: ...

Требования:
- Если поле отсутствует — пиши строго: Н/Д (кириллица, заглавные Н и Д, слэш /).
- В ИНН — РОВНО 10 ЦИФР (иначе Н/Д).
- НИКАКИХ дополнительных строк, пустых строк, комментариев, описаний, JSON и т.п.
- Если вернёшь без префиксов — ответ считается НЕВАЛИДНЫМ. Всегда пиши указанные префиксы в начале строк.
""".strip()

SYSTEM_PROMPT_INN_FALLBACK = """
Ты получаешь ОДИН кредитный блок целиком (Markdown-текст отчёта НБКИ).

НАЙДИ в нём ИНН организации — последовательность РОВНО из 10 цифр.

Требования к ответу:
- Ответь строго ОДНОЙ строкой.
- Если ИНН найден — верни ТОЛЬКО эти 10 цифр подряд, без пробелов, без комментариев, без дополнительных слов.
- Если ИНН нет или его нельзя однозначно определить по тексту — верни ровно: Н/Д (кириллица, заглавные Н и Д, слэш /).

Никаких JSON, описаний, пояснений, перевода строки или других символов, только одна строка по правилам выше.
""".strip()

SYSTEM_PROMPT_STAGE4_FALLBACK = """
Ты получаешь фрагмент таблицы из отчёта НБКИ в текстовом виде (строки с датами и числовыми значениями).

Твоя задача — найти сумму срочной задолженности.

Правила:

1. Найди строку с САМОЙ ПОЗДНЕЙ датой.
   Формат даты: dd.mm.yyyy или dd-mm-yyyy.
2. В этой строке сумма срочной задолженности находится в ТРЕТЬЕМ столбце слева:
   - 1-й столбец — дата.
   - 2-й столбец — какое-то значение (может быть числом или «Н/Д») — его нужно игнорировать.
   - 3-й столбец — искомая сумма срочной задолженности.
3. Сумма может быть:
   - целым числом (например, 32934),
   - числом с пробелами как разделителями тысяч (например, 32 934),
   - числом с запятой для копеек (например, 32934,00 или 12 345,67).
   Считай пробелы внутри числа частью записи числа.
4. НИЧЕГО не придумывай. Используй только те данные, которые есть в тексте.
   Если по тексту невозможно однозначно определить сумму срочной задолженности, нужно вернуть «Срочная задолженность: Н/Д».

Пример:
Строка:
|  14-12-2021 | Н/Д | 32934 | Н/Д | Н/Д | Н/Д | Нет  |
В этом случае:
- самая поздняя дата — 14-12-2021;
- 3-й столбец — 32934;
- правильный ответ: Срочная задолженность: 32934

Формат ответа:

- Ответь СТРОГО ОДНОЙ строкой вида:
  Срочная задолженность: <значение>

Где <значение> — найденная сумма ровно в том виде, как она указана в тексте (без добавления валюты), либо Н/Д.
Никакого дополнительного текста, комментариев, объяснений, JSON, кавычек или лишних строк.
""".strip()

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


def extract_urgent_debt(text: str) -> str:
    """
    Принимает многосрочный текст с таблицей, находит строку с самой поздней датой
    (формат dd-mm-yyyy или dd.mm.yyyy) и пытается извлечь сумму срочной задолженности
    из третьей колонки слева (1 — дата, 2 — игнорируемое значение, 3 — сумма).
    Если по колонкам не получается, использует fallback по списку чисел в строке.
    Возвращает строку:
    - "Срочная задолженность: <значение>"
    - либо "Срочная задолженность: Н/Д", если ничего не найдено.
    """
    if not text:
        return "Срочная задолженность: Н/Д"

    date_re = re.compile(r"\b(\d{2}[.-]\d{2}[.-]\d{4})\b")
    lines_with_dates = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = date_re.search(line)
        if not m:
            continue
        date_str = m.group(1)
        try:
            dt = datetime.strptime(date_str.replace(".", "-"), "%d-%m-%Y")
        except ValueError:
            continue
        lines_with_dates.append((dt, date_str, line, m))

    if not lines_with_dates:
        return "Срочная задолженность: Н/Д"

    latest_dt, latest_date_str, latest_line, latest_match = max(
        lines_with_dates, key=lambda t: t[0]
    )

    value: Optional[str] = None

    # 1) Попытка извлечения по третьей колонке (таблица с разделителем '|')
    if "|" in latest_line:
        parts = [p.strip() for p in latest_line.split("|")]
        cols = [p for p in parts if p]
        if cols:
            date_idx: Optional[int] = None
            for i, col in enumerate(cols):
                if latest_date_str in col:
                    date_idx = i
                    break
            if date_idx is None:
                for i, col in enumerate(cols):
                    if date_re.search(col):
                        date_idx = i
                        break
            if date_idx is not None:
                target_idx = date_idx + 2
                if target_idx < len(cols):
                    target_col = cols[target_idx].strip()
                    upper = target_col.upper().replace(" ", "")
                    if upper not in {"Н/Д", "H/Д"} and target_col:
                        m_num = re.search(r"\d[\d ]*(?:[.,]\d{2})?", target_col)
                        if m_num:
                            value = m_num.group(0).strip()

    # 2) Fallback: по списку чисел в строке после даты
    if value is None:
        rest = latest_line[latest_match.end() :]
        numbers = re.findall(r"\d[\d ]*(?:[.,]\d{2})?", rest)
        if numbers:
            if len(numbers) >= 2:
                value = numbers[1].strip()
            else:
                value = numbers[0].strip()

    if value:
        return f"Срочная задолженность: {value}"

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


# ---------- основной LLM-пайплайн ----------

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
                "Приобретатель прав кредитора": "Н/Д",
                "ИНН приобретателя прав кредитора": "Н/Д",
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
            "model": "mistral-small-latest",
            "api_key_id": "FREE_API_KEY",
            "file_name": original_pdf_name,
        },
    )
    stage2_start = time.perf_counter()

    idxs_stage2: List[int] = list(chunks_df.index)
    if idxs_stage2:
        futures_map = {}
        with ThreadPoolExecutor(max_workers=len(idxs_stage2)) as ex:
            for idx in idxs_stage2:
                number = main_df.at[idx, "Номер"]
                title = main_df.at[idx, "Заголовок блока"]
                full_block = str(chunks_df.at[idx, "text"] or "")
                part_for_stage2 = slice_until_phrase(full_block, PHRASE_SVEDENIYA_ISPOLN)

                futures_map[
                    ex.submit(
                        chat_client.chat,
                        "mistral-small-latest",
                        SYSTEM_PROMPT_STAGE2,
                        part_for_stage2,
                        "stage2",
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
                            "model": "mistral-small-latest",
                            "api_key_id": "N/A",
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
            "model": "mistral-small-latest",
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

        with ThreadPoolExecutor(max_workers=len(idxs_inn_llm)) as ex:
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

    # Этап 4: срочная задолженность (локальный парсер + LLM-fallback)
    logger.info(
        "Этап 4: расчёт 'Сумма задолженности' (локальный парсер + LLM-fallback).",
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
    fallback_idxs: List[int] = []
    fallback_payloads: Dict[int, str] = {}

    for idx in main_df.index[need_more]:
        full_block = str(chunks_df.at[idx, "text"] or "")
        slice_500 = slice_500_before_phrase(full_block, PHRASE_SROCHNAYA_ZADOLZH)
        if not slice_500:
            main_df.at[idx, "Сумма задолженности"] = "Н/Д"
            continue

        resp_text_local = extract_urgent_debt(slice_500)
        value_local = nd_normalize(parse_stage4_response(resp_text_local))
        if is_valid_debt_value(value_local):
            main_df.at[idx, "Сумма задолженности"] = value_local
        else:
            main_df.at[idx, "Сумма задолженности"] = "Н/Д"
            fallback_idxs.append(idx)
            fallback_payloads[idx] = slice_500

    # LLM-fallback только для тех строк, где локальный парсер не дал валидного числа
    if fallback_idxs:
        logger.info(
            "Этап 4: LLM-fallback для 'Сумма задолженности' будет выполнен для %d строк.",
            len(fallback_idxs),
            extra={
                "stage": "llm_stage4_fallback_start",
                "telegram_user_id": telegram_user_id,
                "telegram_username": telegram_username,
                "request_id": request_id,
                "duration_seconds": 0,
                "model": "mistral-small-latest",
                "api_key_id": "FREE_API_KEY",
                "file_name": original_pdf_name,
            },
        )
        fallback_start = time.perf_counter()
        futures_map_stage4: Dict[object, int] = {}
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=len(fallback_idxs)) as ex:
            for idx in fallback_idxs:
                number = main_df.at[idx, "Номер"]
                title = main_df.at[idx, "Заголовок блока"]
                payload = fallback_payloads.get(idx, "")
                futures_map_stage4[
                    ex.submit(
                        chat_client.chat,
                        "mistral-small-latest",
                        SYSTEM_PROMPT_STAGE4_FALLBACK,
                        payload,
                        "stage4_fallback",
                        idx,
                        number,
                        title,
                    )
                ] = idx

            for f in as_completed(futures_map_stage4):
                idx = futures_map_stage4[f]
                try:
                    resp_text_llm = f.result()
                except Exception as e:
                    logger.error(
                        "[stage4_fallback] Ошибка будущего результата для строки %d: %s",
                        idx,
                        e,
                        exc_info=True,
                        extra={
                            "stage": "llm_stage4_fallback_error",
                            "telegram_user_id": telegram_user_id,
                            "telegram_username": telegram_username,
                            "request_id": request_id,
                            "duration_seconds": 0,
                            "model": "mistral-small-latest",
                            "api_key_id": "N/A",
                            "file_name": original_pdf_name,
                        },
                    )
                    resp_text_llm = ""
                value_llm = nd_normalize(parse_stage4_response(resp_text_llm))
                if is_valid_debt_value(value_llm):
                    main_df.at[idx, "Сумма задолженности"] = value_llm
                else:
                    main_df.at[idx, "Сумма задолженности"] = "Н/Д"

        fallback_duration = time.perf_counter() - fallback_start
        logger.info(
            "Этап 4: LLM-fallback для 'Сумма задолженности' завершён за %.3f с.",
            fallback_duration,
            extra={
                "stage": "llm_stage4_fallback_done",
                "telegram_user_id": telegram_user_id,
                "telegram_username": telegram_username,
                "request_id": request_id,
                "duration_seconds": round(fallback_duration, 3),
                "model": "mistral-small-latest",
                "api_key_id": "FREE_API_KEY",
                "file_name": original_pdf_name,
            },
        )

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
        with ThreadPoolExecutor(max_workers=len(idxs_stage5)) as ex:
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
