from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Dict, Optional, List

import requests
import pandas as pd

from logging import Logger

from config import FREE_API_KEY, PAID_API_KEY

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
— Формат обязателен: 8-4-4-4-12-1 (пример: c456294b-dcd0-11ед-81b3-efa2ccd7b24f-7).
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

PROMPT_STAGE4 = """
Ты получаешь ОДИН блок текста (таблица/строки) и ДОЛЖЕН:
1) Найти САМУЮ ПОЗДНЮЮ дату в формате dd-mm-yyyy ИЛИ dd.mm.yyyy (например: 27-06-2025 или 27.06.2025).
2) В той же строке, ГДЕ НАЙДЕНА ЭТА САМАЯ ПОЗДНЯЯ ДАТА, найти ПЕРВОЕ ЧИСЛО ПОСЛЕ ДАТЫ (например: 272996,60).
3) Вывести результат СТРОГО в формате:
Срочная задолженность: 272996,60

Правила:
- Никаких других строк, таблиц, комментариев или пояснений.
- Если данных нет — напиши: Срочная задолженность: Н/Д
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
    m = _re.match(r"^\s*\d+\.\s*(.*?)\s*-\s+", full_title)
    return (m.group(1).strip() if m and m.group(1).strip() else "Н/Д")


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


# ---------- Mistral-клиент с переключением FREE -> PAID ----------

class MistralChatClient:
    """
    Клиент для Mistral Chat Completions с переключением FREE_API_KEY -> PAID_API_KEY
    при ошибках сервера (429, 5xx).
    """

    def __init__(self, free_key: str, paid_key: str, logger: Logger):
        self.free_key = free_key.strip() if free_key else ""
        self.paid_key = paid_key.strip() if paid_key else ""
        self.current_key = self.free_key
        self.current_label = "FREE"
        self.switched = False
        self.logger = logger

        if not self.free_key:
            raise RuntimeError("FREE_API_KEY не задан в .env, невозможен вызов Mistral Chat.")

        if not self.paid_key:
            self.logger.warning("PAID_API_KEY не задан. Переключение на платный ключ будет невозможно.")

        self.logger.info(
            "Инициализация MistralChatClient: текущий ключ=%s (FREE_API_KEY).", self.current_label
        )

    def _switch_to_paid(self, reason: str) -> None:
        if self.switched:
            return
        if not self.paid_key:
            self.logger.error(
                "Хотели переключиться на PAID_API_KEY, но он не задан. Остаёмся на FREE_API_KEY. Причина: %s",
                reason,
            )
            return
        self.current_key = self.paid_key
        self.current_label = "PAID"
        self.switched = True
        self.logger.warning(
            "Переключаюсь на PAID_API_KEY из-за ошибок сервера Mistral. Причина: %s",
            reason,
        )

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
        while attempt <= max_retries:
            attempt += 1
            key_label = self.current_label
            headers = {
                **headers_base,
                "Authorization": f"Bearer {self.current_key}",
            }

            self.logger.info(
                "[%s] Запрос к Mistral (попытка %d/%d, ключ=%s, модель=%s, row=%d).",
                stage,
                attempt,
                max_retries,
                key_label,
                model_name,
                row_index,
            )

            try:
                resp = requests.post(
                    MISTRAL_API_URL,
                    headers=headers,
                    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    timeout=timeout,
                )
            except Exception as e:  # сетевые/прочие ошибки
                last_err = str(e)
                self.logger.error(
                    "[%s] Исключение при запросе к Mistral для строки %d: %s",
                    stage,
                    row_index,
                    last_err,
                )
                # считаем это тоже серверной проблемой: пробуем переключиться
                if not self.switched and self.paid_key:
                    self._switch_to_paid(f"network/exception: {last_err}")
                    attempt = 0  # начать попытки заново с платным
                    continue
                # если уже переключены — просто делаем бэкофф
                time.sleep(min(10, 2 * attempt) + random.uniform(0.0, 0.5))
                continue

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
                )
                self.logger.debug(
                    "[%s] REQUEST system:\n%s\n\nREQUEST user (первые 2000 симв.):\n%s\n\nRESPONSE:\n%s",
                    stage,
                    system_prompt,
                    (user_content or "")[:2000],
                    text,
                )
                return text

            # Ошибки сервера
            if resp.status_code in (429, 500, 502, 503, 504):
                body = resp.text[:1000]
                last_err = f"HTTP {resp.status_code}: {body}"
                self.logger.warning(
                    "[%s] Ошибка сервера Mistral для строки %d: %s",
                    stage,
                    row_index,
                    last_err,
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
            last_err = f"HTTP {resp.status_code}: {resp.text[:1000]}"
            self.logger.error(
                "[%s] Фатальная ошибка Mistral для строки %d (ключ=%s): %s",
                stage,
                row_index,
                key_label,
                last_err,
            )
            break

        self.logger.error(
            "[%s] Не удалось получить ответ от Mistral для строки %d после повторов. Последняя ошибка: %s",
            stage,
            row_index,
            last_err,
        )
        return ""


# ---------- основной LLM-пайплайн ----------

def run_llm_pipeline(
    chunks_csv_path: Path,
    original_pdf_name: str,
    request_dir: Path,
    request_ts: str,
    logger: Logger,
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

    try:
        chunks_df = pd.read_csv(chunks_csv_path, encoding="utf-8-sig")
    except Exception:
        chunks_df = pd.read_csv(chunks_csv_path, encoding="utf-8")

    if "text" not in chunks_df.columns:
        raise ValueError("В файле chunks.csv отсутствует колонка 'text'.")

    logger.info(
        "LLM-пайплайн: загружено %d блок(ов) из %s.",
        len(chunks_df),
        chunks_csv_path,
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
    logger.info("Этап 1 LLM: базовые поля сформированы.")

    # Клиент Mistral с переключением FREE -> PAID
    chat_client = MistralChatClient(FREE_API_KEY, PAID_API_KEY, logger)

    # Этап 2: 4 поля
    logger.info("Этап 2 LLM: извлечение полей 'Прекращение обязательства', 'Дата сделки', 'Сумма и валюта', 'УИд договора'.")
    from concurrent.futures import ThreadPoolExecutor, as_completed

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
                    )
                    resp_text = ""
                parsed = parse_stage2_response(resp_text)
                main_df.at[idx, "Прекращение обязательства"] = parsed[
                    "Прекращение обязательства"
                ]
                main_df.at[idx, "Дата сделки"] = parsed["Дата сделки"]
                main_df.at[idx, "Сумма и валюта"] = parsed["Сумма и валюта"]
                main_df.at[idx, "УИд договора"] = parsed["УИд договора"]

    need_more = main_df["Прекращение обязательства"] == "Н/Д"
    logger.info(
        "Этап 2 LLM: завершён. В Этапы 3–5 пойдут %d строк(и).",
        need_more.sum(),
    )

    # Этап 3: ИНН (10 цифр) локально
    logger.info("Этап 3 LLM: извлекаем ИНН (10 цифр) скриптом на Python.")
    for idx in main_df.index[need_more]:
        full_block = str(chunks_df.at[idx, "text"] or "")
        slice_inn = slice_from_phrase_to_end(full_block, PHRASE_SOURCE_CREDIT_HISTORY)
        main_df.at[idx, "ИНН"] = extract_inn_10_digits(slice_inn)

    for idx in main_df.index[~need_more]:
        for col in [
            "ИНН",
            "Сумма задолженности",
            "Приобретатель прав кредитора",
            "ИНН приобретателя прав кредитора",
        ]:
            main_df.at[idx, col] = nd_normalize(main_df.at[idx, col])

    # Этап 4: срочная задолженность
    logger.info("Этап 4 LLM: параллельные запросы к ЛЛМ для 'Сумма задолженности'.")
    idxs_stage4: List[int] = []
    user_payloads_stage4: Dict[int, str] = {}
    for idx in main_df.index[need_more]:
        full_block = str(chunks_df.at[idx, "text"] or "")
        slice_500 = slice_500_before_phrase(full_block, PHRASE_SROCHNAYA_ZADOLZH)
        if not slice_500:
            main_df.at[idx, "Сумма задолженности"] = "Н/Д"
        else:
            idxs_stage4.append(idx)
            user_payloads_stage4[idx] = (
                f"{PROMPT_STAGE4}\n\nТЕКСТ ДЛЯ АНАЛИЗА:\n\n{slice_500}"
            )

    if idxs_stage4:
        futures_map = {}
        with ThreadPoolExecutor(max_workers=len(idxs_stage4)) as ex:
            for idx in idxs_stage4:
                number = main_df.at[idx, "Номер"]
                title = main_df.at[idx, "Заголовок блока"]
                futures_map[
                    ex.submit(
                        chat_client.chat,
                        "mistral-small-latest",
                        "",
                        user_payloads_stage4[idx],
                        "stage4",
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
                        "[stage4] Ошибка будущего результата для строки %d: %s",
                        idx,
                        e,
                    )
                    resp_text = ""
                main_df.at[idx, "Сумма задолженности"] = nd_normalize(
                    parse_stage4_response(resp_text)
                )

    # Этап 5: приобретатель прав кредитора
    logger.info(
        "Этап 5 LLM: параллельные запросы к ЛЛМ для 'Приобретатель прав кредитора' и 'ИНН приобретателя прав кредитора'."
    )
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
                    )
                    resp_text = ""
                name, inn = parse_stage5_response(
                    resp_text, context_text=context_stage5.get(idx, "")
                )
                main_df.at[idx, "Приобретатель прав кредитора"] = nd_normalize(name)
                main_df.at[idx, "ИНН приобретателя прав кредитора"] = nd_normalize(inn)

    for col in OUTPUT_COLUMNS:
        if col == "УИд договора":
            main_df[col] = main_df[col].fillna("Не найдено")
        else:
            main_df[col] = main_df[col].fillna("Н/Д").apply(nd_normalize)

    base_pdf = Path(original_pdf_name).stem
    result_csv_name = f"{base_pdf}_result_{request_ts}.csv"
    result_csv_path = request_dir / result_csv_name
    main_df.to_csv(result_csv_path, index=False, encoding="utf-8-sig")
    logger.info("LLM-пайплайн завершён. Итоговый CSV: %s", result_csv_path)

    return result_csv_path
