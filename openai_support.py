from __future__ import annotations

import time
import json
from pathlib import Path
from typing import Dict, Optional

import requests
from logging import Logger

from logging_setup import log_llm_call
from logging_setup import FileStats

OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"

_client = None


def init_openai_support(
    api_key: str,
    logger: Logger,
    *,
    base_log_extra: Optional[Dict[str, object]] = None,
    stats: Optional[FileStats] = None,
) -> None:
    global _client
    _client = OpenAIResponsesClient(
        api_key=api_key,
        logger=logger,
        base_log_extra=base_log_extra or {},
        stats=stats,
    )


def _extract_text_from_responses(data: object) -> str:
    if not isinstance(data, dict):
        return ""

    out_text = data.get("output_text")
    if isinstance(out_text, str) and out_text.strip():
        return out_text.strip()

    output = data.get("output")
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for c in content:
                if not isinstance(c, dict):
                    continue
                t = c.get("text")
                if isinstance(t, str) and t.strip():
                    parts.append(t)
        joined = "\n".join([p for p in parts if p.strip()]).strip()
        if joined:
            return joined

    # Fallbacks на случай иных форматов
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        ch0 = choices[0] if isinstance(choices[0], dict) else {}
        msg = ch0.get("message") if isinstance(ch0, dict) else {}
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, str):
                return content.strip()

    text = data.get("text")
    if isinstance(text, str):
        return text.strip()

    return ""


class OpenAIResponsesClient:
    def __init__(
        self,
        *,
        api_key: str,
        logger: Logger,
        base_log_extra: Optional[Dict[str, object]] = None,
        stats: Optional[FileStats] = None,
        timeout_seconds: int = 10,
    ):
        self.api_key = (api_key or "").strip()
        self.logger = logger
        self.base_log_extra = base_log_extra or {}
        self.stats = stats
        self.timeout_seconds = timeout_seconds

    def _log_llm_io(
        self,
        *,
        llm_io_path: Optional[Path],
        model_name: str,
        request_text: str,
        response_text: str,
        latency_seconds: float,
        status: str,
        error_type: Optional[str] = None,
        error_code: Optional[str] = None,
    ) -> None:
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
                    api_key_id="GPT_API_KEY",
                    request_text=request_text,
                    response_text=response_text,
                    latency_seconds=latency_seconds,
                    status=status,
                    error_type=error_type,
                    error_code=error_code,
                )
            except Exception:
                pass

        if self.stats is not None:
            try:
                self.stats.register_llm_call(
                    model=model_name,
                    api_key_id="GPT_API_KEY",
                    request_text=request_text,
                    response_text=response_text,
                    latency_seconds=latency_seconds,
                )
            except Exception:
                pass

    def _one_call(
        self,
        *,
        model: str,
        instructions: str,
        input_text: str,
        mode: str,
        stage: str,
        row_index: int,
    ) -> tuple[bool, int, str]:
        if not self.api_key:
            self.logger.error(
                "[%s] GPT_API_KEY не задан, OpenAI вызов пропущен (model=%s, mode=%s, row=%d).",
                stage,
                model,
                mode,
                row_index,
                extra={
                    **self.base_log_extra,
                    "stage": f"{stage}_openai_error",
                    "duration_seconds": 0,
                    "model": model,
                    "api_key_id": "GPT_API_KEY",
                },
            )
            return False, 0, ""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload: Dict[str, object] = {
            "model": model,
            "instructions": instructions or "",
            "input": input_text or "",
        }
        if mode == "flex":
            payload["service_tier"] = "flex"

        llm_io_path = self.base_log_extra.get("llm_io_path")
        if isinstance(llm_io_path, str):
            llm_io_path = Path(llm_io_path)

        request_text_for_io = (
            "[MODE]\n"
            f"{mode}\n\n"
            "[INSTRUCTIONS]\n"
            f"{instructions or ''}\n\n"
            "[INPUT]\n"
            f"{input_text or ''}"
        )

        send_start = time.perf_counter()

        self.logger.info(
            "[%s] Отправлен запрос к OpenAI (model=%s, mode=%s, row=%d).",
            stage,
            model,
            mode,
            row_index,
            extra={
                **self.base_log_extra,
                "stage": f"{stage}_openai_request",
                "duration_seconds": 0,
                "model": model,
                "api_key_id": "GPT_API_KEY",
            },
        )

        try:
            resp = requests.post(
                OPENAI_RESPONSES_URL,
                headers=headers,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                timeout=self.timeout_seconds,
            )
        except Exception as e:
            call_duration = time.perf_counter() - send_start
            self.logger.error(
                "[%s] Исключение при запросе к OpenAI для строки %d (model=%s, mode=%s): %s",
                stage,
                row_index,
                model,
                mode,
                e,
                exc_info=True,
                extra={
                    **self.base_log_extra,
                    "stage": f"{stage}_openai_error",
                    "duration_seconds": round(call_duration, 3),
                    "model": model,
                    "api_key_id": "GPT_API_KEY",
                },
            )
            self._log_llm_io(
                llm_io_path=llm_io_path,
                model_name=model,
                request_text=request_text_for_io,
                response_text=str(e),
                latency_seconds=call_duration,
                status="error",
                error_type=type(e).__name__,
                error_code=None,
            )
            return False, 0, ""

        call_duration = time.perf_counter() - send_start

        if resp.status_code != 200:
            body = (resp.text or "")[:1000]
            self.logger.error(
                "[%s] Ошибка OpenAI для строки %d (model=%s, mode=%s): HTTP %d: %s",
                stage,
                row_index,
                model,
                mode,
                resp.status_code,
                body,
                extra={
                    **self.base_log_extra,
                    "stage": f"{stage}_openai_error",
                    "duration_seconds": round(call_duration, 3),
                    "model": model,
                    "api_key_id": "GPT_API_KEY",
                },
            )
            self._log_llm_io(
                llm_io_path=llm_io_path,
                model_name=model,
                request_text=request_text_for_io,
                response_text=body,
                latency_seconds=call_duration,
                status="error",
                error_type="HTTPError",
                error_code=str(resp.status_code),
            )
            return False, int(resp.status_code), ""

        try:
            data = resp.json()
        except Exception as e:
            body = (resp.text or "")[:1000]
            self.logger.error(
                "[%s] Ошибка парсинга JSON от OpenAI для строки %d (model=%s, mode=%s): %s",
                stage,
                row_index,
                model,
                mode,
                e,
                exc_info=True,
                extra={
                    **self.base_log_extra,
                    "stage": f"{stage}_openai_error",
                    "duration_seconds": round(call_duration, 3),
                    "model": model,
                    "api_key_id": "GPT_API_KEY",
                },
            )
            self._log_llm_io(
                llm_io_path=llm_io_path,
                model_name=model,
                request_text=request_text_for_io,
                response_text=f"JSON parse error: {e}. Raw body: {body}",
                latency_seconds=call_duration,
                status="error",
                error_type=type(e).__name__,
                error_code=None,
            )
            return False, 0, ""

        text = _extract_text_from_responses(data).strip()

        self.logger.info(
            "[%s] Успешный ответ от OpenAI (model=%s, mode=%s, row=%d).",
            stage,
            model,
            mode,
            row_index,
            extra={
                **self.base_log_extra,
                "stage": f"{stage}_openai_response",
                "duration_seconds": round(call_duration, 3),
                "model": model,
                "api_key_id": "GPT_API_KEY",
            },
        )

        self._log_llm_io(
            llm_io_path=llm_io_path,
            model_name=model,
            request_text=request_text_for_io,
            response_text=text,
            latency_seconds=call_duration,
            status="success",
            error_type=None,
            error_code=None,
        )
        return True, 200, text

    def call_with_flex_fallback(
        self,
        *,
        model: str,
        instructions: str,
        input_text: str,
        stage: str,
        row_index: int,
    ) -> str:
        ok, code, text = self._one_call(
            model=model,
            instructions=instructions,
            input_text=input_text,
            mode="flex",
            stage=stage,
            row_index=row_index,
        )
        if ok:
            return text
        if code == 429:
            ok2, _code2, text2 = self._one_call(
                model=model,
                instructions=instructions,
                input_text=input_text,
                mode="normal",
                stage=stage,
                row_index=row_index,
            )
            return text2 if ok2 else ""
        return ""


def openai_call_with_flex_fallback(
    *,
    model: str,
    system: str,
    input: str,
    stage: str = "openai",
    row_index: int = -1,
) -> str:
    if _client is None:
        return ""
    try:
        return _client.call_with_flex_fallback(
            model=model,
            instructions=system,
            input_text=input,
            stage=stage,
            row_index=row_index,
        )
    except Exception:
        return ""
