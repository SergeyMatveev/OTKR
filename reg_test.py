from __future__ import annotations

import sys
import time
import shutil
import re
from datetime import datetime
from pathlib import Path
from logging import Logger

import pandas as pd

from config import ensure_directories, DATA_DIR
from logging_setup import setup_logging, FileStats, append_stats_row
from pipeline import process_nbki_md
from report_builder import build_credit_report_from_csv
from llm_support import extract_urgent_debt, parse_stage4_response


TEST_MD_PATH = Path("tests/test1.md")
EXPECTED_CSV_PATH = Path("tests/result1.csv")

TEST_TELEGRAM_USER_ID = "464483163"
TEST_TELEGRAM_USERNAME = "LOCAL_REG_TEST"

_DEBT_COL_NAMES = {"Сумма задолженности", "Задолженность"}


def normalize_debt_value(v: str) -> str:
    s = "" if v is None else str(v)
    s = s.replace("\u00A0", " ").replace("\u202F", " ")
    s_strip = s.strip()
    if s_strip == "":
        return s_strip

    compact = re.sub(r"\s+", "", s_strip)
    compact_upper = compact.upper()

    if compact_upper in ("Н/Д", "H/Д"):
        return "ND_OR_ZERO"

    if re.fullmatch(r"0(?:[.,]0{1,2})?", compact) is not None:
        return "ND_OR_ZERO"

    return s_strip


def _now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _read_csv_strict(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    except Exception:
        df = pd.read_csv(path, encoding="utf-8", dtype=str)
    return df.fillna("")


def _sanitize_field_name_for_comment(col: str) -> str:
    c = (col or "").strip()
    if c == "Сумма задолженности":
        return "Задолженность"
    return c


def _diff_comment(cols: list[str], *, is_expected: bool) -> str:
    cols_norm = [_sanitize_field_name_for_comment(c) for c in cols if c]
    cols_norm = [c for c in cols_norm if c]
    if not cols_norm:
        return "Строка не совпадает."
    if len(cols_norm) == 1:
        field_part = f"Поле {cols_norm[0]}"
    else:
        field_part = f"Поля {', '.join(cols_norm)}"
    if is_expected:
        return f"{field_part} не совпадает."
    return f"{field_part} не совпадает с ожидаемым результатом"


def _build_diff_csv(
    actual_df: pd.DataFrame,
    expected_df: pd.DataFrame,
    *,
    columns: list[str],
) -> pd.DataFrame:
    out_rows: list[dict[str, str]] = []

    def _empty_row() -> dict[str, str]:
        d = {c: "" for c in columns}
        d["Комментарий"] = ""
        return d

    def _row_from_df(df: pd.DataFrame, i: int) -> dict[str, str]:
        d = {c: (df.at[i, c] if c in df.columns else "") for c in columns}
        d["Комментарий"] = ""
        return d

    n_actual = len(actual_df)
    n_expected = len(expected_df)
    n = max(n_actual, n_expected)

    for i in range(n):
        if i >= n_actual:
            # нет строки в actual
            r1 = _empty_row()
            r1["Комментарий"] = "Эта строка из результата обработки. Строка отсутствует (в эталоне есть)."
            r2 = _row_from_df(expected_df, i)
            r2["Комментарий"] = "Эта строка из эталонного файла. Строка отсутствует в результате обработки."
            out_rows.append(r1)
            out_rows.append(r2)
            continue

        if i >= n_expected:
            # лишняя строка в actual
            r1 = _row_from_df(actual_df, i)
            r1["Комментарий"] = "Эта строка из результата обработки. Лишняя строка (в эталоне отсутствует)."
            r2 = _empty_row()
            r2["Комментарий"] = "Эта строка из эталонного файла. Строка отсутствует (в результате обработки лишняя)."
            out_rows.append(r1)
            out_rows.append(r2)
            continue

        # обе строки есть
        diffs: list[str] = []
        for c in columns:
            av = actual_df.at[i, c] if c in actual_df.columns else ""
            ev = expected_df.at[i, c] if c in expected_df.columns else ""
            if c in _DEBT_COL_NAMES:
                if normalize_debt_value("" if av is None else str(av)) != normalize_debt_value(
                    "" if ev is None else str(ev)
                ):
                    diffs.append(c)
            else:
                if ("" if av is None else str(av)) != ("" if ev is None else str(ev)):
                    diffs.append(c)

        if not diffs:
            # совпало полностью — просто записываем строку
            r = _row_from_df(actual_df, i)
            out_rows.append(r)
            continue

        # расхождение — две строки подряд (actual, затем expected)
        r1 = _row_from_df(actual_df, i)
        r1["Комментарий"] = (
            "Эта строка из результата обработки. " + _diff_comment(diffs, is_expected=False)
        )
        r2 = _row_from_df(expected_df, i)
        r2["Комментарий"] = (
            "Эта строка из эталонного файла. " + _diff_comment(diffs, is_expected=True)
        )
        out_rows.append(r1)
        out_rows.append(r2)

    diff_df = pd.DataFrame(out_rows, columns=columns + ["Комментарий"])
    return diff_df


def _save_report_messages(messages: list[str], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for i, msg in enumerate(messages, start=1):
            f.write(f"----- MESSAGE {i} -----\n")
            f.write(msg)
            if not msg.endswith("\n"):
                f.write("\n")
            f.write("\n")


def _run_extract_urgent_debt_unit_tests(logger: Logger, *, request_id: str) -> bool:
    cases: list[tuple[str, str, str]] = [
        (
            "multiline_row_merge",
            """|  11-12-2024 | H/Д
448003,36 | 346199,67 | 83268,10  |
|  29-12-2024 | Да
443201,36 | 346199,67 | 83268,10  |
|  28-01-2025 | Нет
443201,36 | 346199,67 | 83268,10  |
|  27-02-2025 | Нет
443201,36 | 346199,67 | 83268,10  |
|  29-03-2025 | Нет
443201,36 | 346199,67 | 83268,10  |
|  28-04-2025 | Нет
443201,36 | 346199,67 | 83268,10  |
|  28-05-2025 | Нет
443201,36 | 346199,67 | 83268,10  |
|  19-06-2025 | Да
438193,36 | 346199,67 | 80223,10  |

Срочная задолженность
""",
            "438193,36",
        ),
        (
            "single_line_row_ok",
            """|  28-04-2025 | Нет | 272996,60 | 193330,98 | 75199,98 | 4465,64 | Нет  |
|  27-05-2025 | Да | 272996,60 | 193330,98 | 75199,98 | 4465,64 | Нет  |
""",
            "272996,60",
        ),
        (
            "dot_date_row_ok",
            """|  14.12.2021 | Н/Д | 32934,00 | Н/Д | Н/Д | Н/Д | Нет  |
""",
            "32934,00",
        ),
    ]

    for name, input_text, expected in cases:
        got = parse_stage4_response(extract_urgent_debt(input_text))
        if got != expected:
            logger.error(
                "REG_TEST: unit-test extract_urgent_debt failed (%s): expected=%s, got=%s",
                name,
                expected,
                got,
                extra={
                    "stage": "reg_test_unit_failed",
                    "telegram_user_id": TEST_TELEGRAM_USER_ID,
                    "telegram_username": TEST_TELEGRAM_USERNAME,
                    "request_id": request_id,
                    "duration_seconds": 0,
                    "model": "N/A",
                    "api_key_id": "N/A",
                    "file_name": TEST_MD_PATH.name,
                },
            )
            return False

    logger.info(
        "REG_TEST: unit-test extract_urgent_debt OK (cases=%d).",
        len(cases),
        extra={
            "stage": "reg_test_unit_ok",
            "telegram_user_id": TEST_TELEGRAM_USER_ID,
            "telegram_username": TEST_TELEGRAM_USERNAME,
            "request_id": request_id,
            "duration_seconds": 0,
            "model": "N/A",
            "api_key_id": "N/A",
            "file_name": TEST_MD_PATH.name,
        },
    )
    return True


def main() -> int:
    ensure_directories()
    logger, _log_path = setup_logging()

    overall_start = time.perf_counter()

    if not TEST_MD_PATH.exists():
        print("FAIL")
        return 1
    if not EXPECTED_CSV_PATH.exists():
        print("FAIL")
        return 1

    ts = _now_ts()
    md_stem = TEST_MD_PATH.stem
    user_dir = DATA_DIR / TEST_TELEGRAM_USER_ID
    request_dir = user_dir / f"{ts}_{md_stem}"
    request_dir.mkdir(parents=True, exist_ok=True)
    request_id = request_dir.name

    start_dt = datetime.strptime(ts, "%Y%m%d_%H%M%S")
    file_stats = FileStats(
        start_dt=start_dt,
        telegram_user_id=TEST_TELEGRAM_USER_ID,
        telegram_username=TEST_TELEGRAM_USERNAME,
        request_id=request_id,
        pdf_filename=TEST_MD_PATH.name,
    )

    processing_error = False
    result_csv_path: Path | None = None

    local_md_path = request_dir / TEST_MD_PATH.name

    try:
        shutil.copy2(TEST_MD_PATH, local_md_path)

        logger.info(
            "REG_TEST: старт. MD=%s, expected=%s, request_dir=%s",
            local_md_path,
            EXPECTED_CSV_PATH,
            request_dir,
            extra={
                "stage": "reg_test_start",
                "telegram_user_id": TEST_TELEGRAM_USER_ID,
                "telegram_username": TEST_TELEGRAM_USERNAME,
                "request_id": request_id,
                "duration_seconds": 0,
                "model": "N/A",
                "api_key_id": "N/A",
                "file_name": TEST_MD_PATH.name,
            },
        )

        if not _run_extract_urgent_debt_unit_tests(logger, request_id=request_id):
            raise RuntimeError("REG_TEST: extract_urgent_debt unit tests failed.")

        result_csv_path = process_nbki_md(
            md_path=local_md_path,
            original_name=TEST_MD_PATH.name,
            request_dir=request_dir,
            request_ts=ts,
            logger=logger,
            file_stats=file_stats,
        )

        report_messages = build_credit_report_from_csv(
            result_csv_path,
            logger,
            telegram_user_id=TEST_TELEGRAM_USER_ID,
            telegram_username=TEST_TELEGRAM_USERNAME,
            request_id=request_id,
            file_name=TEST_MD_PATH.name,
        )

        _save_report_messages(report_messages, request_dir / "report_messages.txt")

        file_stats.mark_success()

    except Exception as e:
        processing_error = True
        duration = time.perf_counter() - overall_start
        logger.error(
            "REG_TEST: ошибка пайплайна: %s",
            e,
            exc_info=True,
            extra={
                "stage": "reg_test_error",
                "telegram_user_id": TEST_TELEGRAM_USER_ID,
                "telegram_username": TEST_TELEGRAM_USERNAME,
                "request_id": request_id,
                "duration_seconds": round(duration, 3),
                "model": "N/A",
                "api_key_id": "N/A",
                "file_name": TEST_MD_PATH.name,
            },
        )
        file_stats.mark_error()

    finally:
        append_stats_row(file_stats.to_row())

    if processing_error or result_csv_path is None:
        print("FAIL")
        return 1

    # Сравнение итогового CSV (после report_builder, уже с "Решение") с expected
    try:
        actual_df = _read_csv_strict(result_csv_path)
        expected_df = _read_csv_strict(EXPECTED_CSV_PATH)
    except Exception:
        print("FAIL")
        return 1

    ok = True
    if list(actual_df.columns) != list(expected_df.columns):
        ok = False
    elif len(actual_df) != len(expected_df):
        ok = False
    else:
        cols = list(actual_df.columns)
        if not any(c in _DEBT_COL_NAMES for c in cols):
            ok = actual_df.equals(expected_df)
        else:
            ok = True
            for i in range(len(actual_df)):
                for c in cols:
                    av = actual_df.at[i, c]
                    ev = expected_df.at[i, c]
                    if c in _DEBT_COL_NAMES:
                        if normalize_debt_value("" if av is None else str(av)) != normalize_debt_value(
                            "" if ev is None else str(ev)
                        ):
                            ok = False
                            break
                    else:
                        if ("" if av is None else str(av)) != ("" if ev is None else str(ev)):
                            ok = False
                            break
                if not ok:
                    break

    if ok:
        print("PASS")
        return 0

    # FAIL по сравнению: сохраняем артефакты, но пайплайн успешен (file_stats уже success)
    try:
        shutil.copy2(EXPECTED_CSV_PATH, request_dir / "expected.csv")
    except Exception:
        pass
    try:
        shutil.copy2(result_csv_path, request_dir / "actual.csv")
    except Exception:
        pass

    try:
        if list(actual_df.columns) != list(expected_df.columns):
            cols_comment = [
                f"ОШИБКА: колонки не совпадают.",
                f"ACTUAL: {list(actual_df.columns)}",
                f"EXPECTED: {list(expected_df.columns)}",
            ]
            diff_df = pd.DataFrame(
                [{"Комментарий": " | ".join(cols_comment)}],
                columns=["Комментарий"],
            )
        else:
            cols = list(actual_df.columns)
            diff_df = _build_diff_csv(actual_df, expected_df, columns=cols)

        diff_path = request_dir / "diff.csv"
        diff_df.to_csv(diff_path, index=False, encoding="utf-8-sig")
    except Exception:
        pass

    print("FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())


