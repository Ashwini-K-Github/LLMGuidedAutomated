"""Checks the loader survives the file shapes that arrive from the real world."""

import sys, os, json, math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from backend.utils.dataio import read_tabular, FileLoadError, df_to_records, sanitize_for_json, normalize_columns
from backend.cleaning.profiler import profile_dataset
from backend.cleaning.statistical import detect_anomalies


def _is_strict_json(obj) -> bool:
    """json.dumps happily emits the bare tokens NaN/Infinity, which no browser accepts."""
    text = json.dumps(obj)
    try:
        json.loads(text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
        return True
    except ValueError:
        return False


def run(results):
    # Encodings
    payload = "Name,City,Salary\nJose Garcia,Malaga,50000\nFrancois,Nimes,60000\n"
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            df, meta = read_tabular(payload.replace("Jose", "José").encode(encoding), "clients.csv")
            results.check(f"reads {encoding} CSV", df.shape == (2, 3), f"shape={df.shape}")
        except Exception as exc:
            results.fail(f"reads {encoding} CSV", str(exc))

    # Delimiters
    for delim, label in ((";", "semicolon"), ("\t", "tab"), ("|", "pipe")):
        text = f"Name{delim}City{delim}Salary\nAnna{delim}Berlin{delim}50000\nBo{delim}Oslo{delim}60000\n"
        df, meta = read_tabular(text.encode("utf-8"), "export.csv")
        results.check(f"detects {label} delimiter", df.shape[1] == 3, f"cols={df.shape[1]}")

    # Column labels that are not strings
    messy = pd.DataFrame({2020: [1, 2], 2021: [3, 4], None: [5, 6], "A": [7, 8], "A ": [9, 10]})
    fixed = normalize_columns(messy.copy())
    results.check("numeric/blank/duplicate headers normalised",
                  list(fixed.columns) == ["2020", "2021", "Column_3", "A", "A_1"],
                  str(list(fixed.columns)))
    try:
        profile_dataset(fixed)
        results.ok("profiling works after header normalisation")
    except Exception as exc:
        results.fail("profiling works after header normalisation", str(exc))

    # Non-finite values must never reach the browser
    df = pd.DataFrame({"ratio": [1.0, 2.0, float("inf"), float("-inf"), float("nan")],
                       "label": ["a", "b", "c", "d", "e"]})
    body = sanitize_for_json({
        "profile": profile_dataset(df),
        "anomalies": detect_anomalies(df),
        "preview": df_to_records(df, limit=10),
    })
    results.check("response is strictly valid JSON with inf/NaN present", _is_strict_json(body))

    # Pathological shapes should not raise
    shapes = {
        "all-empty column": pd.DataFrame({"A": [1, 2, 3], "Empty": [None, None, None]}),
        "single row": pd.DataFrame({"A": [1], "B": ["x"]}),
        "zero rows": pd.DataFrame({"A": pd.Series([], dtype=float)}),
        "datetime dtype": pd.DataFrame({"D": pd.to_datetime(["2024-01-01", None])}),
        "mixed types": pd.DataFrame({"M": [1, "two", 3.0, None, True]}),
    }
    for label, frame in shapes.items():
        try:
            profile_dataset(frame)
            detect_anomalies(frame)
            results.ok(f"profiles without raising: {label}")
        except Exception as exc:
            results.fail(f"profiles without raising: {label}", f"{type(exc).__name__}: {exc}")

    # Errors are actionable
    for data, filename, expect in (
        (b"", "empty.csv", "empty"),
        (b"whatever", "notes.docx", "Unsupported"),
    ):
        try:
            read_tabular(data, filename)
            results.fail(f"rejects {filename}", "no error raised")
        except FileLoadError as exc:
            results.check(f"rejects {filename} with a clear message", expect.lower() in str(exc).lower(), str(exc))

    # Size guard
    try:
        read_tabular(b"x" * (51 * 1024 * 1024), "huge.csv")
        results.fail("enforces the 50MB limit", "no error raised")
    except FileLoadError as exc:
        results.check("enforces the 50MB limit", "exceeds" in str(exc), str(exc))
