"""File loading and JSON serialisation helpers.

Real-world spreadsheets are rarely well-formed UTF-8 comma-separated files. The
upload endpoint used to call pd.read_csv() directly, so a Latin-1 export, a
semicolon-separated European CSV or an Excel sheet with numeric headers returned
an HTTP error and the wizard never advanced past step 1. These helpers make the
loader tolerant and make every response strictly valid JSON.
"""

import io
import csv
import math
import re

import numpy as np
import pandas as pd

# Tried in order; the first that decodes without error wins.
CANDIDATE_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")
CANDIDATE_DELIMITERS = (",", ";", "\t", "|")

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # matches the 50MB advertised in the UI


class FileLoadError(Exception):
    """Raised with a human-readable, actionable message when loading fails."""


def _decode(contents: bytes) -> tuple:
    """Returns (text, encoding_used). Falls back to lossy latin-1 rather than failing."""
    for enc in CANDIDATE_ENCODINGS:
        try:
            return contents.decode(enc), enc
        except (UnicodeDecodeError, LookupError):
            continue
    return contents.decode("latin-1", errors="replace"), "latin-1 (with replacements)"


def _sniff_delimiter(text: str) -> str:
    """Picks the delimiter that yields the most columns on the header line."""
    sample = text[:65536]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="".join(CANDIDATE_DELIMITERS))
        if dialect.delimiter in CANDIDATE_DELIMITERS:
            return dialect.delimiter
    except csv.Error:
        pass

    header = sample.splitlines()[0] if sample.splitlines() else ""
    best, best_count = ",", 0
    for delim in CANDIDATE_DELIMITERS:
        count = len(next(csv.reader([header], delimiter=delim), []))
        if count > best_count:
            best, best_count = delim, count
    return best


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Forces column labels to non-empty, unique strings.

    Excel sheets whose header row holds numbers (e.g. years) produce integer
    labels, which used to crash profiling with
    AttributeError: 'int' object has no attribute 'lower'.
    """
    seen = {}
    new_cols = []
    for i, col in enumerate(df.columns):
        # Render whole-number labels as integers so an Excel year header
        # becomes "2020" rather than "2020.0".
        if isinstance(col, float) and not math.isnan(col) and col.is_integer():
            col = int(col)
        name = str(col).strip()
        if not name or name.lower() in ("nan", "none"):
            name = f"Column_{i + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        new_cols.append(name)
    df.columns = new_cols
    return df


def read_tabular(contents: bytes, filename: str) -> tuple:
    """Reads CSV/Excel/ODS bytes into a DataFrame.

    Returns (df, meta) where meta records how the file was interpreted so the UI
    can show it. Raises FileLoadError with an actionable message on failure.
    """
    if not contents:
        raise FileLoadError("The uploaded file is empty.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise FileLoadError(
            f"File is {len(contents) / 1024 / 1024:.1f} MB, which exceeds the "
            f"{MAX_UPLOAD_BYTES // 1024 // 1024} MB limit."
        )

    name = (filename or "").lower()
    meta = {"filename": filename}

    if name.endswith(".csv") or name.endswith(".txt") or name.endswith(".tsv"):
        text, encoding = _decode(contents)
        delimiter = "\t" if name.endswith(".tsv") else _sniff_delimiter(text)
        meta.update({"format": "csv", "encoding": encoding, "delimiter": delimiter})
        try:
            df = pd.read_csv(io.StringIO(text), sep=delimiter)
        except Exception as exc:
            raise FileLoadError(f"Could not parse the CSV: {exc}") from exc
        # A single wide column usually means the delimiter guess was wrong.
        if df.shape[1] == 1:
            for alt in CANDIDATE_DELIMITERS:
                if alt == delimiter:
                    continue
                try:
                    alt_df = pd.read_csv(io.StringIO(text), sep=alt)
                except Exception:
                    continue
                if alt_df.shape[1] > df.shape[1]:
                    df, meta["delimiter"] = alt_df, alt

    elif name.endswith(".xlsx") or name.endswith(".xlsm"):
        meta["format"] = "xlsx"
        try:
            df = pd.read_excel(io.BytesIO(contents), engine="openpyxl")
        except ImportError as exc:
            raise FileLoadError("Reading .xlsx needs the 'openpyxl' package: pip install openpyxl") from exc
        except Exception as exc:
            raise FileLoadError(f"Could not read the Excel file: {exc}") from exc

    elif name.endswith(".xls"):
        meta["format"] = "xls"
        try:
            df = pd.read_excel(io.BytesIO(contents), engine="xlrd")
        except ImportError as exc:
            raise FileLoadError(
                "Reading legacy .xls files needs the 'xlrd' package "
                "(pip install xlrd), or re-save the file as .xlsx."
            ) from exc
        except Exception as exc:
            raise FileLoadError(f"Could not read the .xls file: {exc}") from exc

    elif name.endswith(".ods"):
        meta["format"] = "ods"
        try:
            df = pd.read_excel(io.BytesIO(contents), engine="odf")
        except ImportError as exc:
            raise FileLoadError("Reading .ods needs the 'odfpy' package: pip install odfpy") from exc
        except Exception as exc:
            raise FileLoadError(f"Could not read the .ods file: {exc}") from exc

    else:
        raise FileLoadError(
            "Unsupported file format. Upload a .csv, .tsv, .xlsx, .xls or .ods file."
        )

    if df is None or df.shape[1] == 0:
        raise FileLoadError("No columns could be read from the file.")

    df = normalize_columns(df)
    meta["rows"], meta["columns"] = int(len(df)), int(df.shape[1])
    return df, meta


def safe_number(value):
    """Returns a JSON-safe float, or None for NaN/Infinity."""
    if value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(num) or math.isinf(num):
        return None
    return num


def sanitize_for_json(obj):
    """Recursively replaces NaN/Infinity with None.

    json.dumps() emits the bare tokens NaN and Infinity, which are not valid JSON;
    the browser's response.json() then throws and the upload appears to fail for
    no reason. Everything leaving the API goes through here.
    """
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return None if (math.isnan(val) or math.isinf(val)) else val
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if obj is pd.NaT:
        return None
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    try:
        if obj is not None and not isinstance(obj, (str, bool, int)) and pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return obj


def df_to_records(df: pd.DataFrame, limit: int = None) -> list:
    """Converts a DataFrame to JSON-safe row dicts (NaN/NaT/Infinity become null)."""
    if df is None:
        return []
    frame = df.head(limit) if limit is not None else df
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.astype(object).where(pd.notnull(frame), None)
    return sanitize_for_json(frame.to_dict(orient="records"))
