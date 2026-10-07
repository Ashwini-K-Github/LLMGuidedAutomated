"""Scores semantic type inference against labelled ground truth."""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.cleaning.profiler import infer_semantic_type
from tests.synthetic import all_datasets


def run(results):
    total_cols = 0
    correct_cols = 0
    per_dataset = []

    for name, df, expected in all_datasets():
        hits, misses = 0, []
        for column, want in expected.items():
            if column not in df.columns:
                misses.append(f"{column} (missing from data)")
                continue
            got = infer_semantic_type(column, df[column])
            if got == want:
                hits += 1
            else:
                misses.append(f"{column}: expected {want}, got {got}")
        n = len(expected)
        total_cols += n
        correct_cols += hits
        per_dataset.append((name, hits, n, misses))
        results.check(
            f"{name}: {hits}/{n} columns typed correctly",
            hits == n,
            "" if hits == n else "; ".join(misses),
        )

    accuracy = (correct_cols / total_cols * 100) if total_cols else 0.0
    print(f"\n  Overall semantic type accuracy: {correct_cols}/{total_cols} = {accuracy:.1f}%")
    results.check("overall accuracy >= 95%", accuracy >= 95.0, f"{accuracy:.1f}%")


def run_regression_guards(results):
    """Specific misclassifications that previously broke real datasets."""
    import numpy as np
    import pandas as pd

    r = np.random.default_rng(11)
    n = 600
    cases = [
        # A substring of an id keyword in the name must not make it an identifier
        ("Humidity", np.round(r.uniform(20, 95, n), 2), "Numeric"),
        ("Width", np.round(r.uniform(1, 50, n), 3), "Numeric"),
        ("Residual", np.round(r.normal(0, 1, n), 4), "Numeric"),
        ("VideoLength", np.round(r.uniform(1, 300, n), 2), "Numeric"),
        # High uniqueness alone must not make a measurement an identifier
        ("Salary", r.integers(30000, 120000, n).astype(float), "Numeric"),
        ("Latitude", np.round(r.uniform(-90, 90, n), 6), "Numeric"),
        # A 10-digit account number is not a phone number
        ("AccountNumber", r.permutation(np.arange(10**9, 10**9 + n)), "Identifier"),
        # A name column is not phone data, and must not be nullified as such
        ("ContactPerson", [f"Given{i} Family{i}" for i in range(n)], "Free Text"),
        # An address is not an email address
        ("MailingAddress", [f"{i} Main Street, Apt {i % 30}" for i in range(n)], "Free Text"),
        # Genuine identifiers and structured columns still resolve
        ("StudentID", [f"STU{1000 + i}" for i in range(n)], "Identifier"),
        ("Phone", [f"98765{i % 100000:05d}" for i in range(n)], "Phone Number"),
        ("Email", [f"u{i}@mail.com" for i in range(n)], "Email"),
    ]
    for column, values, want in cases:
        got = infer_semantic_type(column, pd.Series(values))
        results.equals(f"regression guard: {column}", got, want)
