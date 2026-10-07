"""Labelled datasets used as ground truth for semantic type inference.

Each entry pairs a DataFrame with the semantic type every column *should*
receive. These are deliberately realistic in shape - hundreds of rows, continuous
measurements that are nearly all distinct, and column names that contain 'id' or
'no' as substrings without being identifiers.

Drop real CSVs into tests/datasets/ alongside a <name>.expected.json mapping
column names to expected types and they are scored by the same harness.
"""

import json
import os

import numpy as np
import pandas as pd

DATASETS_DIR = os.path.join(os.path.dirname(__file__), "datasets")


def _rng(seed=42):
    return np.random.default_rng(seed)


def titanic_like(n=891):
    r = _rng(1)
    return pd.DataFrame({
        "PassengerId": np.arange(1, n + 1),
        "Survived": r.choice([0, 1], n),
        "Pclass": r.choice([1, 2, 3], n),
        "Name": [f"Surname{i}, Mr. Given{i}" for i in range(n)],
        "Sex": r.choice(["male", "female"], n),
        "Age": np.round(r.uniform(0.5, 80, n), 1),
        "Fare": np.round(r.uniform(4, 512, n), 4),
        "Cabin": [f"C{i%140}" for i in range(n)],
        "Embarked": r.choice(["S", "C", "Q"], n),
    }), {
        "PassengerId": "Identifier", "Survived": "Boolean", "Pclass": "Categorical",
        "Name": "Free Text", "Sex": "Categorical", "Age": "Numeric",
        "Fare": "Numeric", "Cabin": "Categorical", "Embarked": "Categorical",
    }


def housing_like(n=1460):
    r = _rng(2)
    return pd.DataFrame({
        "Id": np.arange(1, n + 1),
        "LotArea": r.integers(1300, 215245, n),
        "OverallQual": r.integers(1, 11, n),
        "YearBuilt": r.integers(1872, 2011, n),
        "GrLivArea": r.integers(334, 5642, n),
        "Neighborhood": r.choice([f"Nbhd{i}" for i in range(25)], n),
        "CentralAir": r.choice(["Y", "N"], n),
        "SalePrice": np.round(r.uniform(34900, 755000, n), 2),
    }), {
        "Id": "Identifier", "LotArea": "Numeric", "OverallQual": "Numeric",
        "YearBuilt": "Numeric", "GrLivArea": "Numeric",
        "Neighborhood": "Categorical", "CentralAir": "Boolean", "SalePrice": "Numeric",
    }


def weather_like(n=2000):
    r = _rng(3)
    return pd.DataFrame({
        "StationCode": [f"STN{i:05d}" for i in range(n)],
        "ObservationDate": pd.date_range("2020-01-01", periods=n).strftime("%Y-%m-%d"),
        "Humidity": np.round(r.uniform(10, 100, n), 2),
        "Temperature": np.round(r.normal(22, 8, n), 2),
        "WindSpeed": np.round(r.uniform(0, 120, n), 3),
        "Latitude": np.round(r.uniform(-90, 90, n), 5),
        "Longitude": np.round(r.uniform(-180, 180, n), 5),
        "ClimateZone": r.choice(["Arid", "Temperate", "Tropical", "Polar"], n),
        "Notes": [f"observation recorded by field team number {i} on site" for i in range(n)],
    }), {
        "StationCode": "Identifier", "ObservationDate": "Date", "Humidity": "Numeric",
        "Temperature": "Numeric", "WindSpeed": "Numeric", "Latitude": "Numeric",
        "Longitude": "Numeric", "ClimateZone": "Categorical", "Notes": "Free Text",
    }


def hr_like(n=1200):
    r = _rng(4)
    return pd.DataFrame({
        "EmployeeID": [f"EMP{i:06d}" for i in range(n)],
        "FullName": [f"Given{i} Family{i}" for i in range(n)],
        "WorkEmail": [f"user{i}@corp.example.com" for i in range(n)],
        "ContactNumber": [f"98{i%100000000:08d}" for i in range(n)],
        "Department": r.choice(["Sales", "HR", "Engineering", "Finance"], n),
        "JoiningDate": pd.date_range("2015-01-01", periods=n).strftime("%d-%m-%Y"),
        "Income": np.round(r.uniform(25000, 250000, n), 2),
        "IsActive": r.choice(["Yes", "No"], n),
        "Width": np.round(r.uniform(1, 50, n), 3),
        "BidAmount": np.round(r.uniform(10, 9000, n), 2),
    }), {
        "EmployeeID": "Identifier", "FullName": "Free Text", "WorkEmail": "Email",
        "ContactNumber": "Phone Number", "Department": "Categorical",
        "JoiningDate": "Date", "Income": "Numeric", "IsActive": "Boolean",
        "Width": "Numeric", "BidAmount": "Numeric",
    }


def retail_like(n=1500):
    r = _rng(5)
    return pd.DataFrame({
        "OrderNo": [f"ORD-{i:07d}" for i in range(n)],
        "CustomerEmail": [f"buyer{i%700}@shop.example.org" for i in range(n)],
        "ProductCategory": r.choice(["Electronics", "Clothing", "Home", "Toys", "Books"], n),
        "UnitPrice": np.round(r.uniform(1.5, 2500, n), 2),
        "Quantity": r.integers(1, 40, n),
        "OrderDate": pd.date_range("2023-01-01", periods=n).strftime("%Y-%m-%d"),
        "ZipCode": r.choice(["560001", "560002", "560003", "560004"], n),
        "PaymentStatus": r.choice(["true", "false"], n),
        "Diagnosis": r.choice(["A", "B", "C"], n),
    }), {
        "OrderNo": "Identifier", "CustomerEmail": "Email", "ProductCategory": "Categorical",
        "UnitPrice": "Numeric", "Quantity": "Numeric", "OrderDate": "Date",
        "ZipCode": "Categorical", "PaymentStatus": "Boolean", "Diagnosis": "Categorical",
    }


def small_messy(n=8):
    """The development-sized dataset, kept so small-input behaviour stays covered."""
    return pd.DataFrame({
        "StudentID": [101, 102, 103, 104, 105, 106, 107, 101],
        "Name": ["John Doe", "Alice Smith", "Bob Jones", "Charlie Brown",
                 "Diana Prince", "Ethan Hunt", "Fiona Gallagher", "John Doe"],
        "Email": ["john@gmail.com", "alice@gmail", "bob@yahoo.com", None,
                  "diana@gmail.com", "ethan@corp.com", "fiona@gmail.com", "john@gmail.com"],
        "Phone": ["9876543210", "123-456", "9998887776", "98765 43210",
                  None, "8887776665", "1234567", "9876543210"],
        "Age": [20, 21, None, 22, 999, 21, 20, 20],
        "GPA": [3.8, 3.9, 3.5, 6.5, None, 3.7, 3.6, 3.8],
        "EnrollmentDate": ["2023-09-01", "09/02/2023", "03-09-2023", "2023-09-04",
                           "2023-09-05", "2023-09-06", None, "2023-09-01"],
        "Major": ["CS", "cs", "Computer Science", "CS", "CSE", "unknown", "CS", "CS"],
    }), {
        "StudentID": "Identifier", "Name": "Free Text", "Email": "Email",
        "Phone": "Phone Number", "Age": "Numeric", "GPA": "Numeric",
        "EnrollmentDate": "Date", "Major": "Categorical",
    }


BUILTIN = {
    "titanic_like": titanic_like,
    "housing_like": housing_like,
    "weather_like": weather_like,
    "hr_like": hr_like,
    "retail_like": retail_like,
    "small_messy": small_messy,
}


def load_user_datasets():
    """Yields (name, df, expected) for real CSVs dropped into tests/datasets/.

    Pair each <name>.csv with a <name>.expected.json of {"column": "Type", ...}.
    """
    if not os.path.isdir(DATASETS_DIR):
        return
    for filename in sorted(os.listdir(DATASETS_DIR)):
        if not filename.lower().endswith(".csv"):
            continue
        base = os.path.splitext(filename)[0]
        spec_path = os.path.join(DATASETS_DIR, f"{base}.expected.json")
        if not os.path.exists(spec_path):
            continue
        try:
            with open(spec_path, "r", encoding="utf-8") as fh:
                expected = json.load(fh)
            from backend.utils.dataio import read_tabular
            with open(os.path.join(DATASETS_DIR, filename), "rb") as fh:
                df, _ = read_tabular(fh.read(), filename)
            yield base, df, expected
        except Exception as exc:
            print(f"  (could not load {filename}: {exc})")


def all_datasets():
    for name, builder in BUILTIN.items():
        df, expected = builder()
        yield name, df, expected
    for item in load_user_datasets():
        yield item
