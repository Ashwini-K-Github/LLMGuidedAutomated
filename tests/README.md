# Cocoon verification suite

```
python tests/run_all.py
```

Exits non-zero when a check fails. Sections that need scikit-learn are skipped
with a message if it is not installed.

## What is covered

| Section | Verifies |
|---|---|
| 1. Semantic type inference | Every column of six labelled datasets is typed correctly, and overall accuracy is at least 95% |
| 2. Regression guards | Specific misclassifications that previously broke real datasets, e.g. `Humidity` read as an identifier or `AccountNumber` read as a phone number |
| 3. Upload robustness | Encodings, delimiters, non-string headers, non-finite values, pathological shapes, actionable errors, the size limit |
| 4. Problem type resolution | A continuous target requested as classification is corrected rather than accepted |
| 5. Preprocessing isolation | Scaling and encoding statistics come only from the training partition, and unseen categories do not collide with known ones |
| 6. Feature selection and recommendation | A non-monotonic categorical driver is retained, noise is discarded, target leakage is flagged, and a model with no signal is reported as not beating its baseline |

## Adding your own datasets

This is the part to use when testing against real or other students' data. Put a
CSV in `tests/datasets/` with a sibling `<name>.expected.json` naming the
semantic type you expect for each column:

```
tests/datasets/titanic.csv
tests/datasets/titanic.expected.json
```

```json
{
  "PassengerId": "Identifier",
  "Survived": "Boolean",
  "Age": "Numeric",
  "Name": "Free Text",
  "Embarked": "Categorical"
}
```

Valid types: `Numeric`, `Categorical`, `Boolean`, `Date`, `Email`,
`Phone Number`, `Identifier`, `Free Text`.

The file is loaded through the same loader the upload endpoint uses, so encoding
and delimiter detection are exercised too. Every added dataset is scored into the
overall accuracy figure, which gives you a number to quote rather than an
impression: *"semantic type inference: N% accuracy across M unseen datasets"*.

Datasets worth adding, each stressing a different axis:

- **Titanic** - mixed types, missing values, a high-cardinality cabin code
- **Ames Housing** - 80 columns, many categorical
- **Adult Income** - imbalanced binary target
- **Wine Quality** - ordinal numeric target
- **Bike Sharing** - datetime and seasonality
- a European export - semicolon separator, comma decimals, non-UTF-8 encoding
- anything above 100k rows - upload responsiveness

`tests/datasets/` is git-ignored apart from this documentation, so large files
are not committed.
