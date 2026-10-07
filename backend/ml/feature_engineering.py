import numpy as np
import pandas as pd

# Above this many distinct levels a categorical is frequency-encoded instead of
# one-hot encoded, to stop the design matrix exploding on high-cardinality columns.
MAX_ONEHOT_LEVELS = 25
# Levels covering less than this share of the training rows are folded into a
# single "__other__" bucket so rare noise does not become its own dimension.
RARE_LEVEL_MIN_SHARE = 0.01

_OTHER = "__other__"
_MISSING = "__missing__"


class FeatureEngineer:
    """
    Preprocessing and feature engineering for train/test splits, fitting every
    statistic on the training set only.

    Two representations are produced from the same fitted state:

    * the *model* matrix - one-hot encoded categoricals and standardised
      numerics, which is what the estimators are trained on. Nominal categories
      used to be label-encoded to arbitrary alphabetical integers, which implies
      an order that does not exist and misleads every linear and distance-based
      model.
    * the *selection* matrix - one column per original feature, with categoricals
      as integer codes. Mutual information is invariant to how a discrete
      variable is coded, so this gives one honest score per source column
      instead of a score per dummy column.
    """

    def __init__(self, target_col: str, feature_cols: list, col_types: dict,
                 max_onehot_levels: int = MAX_ONEHOT_LEVELS):
        self.target_col = target_col
        self.feature_cols = list(feature_cols)
        self.col_types = col_types
        self.max_onehot_levels = max_onehot_levels

        # Learned from the training data
        self.numeric_sources = []      # numeric + boolean + date-derived
        self.onehot_sources = []       # low-cardinality categoricals
        self.frequency_sources = []    # high-cardinality categoricals
        self.date_sources = []

        self.impute_values = {}        # source/derived col -> fill value
        self.scaling_params = {}       # numeric col -> {mean, std}
        self.category_levels = {}      # source -> kept levels
        self.category_has_other = {}   # source -> bool
        self.frequency_maps = {}       # source -> {level: share}
        self.category_codes = {}       # source -> {level: int} (selection matrix)

        self.feature_names = []        # model matrix column names
        self.feature_groups = {}       # model column -> source column
        self.selection_names = []      # one entry per source feature
        self.selection_discrete = []   # bool mask aligned to selection_names

        self.target_mapping = {}       # classification target label -> code
        self.target_classes = []       # ordered class labels
        self.target_is_categorical = False

        # Backwards-compatible alias used by older callers/tests
        self.category_mappings = self.category_codes
        self.engineered_features = []

    # ------------------------------------------------------------------ dates
    def _date_parts(self, col: str, series: pd.Series) -> dict:
        """Calendar parts plus cyclical encodings for month and weekday.

        Month 12 and month 1 are adjacent in time but maximally distant as plain
        integers, so the sine/cosine pair is what lets a model see seasonality.
        """
        dt = pd.to_datetime(series, errors="coerce")
        month = dt.dt.month
        dow = dt.dt.dayofweek
        return {
            f"{col}_year": dt.dt.year,
            f"{col}_month": month,
            f"{col}_day": dt.dt.day,
            f"{col}_dayofweek": dow,
            f"{col}_month_sin": np.sin(2 * np.pi * month / 12.0),
            f"{col}_month_cos": np.cos(2 * np.pi * month / 12.0),
            f"{col}_dow_sin": np.sin(2 * np.pi * dow / 7.0),
            f"{col}_dow_cos": np.cos(2 * np.pi * dow / 7.0),
        }

    def _expand(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds date-derived numeric columns; leaves everything else untouched."""
        out = df.copy()
        for col in self.feature_cols:
            if self.col_types.get(col) == "Date" and col in out.columns:
                try:
                    for name, values in self._date_parts(col, out[col]).items():
                        out[name] = values
                except Exception as exc:
                    print(f"Error parsing date column {col}: {exc}")
        return out

    @staticmethod
    def _as_label(value) -> str:
        if pd.isna(value):
            return _MISSING
        return str(value).strip()

    # -------------------------------------------------------------------- fit
    def fit(self, train_df: pd.DataFrame):
        expanded = self._expand(train_df)

        self.numeric_sources, self.onehot_sources = [], []
        self.frequency_sources, self.date_sources = [], []

        for col in self.feature_cols:
            ctype = self.col_types.get(col, "Numeric")

            if ctype == "Date":
                self.date_sources.append(col)
                for name in self._date_parts(col, train_df[col]) if col in train_df.columns else {}:
                    self.numeric_sources.append(name)
                continue

            if col not in expanded.columns:
                continue

            if ctype in ("Categorical", "Boolean", "Free Text", "Email", "Phone Number", "Identifier"):
                labels = expanded[col].map(self._as_label)
                counts = labels.value_counts(normalize=True)
                if len(counts) > self.max_onehot_levels:
                    self.frequency_sources.append(col)
                    self.frequency_maps[col] = counts.to_dict()
                else:
                    kept = [lvl for lvl, share in counts.items() if share >= RARE_LEVEL_MIN_SHARE]
                    if not kept:
                        kept = list(counts.index)
                    self.onehot_sources.append(col)
                    self.category_levels[col] = sorted(kept)
                    self.category_has_other[col] = len(kept) < len(counts)
                # Integer codes for the selection matrix (coding is irrelevant to MI)
                self.category_codes[col] = {lvl: i for i, lvl in enumerate(sorted(counts.index))}
            else:
                self.numeric_sources.append(col)

        # Imputation and scaling parameters for every numeric column
        for col in self.numeric_sources:
            if col not in expanded.columns:
                self.impute_values[col] = 0.0
                self.scaling_params[col] = {"mean": 0.0, "std": 1.0}
                continue
            values = pd.to_numeric(expanded[col], errors="coerce")
            median = values.median()
            median = 0.0 if pd.isna(median) else float(median)
            self.impute_values[col] = median
            filled = values.fillna(median)
            std = float(filled.std())
            self.scaling_params[col] = {
                "mean": float(filled.mean()),
                "std": std if std and std > 0 else 1.0,
            }

        # Frequency-encoded columns are scaled too, so they share the numeric range
        for col in self.frequency_sources:
            shares = expanded[col].map(self._as_label).map(self.frequency_maps[col]).fillna(0.0)
            std = float(shares.std())
            self.scaling_params[col] = {"mean": float(shares.mean()), "std": std if std and std > 0 else 1.0}

        self._build_schema()
        self._fit_target(train_df)
        return self

    def _build_schema(self):
        """Fixes the model-matrix column order and the source-column mapping."""
        self.feature_names, self.feature_groups = [], {}

        date_derived = {}
        for src in self.date_sources:
            date_derived[src] = [n for n in self.numeric_sources if n.startswith(f"{src}_")]

        for col in self.feature_cols:
            ctype = self.col_types.get(col, "Numeric")
            if ctype == "Date":
                for name in date_derived.get(col, []):
                    self.feature_names.append(name)
                    self.feature_groups[name] = col
            elif col in self.onehot_sources:
                for level in self.category_levels[col]:
                    name = f"{col}={level}"
                    self.feature_names.append(name)
                    self.feature_groups[name] = col
                if self.category_has_other.get(col):
                    name = f"{col}={_OTHER}"
                    self.feature_names.append(name)
                    self.feature_groups[name] = col
            elif col in self.frequency_sources:
                name = f"{col}__freq"
                self.feature_names.append(name)
                self.feature_groups[name] = col
            elif col in self.numeric_sources:
                self.feature_names.append(col)
                self.feature_groups[col] = col

        self.engineered_features = list(self.feature_names)

        # Selection matrix: exactly one column per original feature
        self.selection_names, self.selection_discrete = [], []
        for col in self.feature_cols:
            ctype = self.col_types.get(col, "Numeric")
            if ctype == "Date":
                for name in date_derived.get(col, []):
                    self.selection_names.append(name)
                    self.selection_discrete.append(False)
            elif col in self.onehot_sources or col in self.frequency_sources:
                self.selection_names.append(col)
                self.selection_discrete.append(True)
            elif col in self.numeric_sources:
                self.selection_names.append(col)
                self.selection_discrete.append(False)

    def _fit_target(self, train_df: pd.DataFrame):
        if self.target_col not in train_df.columns:
            return
        series = train_df[self.target_col]
        ttype = self.col_types.get(self.target_col, "Numeric")
        self.target_is_categorical = ttype in ("Categorical", "Boolean", "Free Text", "Identifier")

        if self.target_is_categorical:
            labels = series.map(self._as_label)
            mode = labels.mode()
            self.impute_values[self.target_col] = mode.iloc[0] if not mode.empty else _MISSING
            self.target_classes = sorted(labels.dropna().unique().tolist())
            self.target_mapping = {lbl: i for i, lbl in enumerate(self.target_classes)}
        else:
            values = pd.to_numeric(series, errors="coerce")
            median = values.median()
            self.impute_values[self.target_col] = 0.0 if pd.isna(median) else float(median)

    # -------------------------------------------------------------- transform
    def _numeric_column(self, frame: pd.DataFrame, col: str) -> np.ndarray:
        fill = self.impute_params(col)
        if col not in frame.columns:
            return np.zeros(len(frame), dtype=float)
        values = pd.to_numeric(frame[col], errors="coerce").fillna(fill)
        params = self.scaling_params.get(col, {"mean": 0.0, "std": 1.0})
        return ((values - params["mean"]) / params["std"]).to_numpy(dtype=float)

    def impute_params(self, col):
        return self.impute_values.get(col, 0.0)

    def transform(self, df: pd.DataFrame) -> tuple:
        """Returns (X, y, feature_names) for the one-hot encoded model matrix."""
        expanded = self._expand(df)
        columns = {}

        for col in self.feature_cols:
            ctype = self.col_types.get(col, "Numeric")
            if ctype == "Date":
                for name in [n for n in self.feature_names if self.feature_groups.get(n) == col]:
                    columns[name] = self._numeric_column(expanded, name)
            elif col in self.onehot_sources:
                labels = (expanded[col].map(self._as_label) if col in expanded.columns
                          else pd.Series([_MISSING] * len(expanded), index=expanded.index))
                kept = self.category_levels[col]
                for level in kept:
                    columns[f"{col}={level}"] = (labels == level).to_numpy(dtype=float)
                if self.category_has_other.get(col):
                    columns[f"{col}={_OTHER}"] = (~labels.isin(kept)).to_numpy(dtype=float)
            elif col in self.frequency_sources:
                labels = (expanded[col].map(self._as_label) if col in expanded.columns
                          else pd.Series([_MISSING] * len(expanded), index=expanded.index))
                shares = labels.map(self.frequency_maps[col]).fillna(0.0)
                params = self.scaling_params.get(col, {"mean": 0.0, "std": 1.0})
                columns[f"{col}__freq"] = ((shares - params["mean"]) / params["std"]).to_numpy(dtype=float)
            elif col in self.numeric_sources:
                columns[col] = self._numeric_column(expanded, col)

        if self.feature_names:
            X = np.column_stack([columns[name] for name in self.feature_names])
        else:
            X = np.zeros((len(df), 0))

        return X, self.transform_target(df), list(self.feature_names)

    def transform_target(self, df: pd.DataFrame):
        if self.target_col not in df.columns:
            return None
        series = df[self.target_col]
        if self.target_is_categorical:
            labels = series.map(self._as_label)
            fill = self.impute_values.get(self.target_col, _MISSING)
            labels = labels.fillna(fill)
            return labels.map(lambda v: self.target_mapping.get(v, -1)).to_numpy()
        fill = self.impute_values.get(self.target_col, 0.0)
        return pd.to_numeric(series, errors="coerce").fillna(fill).to_numpy(dtype=float)

    def transform_for_selection(self, df: pd.DataFrame) -> tuple:
        """Returns (X, names, discrete_mask) with one column per original feature."""
        expanded = self._expand(df)
        columns = []
        for name, is_discrete in zip(self.selection_names, self.selection_discrete):
            if is_discrete:
                labels = (expanded[name].map(self._as_label) if name in expanded.columns
                          else pd.Series([_MISSING] * len(expanded), index=expanded.index))
                codes = labels.map(lambda v: self.category_codes.get(name, {}).get(v, -1))
                columns.append(codes.to_numpy(dtype=float))
            else:
                columns.append(self._numeric_column(expanded, name))

        X = np.column_stack(columns) if columns else np.zeros((len(df), 0))
        return X, list(self.selection_names), list(self.selection_discrete)

    def model_columns_for(self, sources: list) -> list:
        """Indices of the model-matrix columns belonging to the given source features."""
        wanted = set(sources)
        return [i for i, name in enumerate(self.feature_names)
                if self.feature_groups.get(name, name) in wanted or name in wanted]
