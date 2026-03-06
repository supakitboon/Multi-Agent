"""
Individual preprocessing tools for the data analyst agent.

Each tool runs code inside a shared CodeInterpreterSession so the sandbox
state (loaded DataFrame, cleaned columns, etc.) persists across calls.

Usage:
    session = CodeInterpreterSession()
    session.start()
    session.upload_csv(csv_content)
    tools = create_preprocessing_tools(session)
    # pass `tools` to a Strands Agent
"""

import json
from strands import tool as strands_tool
from tools.code_interpreter import CodeInterpreterSession


def create_preprocessing_tools(session: CodeInterpreterSession) -> list:
    """Create preprocessing tools bound to an active sandbox session.

    The session must already be started and have the CSV uploaded as
    ``dataset.csv`` before the agent calls any of these tools.
    """

    # ------------------------------------------------------------------
    # 1. DATA PROFILING
    # ------------------------------------------------------------------
    @strands_tool
    def profile_dataset() -> str:
        """Profile the dataset: shape, dtypes, missing values, basic stats,
        outlier counts, and cardinality for every column. Always call this
        first so you can decide which preprocessing steps are needed.
        """
        code = """
import pandas as pd, numpy as np, json, warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('dataset.csv')

profile = {
    "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
    "columns": df.columns.tolist(),
    "dtypes": df.dtypes.astype(str).to_dict(),
    "duplicate_rows": int(df.duplicated().sum()),
}

col_stats = {}
for col in df.columns:
    s = df[col]
    info = {
        "dtype": str(s.dtype),
        "missing": int(s.isna().sum()),
        "missing_pct": round(s.isna().mean() * 100, 2),
        "unique": int(s.nunique()),
        "cardinality": "high" if s.nunique() > 50 else "low",
    }
    if pd.api.types.is_numeric_dtype(s):
        desc = s.describe()
        info.update({
            "min": float(s.min()), "max": float(s.max()),
            "mean": round(float(s.mean()), 4),
            "median": float(s.median()),
            "std": round(float(s.std()), 4),
            "skewness": round(float(s.skew()), 4),
            "kurtosis": round(float(s.kurtosis()), 4),
        })
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        info["outlier_count"] = int(((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum())
        info["distribution_shape"] = (
            "roughly_normal" if abs(info["skewness"]) < 0.5
            else ("right_skewed" if info["skewness"] > 0 else "left_skewed")
        )
    else:
        info["top_values"] = s.value_counts().head(5).to_dict()

    col_stats[col] = info

print(json.dumps({"profile": profile, "column_stats": col_stats}, default=str))
"""
        return session.run_code(code)

    # ------------------------------------------------------------------
    # 2. MISSING VALUE CLEANING
    # ------------------------------------------------------------------
    @strands_tool
    def clean_missing_values(column_strategies: str) -> str:
        """Clean missing values in the dataset.

        Args:
            column_strategies: A JSON object mapping column names to their
                cleaning strategy. Supported strategies:
                - "median"  : fill with column median  (numeric only)
                - "mean"    : fill with column mean    (numeric only)
                - "mode"    : fill with most frequent value
                - "drop"    : drop rows where this column is NaN
                - "ffill"   : forward fill
                - "bfill"   : backward fill
                - "constant:<value>" : fill with a literal value
                Example: {"age": "median", "name": "mode", "score": "constant:0"}
        """
        code = f"""
import pandas as pd, json

strategies = json.loads('''{column_strategies}''')
df = pd.read_csv('dataset.csv')
actions = {{}}

for col, strategy in strategies.items():
    if col not in df.columns:
        actions[col] = {{"error": "column not found"}}
        continue
    n_before = int(df[col].isna().sum())
    if n_before == 0:
        actions[col] = {{"skipped": True, "reason": "no missing values"}}
        continue

    if strategy == "median":
        val = df[col].median()
        df[col].fillna(val, inplace=True)
    elif strategy == "mean":
        val = round(df[col].mean(), 4)
        df[col].fillna(val, inplace=True)
    elif strategy == "mode":
        val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
        df[col].fillna(val, inplace=True)
    elif strategy == "drop":
        df.dropna(subset=[col], inplace=True)
        val = None
    elif strategy == "ffill":
        df[col].ffill(inplace=True)
        val = None
    elif strategy == "bfill":
        df[col].bfill(inplace=True)
        val = None
    elif strategy.startswith("constant:"):
        val = strategy.split(":", 1)[1]
        if pd.api.types.is_numeric_dtype(df[col]):
            val = float(val)
        df[col].fillna(val, inplace=True)
    else:
        actions[col] = {{"error": f"unknown strategy: {{strategy}}"}}
        continue

    n_after = int(df[col].isna().sum())
    actions[col] = {{
        "strategy": strategy,
        "filled": n_before - n_after,
        "remaining_missing": n_after,
    }}

df.to_csv('dataset.csv', index=False)
print(json.dumps({{"nan_cleaning": actions, "shape_after": list(df.shape)}}, default=str))
"""
        return session.run_code(code)

    # ------------------------------------------------------------------
    # 3. NORMALIZATION / SCALING
    # ------------------------------------------------------------------
    @strands_tool
    def normalize_columns(columns: str, method: str) -> str:
        """Normalize or scale numeric columns.

        Args:
            columns: JSON array of column names to normalize.
                Example: ["price", "quantity", "age"]
            method: Normalization method. One of:
                - "min-max"  : scale to [0, 1]
                - "z-score"  : standardize to mean=0, std=1
                - "log"      : apply log1p transform (good for right-skewed data)
                - "robust"   : scale using median and IQR (outlier-resistant)
        """
        code = f"""
import pandas as pd, numpy as np, json

cols = json.loads('''{columns}''')
method = "{method}"
df = pd.read_csv('dataset.csv')
actions = {{}}

for col in cols:
    if col not in df.columns:
        actions[col] = {{"error": "column not found"}}
        continue
    if not pd.api.types.is_numeric_dtype(df[col]):
        actions[col] = {{"error": "not numeric"}}
        continue

    original_min = float(df[col].min())
    original_max = float(df[col].max())
    original_mean = round(float(df[col].mean()), 4)
    original_std = round(float(df[col].std()), 4)

    if method == "min-max":
        col_min, col_max = df[col].min(), df[col].max()
        if col_max - col_min == 0:
            actions[col] = {{"error": "zero range, cannot normalize"}}
            continue
        df[col] = (df[col] - col_min) / (col_max - col_min)
    elif method == "z-score":
        col_mean, col_std = df[col].mean(), df[col].std()
        if col_std == 0:
            actions[col] = {{"error": "zero std, cannot standardize"}}
            continue
        df[col] = (df[col] - col_mean) / col_std
    elif method == "log":
        if (df[col] < 0).any():
            actions[col] = {{"error": "negative values, cannot apply log"}}
            continue
        df[col] = np.log1p(df[col])
    elif method == "robust":
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            actions[col] = {{"error": "zero IQR, cannot robust-scale"}}
            continue
        median = df[col].median()
        df[col] = (df[col] - median) / iqr

    actions[col] = {{
        "method": method,
        "original_range": [original_min, original_max],
        "original_mean": original_mean,
        "new_min": round(float(df[col].min()), 4),
        "new_max": round(float(df[col].max()), 4),
        "new_mean": round(float(df[col].mean()), 4),
    }}

df.to_csv('dataset.csv', index=False)
print(json.dumps({{"normalization": actions}}, default=str))
"""
        return session.run_code(code)

    # ------------------------------------------------------------------
    # 4. OUTLIER DETECTION
    # ------------------------------------------------------------------
    @strands_tool
    def detect_outliers(columns: str, method: str) -> str:
        """Detect outliers in numeric columns.

        Args:
            columns: JSON array of column names to check.
                Example: ["price", "age"]
            method: Detection method. One of:
                - "iqr"     : values beyond 1.5 * IQR from Q1/Q3
                - "z-score" : values with |z| > 3
        """
        code = f"""
import pandas as pd, numpy as np, json

cols = json.loads('''{columns}''')
method = "{method}"
df = pd.read_csv('dataset.csv')
results = {{}}

for col in cols:
    if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
        results[col] = {{"error": "column not found or not numeric"}}
        continue

    s = df[col].dropna()
    if method == "iqr":
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (s < lower) | (s > upper)
    elif method == "z-score":
        z = (s - s.mean()) / s.std()
        mask = z.abs() > 3
        lower, upper = float(s.mean() - 3*s.std()), float(s.mean() + 3*s.std())

    outlier_count = int(mask.sum())
    results[col] = {{
        "method": method,
        "outlier_count": outlier_count,
        "outlier_pct": round(outlier_count / len(s) * 100, 2),
        "bounds": [round(float(lower), 4), round(float(upper), 4)],
    }}
    if outlier_count > 0 and outlier_count <= 10:
        results[col]["outlier_values"] = s[mask].tolist()

print(json.dumps({{"outlier_detection": results}}, default=str))
"""
        return session.run_code(code)

    # ------------------------------------------------------------------
    # 5. CATEGORICAL ENCODING
    # ------------------------------------------------------------------
    @strands_tool
    def encode_categoricals(columns: str, method: str) -> str:
        """Encode categorical columns into numeric representations.

        Args:
            columns: JSON array of column names to encode.
                Example: ["status", "category"]
            method: Encoding method. One of:
                - "label"   : integer label encoding (0, 1, 2, ...)
                - "one-hot" : one-hot / dummy encoding (drops first)
        """
        code = f"""
import pandas as pd, json

cols = json.loads('''{columns}''')
method = "{method}"
df = pd.read_csv('dataset.csv')
actions = {{}}

for col in cols:
    if col not in df.columns:
        actions[col] = {{"error": "column not found"}}
        continue

    unique_vals = df[col].nunique()
    if method == "label":
        mapping = {{v: i for i, v in enumerate(df[col].dropna().unique())}}
        df[col] = df[col].map(mapping)
        actions[col] = {{"method": "label", "mapping": {{str(k): v for k, v in mapping.items()}}}}
    elif method == "one-hot":
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=int)
        df = df.drop(columns=[col])
        df = pd.concat([df, dummies], axis=1)
        actions[col] = {{"method": "one-hot", "new_columns": dummies.columns.tolist()}}

df.to_csv('dataset.csv', index=False)
print(json.dumps({{"encoding": actions, "columns_after": df.columns.tolist(), "shape_after": list(df.shape)}}, default=str))
"""
        return session.run_code(code)

    # ------------------------------------------------------------------
    # 6. REMOVE DUPLICATES
    # ------------------------------------------------------------------
    @strands_tool
    def remove_duplicates() -> str:
        """Remove duplicate rows from the dataset."""
        code = """
import pandas as pd, json

df = pd.read_csv('dataset.csv')
n_before = len(df)
df.drop_duplicates(inplace=True)
n_after = len(df)
df.to_csv('dataset.csv', index=False)
print(json.dumps({"duplicates_removed": n_before - n_after, "rows_remaining": n_after}))
"""
        return session.run_code(code)

    # ------------------------------------------------------------------
    # 7. CORRELATION ANALYSIS
    # ------------------------------------------------------------------
    @strands_tool
    def compute_correlations() -> str:
        """Compute the correlation matrix for all numeric columns and
        report pairs with moderate (|r| > 0.5) or strong (|r| > 0.8)
        correlations."""
        code = """
import pandas as pd, json

df = pd.read_csv('dataset.csv')
numeric_cols = df.select_dtypes(include='number').columns.tolist()
result = {"numeric_columns": numeric_cols, "strong_correlations": []}

if len(numeric_cols) >= 2:
    corr = df[numeric_cols].corr().round(4)
    for i, c1 in enumerate(numeric_cols):
        for c2 in numeric_cols[i+1:]:
            r = float(corr.loc[c1, c2])
            if abs(r) > 0.5:
                strength = "strong" if abs(r) > 0.8 else "moderate"
                direction = "positive" if r > 0 else "negative"
                result["strong_correlations"].append({
                    "col_a": c1, "col_b": c2,
                    "r": r, "strength": strength, "direction": direction,
                })

print(json.dumps(result, default=str))
"""
        return session.run_code(code)

    # ------------------------------------------------------------------
    # 8. GET CLEANED DATA SUMMARY
    # ------------------------------------------------------------------
    @strands_tool
    def get_cleaned_summary() -> str:
        """Return a summary of the current (cleaned) dataset: shape, dtypes,
        sample rows, and basic stats. Call this after all preprocessing is
        done to get the final state."""
        code = """
import pandas as pd, json

df = pd.read_csv('dataset.csv')
summary = {
    "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
    "columns": df.columns.tolist(),
    "dtypes": df.dtypes.astype(str).to_dict(),
    "missing_total": int(df.isna().sum().sum()),
    "sample_rows": df.head(3).to_dict(orient='records'),
}
print(json.dumps(summary, default=str))
"""
        return session.run_code(code)

    return [
        profile_dataset,
        clean_missing_values,
        normalize_columns,
        detect_outliers,
        encode_categoricals,
        remove_duplicates,
        compute_correlations,
        get_cleaned_summary,
    ]
