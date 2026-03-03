import json
import os

from strands import tool

from tools.csv_tools import upload_csv_to_s3
from tools.code_interpreter import run_analysis
from tools.memory_tools import save_analysis

# This code runs in the AgentCore Code Interpreter sandbox.
# It produces a comprehensive analysis that becomes the tutor's private "answer key".
# The results are stored in memory but NOT shown verbatim to the student.
_ANALYSIS_CODE = """
import pandas as pd
import numpy as np
import json, warnings
warnings.filterwarnings('ignore')

df_raw = pd.read_csv('dataset.csv')

# ── 1. PROFILING ──────────────────────────────────────────────────────────────
profile = {
    "shape": {"rows": int(df_raw.shape[0]), "columns": int(df_raw.shape[1])},
    "columns": df_raw.columns.tolist(),
    "dtypes": df_raw.dtypes.astype(str).to_dict(),
    "duplicate_rows": int(df_raw.duplicated().sum()),
}

col_stats = {}
for col in df_raw.columns:
    s = df_raw[col]
    info = {
        "dtype": str(s.dtype),
        "missing": int(s.isna().sum()),
        "missing_pct": round(s.isna().mean() * 100, 2),
        "unique": int(s.nunique()),
        "cardinality": "high" if s.nunique() > 50 else "low",
    }
    if pd.api.types.is_numeric_dtype(s):
        info.update({
            "min": float(s.min()), "max": float(s.max()),
            "mean": float(s.mean()), "median": float(s.median()),
            "std": float(s.std()), "skewness": float(s.skew()),
            "kurtosis": float(s.kurtosis()),
        })
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        outlier_count = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
        info["outlier_count"] = outlier_count
        info["distribution_shape"] = (
            "roughly_normal" if abs(info["skewness"]) < 0.5
            else ("right_skewed" if info["skewness"] > 0 else "left_skewed")
        )
    else:
        info["top_values"] = s.value_counts().head(5).to_dict()
    col_stats[col] = info

# ── 2. NaN CLEANING ───────────────────────────────────────────────────────────
df = df_raw.copy()
nan_actions = {}
for col in df.columns:
    n_missing = int(df[col].isna().sum())
    if n_missing == 0:
        continue
    if pd.api.types.is_numeric_dtype(df[col]):
        fill_val = df[col].median()
        df[col].fillna(fill_val, inplace=True)
        nan_actions[col] = {"count": n_missing, "strategy": "median", "value": float(fill_val)}
    else:
        fill_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
        df[col].fillna(fill_val, inplace=True)
        nan_actions[col] = {"count": n_missing, "strategy": "mode", "value": str(fill_val)}

# ── 3. NORMALIZATION CHECK ────────────────────────────────────────────────────
numeric_cols = df.select_dtypes(include='number').columns.tolist()
norm_actions = {}
for col in numeric_cols:
    col_range = df[col].max() - df[col].min()
    if col_range > 100:
        col_min, col_max = float(df[col].min()), float(df[col].max())
        df[col] = (df[col] - col_min) / (col_max - col_min)
        norm_actions[col] = {"original_min": col_min, "original_max": col_max, "method": "min-max"}

# ── 4. CORRELATION MATRIX ─────────────────────────────────────────────────────
corr_info = {}
if len(numeric_cols) >= 2:
    corr_matrix = df[numeric_cols].corr().round(3)
    strong = []
    for i, c1 in enumerate(numeric_cols):
        for c2 in numeric_cols[i + 1:]:
            r = float(corr_matrix.loc[c1, c2])
            if abs(r) > 0.5:
                strength = "strong" if abs(r) > 0.8 else "moderate"
                direction = "positive" if r > 0 else "negative"
                strong.append({"col_a": c1, "col_b": c2, "r": r, "strength": strength, "direction": direction})
    corr_info["strong_correlations"] = strong

# ── OUTPUT ────────────────────────────────────────────────────────────────────
result = {
    "profile": profile,
    "column_stats": col_stats,
    "nan_cleaning": nan_actions,
    "normalization": norm_actions,
    "correlations": corr_info,
    "cleaned_sample": df.head(3).to_dict(orient='records'),
}
print(json.dumps(result, default=str))
"""


@tool
def analyze_dataset(user_id: str, csv_content: str) -> str:
    """
    Run a comprehensive expert analysis on the CSV, persist the results,
    and return a structured summary for the tutor's internal use only.

    Args:
        user_id: The student's username (used as storage key).
        csv_content: Raw text content of the uploaded CSV file.
    """
    import time
    t0 = time.time()

    # Step 1: Run the analysis code in the Code Interpreter sandbox
    print(f"[analyze_dataset] Step 1/3: Running code interpreter...", flush=True)
    analysis_output = run_analysis(
        csv_content=csv_content,
        code=_ANALYSIS_CODE,
    )
    print(f"[analyze_dataset] Step 1/3 done ({time.time() - t0:.1f}s)", flush=True)

    # Step 2: Upload the raw CSV to S3 for future sessions
    print(f"[analyze_dataset] Step 2/3: Uploading CSV to S3...", flush=True)
    upload_csv_to_s3(
        user_id=user_id,
        csv_content=csv_content,
    )
    print(f"[analyze_dataset] Step 2/3 done ({time.time() - t0:.1f}s)", flush=True)

    # Step 3: Parse & save the analysis results to memory
    print(f"[analyze_dataset] Step 3/3: Saving analysis to memory...", flush=True)
    try:
        summary = json.loads(analysis_output)
    except (json.JSONDecodeError, TypeError):
        summary = {"raw_output": str(analysis_output)}

    save_analysis(
        username=user_id,
        summary=summary,
    )
    print(f"[analyze_dataset] Step 3/3 done ({time.time() - t0:.1f}s)", flush=True)

    return json.dumps(summary, indent=2, default=str)
