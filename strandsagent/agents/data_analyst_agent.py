import io
import json
import time
import warnings

import numpy as np
import pandas as pd
from strands import tool

from tools.memory_tools import _save_analysis


def _run_analysis(csv_content: str) -> dict:
    """Run the full analysis locally with pandas — no sandbox needed."""
    warnings.filterwarnings("ignore")
    df = pd.read_csv(io.StringIO(csv_content))

    # ── 1. PROFILE ────────────────────────────────────────────────────────
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
            info.update({
                "min": float(s.min()) if not s.isna().all() else None,
                "max": float(s.max()) if not s.isna().all() else None,
                "mean": round(float(s.mean()), 4) if not s.isna().all() else None,
                "median": float(s.median()) if not s.isna().all() else None,
                "std": round(float(s.std()), 4) if not s.isna().all() else None,
                "skewness": round(float(s.skew()), 4) if len(s.dropna()) > 2 else None,
                "kurtosis": round(float(s.kurtosis()), 4) if len(s.dropna()) > 3 else None,
            })
            if not s.isna().all():
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                info["outlier_count"] = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
                info["distribution_shape"] = (
                    "roughly_normal" if info["skewness"] is not None and abs(info["skewness"]) < 0.5
                    else ("right_skewed" if (info["skewness"] or 0) > 0 else "left_skewed")
                )
        else:
            info["top_values"] = s.value_counts().head(5).to_dict()
        col_stats[col] = info

    # ── 2. REMOVE DUPLICATES ─────────────────────────────────────────────
    preprocessing_steps = []
    n_before = len(df)
    df.drop_duplicates(inplace=True)
    n_removed = n_before - len(df)
    if n_removed > 0:
        preprocessing_steps.append(f"Removed {n_removed} duplicate rows")

    # ── 3. CLEAN MISSING VALUES ──────────────────────────────────────────
    nan_actions = {}
    for col in list(df.columns):
        n_missing = int(df[col].isna().sum())
        if n_missing == 0:
            continue
        missing_pct = n_missing / len(df) * 100
        if missing_pct > 50:
            df.drop(columns=[col], inplace=True)
            nan_actions[col] = {"strategy": "drop_column", "reason": f"{missing_pct:.1f}% missing"}
            preprocessing_steps.append(f"Dropped column '{col}' ({missing_pct:.1f}% missing)")
        elif pd.api.types.is_numeric_dtype(df[col]):
            skew = df[col].skew() if len(df[col].dropna()) > 2 else 0
            if abs(skew) > 1:
                val = df[col].median()
                df[col].fillna(val, inplace=True)
                nan_actions[col] = {"strategy": "median", "value": float(val)}
                preprocessing_steps.append(f"Filled '{col}' NaNs with median ({val:.4g})")
            else:
                val = df[col].mean()
                df[col].fillna(val, inplace=True)
                nan_actions[col] = {"strategy": "mean", "value": round(float(val), 4)}
                preprocessing_steps.append(f"Filled '{col}' NaNs with mean ({val:.4g})")
        else:
            val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
            df[col].fillna(val, inplace=True)
            nan_actions[col] = {"strategy": "mode", "value": str(val)}
            preprocessing_steps.append(f"Filled '{col}' NaNs with mode ('{val}')")

    # ── 4. NORMALIZE NUMERIC COLUMNS ─────────────────────────────────────
    normalization = {}
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    for col in numeric_cols:
        s = df[col]
        col_std = s.std()
        if col_std == 0:
            continue
        col_range = s.max() - s.min()
        if col_range == 0:
            continue

        skew = s.skew() if len(s.dropna()) > 2 else 0
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        outlier_count = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
        outlier_pct = outlier_count / len(s) * 100

        if skew > 1 and (s >= 0).all():
            method = "log"
            df[col] = np.log1p(s)
        elif outlier_pct > 5:
            method = "robust"
            median = s.median()
            df[col] = (s - median) / iqr if iqr != 0 else s
        elif abs(skew) < 0.5:
            method = "min-max"
            df[col] = (s - s.min()) / col_range
        else:
            method = "z-score"
            df[col] = (s - s.mean()) / col_std

        normalization[col] = {"method": method}
        preprocessing_steps.append(f"Normalized '{col}' using {method}")

    # ── 5. CORRELATIONS ──────────────────────────────────────────────────
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    strong_correlations = []
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr().round(4)
        for i, c1 in enumerate(numeric_cols):
            for c2 in numeric_cols[i + 1:]:
                r = float(corr.loc[c1, c2])
                if abs(r) > 0.5:
                    strength = "strong" if abs(r) > 0.8 else "moderate"
                    direction = "positive" if r > 0 else "negative"
                    strong_correlations.append({
                        "col_a": c1, "col_b": c2,
                        "r": r, "strength": strength, "direction": direction,
                    })

    # ── 6. CLEANED SUMMARY ───────────────────────────────────────────────
    cleaned_summary = {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_total": int(df.isna().sum().sum()),
        "sample_rows": df.head(3).to_dict(orient="records"),
    }

    return {
        "profile": {**profile, "column_stats": col_stats},
        "preprocessing_steps": preprocessing_steps,
        "nan_cleaning": nan_actions,
        "normalization": normalization,
        "correlations": strong_correlations,
        "cleaned_summary": cleaned_summary,
    }


def _analyze_dataset(user_id: str, csv_content: str) -> str:
    """Plain helper: run analysis and save results."""
    t0 = time.time()

    print("[analyze_dataset] Running analysis...", flush=True)
    summary = _run_analysis(csv_content)
    print(f"[analyze_dataset] Analysis done ({time.time() - t0:.1f}s)", flush=True)

    print("[analyze_dataset] Saving results...", flush=True)
    _save_analysis(username=user_id, summary=summary)
    print(f"[analyze_dataset] Done ({time.time() - t0:.1f}s)", flush=True)

    return json.dumps(summary, indent=2, default=str)


@tool
def analyze_dataset(user_id: str, csv_content: str) -> str:
    """
    Run a comprehensive analysis on the CSV locally with pandas.
    Results are stored in memory for the tutor.

    Args:
        user_id: The student's username (used as storage key).
        csv_content: Raw text content of the uploaded CSV file.
    """
    return _analyze_dataset(user_id, csv_content)
