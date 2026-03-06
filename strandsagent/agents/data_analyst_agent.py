import io
import json
import os
import time
import warnings

import pandas as pd
from botocore.config import Config
from strands import Agent, tool
from strands.models import BedrockModel

from tools.memory_tools import _save_analysis

_BOTO_CONFIG = Config(read_timeout=300, connect_timeout=60)

# ═══════════════════════════════════════════════════════════════════════
# Individual analysis step functions (used by both deterministic & LLM paths)
# ═══════════════════════════════════════════════════════════════════════

def _profile(csv_content: str) -> dict:
    """Profile the dataset: shape, dtypes, missing values, per-column stats."""
    warnings.filterwarnings("ignore")
    df = pd.read_csv(io.StringIO(csv_content))

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

    profile["column_stats"] = col_stats
    return profile


def _remove_duplicates(csv_content: str) -> dict:
    """Remove duplicate rows and return the count removed."""
    df = pd.read_csv(io.StringIO(csv_content))
    n_before = len(df)
    df.drop_duplicates(inplace=True)
    n_removed = n_before - len(df)
    return {
        "duplicates_removed": n_removed,
        "rows_before": n_before,
        "rows_after": len(df),
    }


def _clean_missing(csv_content: str) -> dict:
    """Analyze and clean missing values with appropriate strategies."""
    df = pd.read_csv(io.StringIO(csv_content))
    df.drop_duplicates(inplace=True)

    nan_actions = {}
    steps = []
    for col in list(df.columns):
        n_missing = int(df[col].isna().sum())
        if n_missing == 0:
            continue
        missing_pct = n_missing / len(df) * 100
        if missing_pct > 50:
            df.drop(columns=[col], inplace=True)
            nan_actions[col] = {"strategy": "drop_column", "reason": f"{missing_pct:.1f}% missing"}
            steps.append(f"Dropped column '{col}' ({missing_pct:.1f}% missing)")
        elif pd.api.types.is_numeric_dtype(df[col]):
            skew = df[col].skew() if len(df[col].dropna()) > 2 else 0
            if abs(skew) > 1:
                val = df[col].median()
                df[col].fillna(val, inplace=True)
                nan_actions[col] = {"strategy": "median", "value": float(val)}
                steps.append(f"Filled '{col}' NaNs with median ({val:.4g})")
            else:
                val = df[col].mean()
                df[col].fillna(val, inplace=True)
                nan_actions[col] = {"strategy": "mean", "value": round(float(val), 4)}
                steps.append(f"Filled '{col}' NaNs with mean ({val:.4g})")
        else:
            val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
            df[col].fillna(val, inplace=True)
            nan_actions[col] = {"strategy": "mode", "value": str(val)}
            steps.append(f"Filled '{col}' NaNs with mode ('{val}')")

    return {"nan_cleaning": nan_actions, "steps": steps}


def _detect_outliers(csv_content: str) -> dict:
    """Detect outliers in numeric columns using the IQR method."""
    df = pd.read_csv(io.StringIO(csv_content))
    outliers = {}
    for col in df.select_dtypes(include="number").columns:
        s = df[col].dropna()
        if len(s) < 4:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        count = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
        if count > 0:
            outliers[col] = {
                "count": count,
                "pct": round(count / len(s) * 100, 2),
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(iqr),
            }
    return {"outliers": outliers}


def _compute_correlations(csv_content: str) -> dict:
    """Find strong correlations (|r| > 0.5) between numeric columns."""
    df = pd.read_csv(io.StringIO(csv_content))
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    strong = []
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr().round(4)
        for i, c1 in enumerate(numeric_cols):
            for c2 in numeric_cols[i + 1:]:
                r = float(corr.loc[c1, c2])
                if abs(r) > 0.5:
                    strength = "strong" if abs(r) > 0.8 else "moderate"
                    direction = "positive" if r > 0 else "negative"
                    strong.append({
                        "col_a": c1, "col_b": c2,
                        "r": r, "strength": strength, "direction": direction,
                    })
    return {"correlations": strong}


def _normalize(csv_content: str) -> dict:
    """Determine appropriate normalization methods for numeric columns."""
    df = pd.read_csv(io.StringIO(csv_content))
    normalization = {}
    steps = []
    for col in df.select_dtypes(include="number").columns:
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
        elif outlier_pct > 5:
            method = "robust"
        elif abs(skew) < 0.5:
            method = "min-max"
        else:
            method = "z-score"

        normalization[col] = {"method": method}
        steps.append(f"Normalized '{col}' using {method}")

    return {"normalization": normalization, "steps": steps}


# ═══════════════════════════════════════════════════════════════════════
# Deterministic full analysis (fast path — runs ALL steps)
# ═══════════════════════════════════════════════════════════════════════

def _run_analysis(csv_content: str) -> dict:
    """Run the full analysis locally with pandas — no sandbox needed."""
    profile = _profile(csv_content)
    duplicates = _remove_duplicates(csv_content)
    missing = _clean_missing(csv_content)
    outlier_info = _detect_outliers(csv_content)
    correlations = _compute_correlations(csv_content)
    normalization = _normalize(csv_content)

    preprocessing_steps = []
    if duplicates["duplicates_removed"] > 0:
        preprocessing_steps.append(f"Removed {duplicates['duplicates_removed']} duplicate rows")
    preprocessing_steps.extend(missing["steps"])
    preprocessing_steps.extend(normalization["steps"])

    # Build cleaned summary from profile
    df = pd.read_csv(io.StringIO(csv_content))
    df.drop_duplicates(inplace=True)
    cleaned_summary = {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_total": int(df.isna().sum().sum()),
        "sample_rows": df.head(3).to_dict(orient="records"),
    }

    return {
        "profile": profile,
        "preprocessing_steps": preprocessing_steps,
        "nan_cleaning": missing["nan_cleaning"],
        "normalization": normalization["normalization"],
        "outliers": outlier_info["outliers"],
        "correlations": correlations["correlations"],
        "cleaned_summary": cleaned_summary,
    }


def _analyze_dataset(user_id: str, csv_content: str) -> str:
    """Deterministic helper: run full analysis and save results."""
    t0 = time.time()

    print("[analyze_dataset] Running deterministic analysis...", flush=True)
    summary = _run_analysis(csv_content)
    print(f"[analyze_dataset] Analysis done ({time.time() - t0:.1f}s)", flush=True)

    print("[analyze_dataset] Saving results...", flush=True)
    _save_analysis(username=user_id, summary=summary)
    print(f"[analyze_dataset] Done ({time.time() - t0:.1f}s)", flush=True)

    return json.dumps(summary, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════════
# LLM-powered smart analysis (fallback — LLM decides what to run)
# ═══════════════════════════════════════════════════════════════════════

_ANALYST_SYSTEM_PROMPT = """You are a data analysis expert. You have tools to
analyze a dataset. Your job is to look at the data and decide which analysis
steps are actually needed.

WORKFLOW:
1. ALWAYS start by calling profile_data to understand the dataset
2. Based on the profile, decide which additional steps are relevant:
   - If there are missing values -> call clean_missing_values
   - If there are numeric columns with potential outliers -> call detect_outliers
   - If there are 2+ numeric columns -> call find_correlations
   - If numeric columns have very different scales -> call suggest_normalization
   - If there are duplicate rows -> call remove_duplicates
3. Skip steps that aren't needed (e.g., no correlations if only 1 numeric column)
4. After running the relevant tools, compile ALL results into a single JSON summary

OUTPUT: Return a JSON object with all the findings from the tools you called.
Only include sections for the analyses you actually ran."""


def _smart_analyze_dataset(user_id: str, csv_content: str) -> str:
    """LLM-powered analysis: the model decides which steps to run."""
    from strands import tool as strands_tool

    # Capture csv_content in closure for each tool
    @strands_tool
    def profile_data() -> str:
        """Profile the dataset: shape, dtypes, missing values, per-column statistics.
        Always call this first to understand the data before deciding on further steps."""
        result = _profile(csv_content)
        return json.dumps(result, indent=2, default=str)

    @strands_tool
    def remove_duplicate_rows() -> str:
        """Remove duplicate rows from the dataset. Call this if the profile shows duplicate_rows > 0."""
        result = _remove_duplicates(csv_content)
        return json.dumps(result, indent=2, default=str)

    @strands_tool
    def clean_missing_values() -> str:
        """Analyze and clean missing values using appropriate strategies
        (median, mean, mode, or drop). Call this if the profile shows columns with missing values."""
        result = _clean_missing(csv_content)
        return json.dumps(result, indent=2, default=str)

    @strands_tool
    def detect_outliers() -> str:
        """Detect outliers in numeric columns using the IQR method.
        Call this if there are numeric columns that might have extreme values."""
        result = _detect_outliers(csv_content)
        return json.dumps(result, indent=2, default=str)

    @strands_tool
    def find_correlations() -> str:
        """Find strong correlations between numeric columns.
        Call this only if there are 2 or more numeric columns."""
        result = _compute_correlations(csv_content)
        return json.dumps(result, indent=2, default=str)

    @strands_tool
    def suggest_normalization() -> str:
        """Determine appropriate normalization methods for numeric columns.
        Call this if numeric columns have very different scales or distributions."""
        result = _normalize(csv_content)
        return json.dumps(result, indent=2, default=str)

    t0 = time.time()
    print("[smart_analyze] Starting LLM-powered analysis...", flush=True)

    agent = Agent(
        model=BedrockModel(
            model_id="us.anthropic.claude-sonnet-4-6",
            region_name=os.environ.get("AWS_REGION", "us-east-2"),
            boto_client_config=_BOTO_CONFIG,
        ),
        system_prompt=_ANALYST_SYSTEM_PROMPT,
        tools=[
            profile_data, remove_duplicate_rows, clean_missing_values,
            detect_outliers, find_correlations, suggest_normalization,
        ],
    )

    prompt = (
        f"Analyze this dataset for user '{user_id}'. "
        f"Start by profiling, then decide which additional analyses are needed.\n\n"
        f"Dataset preview (first 3000 chars):\n{csv_content[:3000]}"
    )

    result = str(agent(prompt))
    print(f"[smart_analyze] Done ({time.time() - t0:.1f}s)", flush=True)

    _save_analysis(username=user_id, summary=result)
    return result


# ═══════════════════════════════════════════════════════════════════════
# Public tools (exposed to the tutor agent)
# ═══════════════════════════════════════════════════════════════════════

@tool
def analyze_dataset(user_id: str, csv_content: str) -> str:
    """
    Run a comprehensive deterministic analysis on the CSV locally with pandas.
    Runs ALL analysis steps. Results are stored in memory for the tutor.
    Use this as the default analysis path.

    Args:
        user_id: The student's username (used as storage key).
        csv_content: Raw text content of the uploaded CSV file.
    """
    return _analyze_dataset(user_id, csv_content)


@tool
def smart_analyze_dataset(user_id: str, csv_content: str) -> str:
    """
    Run an LLM-powered smart analysis on the CSV. The LLM decides which
    analysis steps are relevant based on the dataset characteristics.
    Use this when you need targeted analysis rather than running everything.

    Args:
        user_id: The student's username (used as storage key).
        csv_content: Raw text content of the uploaded CSV file.
    """
    return _smart_analyze_dataset(user_id, csv_content)
