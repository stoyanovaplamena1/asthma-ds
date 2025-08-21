from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Optional
import numpy as np
import pandas as pd

def camel_to_snake(name: str) -> str:
    """
    Convert CamelCase or mixed column names to snake_case.
    
    Parameters
    ----------
    name : str
        Column name in CamelCase or mixed format.
    
    Returns
    -------
    str
        Column name in snake_case.
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
    return s2.lower()


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize dataframe column names to snake_case.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with mixed-style column names.
    
    Returns
    -------
    pd.DataFrame
        Copy of dataframe with snake_case column names.
    """
    df = df.copy()
    df.columns = [camel_to_snake(c) for c in df.columns]
    return df


def drop_irrelevant_columns(df: pd.DataFrame, cols_to_drop=None) -> pd.DataFrame:
    """
    Drop irrelevant columns (like identifiers or constant features).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    cols_to_drop : list, optional
        List of columns to drop. If None, defaults to ["patient_id", "doctor_in_charge"].
    
    Returns
    -------
    pd.DataFrame
        Dataframe without irrelevant columns.
    """
    df = df.copy()
    if cols_to_drop is None:
        cols_to_drop = ["patient_id", "doctor_in_charge"]
    return df.drop(columns=[c for c in cols_to_drop if c in df.columns])

def add_fev1_fvc(
    df: pd.DataFrame,
    fev1_col: str = "lung_function_fev1",
    fvc_col: str = "lung_function_fvc",
    new_col: str | None = None,
    as_percent: bool = False,
    clip_range: tuple[float, float] | None = (0.0, 2.0),
) -> pd.DataFrame:
    """
    Add the FEV1/FVC feature (ratio or percentage) to the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (will not be modified in place).
    fev1_col : str, default "lung_function_fev1"
        Column name for FEV1 values.
    fvc_col : str, default "lung_function_fvc"
        Column name for FVC values.
    new_col : str or None, default None
        Name of the new column. If None, uses:
        - "fev1_fvc_pct" when as_percent=True
        - "fev1_fvc_ratio" otherwise
    as_percent : bool, default False
        If True, create percentage (0–100+). If False, ratio (0–~1+).
    clip_range : (float, float) or None, default (0.0, 2.0)
        Clip extreme values to a sensible range; set to None to disable.

    Returns
    -------
    pd.DataFrame
        Copy of df with a new FEV1/FVC column.
    """
    df = df.copy()

    if fev1_col not in df.columns or fvc_col not in df.columns:
        # nothing to do if columns are missing
        return df

    # Avoid division by zero or negative FVC by converting invalid to NaN
    fvc = df[fvc_col].replace({0: np.nan})
    fvc = fvc.where(fvc > 0, np.nan)

    ratio = df[fev1_col] / fvc

    if as_percent:
        vals = ratio * 100.0
        default_name = "fev1_fvc_pct"
    else:
        vals = ratio
        default_name = "fev1_fvc_ratio"

    if clip_range is not None:
        lo, hi = clip_range
        vals = vals.clip(lower=lo, upper=hi)

    out_name = new_col or default_name
    df[out_name] = vals
    return df


def univariate_summary(
    df: pd.DataFrame,
    cat_max_unique: int = 20,
    exclude: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Produce univariate summaries for numeric and categorical columns.
    (No plots; suitable for notebooks to render as tables.)

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    cat_max_unique : int, default 20
        Columns with <= this many unique non-null values will be treated as categorical.
    exclude : list of str, optional
        Columns to exclude from the summary (e.g., identifiers).

    Returns
    -------
    dict
        {
          "numeric": DataFrame[mean, std, min, p25, median, p75, max, skew, kurtosis, missing],
          "categorical": DataFrame[n_unique, top, top_freq, missing]
        }
    """
    exclude = set(exclude or [])
    cols = [c for c in df.columns if c not in exclude]

    # split columns by type/uniqueness
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [
        c for c in cols
        if c not in numeric_cols and df[c].nunique(dropna=True) <= cat_max_unique
    ]

    # Numeric summary
    num = df[numeric_cols].copy()
    numeric_summary = pd.DataFrame({
        "mean": num.mean(),
        "std": num.std(),
        "min": num.min(),
        "p25": num.quantile(0.25),
        "median": num.median(),
        "p75": num.quantile(0.75),
        "max": num.max(),
        "skew": num.skew(numeric_only=True),
        "kurtosis": num.kurtosis(numeric_only=True),
        "missing": num.isna().sum()
    }).sort_index()

    # Categorical summary
    cat_rows = []
    for c in categorical_cols:
        s = df[c]
        vc = s.value_counts(dropna=True)
        top = vc.index[0] if not vc.empty else np.nan
        top_freq = vc.iloc[0] if not vc.empty else np.nan
        cat_rows.append({
            "column": c,
            "n_unique": s.nunique(dropna=True),
            "top": top,
            "top_freq": top_freq,
            "missing": s.isna().sum()
        })
    categorical_summary = pd.DataFrame(cat_rows).set_index("column") if cat_rows else pd.DataFrame(
        columns=["n_unique", "top", "top_freq", "missing"]
    )

    return {"numeric": numeric_summary, "categorical": categorical_summary}


def detect_outliers(
    series: pd.Series,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
    iqr_multiplier: float = 1.5
) -> Tuple[pd.Series, Tuple[float, float, float, float, float]]:
    """
    Detect outliers in a pandas Series using IQR + optional medical cutoffs.
    
    Parameters
    ----------
    series : pd.Series
        Numeric column to check.
    lower_bound : float, optional
        Hard lower bound (e.g. medically plausible minimum).
    upper_bound : float, optional
        Hard upper bound (e.g. medically plausible maximum).
    iqr_multiplier : float, default=1.5
        Multiplier for IQR rule.
    
    Returns
    -------
    outliers : pd.Series
        Subset of series containing outliers.
    bounds : tuple
        (low, high, Q1, Q3, IQR) used for detection.
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    low_iqr = Q1 - iqr_multiplier * IQR
    high_iqr = Q3 + iqr_multiplier * IQR
    
    low = max(low_iqr, lower_bound) if lower_bound is not None else low_iqr
    high = min(high_iqr, upper_bound) if upper_bound is not None else high_iqr
    
    outliers = series[(series < low) | (series > high)]
    return outliers, (low, high, Q1, Q3, IQR)

def check_missing_values(df: pd.DataFrame, sort: bool = True) -> pd.DataFrame:
    """
    Check missing values in a dataframe (counts and percentages).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    sort : bool, default True
        If True, sort output by descending percentage of missing values.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - n_missing: number of missing values per column
        - pct_missing: percentage of missing values per column
    """
    missing = df.isna().sum()
    pct = 100 * missing / len(df)

    out = pd.DataFrame({
        "n_missing": missing,
        "pct_missing": pct.round(2)
    })

    if sort:
        out = out.sort_values("pct_missing", ascending=False)

    return out


ETHNICITY_MAP: Dict[int, str] = {
    0: "caucasian",
    1: "african_american",
    2: "asian",
    3: "other",
}


@dataclass(frozen=True)
class RatioBounds:
    lower: float = 0.0   # percent
    upper: float = 120.0 # percent


def ensure_fev1_fvc_pct(
    df: pd.DataFrame,
    fev1_col: str = "lung_function_fev1",
    fvc_col: str = "lung_function_fvc",
    out_col: str = "fev1_fvc_pct",
    bounds: RatioBounds = RatioBounds(),
) -> pd.DataFrame:
    """
    Ensure FEV1/FVC percent column exists and is clipped to [lower, upper].
    Does not mutate input; returns a copy.
    """
    d = df.copy()
    if out_col not in d.columns:
        if fev1_col not in d.columns or fvc_col not in d.columns:
            raise KeyError(
                f"Cannot compute {out_col}: missing {fev1_col} or {fvc_col}"
            )
        ratio = (d[fev1_col] / d[fvc_col]) * 100.0
        ratio = ratio.replace([np.inf, -np.inf], np.nan)
        d[out_col] = ratio

    # clip in-place (idempotent)
    d[out_col] = d[out_col].clip(lower=bounds.lower, upper=bounds.upper)
    return d


def one_hot_ethnicity(
    df: pd.DataFrame,
    src_col: str = "ethnicity",
    prefix: str = "ethnicity",
    mapping: Dict[int, str] = ETHNICITY_MAP,
    drop_original: bool = True,
) -> pd.DataFrame:
    """
    One-hot encode ethnicity using a human-readable mapping (e.g., caucasian).
    Safe to call even if src_col already replaced (idempotent).
    """
    d = df.copy()
    if src_col not in d.columns:
        # already encoded earlier or dropped; return as-is
        return d

    series = d[src_col].map(mapping)
    dummies = pd.get_dummies(series, prefix=prefix, dtype="int8")
    d = pd.concat([d.drop(columns=[src_col]), dummies], axis=1)
    if not drop_original:
        d[src_col] = series
    return d


def drop_if_present(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """Drop columns if they exist. Returns a copy."""
    d = df.copy()
    existing = [c for c in cols if c in d.columns]
    if existing:
        d = d.drop(columns=existing)
    return d


def drop_low_variance(
    df: pd.DataFrame,
    numeric_only: bool = True,
    threshold: float = 0.0,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Drop columns with variance <= threshold. Returns (df, dropped_cols).
    threshold=0 keeps non-constant columns; use small >0 for near-constants.
    """
    d = df.copy()
    if numeric_only:
        num = d.select_dtypes(include=[np.number])
    else:
        num = d

    var = num.var(axis=0, ddof=0)
    drop_cols = var.index[var <= threshold].tolist()
    d = d.drop(columns=drop_cols)
    return d, drop_cols


def prepare_clean_dataset(
    df: pd.DataFrame,
    *,
    ratio_bounds: RatioBounds = RatioBounds(),
    drop_cols: Iterable[str] = ("doctor_in_charge", "patient_id"),
    drop_fvc: bool = True,
    do_one_hot_ethnicity: bool = True,
) -> pd.DataFrame:
    """
    Full cleaning pass:
      - ensure & clip fev1_fvc_pct in [bounds]
      - drop low-information columns (ids, singletons)
      - optionally drop FVC to reduce multicollinearity (keep FEV1 + fev1_fvc_pct)
      - one-hot encode ethnicity with readable names
    """
    d = ensure_fev1_fvc_pct(df, bounds=ratio_bounds)
    d = drop_if_present(d, drop_cols)

    if drop_fvc and "lung_function_fvc" in d.columns:
        d = d.drop(columns=["lung_function_fvc"])

    if do_one_hot_ethnicity:
        d = one_hot_ethnicity(d)

    # enforce consistent dtypes: binaries/int->int8, floats stay float64
    for c in d.columns:
        if d[c].dropna().isin([0, 1]).all():
            d[c] = d[c].astype("int8")
    return d