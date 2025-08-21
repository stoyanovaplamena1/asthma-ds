from __future__ import annotations

from typing import Dict, Iterable, List, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


def feature_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Detect groups by name patterns to avoid hardcoding."""
    symptoms = [
        c for c in df.columns
        if c in {"wheezing","shortness_of_breath","chest_tightness",
                 "coughing","nighttime_symptoms","exercise_induced"}
    ]
    atopy = [c for c in df.columns if c in {"eczema","hay_fever","history_of_allergies","pet_allergy"}]
    exposures = [c for c in df.columns if c in {"pollution_exposure","pollen_exposure","dust_exposure"}]
    lifestyle = [c for c in df.columns if c in {"physical_activity","diet_quality","sleep_quality"}]
    ethnicity = [c for c in df.columns if c.startswith("ethnicity_")]

    core_numeric = [c for c in ["age","bmi","lung_function_fev1","fev1_fvc_pct"] if c in df.columns]
    binary_other = [c for c in df.columns if c in {
        "gender","smoking","family_history_asthma","gastroesophageal_reflux"
    }]

    return {
        "symptoms": symptoms,
        "atopy": atopy,
        "exposures": exposures,
        "lifestyle": lifestyle,
        "ethnicity": ethnicity,
        "core_numeric": core_numeric,
        "binary_other": binary_other,
    }


def engineer_features(
    df: pd.DataFrame,
    *,
    add_bmi_quadratic: bool = True,
    add_interaction: bool = True,
) -> pd.DataFrame:
    """
    Add clinically motivated features; returns a copy with new columns.
    """
    d = df.copy()
    groups = feature_groups(d)

    # symptom burden / atopy breadth
    if groups["symptoms"]:
        d["symptom_count"] = d[groups["symptoms"]].sum(axis=1).astype(float)
    else:
        d["symptom_count"] = 0.0

    if groups["atopy"]:
        d["atopy_score"] = d[groups["atopy"]].sum(axis=1).astype(float)
    else:
        d["atopy_score"] = 0.0

    # indices (means of 0-10 scales)
    for new_col, cols in {
        "lifestyle_index": groups["lifestyle"],
        "exposure_index": groups["exposures"],
    }.items():
        d[new_col] = d[cols].mean(axis=1) if cols else 0.0

    # interaction hypothesis: obstruction * symptoms
    if add_interaction and "fev1_fvc_pct" in d.columns:
        d["ratio_x_symptoms"] = d["fev1_fvc_pct"] * d["symptom_count"]

    # BMI nonlinearity
    if add_bmi_quadratic and "bmi" in d.columns:
        d["bmi_c"] = d["bmi"] - d["bmi"].mean()
        d["bmi_c2"] = d["bmi_c"] ** 2

    return d


def to_numeric_float(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast all columns to float where appropriate (binaries become 0/1 floats).
    """
    d = df.copy()
    for c in d.columns:
        if str(d[c].dtype).startswith(("int", "uint", "bool")):
            d[c] = d[c].astype(float)
        elif d[c].dtype == "boolean":
            d[c] = d[c].astype(float)
    return d


def build_feature_matrix(
    df: pd.DataFrame,
    target_col: str = "diagnosis",
    *,
    scale_extra: Iterable[str] = ("symptom_count","atopy_score","lifestyle_index",
                                  "exposure_index","ratio_x_symptoms","bmi_c","bmi_c2"),
    return_transformer: bool = False,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, List[str]]] | Tuple[pd.DataFrame, pd.Series, Dict[str, List[str]], ColumnTransformer]:
    """
    Prepare an all-numeric, standardized feature matrix (X) and target (y).
    - Standardizes continuous columns; leaves binaries/dummies as 0/1.
    - Returns (X, y, meta) or (X, y, meta, preprocessor).
    """
    if target_col not in df.columns:
        raise KeyError(f"Target '{target_col}' not found.")

    groups = feature_groups(df)
    y = df[target_col].astype(int)

    # candidate scale columns: core numeric + engineered extras present in df
    scale_cols = [c for c in (groups["core_numeric"] + list(scale_extra)) if c in df.columns]

    X = df.drop(columns=[target_col]).copy()
    X = to_numeric_float(X)

    passthrough_cols = [c for c in X.columns if c not in scale_cols]

    preproc = ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(with_mean=True, with_std=True), scale_cols),
            ("keep", "passthrough", passthrough_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    X_arr = preproc.fit_transform(X)
    X_final = pd.DataFrame(X_arr, columns=list(preproc.get_feature_names_out()), index=X.index)

    meta = {
        "scale_cols": scale_cols,
        "passthrough_cols": passthrough_cols,
        "groups": groups,
    }
    if return_transformer:
        return X_final, y, meta, preproc
    return X_final, y, meta