"""
Prediction helpers for Manodayam Streamlit deployment.

Supports both local layouts:
1) repo/deployment/models/*.joblib
2) repo/models/*.joblib

Only deployment-package joblib files are loaded. Legacy joblib files that do not
contain the required package dictionary are skipped safely.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from feature_extraction import extract_audio_features, prefix_features


DEFAULT_DEPLOYMENT_DIR = Path("deployment")


def detect_deployment_dir() -> Path:
    """Auto-detect deployment directory.

    Returns:
        Path("deployment") if deployment/models exists.
        Path(".") if models exists in the current folder.
        Path("deployment") as default fallback.
    """
    if (Path("deployment") / "models").exists():
        return Path("deployment")

    if Path("models").exists():
        return Path(".")

    return DEFAULT_DEPLOYMENT_DIR


def get_model_dir(deployment_dir: Path | str) -> Path:
    deployment_dir = Path(deployment_dir)

    # If deployment_dir is '.', model directory is ./models
    model_dir = deployment_dir / "models"

    if model_dir.exists():
        return model_dir

    # Fallback for local execution
    fallback_model_dir = Path("models")
    if fallback_model_dir.exists():
        return fallback_model_dir

    raise FileNotFoundError(
        f"Model directory not found. Checked: {model_dir} and {fallback_model_dir}"
    )


def is_valid_model_package(package: Any) -> Tuple[bool, List[str]]:
    required_keys = [
        "model",
        "feature_cols",
        "input_name",
        "target_name",
        "target_col",
        "task_type",
    ]

    if not isinstance(package, dict):
        return False, ["not_a_package_dict"]

    missing = [k for k in required_keys if k not in package]
    return len(missing) == 0, missing


def load_model_packages(deployment_dir: Path | str = DEFAULT_DEPLOYMENT_DIR) -> Dict[str, Dict[str, Any]]:
    """Load valid deployment model packages.

    This skips old/legacy .joblib files such as depression_q1_logistic.joblib
    if they are plain sklearn models instead of deployment package dicts.
    """
    model_dir = get_model_dir(deployment_dir)

    packages: Dict[str, Dict[str, Any]] = {}
    skipped_files: List[Dict[str, str]] = []

    for model_path in sorted(model_dir.glob("*.joblib")):
        try:
            package = joblib.load(model_path)
        except Exception as e:
            skipped_files.append({
                "file": model_path.name,
                "reason": f"load_failed: {e}",
            })
            continue

        is_valid, missing = is_valid_model_package(package)
        if not is_valid:
            skipped_files.append({
                "file": model_path.name,
                "reason": f"invalid_or_legacy_package: {missing}",
            })
            continue

        package["model_path"] = str(model_path)
        packages[str(package["target_col"])] = package

    if len(packages) == 0:
        skipped_msg = "; ".join([f"{x['file']} -> {x['reason']}" for x in skipped_files])
        raise FileNotFoundError(
            f"No valid deployment model packages found in {model_dir}. "
            f"Skipped files: {skipped_msg}"
        )

    # Store loader diagnostics in each package so app can display/debug if needed
    for pkg in packages.values():
        pkg["_model_dir"] = str(model_dir)
        pkg["_skipped_files"] = skipped_files

    return packages


def model_needs_input(input_name: str) -> Tuple[bool, bool]:
    if input_name == "Q1 Full Audio":
        return True, False
    if input_name == "Q2 Full Audio":
        return False, True
    if input_name == "Q1+Q2 Full Audio":
        return True, True
    raise ValueError(f"Unknown input_name: {input_name}")


def build_model_input(
    model_package: Dict[str, Any],
    q1_feature_dict: Optional[Dict[str, Any]] = None,
    q2_feature_dict: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    input_name = model_package["input_name"]
    feature_cols = list(model_package["feature_cols"])

    needs_q1, needs_q2 = model_needs_input(input_name)

    if needs_q1 and q1_feature_dict is None:
        raise ValueError(f"{model_package['target_name']} requires Q1 audio features.")
    if needs_q2 and q2_feature_dict is None:
        raise ValueError(f"{model_package['target_name']} requires Q2 audio features.")

    row: Dict[str, Any] = {}

    if input_name in ["Q1 Full Audio", "Q1+Q2 Full Audio"]:
        row.update(prefix_features(q1_feature_dict, "q1"))

    if input_name in ["Q2 Full Audio", "Q1+Q2 Full Audio"]:
        row.update(prefix_features(q2_feature_dict, "q2"))

    X = pd.DataFrame([row])

    for col in feature_cols:
        if col not in X.columns:
            X[col] = np.nan

    X = X[feature_cols]

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    return X


def predict_from_package(
    model_package: Dict[str, Any],
    q1_feature_dict: Optional[Dict[str, Any]] = None,
    q2_feature_dict: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    model = model_package["model"]
    class_mapping = model_package.get("class_mapping", {})

    X = build_model_input(
        model_package,
        q1_feature_dict=q1_feature_dict,
        q2_feature_dict=q2_feature_dict,
    )

    pred = int(model.predict(X.values)[0])

    result: Dict[str, Any] = {
        "target_name": model_package.get("target_name"),
        "target_col": model_package.get("target_col"),
        "task_type": model_package.get("task_type"),
        "model_name": model_package.get("model_name"),
        "input_name": model_package.get("input_name"),
        "status": model_package.get("status", "unknown"),
        "prediction": pred,
        "label": class_mapping.get(str(pred), str(pred)),
        "warning": model_package.get(
            "warning",
            "This is a speech-based screening model, not a clinical diagnosis.",
        ),
    }

    if hasattr(model, "predict_proba"):
        try:
            prob = np.asarray(model.predict_proba(X.values))
            result["probability"] = prob[0].tolist()
            if prob.ndim == 2 and pred < prob.shape[1]:
                result["predicted_class_probability"] = float(prob[0, pred])
        except Exception as e:
            result["probability_error"] = str(e)

    if hasattr(model, "decision_function"):
        try:
            score = np.asarray(model.decision_function(X.values))
            result["decision_score"] = score.tolist()
        except Exception as e:
            result["decision_score_error"] = str(e)

    return result


def extract_needed_features(
    q1_audio_path=None,
    q2_audio_path=None,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Dict[str, Any]]:
    q1_features = extract_audio_features(q1_audio_path) if q1_audio_path else None
    q2_features = extract_audio_features(q2_audio_path) if q2_audio_path else None

    debug = {
        "q1_status": q1_features.get("feature_extraction_status") if q1_features else "not_provided",
        "q1_error": q1_features.get("feature_extraction_error") if q1_features else "",
        "q1_duration_sec": q1_features.get("raw_audio_duration_sec") if q1_features else None,
        "q2_status": q2_features.get("feature_extraction_status") if q2_features else "not_provided",
        "q2_error": q2_features.get("feature_extraction_error") if q2_features else "",
        "q2_duration_sec": q2_features.get("raw_audio_duration_sec") if q2_features else None,
    }

    return q1_features, q2_features, debug


def predict_all(
    q1_audio_path=None,
    q2_audio_path=None,
    deployment_dir: Path | str = DEFAULT_DEPLOYMENT_DIR,
    packages: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if packages is None:
        packages = load_model_packages(deployment_dir)

    q1_features, q2_features, debug = extract_needed_features(q1_audio_path, q2_audio_path)

    results: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for target_col, package in packages.items():
        input_name = package["input_name"]
        needs_q1, needs_q2 = model_needs_input(input_name)

        if needs_q1 and q1_features is None:
            skipped.append({
                "target_col": target_col,
                "target_name": package.get("target_name"),
                "reason": "Q1 audio required but not provided",
            })
            continue

        if needs_q2 and q2_features is None:
            skipped.append({
                "target_col": target_col,
                "target_name": package.get("target_name"),
                "reason": "Q2 audio required but not provided",
            })
            continue

        try:
            results.append(
                predict_from_package(
                    package,
                    q1_feature_dict=q1_features,
                    q2_feature_dict=q2_features,
                )
            )
        except Exception as e:
            skipped.append({
                "target_col": target_col,
                "target_name": package.get("target_name"),
                "reason": str(e),
            })

    return {
        "results": results,
        "skipped": skipped,
        "debug": debug,
    }
