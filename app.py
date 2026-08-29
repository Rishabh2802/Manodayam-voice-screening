from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from predict import load_model_packages, predict_all, detect_deployment_dir


# ============================================================
# Page setup
# ============================================================

st.set_page_config(
    page_title="Manodayam Voice Screening",
    page_icon="🎙️",
    layout="wide",
)

DEPLOYMENT_DIR = detect_deployment_dir()


# ============================================================
# Constants
# ============================================================

TARGET_ORDER = {
    "Depression Multiclass 3": 0,
    "Anxiety Multiclass 3": 1,
    "Depression Binary": 2,
    "Anxiety Binary": 3,
}

DISPLAY_NAMES = {
    "Depression Binary": "Depression screening flag",
    "Anxiety Binary": "Anxiety screening flag",
    "Depression Multiclass 3": "Depression category",
    "Anxiety Multiclass 3": "Anxiety category",
}

# Participant-facing trained categories requested by the project.
# These are screening categories, not clinical diagnostic labels.
TRAINED_CATEGORIES = {
    "Depression Multiclass 3": {
        0: "Normal",
        1: "Mild",
        2: "Moderate or severe",
    },
    "Anxiety Multiclass 3": {
        0: "Normal",
        1: "Mild",
        2: "Moderate or severe",
    },
    # Binary outputs are kept as supportive screening flags only.
    "Depression Binary": {
        0: "Normal / no clear elevated pattern",
        1: "Elevated pattern detected",
    },
    "Anxiety Binary": {
        0: "Normal / no clear elevated pattern",
        1: "Elevated pattern detected",
    },
}

CATEGORY_EXPLANATIONS = {
    "Normal": {
        "summary": "The speech pattern was closest to the normal category learned during training.",
        "next_step": "No elevated speech-pattern flag was produced. This does not rule out concerns; seek support if the participant feels distressed.",
    },
    "Mild": {
        "summary": "The speech pattern was closest to the mild category learned during training.",
        "next_step": "A standard screening questionnaire can be completed for follow-up, especially if symptoms are present.",
    },
    "Moderate or severe": {
        "summary": "The speech pattern was closest to the moderate-or-severe category learned during training.",
        "next_step": "A follow-up questionnaire or review by a qualified mental health professional is recommended.",
    },
}

BINARY_EXPLANATIONS = {
    "Depression Binary": {
        0: "The binary depression screening model did not show a clear elevated depression-related speech pattern.",
        1: "The binary depression screening model showed an elevated depression-related speech pattern.",
    },
    "Anxiety Binary": {
        0: "The binary anxiety screening model did not show a clear elevated anxiety-related speech pattern.",
        1: "The binary anxiety screening model showed an elevated anxiety-related speech pattern.",
    },
}


# ============================================================
# Cached loading
# ============================================================

@st.cache_resource(show_spinner=False)
def get_model_packages(deployment_dir_str: str):
    return load_model_packages(Path(deployment_dir_str))


# ============================================================
# File helpers
# ============================================================

def save_uploaded_audio(uploaded_file, prefix: str) -> Optional[str]:
    if uploaded_file is None:
        return None

    suffix = Path(uploaded_file.name).suffix or ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix=prefix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def remove_temp_file(path: Optional[str]) -> None:
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass


# ============================================================
# Result helpers
# ============================================================

def sort_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(results, key=lambda r: TARGET_ORDER.get(r.get("target_name"), 99))


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def selected_category(result: Dict[str, Any]) -> str:
    target_name = result.get("target_name", "")
    prediction = safe_int(result.get("prediction"), default=-1)

    if target_name in TRAINED_CATEGORIES and prediction in TRAINED_CATEGORIES[target_name]:
        return TRAINED_CATEGORIES[target_name][prediction]

    return str(result.get("label", "Result unavailable"))


def get_domain(target_name: str) -> str:
    if "Depression" in target_name:
        return "Depression"
    if "Anxiety" in target_name:
        return "Anxiety"
    return target_name


def category_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Class": 0, "Category": "Normal"},
            {"Class": 1, "Category": "Mild"},
            {"Class": 2, "Category": "Moderate or severe"},
        ]
    )


def binary_flag_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Class": 0, "Meaning": "Normal / no clear elevated pattern"},
            {"Class": 1, "Meaning": "Elevated pattern detected"},
        ]
    )


def participant_summary_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Participant-facing summary.
    Only show the 3 trained categories: Normal, Mild, Moderate or severe.
    Binary screening flags are intentionally hidden from the participant UI.
    """
    rows = []

    multiclass_results = [
        result for result in sort_results(results)
        if result.get("task_type") == "multiclass"
    ]

    for result in multiclass_results:
        target_name = result.get("target_name")

        rows.append(
            {
                "Output": DISPLAY_NAMES.get(target_name, target_name),
                "Selected category": selected_category(result),
                "Categories used": "Normal / Mild / Moderate or severe",
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# UI helpers
# ============================================================

def show_audio_status(debug: Dict[str, Any]) -> None:
    st.subheader("Audio check")

    col1, col2 = st.columns(2)

    q1_status = debug.get("q1_status")
    q2_status = debug.get("q2_status")
    q1_duration = debug.get("q1_duration_sec")
    q2_duration = debug.get("q2_duration_sec")

    with col1:
        st.write("**Q1: Daily routine**")
        if q1_status == "ok":
            st.success(f"Processed successfully. Duration: {float(q1_duration):.1f} seconds")
        elif q1_status == "not_provided":
            st.info("Q1 audio was not provided.")
        else:
            st.error(f"Q1 could not be processed: {debug.get('q1_error', '')}")

    with col2:
        st.write("**Q2: Environment near home**")
        if q2_status == "ok":
            st.success(f"Processed successfully. Duration: {float(q2_duration):.1f} seconds")
        elif q2_status == "not_provided":
            st.info("Q2 audio was not provided.")
        else:
            st.error(f"Q2 could not be processed: {debug.get('q2_error', '')}")


def category_status_label(category: str) -> str:
    if category == "Normal":
        return "Normal"
    if category == "Mild":
        return "Mild"
    if category == "Moderate or severe":
        return "Moderate or severe"
    return category


def show_overall_summary(multiclass_results: List[Dict[str, Any]]) -> None:
    if not multiclass_results:
        return

    st.subheader("Overall category summary")

    cols = st.columns(len(multiclass_results))

    for col, result in zip(cols, multiclass_results):
        domain = get_domain(result.get("target_name", ""))
        category = selected_category(result)

        if category == "Normal":
            col.success(f"{domain}: {category_status_label(category)}")
        elif category == "Mild":
            col.warning(f"{domain}: {category_status_label(category)}")
        else:
            col.error(f"{domain}: {category_status_label(category)}")

    highest = "Normal"
    categories = [selected_category(r) for r in multiclass_results]
    if "Moderate or severe" in categories:
        highest = "Moderate or severe"
    elif "Mild" in categories:
        highest = "Mild"

    if highest == "Normal":
        st.info(
            "The uploaded speech responses were categorized as Normal by the trained 3-category models. "
            "This does not rule out mental health concerns."
        )
    elif highest == "Mild":
        st.info(
            "At least one output was categorized as Mild. This is not a diagnosis, but follow-up screening may be useful."
        )
    else:
        st.info(
            "At least one output was categorized as Moderate or severe. This is not a diagnosis, but professional review is recommended."
        )


def show_trained_category_card(result: Dict[str, Any]) -> None:
    target_name = result.get("target_name", "")
    domain = get_domain(target_name)
    category = selected_category(result)
    explanation = CATEGORY_EXPLANATIONS.get(category, CATEGORY_EXPLANATIONS["Normal"])

    with st.container(border=True):
        st.markdown(f"### {domain} category")

        if category == "Normal":
            st.success(f"Selected category: {category}")
        elif category == "Mild":
            st.warning(f"Selected category: {category}")
        else:
            st.error(f"Selected category: {category}")

        st.write("**Trained categories used by the model**")
        st.dataframe(category_table(), use_container_width=True, hide_index=True)

        st.write("**What this means**")
        st.write(explanation["summary"])

        st.write("**Suggested next step**")
        st.write(explanation["next_step"])

        st.caption(
            "This is a speech-based screening category learned from training data. "
            "It should not be treated as a clinical diagnosis or final clinical severity label."
        )


def show_binary_flag_card(result: Dict[str, Any]) -> None:
    target_name = result.get("target_name", "")
    domain = get_domain(target_name)
    prediction = safe_int(result.get("prediction"))
    category = selected_category(result)
    explanation = BINARY_EXPLANATIONS.get(target_name, {}).get(
        prediction,
        "The binary screening model selected one of the two trained classes.",
    )

    with st.container(border=True):
        st.markdown(f"### {domain} binary screening flag")

        if prediction == 1:
            st.warning(f"Flag: {category}")
        else:
            st.success(f"Flag: {category}")

        st.write(explanation)

        st.write("**Binary classes used during training**")
        st.dataframe(binary_flag_table(), use_container_width=True, hide_index=True)

        st.caption(
            "The binary output is used as an additional screening flag. "
            "The main participant category is shown using the 3-category model: Normal, Mild, or Moderate or severe."
        )


def show_researcher_details(output: Dict[str, Any], packages: Dict[str, Dict[str, Any]]) -> None:
    with st.expander("Researcher/debug details", expanded=False):
        st.write("These details are hidden from participants by default.")

        results = output.get("results", [])
        if results:
            rows = []
            for r in sort_results(results):
                rows.append(
                    {
                        "Target": r.get("target_name"),
                        "Prediction class": r.get("prediction"),
                        "Participant label": selected_category(r),
                        "Raw label": r.get("label"),
                        "Model": r.get("model_name"),
                        "Input": r.get("input_name"),
                        "Status": r.get("status"),
                        "Predicted-class probability": r.get("predicted_class_probability"),
                        "Decision score": r.get("decision_score"),
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        if output.get("skipped"):
            st.write("Skipped outputs")
            st.dataframe(pd.DataFrame(output.get("skipped")), use_container_width=True)

        st.write("Raw prediction output")
        st.json(output)

        st.write("Loaded model packages")
        package_rows = []
        for _, pkg in packages.items():
            package_rows.append(
                {
                    "Target": pkg.get("target_name"),
                    "Input": pkg.get("input_name"),
                    "Model": pkg.get("model_name"),
                    "Status": pkg.get("status"),
                    "Features": len(pkg.get("feature_cols", [])),
                }
            )
        st.dataframe(pd.DataFrame(package_rows), use_container_width=True)


def show_intro() -> None:
    st.title("🎙️ Manodayam Voice-Based Mental Health Screening")
    st.warning(
        "This tool provides speech-based screening categories only. "
        "It is not a clinical diagnosis and should not replace assessment by a qualified professional."
    )

    st.write(
        "Please upload the two speech responses used during data collection. "
        "The participant-facing output categorizes the speech responses into: **Normal**, **Mild**, or **Moderate or severe**."
    )

    with st.expander("What does this tool analyze?"):
        st.write(
            "The tool extracts acoustic and speech-pattern features from the uploaded audio. "
            "It then compares those features with categories learned during model training."
        )
        st.write(
            "The output should be interpreted as screening-support information only, not as a diagnosis."
        )


# ============================================================
# Main app
# ============================================================

def main() -> None:
    show_intro()

    with st.sidebar:
        st.header("Session setup")
        researcher_mode = st.checkbox("Show researcher/debug details", value=False)

        try:
            packages = get_model_packages(str(DEPLOYMENT_DIR))
            st.success("Model package ready")
            if researcher_mode:
                st.caption(f"Deployment directory: `{DEPLOYMENT_DIR}`")
                st.write(f"Loaded {len(packages)} model(s)")
                package_table = pd.DataFrame(
                    [
                        {
                            "Target": pkg.get("target_name"),
                            "Input": pkg.get("input_name"),
                            "Status": pkg.get("status"),
                        }
                        for pkg in packages.values()
                    ]
                )
                st.dataframe(package_table, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Could not load deployment models: {e}")
            st.info("Check that the `models/` folder contains the deployment `.joblib` package files.")
            st.stop()

    st.header("Step 1: Consent")
    consent = st.checkbox(
        "I understand this is a research screening demo and not a medical diagnosis.",
        value=False,
    )

    st.header("Step 2: Upload speech responses")
    st.write("Each response should ideally be 30–60 seconds, spoken naturally in a quiet environment.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Q1 audio: Daily routine")
        st.caption("Prompt: Please describe your daily routine.")
        q1_file = st.file_uploader(
            "Upload Q1 audio file",
            type=["wav", "mp3", "m4a", "flac", "ogg", "aac", "wma"],
            key="q1_audio",
        )

    with col2:
        st.subheader("Q2 audio: Environment near home")
        st.caption("Prompt: Please describe the environment near your home.")
        q2_file = st.file_uploader(
            "Upload Q2 audio file",
            type=["wav", "mp3", "m4a", "flac", "ogg", "aac", "wma"],
            key="q2_audio",
        )

    st.header("Step 3: Run screening")

    if not consent:
        st.button("Run screening", type="primary", disabled=True)
        st.info("Please confirm consent before running the screening.")
        return

    run_button = st.button("Run screening", type="primary")

    if not run_button:
        return

    if q1_file is None or q2_file is None:
        st.error("Please upload both Q1 and Q2 audio files to run the full screening.")
        return

    q1_path = None
    q2_path = None

    try:
        q1_path = save_uploaded_audio(q1_file, "q1_")
        q2_path = save_uploaded_audio(q2_file, "q2_")

        with st.spinner("Processing audio and preparing screening categories..."):
            output = predict_all(
                q1_audio_path=q1_path,
                q2_audio_path=q2_path,
                deployment_dir=DEPLOYMENT_DIR,
                packages=packages,
            )

        debug = output.get("debug", {})
        results = output.get("results", [])

        show_audio_status(debug)

        if not results:
            st.error("No screening output could be generated. Please check the uploaded audio files.")
            if researcher_mode:
                show_researcher_details(output, packages)
            return

        multiclass_results = [r for r in sort_results(results) if r.get("task_type") == "multiclass"]

        st.header("Screening outputs")

        show_overall_summary(multiclass_results)

        st.subheader("Participant category results")
        st.caption("Only the 3 trained categories are shown to the participant: Normal, Mild, and Moderate or severe.")
        for result in multiclass_results:
            show_trained_category_card(result)

        st.subheader("Participant summary")
        st.dataframe(
            participant_summary_table(results),
            use_container_width=True,
            hide_index=True,
        )

        st.warning(
            "Reminder: This is a speech-based screening output, not a diagnosis. "
            "A Normal category does not rule out concerns, and a Mild or Moderate-or-severe category does not confirm a disorder. "
            "For clinical evaluation, use validated questionnaires and assessment by a qualified mental health professional."
        )

        if researcher_mode:
            show_researcher_details(output, packages)

    finally:
        remove_temp_file(q1_path)
        remove_temp_file(q2_path)


if __name__ == "__main__":
    main()
