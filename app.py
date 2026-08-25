import json
import tempfile
from pathlib import Path
from datetime import datetime

import streamlit as st
import librosa
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from predict import predict_all


# ============================================================
# Page config
# ============================================================

st.set_page_config(
    page_title="Manodayam Voice Screening",
    page_icon="🎙️",
    layout="centered"
)


# ============================================================
# CSS styling
# ============================================================

st.markdown(
    """
<style>
.result-card {
    padding: 1.2rem;
    border-radius: 14px;
    margin-bottom: 1rem;
    border: 1px solid #e5e7eb;
    background-color: #f9fafb;
}

.positive-card {
    padding: 1.2rem;
    border-radius: 14px;
    margin-bottom: 1rem;
    border: 1px solid #f59e0b;
    background-color: #fffbeb;
}

.negative-card {
    padding: 1.2rem;
    border-radius: 14px;
    margin-bottom: 1rem;
    border: 1px solid #10b981;
    background-color: #ecfdf5;
}

.info-card {
    padding: 1rem;
    border-radius: 12px;
    background-color: #eff6ff;
    border: 1px solid #93c5fd;
    margin-bottom: 1rem;
}

.warning-card {
    padding: 1rem;
    border-radius: 12px;
    background-color: #fff7ed;
    border: 1px solid #fdba74;
    margin-bottom: 1rem;
}

.small-text {
    font-size: 0.92rem;
    color: #4b5563;
}

.big-outcome {
    font-size: 1.25rem;
    font-weight: 700;
}
</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# Helper functions
# ============================================================

def save_audio_to_temp(audio_file, suffix=".wav"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_file.getvalue())
        return tmp.name


def get_audio_info(audio_path):
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    duration = len(y) / sr

    rms = float((y ** 2).mean() ** 0.5) if len(y) > 0 else 0.0
    peak = float(abs(y).max()) if len(y) > 0 else 0.0

    return {
        "duration_sec": duration,
        "sample_rate": sr,
        "samples": len(y),
        "rms": rms,
        "peak": peak,
    }


def interpret_result(probability, symptom_name):
    """
    Binary classifier:
    < 0.50 = screening negative
    >= 0.50 = screening positive

    Signal strength is not clinical severity.
    """

    if probability is None:
        return {
            "is_positive": False,
            "outcome": "Unable to estimate",
            "summary": "The model could not estimate a reliable score for this sample.",
            "signal": "Not available",
            "recommendation": "Please try another recording with clearer audio.",
            "score_text": "Not available",
        }

    score = probability * 100

    if probability < 0.40:
        return {
            "is_positive": False,
            "outcome": f"Screening negative: No clear elevated {symptom_name} symptoms detected",
            "summary": (
                f"The model score is {score:.1f}%, which is below the screening "
                "threshold of 50%."
            ),
            "signal": "Clearly below threshold",
            "recommendation": (
                "No clear elevated pattern was detected from this voice sample. "
                "If the person still feels distressed, clinical consultation is recommended."
            ),
            "score_text": f"{score:.1f} out of 100",
        }

    elif probability < 0.50:
        return {
            "is_positive": False,
            "outcome": f"Screening negative, but close to threshold",
            "summary": (
                f"The model score is {score:.1f}%, which is below but close to the "
                "screening threshold of 50%."
            ),
            "signal": "Close to threshold",
            "recommendation": (
                "The result is close to the decision boundary. A clearer or longer recording "
                "may be useful. If symptoms are present, use PHQ-9/GAD-7 or consult a professional."
            ),
            "score_text": f"{score:.1f} out of 100",
        }

    elif probability < 0.65:
        return {
            "is_positive": True,
            "outcome": f"Screening positive: Possible elevated {symptom_name} symptoms detected",
            "summary": (
                f"The model score is {score:.1f}%, which is slightly above the "
                "screening threshold of 50%."
            ),
            "signal": "Slightly above threshold",
            "recommendation": (
                "This suggests a possible elevated-symptom pattern, but the signal is not strong. "
                "A follow-up PHQ-9/GAD-7 questionnaire or professional review is recommended."
            ),
            "score_text": f"{score:.1f} out of 100",
        }

    elif probability < 0.80:
        return {
            "is_positive": True,
            "outcome": f"Screening positive: Elevated {symptom_name} symptoms detected",
            "summary": (
                f"The model score is {score:.1f}%, which is clearly above the "
                "screening threshold of 50%."
            ),
            "signal": "Moderate model signal",
            "recommendation": (
                "The voice pattern is closer to the elevated-symptom group learned during training. "
                "A standard clinical screening questionnaire or professional evaluation is recommended."
            ),
            "score_text": f"{score:.1f} out of 100",
        }

    else:
        return {
            "is_positive": True,
            "outcome": f"Screening positive: Strong indication of elevated {symptom_name} symptoms",
            "summary": (
                f"The model score is {score:.1f}%, which is strongly above the "
                "screening threshold of 50%."
            ),
            "signal": "Strong model signal",
            "recommendation": (
                "The speech pattern strongly matches the elevated-symptom group learned during training. "
                "Clinical follow-up is recommended."
            ),
            "score_text": f"{score:.1f} out of 100",
        }


def make_gauge_chart(probability, title):
    if probability is None:
        probability = 0.0

    score = probability * 100

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "%"},
            title={"text": title},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#4F46E5"},
                "steps": [
                    {"range": [0, 40], "color": "#DCFCE7"},
                    {"range": [40, 50], "color": "#FEF9C3"},
                    {"range": [50, 65], "color": "#FFEDD5"},
                    {"range": [65, 80], "color": "#FED7AA"},
                    {"range": [80, 100], "color": "#FECACA"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        )
    )

    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return fig


def show_audio_quality_card(audio_info):
    duration = audio_info["duration_sec"]
    rms = audio_info["rms"]

    if duration >= 30:
        quality = "Good"
        message = "Audio duration looks good."
        message_type = "success"
    elif duration >= 10:
        quality = "Acceptable"
        message = "Audio is acceptable, but 30–60 seconds is recommended for better reliability."
        message_type = "info"
    else:
        quality = "Too short"
        message = "Audio is too short. Please record at least 30–60 seconds."
        message_type = "warning"

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Duration", f"{duration:.1f} sec")

    with col2:
        st.metric("Sample Rate", f"{audio_info['sample_rate']} Hz")

    with col3:
        st.metric("Audio Quality", quality)

    if message_type == "success":
        st.success(message)
    elif message_type == "info":
        st.info(message)
    else:
        st.warning(message)

    if rms < 0.005:
        st.warning("The audio may be very low in volume. A clearer recording may improve reliability.")


def show_explanation_chart(result):
    explanation_details = result.get("explanation_details", None)

    if not explanation_details:
        st.write(result.get("explanation", "No explanation available."))
        return

    groups = explanation_details.get("top_feature_groups", [])

    if len(groups) == 0:
        st.write(result.get("explanation", "No explanation available."))
        return

    df = pd.DataFrame(groups)

    if "group" not in df.columns or "contribution" not in df.columns:
        st.write(result.get("explanation", "No explanation available."))
        return

    df["direction"] = df["contribution"].apply(
        lambda x: "Toward elevated symptoms" if x > 0 else "Toward minimal/no symptoms"
    )

    df["relative_influence"] = df["contribution"].abs()
    df = df.sort_values("relative_influence", ascending=True)

    fig = px.bar(
        df,
        x="relative_influence",
        y="group",
        color="direction",
        orientation="h",
        title="Main speech-pattern influence groups",
        labels={
            "relative_influence": "Relative influence",
            "group": "Speech feature group",
            "direction": "Direction",
        },
    )

    fig.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "This chart explains model behavior using speech-feature groups. It is not clinical reasoning."
    )


def show_score_card(title, result, symptom_name):
    probability = result.get("probability_elevated")
    interpretation = interpret_result(probability, symptom_name)

    st.markdown(f"## {title}")

    card_class = "positive-card" if interpretation["is_positive"] else "negative-card"

    st.markdown(
        f"""
<div class="{card_class}">
    <div class="big-outcome">{interpretation["outcome"]}</div>
    <p>{interpretation["summary"]}</p>
    <p><b>Meaning:</b> {interpretation["signal"]}</p>
</div>
""",
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.plotly_chart(
            make_gauge_chart(
                probability,
                "Elevated-symptom score"
            ),
            use_container_width=True
        )

    with col2:
        st.metric(
            label="Screening threshold",
            value="50%"
        )

        st.metric(
            label="Your model score",
            value=interpretation["score_text"]
        )

        st.metric(
            label="Strength of model signal",
            value=interpretation["signal"]
        )

    st.markdown(
        f"""
<div class="info-card">
    <b>Recommended next step:</b><br>
    {interpretation["recommendation"]}
</div>
""",
        unsafe_allow_html=True
    )

    with st.expander("What does this result mean?"):
        st.write(
            """
This model performs **binary screening**, not diagnosis.

- **Below 50%**: the voice pattern is closer to the minimal/no-symptom group.
- **50% or above**: the voice pattern is closer to the elevated-symptom group.

The score is not the percentage of illness. It is a model screening score based on speech features.
"""
        )

    with st.expander("How is this score calculated?"):
        st.write(
            """
The uploaded speech sample is converted into numerical acoustic features such as:

- pause and silence pattern  
- pitch and voicing variation  
- energy/loudness pattern  
- MFCC voice texture  
- spectral voice pattern  

These features are passed to a trained Logistic Regression model. The model outputs a score between 0 and 1 for the elevated-symptom class. The app converts this into a score out of 100.
"""
        )

    with st.expander("Why did the model give this result?"):
        st.write(result.get("explanation", "No explanation available."))

        try:
            show_explanation_chart(result)
        except Exception:
            pass


def show_overall_summary(result):
    dep_prob = result["depression"].get("probability_elevated")
    anx_prob = result["anxiety"].get("probability_elevated")

    dep_interpretation = interpret_result(dep_prob, "depression-related")
    anx_interpretation = interpret_result(anx_prob, "anxiety-related")

    st.markdown("## Overall Screening Summary")

    dep_status = "Positive" if dep_interpretation["is_positive"] else "Negative"
    anx_status = "Positive" if anx_interpretation["is_positive"] else "Negative"

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Depression Screening",
            dep_status,
            f"{dep_prob * 100:.1f}%" if dep_prob is not None else "NA"
        )

    with col2:
        st.metric(
            "Anxiety Screening",
            anx_status,
            f"{anx_prob * 100:.1f}%" if anx_prob is not None else "NA"
        )

    if dep_interpretation["is_positive"] or anx_interpretation["is_positive"]:
        st.warning(
            "At least one screening result is positive. This does not confirm a disorder, "
            "but follow-up screening or professional review is recommended."
        )
    else:
        st.success(
            "No clear elevated symptom pattern was detected in this voice sample. "
            "If distress is still present, professional consultation is recommended."
        )


def create_download_json(result):
    output = {
        "timestamp": datetime.now().isoformat(),
        "input": result.get("input"),
        "depression": result.get("depression"),
        "anxiety": result.get("anxiety"),
        "disclaimer": result.get("disclaimer"),
    }

    return json.dumps(output, indent=4, default=str)


# ============================================================
# Main UI
# ============================================================

st.title("🎙️ Manodayam Voice-Based Screening")

st.markdown(
    """
This demo analyzes a short speech response and gives a **screening-level**
indication of anxiety/depression-related symptoms.
"""
)

st.markdown(
    """
<div class="warning-card">
<b>Question 1:</b> Please describe your daily routine.<br><br>
For best results, speak naturally for at least <b>30–60 seconds</b> in a quiet environment.
</div>
""",
    unsafe_allow_html=True
)

with st.expander("What does this tool analyze?"):
    st.write(
        """
The system extracts acoustic features from the voice sample, including pause patterns,
pitch variation, energy/loudness, MFCC voice texture, and spectral features. These features
are used by trained machine learning models for binary screening.
"""
    )

st.warning(
    "This is not a medical diagnosis. It is only a speech-based screening demo."
)


# ============================================================
# Step 1: Consent
# ============================================================

st.markdown("## Step 1: Consent")

consent = st.checkbox(
    "I consent to upload or record my voice sample for screening analysis."
)


# ============================================================
# Step 2: Audio input
# ============================================================

st.markdown("## Step 2: Provide your Q1 audio")

input_mode = st.radio(
    "Choose input method",
    ["Upload audio file", "Record audio now"],
    horizontal=True
)

audio_file = None
audio_suffix = ".wav"

if input_mode == "Upload audio file":
    uploaded_file = st.file_uploader(
        "Upload Q1 audio file",
        type=["wav", "mp3", "m4a", "flac", "ogg"]
    )

    if uploaded_file is not None:
        audio_file = uploaded_file
        audio_suffix = Path(uploaded_file.name).suffix or ".wav"
        st.audio(uploaded_file)

else:
    recorded_audio = st.audio_input(
        "Record your response to Question 1",
        sample_rate=16000
    )

    if recorded_audio is not None:
        audio_file = recorded_audio
        audio_suffix = ".wav"
        st.audio(recorded_audio)


# ============================================================
# Step 3: Run prediction
# ============================================================

st.markdown("## Step 3: Run screening")

run_button = st.button(
    "Run Voice Screening",
    type="primary",
    use_container_width=True
)

if run_button:

    if not consent:
        st.error("Please provide consent before running the screening.")

    elif audio_file is None:
        st.error("Please upload or record an audio file.")

    else:
        temp_audio_path = save_audio_to_temp(audio_file, audio_suffix)

        try:
            with st.status("Processing voice sample...", expanded=True) as status:
                st.write("Checking audio quality...")
                audio_info = get_audio_info(temp_audio_path)

                show_audio_quality_card(audio_info)

                st.write("Extracting acoustic features...")
                st.write("Running screening models...")

                result = predict_all(temp_audio_path)

                status.update(
                    label="Screening completed.",
                    state="complete",
                    expanded=False
                )

            st.success("Screening completed successfully.")

            show_overall_summary(result)

            st.divider()

            show_score_card(
                "Depression-related Symptoms",
                result["depression"],
                "depression-related"
            )

            st.divider()

            show_score_card(
                "Anxiety-related Symptoms",
                result["anxiety"],
                "anxiety-related"
            )

            st.divider()

            st.markdown(
                """
<div class="info-card">
<b>Important:</b><br>
This does not mean the person has depression or anxiety. It means the uploaded voice sample
showed speech patterns that the model associated with one of the two learned classes.
For clinical evaluation, use validated questionnaires and professional assessment.
</div>
""",
                unsafe_allow_html=True
            )

            st.info(result["disclaimer"])

            st.download_button(
                label="Download screening result as JSON",
                data=create_download_json(result),
                file_name="manodayam_screening_result.json",
                mime="application/json",
                use_container_width=True
            )

        except Exception as e:
            st.error("Prediction failed.")
            st.exception(e)


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    """
Manodayam Voice Screening Demo | Intended for research and screening support only.
For clinical evaluation, consult a qualified mental health professional.
"""
)