"""
Feature extraction utilities for Manodayam speech-based screening deployment.

This file should stay aligned with the Kaggle training notebook feature extraction.
The deployed sklearn pipelines use only numeric acoustic features; status/error
fields are kept for debugging and ignored unless they appear in model feature_cols.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Any

import numpy as np
import scipy.stats as st
import librosa


TARGET_SR = 16000
MIN_AUDIO_DURATION_SEC = 5.0

AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma"
}


def safe_stats(values, prefix: str) -> Dict[str, float]:
    """
    Robust summary statistics.
    Always returns the same keys for a given prefix.
    """
    stat_names = [
        "mean", "std", "min", "max", "median",
        "q25", "q75", "iqr", "skew", "kurtosis",
    ]

    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {f"{prefix}_{stat}": np.nan for stat in stat_names}

    q25 = np.percentile(values, 25)
    q75 = np.percentile(values, 75)
    std = np.std(values)

    if len(values) > 2 and std > 1e-12:
        skew_value = float(st.skew(values))
    else:
        skew_value = 0.0

    if len(values) > 3 and std > 1e-12:
        kurtosis_value = float(st.kurtosis(values))
    else:
        kurtosis_value = 0.0

    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(std),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_q25": float(q25),
        f"{prefix}_q75": float(q75),
        f"{prefix}_iqr": float(q75 - q25),
        f"{prefix}_skew": skew_value,
        f"{prefix}_kurtosis": kurtosis_value,
    }


def empty_pause_features(duration=np.nan) -> Dict[str, Any]:
    features = {
        "duration_sec": duration,
        "speech_duration_sec": np.nan,
        "silence_duration_sec": np.nan,
        "speech_ratio": np.nan,
        "silence_ratio": np.nan,
        "num_speech_segments": 0,
        "pause_count": 0,
        "pause_rate": np.nan,
        "pause_total_sec": np.nan,
        "pause_mean_sec": np.nan,
        "pause_max_sec": np.nan,
    }
    features.update(safe_stats([], "speech_segment_duration"))
    features.update(safe_stats([], "pause_duration"))
    return features


def extract_pause_features(y, sr: int, top_db: int = 30) -> Dict[str, Any]:
    duration = len(y) / sr if sr else 0.0

    if duration <= 0:
        return empty_pause_features(duration=np.nan)

    intervals = librosa.effects.split(y, top_db=top_db)

    if len(intervals) == 0:
        features = {
            "duration_sec": duration,
            "speech_duration_sec": 0.0,
            "silence_duration_sec": duration,
            "speech_ratio": 0.0,
            "silence_ratio": 1.0,
            "num_speech_segments": 0,
            "pause_count": 0,
            "pause_rate": 0.0,
            "pause_total_sec": duration,
            "pause_mean_sec": duration,
            "pause_max_sec": duration,
        }
        features.update(safe_stats([], "speech_segment_duration"))
        features.update(safe_stats([duration], "pause_duration"))
        return features

    speech_durations = (intervals[:, 1] - intervals[:, 0]) / sr
    speech_total = float(np.sum(speech_durations))
    silence_total = max(0.0, duration - speech_total)

    pauses = []
    for i in range(1, len(intervals)):
        pause = (intervals[i, 0] - intervals[i - 1, 1]) / sr
        if pause > 0:
            pauses.append(pause)

    pauses = np.asarray(pauses, dtype=float)

    features = {
        "duration_sec": duration,
        "speech_duration_sec": speech_total,
        "silence_duration_sec": silence_total,
        "speech_ratio": speech_total / duration if duration > 0 else np.nan,
        "silence_ratio": silence_total / duration if duration > 0 else np.nan,
        "num_speech_segments": int(len(intervals)),
        "pause_count": int(len(pauses)),
        "pause_rate": float(len(pauses) / duration) if duration > 0 else np.nan,
        "pause_total_sec": float(np.sum(pauses)) if len(pauses) > 0 else 0.0,
        "pause_mean_sec": float(np.mean(pauses)) if len(pauses) > 0 else 0.0,
        "pause_max_sec": float(np.max(pauses)) if len(pauses) > 0 else 0.0,
    }

    features.update(safe_stats(speech_durations, "speech_segment_duration"))
    features.update(safe_stats(pauses, "pause_duration"))
    return features


def extract_pitch_features(y, sr: int) -> Dict[str, Any]:
    try:
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y,
            fmin=50,
            fmax=500,
            sr=sr,
        )

        valid_f0 = f0[np.isfinite(f0)] if f0 is not None else np.asarray([])

        if voiced_flag is not None:
            voiced_flag = np.asarray(voiced_flag, dtype=float)
            valid_voiced = voiced_flag[np.isfinite(voiced_flag)]
            voiced_ratio = float(np.mean(valid_voiced)) if len(valid_voiced) else np.nan
        else:
            voiced_ratio = np.nan

        features = {"voiced_ratio": voiced_ratio}
        features.update(safe_stats(valid_f0, "f0"))
        features.update(safe_stats(np.diff(valid_f0) if len(valid_f0) > 1 else [], "f0_diff"))
        features.update(safe_stats(voiced_prob if voiced_prob is not None else [], "voiced_prob"))
        return features

    except Exception as e:
        features = {
            "voiced_ratio": np.nan,
            "pitch_extraction_error": str(e),
        }
        features.update(safe_stats([], "f0"))
        features.update(safe_stats([], "f0_diff"))
        features.update(safe_stats([], "voiced_prob"))
        return features


def extract_audio_features(
    audio_path,
    sr_target: int = TARGET_SR,
    min_duration_sec: float = MIN_AUDIO_DURATION_SEC,
) -> Dict[str, Any]:
    """
    Extract acoustic features from one audio file.
    Returns numeric features plus debug status fields.
    """
    features: Dict[str, Any] = {
        "feature_extraction_status": "ok",
        "feature_extraction_error": "",
    }

    try:
        audio_path = str(audio_path)
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        y, sr = librosa.load(audio_path, sr=sr_target, mono=True)
        y = np.asarray(y, dtype=np.float32)
        duration = len(y) / sr if sr > 0 else 0.0

        features["raw_audio_duration_sec"] = float(duration)

        if duration < min_duration_sec:
            features["feature_extraction_status"] = "too_short"
            features["feature_extraction_error"] = (
                f"Audio duration {duration:.2f}s is shorter than minimum "
                f"{min_duration_sec:.2f}s"
            )
            features["audio_rms_global"] = np.nan
            features["audio_peak"] = np.nan
            features["audio_mean_abs"] = np.nan
            features.update(empty_pause_features(duration=duration))
            features.update(extract_pitch_features(y, sr) if len(y) > 0 else {})
            return features

        features["audio_rms_global"] = float(np.sqrt(np.mean(y ** 2)))
        features["audio_peak"] = float(np.max(np.abs(y)))
        features["audio_mean_abs"] = float(np.mean(np.abs(y)))

        features.update(extract_pause_features(y, sr))
        features.update(extract_pitch_features(y, sr))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=30)
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

            for i in range(mfcc.shape[0]):
                features.update(safe_stats(mfcc[i], f"mfcc_{i + 1}"))
                features.update(safe_stats(mfcc_delta[i], f"mfcc_delta_{i + 1}"))
                features.update(safe_stats(mfcc_delta2[i], f"mfcc_delta2_{i + 1}"))

            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
            rms = librosa.feature.rms(y=y)[0]
            zcr = librosa.feature.zero_crossing_rate(y=y)[0]

            features.update(safe_stats(spectral_centroid, "spectral_centroid"))
            features.update(safe_stats(spectral_bandwidth, "spectral_bandwidth"))
            features.update(safe_stats(spectral_rolloff, "spectral_rolloff"))
            features.update(safe_stats(spectral_flatness, "spectral_flatness"))
            features.update(safe_stats(rms, "rms_frame"))
            features.update(safe_stats(zcr, "zcr"))

            try:
                spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
                for i in range(spectral_contrast.shape[0]):
                    features.update(safe_stats(spectral_contrast[i], f"spectral_contrast_{i + 1}"))
                features["spectral_contrast_status"] = "ok"
                features["spectral_contrast_error"] = ""

            except Exception as e:
                features["spectral_contrast_status"] = "failed"
                features["spectral_contrast_error"] = str(e)
                for i in range(7):
                    features.update(safe_stats([], f"spectral_contrast_{i + 1}"))

        return features

    except Exception as e:
        features["feature_extraction_status"] = "failed"
        features["feature_extraction_error"] = str(e)
        features["raw_audio_duration_sec"] = np.nan
        return features


def prefix_features(features: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """Prefix feature names with q1_ or q2_."""
    return {f"{prefix}_{k}": v for k, v in features.items()}
