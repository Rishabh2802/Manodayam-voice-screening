import warnings
import numpy as np
import librosa
import scipy.stats as st


def safe_stats(values, prefix):
    values = np.asarray(values).astype(float).ravel()
    values = values[np.isfinite(values)]

    stats = [
        "mean", "std", "min", "max", "median",
        "q25", "q75", "iqr", "skew", "kurtosis"
    ]

    if len(values) == 0:
        return {f"{prefix}_{stat}": np.nan for stat in stats}

    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_q25": float(np.percentile(values, 25)),
        f"{prefix}_q75": float(np.percentile(values, 75)),
        f"{prefix}_iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
        f"{prefix}_skew": float(st.skew(values)) if len(values) > 2 else 0.0,
        f"{prefix}_kurtosis": float(st.kurtosis(values)) if len(values) > 3 else 0.0,
    }


def extract_pause_features(y, sr, top_db=30):
    duration = len(y) / sr

    if duration <= 0:
        return {}

    intervals = librosa.effects.split(y, top_db=top_db)

    if len(intervals) == 0:
        return {
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
        "speech_ratio": speech_total / duration,
        "silence_ratio": silence_total / duration,
        "num_speech_segments": int(len(intervals)),
        "pause_count": int(len(pauses)),
        "pause_rate": float(len(pauses) / duration),
        "pause_total_sec": float(np.sum(pauses)) if len(pauses) > 0 else 0.0,
        "pause_mean_sec": float(np.mean(pauses)) if len(pauses) > 0 else 0.0,
        "pause_max_sec": float(np.max(pauses)) if len(pauses) > 0 else 0.0,
    }

    features.update(safe_stats(speech_durations, "speech_segment_duration"))
    features.update(safe_stats(pauses, "pause_duration"))

    return features


def extract_pitch_features(y, sr):
    try:
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y,
            fmin=50,
            fmax=500,
            sr=sr
        )

        valid_f0 = f0[np.isfinite(f0)]

        features = {
            "voiced_ratio": float(np.mean(voiced_flag)) if voiced_flag is not None else np.nan
        }

        features.update(safe_stats(valid_f0, "f0"))

        if len(valid_f0) > 1:
            features.update(safe_stats(np.diff(valid_f0), "f0_diff"))
        else:
            features.update(safe_stats([], "f0_diff"))

        if voiced_prob is not None:
            features.update(safe_stats(voiced_prob, "voiced_prob"))
        else:
            features.update(safe_stats([], "voiced_prob"))

        return features

    except Exception:
        features = {"voiced_ratio": np.nan}
        features.update(safe_stats([], "f0"))
        features.update(safe_stats([], "f0_diff"))
        features.update(safe_stats([], "voiced_prob"))
        return features


def extract_audio_features(audio_path, sr_target=16000):
    features = {}

    y, sr = librosa.load(audio_path, sr=sr_target, mono=True)
    y = np.asarray(y, dtype=np.float32)

    if len(y) < sr:
        raise ValueError("Audio is too short. Please upload at least 10–15 seconds.")

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
                features.update(
                    safe_stats(
                        spectral_contrast[i],
                        f"spectral_contrast_{i + 1}"
                    )
                )

        except Exception:
            pass

    return features