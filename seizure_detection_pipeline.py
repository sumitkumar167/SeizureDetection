"""
TinyML-Based EEG Seizure Detection Pipeline
============================================
Dataset: CHB-MIT Scalp EEG (EDF Files)
Target:  Edge Impulse / TinyML compatible binary classifier

Pipeline Stages:
  1. EDF Parsing       - Load EEG signals and parse seizure annotations
  2. Windowing         - Segment EEG into labelled 2-second windows
  3. Feature Extraction- Lightweight statistical + spectral features per window
  4. Model Training    - MLP classifier with probability output
  5. Abnormality Score - EMA-smoothed seizure probability
  6. Prediction-Lite   - Rolling-slope early warning system
  7. Visualization     - Abnormality curves, risk levels, lead-time histogram

Author : Sumit Kumar
Date   : 2026-03-01
"""

# ─────────────────────────────────────────────────────────────────────────────
# Standard Library
# ─────────────────────────────────────────────────────────────────────────────
import os
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Third-party
# ─────────────────────────────────────────────────────────────────────────────
import mne
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import welch
from scipy.stats import differential_entropy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    recall_score,
    classification_report,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("WARNING")


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1 — EDF Parsing
# ─────────────────────────────────────────────────────────────────────────────

def load_edf(edf_path: str) -> Tuple[np.ndarray, float, List[str]]:
    """Load an EDF file and return raw EEG data.

    Parameters
    ----------
    edf_path : str
        Absolute or relative path to the .edf file.

    Returns
    -------
    data : np.ndarray, shape (n_channels, n_samples)
        Raw EEG signal.
    sfreq : float
        Sampling frequency in Hz.
    ch_names : list of str
        Channel names.
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    data = raw.get_data()          # shape: (n_channels, n_samples)
    sfreq = raw.info["sfreq"]
    ch_names = raw.ch_names
    return data, sfreq, ch_names


def parse_summary(summary_path: str) -> Dict[str, List[Tuple[float, float]]]:
    """Parse a CHB-MIT summary text file for seizure start/end times.

    The CHB-MIT summary files follow a repetitive structure such as::

        File Name: chb01_03.edf
        Seizure 1 Start Time: 2996 seconds
        Seizure 1 End Time:   3036 seconds

    Parameters
    ----------
    summary_path : str
        Path to the *-summary.txt file for one patient.

    Returns
    -------
    annotations : dict
        Mapping from EDF filename (basename only) to a list of
        (start_sec, end_sec) tuples.
    """
    annotations: Dict[str, List[Tuple[float, float]]] = {}
    current_file: Optional[str] = None
    starts: List[float] = []
    ends:   List[float] = []

    with open(summary_path, "r") as fh:
        for line in fh:
            line = line.strip()

            # Detect file name header
            file_match = re.match(r"File Name:\s*(.+\.edf)", line, re.IGNORECASE)
            if file_match:
                # Save previous file's seizures before moving on
                if current_file is not None:
                    pairs = list(zip(starts, ends))
                    annotations.setdefault(current_file, []).extend(pairs)
                current_file = file_match.group(1).strip()
                starts, ends = [], []
                continue

            start_match = re.search(
                r"Seizure(?:\s*\d+)?\s*Start\s*Time\s*[:\-]\s*([\d.]+)\s*seconds?",
                line, re.IGNORECASE,
            )
            if start_match:
                starts.append(float(start_match.group(1)))

            end_match = re.search(
                r"Seizure(?:\s*\d+)?\s*End\s*Time\s*[:\-]\s*([\d.]+)\s*seconds?",
                line, re.IGNORECASE,
            )
            if end_match:
                ends.append(float(end_match.group(1)))

    # Flush last file
    if current_file is not None:
        pairs = list(zip(starts, ends))
        annotations.setdefault(current_file, []).extend(pairs)

    return annotations


def seizure_intervals_to_samples(
    intervals_sec: List[Tuple[float, float]], sfreq: float
) -> List[Tuple[int, int]]:
    """Convert seizure intervals from seconds to sample indices.

    Parameters
    ----------
    intervals_sec : list of (float, float)
        Seizure (start, end) pairs in seconds.
    sfreq : float
        Sampling frequency of the recording.

    Returns
    -------
    intervals_samp : list of (int, int)
        Seizure (start_sample, end_sample) pairs.
    """
    return [(int(s * sfreq), int(e * sfreq)) for s, e in intervals_sec]


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2 — Windowing
# ─────────────────────────────────────────────────────────────────────────────

def create_windows(
    data: np.ndarray,
    sfreq: float,
    seizure_intervals_samp: List[Tuple[int, int]],
    window_sec: float = 2.0,
    overlap: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Segment EEG into labelled overlapping windows.

    Parameters
    ----------
    data : np.ndarray, shape (n_channels, n_samples)
        Raw EEG recording.
    sfreq : float
        Sampling frequency.
    seizure_intervals_samp : list of (int, int)
        Seizure sample intervals from :func:`seizure_intervals_to_samples`.
    window_sec : float
        Window duration in seconds (default 2.0).
    overlap : float
        Fractional overlap between consecutive windows (default 0.5 → 50 %).

    Returns
    -------
    windows : np.ndarray, shape (N, n_channels, window_samples)
        Segmented EEG data.
    labels : np.ndarray, shape (N,)
        Binary labels — 1 if window overlaps a seizure, 0 otherwise.
    """
    n_channels, n_samples = data.shape
    win_len = int(window_sec * sfreq)       # samples per window
    step    = int(win_len * (1.0 - overlap))  # hop size

    windows_list: List[np.ndarray] = []
    labels_list:  List[int]        = []

    for start in range(0, n_samples - win_len + 1, step):
        end = start + win_len
        window = data[:, start:end]

        # Label: 1 if this window overlaps any seizure interval
        label = 0
        for sz_start, sz_end in seizure_intervals_samp:
            # Overlap condition: not (end <= sz_start or start >= sz_end)
            if not (end <= sz_start or start >= sz_end):
                label = 1
                break

        windows_list.append(window)
        labels_list.append(label)

    windows = np.array(windows_list, dtype=np.float32)  # (N, C, W)
    labels  = np.array(labels_list,  dtype=np.int32)    # (N,)
    return windows, labels


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3 — Feature Extraction
# ─────────────────────────────────────────────────────────────────────────────

def _band_power(signal: np.ndarray, sfreq: float, low: float, high: float) -> float:
    """Compute average band power via Welch's method.

    Parameters
    ----------
    signal : np.ndarray, shape (n_samples,)
        1-D time-domain signal for a single channel.
    sfreq : float
        Sampling frequency.
    low, high : float
        Frequency band boundaries (Hz).

    Returns
    -------
    power : float
        Mean power spectral density within [low, high] Hz.
    """
    nperseg = min(len(signal), 256)
    freqs, psd = welch(signal, fs=sfreq, nperseg=nperseg)
    idx = np.logical_and(freqs >= low, freqs <= high)
    if not idx.any():
        return 0.0
    return float(np.mean(psd[idx]))


def _hjorth_parameters(signal: np.ndarray) -> Tuple[float, float, float]:
    """Compute Hjorth Activity, Mobility, and Complexity.

    Parameters
    ----------
    signal : np.ndarray, shape (n_samples,)
        1-D EEG window for a single channel.

    Returns
    -------
    activity : float
        Variance of the signal.
    mobility : float
        Ratio of standard deviations of first derivative to signal.
    complexity : float
        Ratio of mobility of second derivative to mobility of first derivative.
    """
    activity = float(np.var(signal))

    d1 = np.diff(signal)
    var_d1 = float(np.var(d1)) if len(d1) > 0 else 0.0

    mobility = np.sqrt(var_d1 / activity) if activity > 0 else 0.0

    d2 = np.diff(d1)
    var_d2 = float(np.var(d2)) if len(d2) > 0 else 0.0
    mobility_d1 = np.sqrt(var_d2 / var_d1) if var_d1 > 0 else 0.0
    complexity = mobility_d1 / mobility if mobility > 0 else 0.0

    return activity, float(mobility), float(complexity)


def _spectral_entropy(signal: np.ndarray, sfreq: float, n_freq_bins: int = 128) -> float:
    """Compute spectral entropy of a signal.

    Parameters
    ----------
    signal : np.ndarray, shape (n_samples,)
        1-D EEG signal.
    sfreq : float
        Sampling frequency.
    n_freq_bins : int
        Number of frequency bins for PSD estimation.

    Returns
    -------
    entropy : float
        Spectral entropy value.
    """
    nperseg = min(len(signal), n_freq_bins * 2)
    _, psd = welch(signal, fs=sfreq, nperseg=nperseg)
    psd_norm = psd / (psd.sum() + 1e-12)
    return float(-np.sum(psd_norm * np.log2(psd_norm + 1e-12)))


def extract_features_window(window: np.ndarray, sfreq: float) -> np.ndarray:
    """Extract a feature vector from a single multi-channel EEG window.

    Features per channel (11 per channel):
        - Mean
        - Variance
        - RMS
        - Hjorth Activity, Mobility, Complexity
        - Band power: delta, theta, alpha, beta, gamma
        - Spectral entropy

    Parameters
    ----------
    window : np.ndarray, shape (n_channels, n_samples)
        EEG data for one window.
    sfreq : float
        Sampling frequency.

    Returns
    -------
    features : np.ndarray, shape (n_channels * 11,)
        Flattened feature vector.
    """
    BANDS = {
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta":  (13.0, 30.0),
        "gamma": (30.0, 40.0),
    }

    feature_vec: List[float] = []
    for ch_signal in window:
        mean     = float(np.mean(ch_signal))
        variance = float(np.var(ch_signal))
        rms      = float(np.sqrt(np.mean(ch_signal ** 2)))

        activity, mobility, complexity = _hjorth_parameters(ch_signal)

        band_powers = [
            _band_power(ch_signal, sfreq, lo, hi)
            for lo, hi in BANDS.values()
        ]

        spec_ent = _spectral_entropy(ch_signal, sfreq)

        feature_vec.extend(
            [mean, variance, rms, activity, mobility, complexity]
            + band_powers
            + [spec_ent]
        )

    return np.array(feature_vec, dtype=np.float32)


def extract_features_batch(
    windows: np.ndarray,
    sfreq: float,
    verbose: bool = True,
) -> np.ndarray:
    """Extract features for a batch of windows.

    Parameters
    ----------
    windows : np.ndarray, shape (N, n_channels, n_samples)
        Batch of EEG windows.
    sfreq : float
        Sampling frequency.
    verbose : bool
        Print progress every 500 windows.

    Returns
    -------
    feature_matrix : np.ndarray, shape (N, n_features)
        One feature vector per window.
    """
    n_windows = len(windows)
    features  = []
    for i, window in enumerate(windows):
        if verbose and i % 500 == 0:
            print(f"  Extracting features: {i}/{n_windows}", end="\r")
        features.append(extract_features_window(window, sfreq))

    if verbose:
        print(f"  Extracting features: {n_windows}/{n_windows} — done.    ")
    return np.array(features, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 4 — Model Training
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    hidden_layer_sizes: Tuple[int, ...] = (32, 16),
    random_state: int = 42,
) -> Tuple[MLPClassifier, StandardScaler, Dict]:
    """Train an MLP binary seizure classifier.

    The architecture mirrors Input → 32 → 16 → 1 (sigmoid), implemented via
    sklearn's MLPClassifier with logistic activation and lbfgs/adam solver.

    Parameters
    ----------
    X_train, X_test : np.ndarray, shape (N, n_features)
        Feature matrices for training and evaluation.
    y_train, y_test : np.ndarray, shape (N,)
        Binary labels.
    hidden_layer_sizes : tuple of int
        Number of neurons in each hidden layer.
    random_state : int
        Reproducibility seed.

    Returns
    -------
    model : MLPClassifier
        Trained classifier.
    scaler : StandardScaler
        Fitted feature scaler (must be applied before inference).
    metrics : dict
        Dictionary with keys: accuracy, sensitivity, roc_auc.
    """
    # Feature normalisation
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # MLP: Input → 32 → 16 → 1 (sigmoid output via predict_proba)
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=random_state,
        verbose=False,
    )

    print("Training MLP …")
    model.fit(X_train_sc, y_train)

    # Evaluation
    y_pred      = model.predict(X_test_sc)
    y_prob      = model.predict_proba(X_test_sc)[:, 1]
    accuracy    = float(accuracy_score(y_test, y_pred))
    sensitivity = float(recall_score(y_test, y_pred, zero_division=0))
    roc_auc     = float(roc_auc_score(y_test, y_prob))

    metrics = {
        "accuracy":    accuracy,
        "sensitivity": sensitivity,
        "roc_auc":     roc_auc,
    }

    print("\n" + "─" * 50)
    print("  Model Performance")
    print("─" * 50)
    print(f"  Accuracy    : {accuracy:.4f}")
    print(f"  Sensitivity : {sensitivity:.4f}  (Recall for seizure class)")
    print(f"  ROC-AUC     : {roc_auc:.4f}")
    print("─" * 50)
    print(classification_report(y_test, y_pred, target_names=["Normal", "Seizure"]))

    return model, scaler, metrics


def predict_probabilities(
    model: MLPClassifier,
    scaler: StandardScaler,
    X: np.ndarray,
) -> np.ndarray:
    """Return seizure probabilities for a feature matrix.

    Parameters
    ----------
    model : MLPClassifier
        Trained MLP classifier.
    scaler : StandardScaler
        Fitted scaler used during training.
    X : np.ndarray, shape (N, n_features)
        Raw (unscaled) feature matrix for N windows.

    Returns
    -------
    proba : np.ndarray, shape (N,)
        Predicted seizure probability for each window.
    """
    X_sc = scaler.transform(X)
    return model.predict_proba(X_sc)[:, 1].astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 5 — Abnormality Score
# ─────────────────────────────────────────────────────────────────────────────

def compute_abnormality_score(
    probabilities: np.ndarray,
    alpha: float = 0.2,
) -> np.ndarray:
    """Apply exponential moving average (EMA) smoothing.

    Formula::

        A_smooth(t) = α * A(t) + (1 - α) * A_smooth(t-1)

    Parameters
    ----------
    probabilities : np.ndarray, shape (N,)
        Raw model seizure probabilities A(t).
    alpha : float
        EMA decay factor (default 0.2).

    Returns
    -------
    smoothed : np.ndarray, shape (N,)
        EMA-smoothed abnormality score.
    """
    smoothed = np.empty_like(probabilities)
    smoothed[0] = probabilities[0]
    for t in range(1, len(probabilities)):
        smoothed[t] = alpha * probabilities[t] + (1.0 - alpha) * smoothed[t - 1]
    return smoothed


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 6 — Prediction-Lite Early Warning
# ─────────────────────────────────────────────────────────────────────────────

def compute_risk_levels(
    smoothed_score: np.ndarray,
    window_sec: float = 2.0,
    overlap: float = 0.5,
    K: int = 30,
    thresh_elevated: float = 0.3,
    thresh_high: float = 0.6,
    slope_threshold: float = 0.005,
    M: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute rolling slope and assign 3-level risk classification.

    Risk levels::

        Level 0 — Normal
        Level 1 — Elevated  : A_smooth > thresh_elevated AND slope > 0
        Level 2 — High Risk : A_smooth > thresh_high AND slope > slope_threshold
                              for M consecutive windows

    Parameters
    ----------
    smoothed_score : np.ndarray, shape (N,)
        EMA-smoothed abnormality scores.
    window_sec : float
        Duration of each analysis window in seconds.
    overlap : float
        Fractional overlap used when creating windows.
    K : int
        Look-back size for slope calculation (number of windows).
    thresh_elevated : float
        Threshold for Level 1 classification.
    thresh_high : float
        Threshold for Level 2 classification.
    slope_threshold : float
        Minimum slope to trigger Level 2.
    M : int
        Minimum consecutive Level-2 windows to confirm high-risk flag.

    Returns
    -------
    risk_levels : np.ndarray, shape (N,) of int
        Per-window risk level (0, 1, or 2).
    slopes : np.ndarray, shape (N,)
        Rolling slope values (undefined for the first K windows → 0.0).
    """
    N     = len(smoothed_score)
    step_sec   = window_sec * (1.0 - overlap)   # time between window starts
    delta_t    = step_sec                        # Δt between windows

    slopes      = np.zeros(N, dtype=np.float32)
    risk_levels = np.zeros(N, dtype=np.int32)

    # Rolling slope
    for t in range(K, N):
        slope = (smoothed_score[t] - smoothed_score[t - K]) / (K * delta_t)
        slopes[t] = slope

    # Preliminary level assignment
    for t in range(N):
        score = smoothed_score[t]
        slope = slopes[t]
        if score > thresh_high and slope > slope_threshold:
            risk_levels[t] = 2
        elif score > thresh_elevated and slope > 0:
            risk_levels[t] = 1
        else:
            risk_levels[t] = 0

    # Apply consecutive-M filter for Level 2 (avoid spurious alerts)
    confirmed  = np.zeros(N, dtype=np.int32)
    consecutive = 0
    for t in range(N):
        if risk_levels[t] == 2:
            consecutive += 1
        else:
            consecutive = 0
        if consecutive >= M:
            confirmed[max(0, t - M + 1): t + 1] = 2

    # Merge: where confirmed says 2, override; otherwise keep preliminary
    for t in range(N):
        if confirmed[t] == 2:
            risk_levels[t] = 2
        elif risk_levels[t] == 2:
            # Downgrade lone Level-2 windows that didn't satisfy M-consecutive
            risk_levels[t] = 1

    return risk_levels, slopes


def evaluate_early_warning(
    risk_levels: np.ndarray,
    seizure_onset_windows: List[int],
    window_sec: float = 2.0,
    overlap: float = 0.5,
) -> List[float]:
    """Compute lead time between Level-2 alert and seizure onset.

    For each seizure onset window, find the earliest preceding Level-2
    flag and compute the time difference.

    Parameters
    ----------
    risk_levels : np.ndarray, shape (N,)
        Per-window risk levels.
    seizure_onset_windows : list of int
        Window indices corresponding to seizure onset.
    window_sec : float
        Window duration in seconds.
    overlap : float
        Fractional overlap.

    Returns
    -------
    lead_times : list of float
        Lead times in seconds (positive means warning before seizure).
        If no warning preceded a seizure, that entry is 0.0.
    """
    step_sec   = window_sec * (1.0 - overlap)
    lead_times = []

    for onset_idx in seizure_onset_windows:
        # Scan backwards from onset to find the earliest Level-2 flag
        alert_idx = None
        for t in range(onset_idx - 1, -1, -1):
            if risk_levels[t] == 2:
                alert_idx = t
            else:
                break  # stop at first non-Level-2 window before onset

        if alert_idx is not None:
            lead_time = (onset_idx - alert_idx) * step_sec
        else:
            lead_time = 0.0

        lead_times.append(lead_time)

    return lead_times


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 7 — Visualization
# ─────────────────────────────────────────────────────────────────────────────

RISK_COLORS = {0: "#4CAF50", 1: "#FF9800", 2: "#F44336"}
RISK_LABELS = {0: "Normal", 1: "Elevated", 2: "High Risk"}


def _window_times(N: int, window_sec: float, overlap: float) -> np.ndarray:
    """Return the start time (seconds) of each window."""
    step_sec = window_sec * (1.0 - overlap)
    return np.arange(N) * step_sec


def plot_abnormality_and_risk(
    raw_proba: np.ndarray,
    smoothed_score: np.ndarray,
    risk_levels: np.ndarray,
    slopes: np.ndarray,
    true_labels: np.ndarray,
    window_sec: float = 2.0,
    overlap: float = 0.5,
    save_path: Optional[str] = None,
) -> None:
    """Plot abnormality score, risk levels, and ground-truth seizure overlay.

    Parameters
    ----------
    raw_proba : np.ndarray, shape (N,)
        Raw model seizure probabilities.
    smoothed_score : np.ndarray, shape (N,)
        EMA-smoothed abnormality scores.
    risk_levels : np.ndarray, shape (N,)
        Per-window risk levels (0, 1, 2).
    slopes : np.ndarray, shape (N,)
        Rolling slope values.
    true_labels : np.ndarray, shape (N,)
        Ground-truth binary seizure labels.
    window_sec : float
        Window duration for time-axis computation.
    overlap : float
        Fractional overlap for time-axis computation.
    save_path : str or None
        If provided, figure is saved to this path instead of displayed.
    """
    times = _window_times(len(raw_proba), window_sec, overlap)

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(4, 1, hspace=0.45)

    # ── Panel 1: Abnormality Score ──
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(times, raw_proba,      alpha=0.35, color="#90CAF9", label="Raw Probability")
    ax1.plot(times, smoothed_score, linewidth=2, color="#1565C0", label="EMA Smoothed")
    ax1.axhline(0.3, color="#FF9800", linestyle="--", linewidth=1.0, label="Elevated threshold (0.3)")
    ax1.axhline(0.6, color="#F44336", linestyle="--", linewidth=1.0, label="High-risk threshold (0.6)")

    # Overlay true seizure regions
    _shade_seizures(ax1, times, true_labels)
    ax1.set_ylabel("Abnormality Score")
    ax1.set_title("Abnormality Score vs Time")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlim(times[0], times[-1])

    # ── Panel 2: Risk Levels ──
    ax2 = fig.add_subplot(gs[1])
    for t, (time, lvl) in enumerate(zip(times, risk_levels)):
        width = window_sec * (1.0 - overlap)
        ax2.bar(time, 1, width=width, align="edge",
                color=RISK_COLORS[lvl], alpha=0.75, linewidth=0)
    _shade_seizures(ax2, times, true_labels, alpha=0.3)
    ax2.set_ylabel("Risk Level")
    ax2.set_title("Prediction-Lite Risk Classification")
    ax2.set_yticks([0.2, 0.5, 0.85])
    ax2.set_yticklabels(["0 - Normal", "1 - Elevated", "2 - High Risk"], fontsize=8)
    ax2.set_xlim(times[0], times[-1])

    # Legend for risk colours
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(color=RISK_COLORS[k], label=RISK_LABELS[k]) for k in sorted(RISK_COLORS)
    ]
    ax2.legend(handles=legend_patches, fontsize=8, loc="upper left")

    # ── Panel 3: Rolling Slope ──
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(times, slopes, color="#6A1B9A", linewidth=1.5)
    ax3.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    _shade_seizures(ax3, times, true_labels)
    ax3.set_ylabel("Slope")
    ax3.set_title("Rolling Slope of Abnormality Score")
    ax3.set_xlim(times[0], times[-1])

    # ── Panel 4: Ground-truth Label ──
    ax4 = fig.add_subplot(gs[3])
    ax4.fill_between(times, true_labels, alpha=0.6, color="#EF5350", step="post",
                     label="Seizure (ground truth)")
    ax4.set_ylabel("Seizure Label")
    ax4.set_xlabel("Time (seconds)")
    ax4.set_title("Ground-Truth Seizure Labels")
    ax4.set_ylim(-0.05, 1.3)
    ax4.set_xlim(times[0], times[-1])
    ax4.legend(fontsize=8)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Figure saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


def _shade_seizures(
    ax: plt.Axes,
    times: np.ndarray,
    true_labels: np.ndarray,
    alpha: float = 0.15,
    color: str = "#EF5350",
) -> None:
    """Add semi-transparent seizure shading to an axes panel.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    times : np.ndarray
        Window start times.
    true_labels : np.ndarray
        Ground-truth binary labels aligned with *times*.
    alpha : float
        Opacity of shading.
    color : str
        Fill colour.
    """
    in_seizure = False
    start_t    = 0.0
    for t, label in zip(times, true_labels):
        if label == 1 and not in_seizure:
            start_t    = t
            in_seizure = True
        elif label == 0 and in_seizure:
            ax.axvspan(start_t, t, color=color, alpha=alpha, label="_nolegend_")
            in_seizure = False
    if in_seizure:
        ax.axvspan(start_t, times[-1], color=color, alpha=alpha, label="_nolegend_")


def plot_lead_time_histogram(
    lead_times: List[float],
    save_path: Optional[str] = None,
) -> None:
    """Plot a histogram of early-warning lead times.

    Parameters
    ----------
    lead_times : list of float
        Lead times in seconds from :func:`evaluate_early_warning`.
    save_path : str or None
        If provided, figure is saved to this path.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    valid = [lt for lt in lead_times if lt > 0]

    if not valid:
        ax.text(0.5, 0.5, "No early warnings detected",
                ha="center", va="center", transform=ax.transAxes, fontsize=14)
    else:
        ax.hist(valid, bins=max(5, len(valid) // 2), color="#1565C0", edgecolor="white",
                alpha=0.85)
        ax.axvline(np.mean(valid), color="#F44336", linestyle="--",
                   linewidth=1.5, label=f"Mean = {np.mean(valid):.1f}s")
        ax.legend(fontsize=10)

    ax.set_xlabel("Lead Time (seconds)")
    ax.set_ylabel("Count")
    ax.set_title("Early Warning Lead Time Distribution")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Figure saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# HIGH-LEVEL UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset_from_patient(
    edf_dir: str,
    summary_path: str,
    window_sec: float = 2.0,
    overlap: float = 0.5,
    max_files: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Load all EDF files for one patient and build labelled feature matrix.

    Parameters
    ----------
    edf_dir : str
        Directory containing *.edf files for a patient.
    summary_path : str
        Path to the corresponding *-summary.txt file.
    window_sec : float
        Window duration in seconds.
    overlap : float
        Fractional overlap.
    max_files : int or None
        If set, only process the first *max_files* EDFs (useful for quick tests).

    Returns
    -------
    X : np.ndarray, shape (N_total, n_features)
        Feature matrix across all EDF files for this patient.
    y : np.ndarray, shape (N_total,)
        Binary labels.
    sfreq : float
        Sampling frequency (from the first EDF).
    """
    annotations = parse_summary(summary_path)
    edf_files   = sorted(Path(edf_dir).glob("*.edf"))
    if max_files is not None:
        edf_files = edf_files[:max_files]

    all_X: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    sfreq_global: Optional[float] = None

    for edf_path in edf_files:
        fname = edf_path.name
        print(f"  Loading {fname} …")

        data, sfreq, _ = load_edf(str(edf_path))

        if sfreq_global is None:
            sfreq_global = sfreq
        elif sfreq != sfreq_global:
            print(f"  ⚠  sfreq mismatch ({sfreq} vs {sfreq_global}) — skipping {fname}")
            continue

        intervals_sec  = annotations.get(fname, [])
        intervals_samp = seizure_intervals_to_samples(intervals_sec, sfreq)

        windows, labels = create_windows(data, sfreq, intervals_samp, window_sec, overlap)
        X = extract_features_batch(windows, sfreq, verbose=True)
        all_X.append(X)
        all_y.append(labels)

    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)
    return X_all, y_all, sfreq_global


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE USAGE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ─── Configuration ────────────────────────────────────────────────────────
    # Update these paths to point at your local CHB-MIT dataset.
    PATIENT_ID   = "chb01"
    DATA_ROOT    = Path("/Users/sumitkumar/Downloads/Lectures/Spring26/Individual Instruction/EEG_Classification/MIT_Scalp_EEG_Dataset/physionet.org/files/chbmit/1.0.0")
    EDF_DIR      = DATA_ROOT / PATIENT_ID
    SUMMARY_FILE = EDF_DIR   / f"{PATIENT_ID}-summary.txt"
    OUT_DIR      = Path("outputs")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ─── Step 1 + 2 + 3  Build dataset ────────────────────────────────────────
    print("=" * 60)
    print(f"Building dataset for patient: {PATIENT_ID}")
    print("=" * 60)

    X, y, sfreq = build_dataset_from_patient(
        edf_dir=str(EDF_DIR),
        summary_path=str(SUMMARY_FILE),
        window_sec=2.0,
        overlap=0.5,
        max_files=None,   # set to e.g. 5 to quicktest on fewer files
    )

    print(f"\nDataset shape : X={X.shape}, y={y.shape}")
    print(f"Seizure ratio : {y.mean()*100:.2f}%")

    # ─── Step 4  Train / test split & model training ──────────────────────────
    # Stratified split — in a multi-patient study, split by patient instead.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model, scaler, metrics = train_model(X_train, y_train, X_test, y_test)

    # ─── Step 5  Abnormality score on test set ────────────────────────────────
    # For the early warning demo we run inference on the FULL sorted sequence.
    # Sorting preserves temporal order in the test set.
    raw_proba      = predict_probabilities(model, scaler, X_test)
    smoothed_score = compute_abnormality_score(raw_proba, alpha=0.2)

    # ─── Step 6  Risk levels & early warning ─────────────────────────────────
    risk_levels, slopes = compute_risk_levels(
        smoothed_score,
        window_sec=2.0,
        overlap=0.5,
        K=30,
        thresh_elevated=0.3,
        thresh_high=0.6,
        slope_threshold=0.005,
        M=5,
    )

    # Find seizure onset windows in the test set
    seizure_onset_windows = []
    prev = 0
    for idx, label in enumerate(y_test):
        if label == 1 and prev == 0:
            seizure_onset_windows.append(idx)
        prev = label

    lead_times = evaluate_early_warning(
        risk_levels, seizure_onset_windows, window_sec=2.0, overlap=0.5
    )
    print(f"\nEarly Warning Lead Times (s): {lead_times}")
    if lead_times:
        print(f"  Mean lead time : {np.mean(lead_times):.1f} s")

    # ─── Step 7  Visualisation ────────────────────────────────────────────────
    plot_abnormality_and_risk(
        raw_proba       = raw_proba,
        smoothed_score  = smoothed_score,
        risk_levels     = risk_levels,
        slopes          = slopes,
        true_labels     = y_test,
        window_sec      = 2.0,
        overlap         = 0.5,
        save_path       = str(OUT_DIR / "abnormality_and_risk.png"),
    )

    plot_lead_time_histogram(
        lead_times = lead_times,
        save_path  = str(OUT_DIR / "lead_time_histogram.png"),
    )

    print("\n✓ Pipeline complete. Outputs saved to:", OUT_DIR.resolve())
