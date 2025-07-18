import os
import numpy as np
import pandas as pd
import mne
from mne import create_info
from mne.io import RawArray
from mne.preprocessing import ICA
from scipy.signal import welch
from scipy.integrate import trapezoid
from scipy.linalg import eigh
from scipy.stats import entropy, skew, kurtosis
from sklearn.preprocessing import StandardScaler
from antropy import hjorth_params, higuchi_fd

# === CONFIGURATION ===
SAMPLE_RATE = 256
WINDOW_SEC = 4
WINDOW_SIZE = SAMPLE_RATE * WINDOW_SEC
OVERLAP = 0.75
STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP))
CHANNELS = ['Fp1', 'F3', 'F7', 'Fz', 'Fp2', 'F4', 'F8',
            'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4', 'O1',
            'O2', 'T3', 'T4', 'T5', 'T6', 'Oz']
BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (12, 30),
    "gamma": (30, 45)
}

# === ICA CLEANING ===
def apply_ica(data):
    info = create_info(CHANNELS, SAMPLE_RATE, ch_types='eeg')
    raw = RawArray(data, info, verbose=False)
    raw.filter(1.0, 45.0, fir_design='firwin', verbose=False)
    raw.notch_filter(50.0, verbose=False)
    if raw.n_times < 3 * SAMPLE_RATE:
        return raw.get_data()
    try:
        ica = ICA(n_components=min(15, raw.info['nchan'] - 1), random_state=97, max_iter='auto', verbose=False)
        ica.fit(raw)
        return ica.apply(raw, verbose=False).get_data()
    except:
        return raw.get_data()

# === GFT FEATURES ===
def compute_gft_features(segment):
    dist = np.linalg.norm(segment[:, None, :] - segment[None, :, :], axis=2)
    W = np.exp(-dist * 2 / (2.0 * np.mean(dist) * 2))
    np.fill_diagonal(W, 0)
    D = np.diag(W.sum(axis=1))
    L = D - W
    eigvals, eigvecs = eigh(L)
    gft = eigvecs.T @ segment
    features = []
    for node_signal in gft:
        features.extend([np.mean(node_signal), np.std(node_signal), np.min(node_signal), np.max(node_signal)])
    return np.array(features)

# === BAND POWER FEATURES ===
def compute_band_power_features(segment):
    features = []
    for ch in segment:
        f, Pxx = welch(ch, fs=SAMPLE_RATE, nperseg=SAMPLE_RATE * 2)
        for band_range in BANDS.values():
            idx = np.logical_and(f >= band_range[0], f <= band_range[1])
            power_vals = Pxx[idx]
            features.extend([
                trapezoid(power_vals, f[idx]),
                np.mean(power_vals), np.std(power_vals),
                skew(power_vals), kurtosis(power_vals),
                entropy(np.abs(power_vals) + 1e-6)
            ])
    return np.array(features)

# === HJORTH + HIGUCHI ===
def compute_complexity_features(segment):
    features = []
    for ch in segment:
        mobility, complexity = hjorth_params(ch)
        fd = higuchi_fd(ch)
        features.extend([mobility, complexity, fd])
    return np.array(features)

# === SEGMENTATION ===
def segment_eeg(eeg_data, subject_id, label):
    handcrafted_feats, labels, subject_ids = [], [], []
    for start in range(0, eeg_data.shape[1] - WINDOW_SIZE + 1, STEP_SIZE):
        segment = eeg_data[:, start:start + WINDOW_SIZE]
        gft = compute_gft_features(segment)
        bp = compute_band_power_features(segment)
        comp = compute_complexity_features(segment)
        handcrafted_feats.append(np.concatenate([gft, bp, comp]))
        labels.append(label)
        subject_ids.append(subject_id)
    return handcrafted_feats, labels, subject_ids

# === FILENAME-BASED LABELING ===
def infer_label_from_filename(filename):
    name = filename.upper()
    if "MDD" in name:
        return 1, "MDD_" + ''.join(filter(str.isdigit, name))
    elif "H" in name:
        return 0, "H_" + ''.join(filter(str.isdigit, name))
    else:
        return None, None

# === PROCESS SINGLE EDF ===
def process_edf_file(filepath, label, subject_id):
    try:
        raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
        raw.pick_types(eeg=True)
        raw.resample(SAMPLE_RATE)
        data = raw.get_data()
        if data.shape[0] > len(CHANNELS):
            data = data[:len(CHANNELS)]
        cleaned = apply_ica(data)
        return segment_eeg(cleaned, subject_id, label)
    except Exception as e:
        print(f"⚠️ Skipping {filepath} due to error: {e}")
        return [], [], []

# === PROCESS DATASET ===
def process_edf_dataset(root_dir):
    X_all, y_all, ids_all = [], [], []
    edf_files = [os.path.join(dp, f) for dp, _, fs in os.walk(root_dir)
                 for f in fs if f.lower().endswith('.edf') and 'ec' in f.lower()]
    edf_files.sort()

    for idx, file in enumerate(edf_files):
        label, subj_id = infer_label_from_filename(os.path.basename(file))
        if label is None:
            print(f"⚠️ Skipping {file} (no label)")
            continue

        print(f"[{idx+1}/{len(edf_files)}] Processing {os.path.basename(file)} - Label: {label}")
        X, y, ids = process_edf_file(file, label, subj_id)
        if X:
            X_all.extend(X)
            y_all.extend(y)
            ids_all.extend(ids)

    return np.array(X_all), np.array(y_all), np.array(ids_all)

# === NORMALIZATION ===
def normalize_per_subject(X, subject_ids):
    X_norm = np.zeros_like(X)
    for subj in np.unique(subject_ids):
        idx = subject_ids == subj
        scaler = StandardScaler()
        X_norm[idx] = scaler.fit_transform(X[idx])
    return X_norm

# === MAIN ===
if __name__ == "__main__":
    data_dir = "files"
    os.makedirs("processed_data", exist_ok=True)

    print("\n=== Processing EC EDF Files ===")
    X_combined, y, subj_ids = process_edf_dataset(data_dir)

    X_norm = normalize_per_subject(X_combined, subj_ids)

    np.save("processed_data/X_EC_enhanced.npy", X_norm)
    np.save("processed_data/y_EC.npy", y)
    np.save("processed_data/subject_ids_EC.npy", subj_ids)

    print("✅ Saved: X_EC_enhanced.npy, y_EC.npy, subject_ids_EC.npy")
