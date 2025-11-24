import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.signal import periodogram
from scipy.integrate import simps

# ===============================================================
# 1️⃣  Frequency‑domain PPG feature extraction
# ===============================================================
def extract_frequency_features(ppg, fs):
    """
    Calculate frequency‑domain HRV / spectral features from a 1‑D PPG waveform.
    Parameters
    ----------
    ppg : 1‑D numpy array
        Pre‑processed PPG signal.
    fs : int or float
        Sampling rate (Hz)
    Returns
    -------
    dict : spectral features
    """

    # Remove mean to avoid DC bias
    ppg = np.asarray(ppg).astype(float)
    ppg = ppg - np.mean(ppg)

    # Compute PSD using periodogram
    f, Pxx = periodogram(ppg, fs=fs, window='hann', scaling='density')

    # --- Define HRV frequency bands (Hz) ---
    bands = {
        "VLF": (0.0033, 0.04),
        "LF" : (0.04, 0.15),
        "HF" : (0.15, 0.4)
    }

    # Integrate power in each band
    total_power = simps(Pxx, f)
    vlf_mask = np.logical_and(f >= bands["VLF"][0], f < bands["VLF"][1])
    lf_mask  = np.logical_and(f >= bands["LF"][0],  f < bands["LF"][1])
    hf_mask  = np.logical_and(f >= bands["HF"][0],  f < bands["HF"][1])

    vlf_power = simps(Pxx[vlf_mask], f[vlf_mask])
    lf_power  = simps(Pxx[lf_mask],  f[lf_mask])
    hf_power  = simps(Pxx[hf_mask],  f[hf_mask])

    lf_hf_ratio = lf_power / (hf_power + 1e-6)
    norm_lf = lf_power / (lf_power + hf_power + 1e-6)
    norm_hf = hf_power / (lf_power + hf_power + 1e-6)

    # Spectral peak frequency (dominant heart‑rate component)
    peak_idx = np.argmax(Pxx)
    dominant_freq = f[peak_idx]
    derived_hr_bpm = dominant_freq * 60.0          # Heart rate from spectrum

    feats = {
        "total_power": total_power,
        "VLF_power": vlf_power,
        "LF_power": lf_power,
        "HF_power": hf_power,
        "LF_HF_ratio": lf_hf_ratio,
        "normalized_LF": norm_lf,
        "normalized_HF": norm_hf,
        "dominant_freq_Hz": dominant_freq,
        "derived_HR_BPM": derived_hr_bpm
    }
    return feats, f, Pxx

# ===============================================================
# 2️⃣  Plot spectrum (optional)
# ===============================================================
def plot_spectrum(f, Pxx, fname, out_dir="spectra"):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(7,4))
    plt.semilogy(f, Pxx, color='navy', lw=2.5, alpha=0.7, solid_capstyle='round', solid_joinstyle='round')
    plt.title(f"Power Spectral Density – {fname}", fontsize=14, fontweight='bold')
    plt.xlabel("Frequency (Hz)", fontsize=12, fontweight='bold')
    plt.ylabel("Power Density", fontsize=12, fontweight='bold')
    plt.xticks(fontsize=10, fontweight='bold')
    plt.yticks(fontsize=10, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    out_path = os.path.join(out_dir, f"{os.path.splitext(fname)[0]}_spectrum.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"🖼️  Saved spectrum plot → {out_path}")

# ===============================================================
# 3️⃣  Batch processing
# ===============================================================
def process_all(folder="preprocessed_signals",
                out_csv="ppg_frequency_domain_features.csv",
                make_plots=True):

    results = []
    for fname in os.listdir(folder):
        if not fname.endswith(".npz"): continue
        fp = os.path.join(folder, fname)
        try:
            data = np.load(fp, allow_pickle=True)
            fs = float(data["fs"])
            sig = data["moving_average"].astype(float)
            sig = sig - np.mean(sig)
            feats, f, Pxx = extract_frequency_features(sig, fs)
            feats["file"] = fname
            feats["fs_Hz"] = fs
            feats["duration_s"] = len(sig)/fs
            results.append(feats)
            if make_plots:
                plot_spectrum(f, Pxx, fname)
        except Exception as e:
            print(f"⚠️ {fname}: {e}")

    if results:
        df = pd.DataFrame(results)
        df.to_csv(out_csv, index=False)
        print(df)
        print(f"✅ Saved frequency‑domain features → {out_csv}")
    else:
        print("⚠️ No valid signals processed.")

# ===============================================================
# 4️⃣  Run
# ===============================================================
if __name__ == "__main__":
    folder = "preprocessed_signals"
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")
    process_all(folder)
