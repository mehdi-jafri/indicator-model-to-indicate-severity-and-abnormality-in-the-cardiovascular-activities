import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
from scipy.stats import skew, kurtosis


# =====================================================
# 1️⃣  Band-pass Filter
# =====================================================
def bandpass_filter(sig, fs, low=0.5, high=8.0, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, sig)


# =====================================================
# 2️⃣  Feature Extraction
# =====================================================
def extract_ppg_features(ppg, fs):
    # Ensure float and 1D
    ppg = np.asarray(ppg).astype(float)
    if ppg.ndim != 1:
        raise ValueError(f"PPG must be 1D, got shape {ppg.shape}")

    if len(ppg) < fs:
        raise ValueError("Signal too short (<1 second)")

    # Band-pass filter
    ppg = bandpass_filter(ppg, fs)

    # Systolic peaks
    systolic_peaks, _ = find_peaks(ppg, distance=0.3*fs, prominence=0.01)

    # Diastolic minima (peaks on inverted signal)
    diastolic_peaks, _ = find_peaks(-ppg, distance=0.3*fs, prominence=0.01)

    # If too few peaks -> no HR features
    if len(systolic_peaks) < 2:
        return {}, systolic_peaks, diastolic_peaks, []

    # PPI intervals
    ppi = np.diff(systolic_peaks)/fs
    hr = 60. / np.mean(ppi)
    sdnn = np.std(ppi)
    rmssd = np.sqrt(np.mean(np.square(np.diff(ppi))))

    # Pulse amplitude
    amp = []
    for pk in systolic_peaks:
        prev_d = diastolic_peaks[diastolic_peaks < pk]
        if prev_d.size:
            amp.append(ppg[pk] - ppg[prev_d[-1]])
    mean_amp = np.mean(amp) if amp else np.nan

    # Dicrotic notch
    notch_vals, notch_idx = [], []
    for pk in systolic_peaks[:-1]:
        next_d = diastolic_peaks[diastolic_peaks > pk]
        if next_d.size:
            seg = ppg[pk : next_d[0]]
            inv, _ = find_peaks(-seg)
            if inv.size:
                n_idx = pk + inv[0]
                notch_vals.append(ppg[n_idx])
                notch_idx.append(n_idx)
    mean_notch = np.mean(notch_vals) if notch_vals else np.nan

    # Feature dictionary
    ftrs = {
        "heart_rate_bpm": hr,
        "mean_PPI_s": np.mean(ppi),
        "SDNN_s": sdnn,
        "RMSSD_s": rmssd,
        "mean_pulse_amplitude": mean_amp,
        "mean_systolic_peak": np.mean(ppg[systolic_peaks]),
        "mean_diastolic_valley": np.mean(ppg[diastolic_peaks]),
        "mean_dicrotic_notch": mean_notch,
        "PTT_s": np.nan,
        "skewness": skew(ppg),
        "kurtosis": kurtosis(ppg),
        "rms": np.sqrt(np.mean(ppg**2)),
        "abs_energy": np.sum(ppg**2),
        "num_beats": len(systolic_peaks)
    }

    return ftrs, systolic_peaks, diastolic_peaks, notch_idx


# =====================================================
# 3️⃣  Plot Function
# =====================================================
def plot_ppg(ppg, fs, systolic, diastolic, notches, fname, out_dir="plots"):
    os.makedirs(out_dir, exist_ok=True)
    t = np.arange(len(ppg)) / fs

    plt.figure(figsize=(10, 4))
    plt.plot(t, ppg, label="PPG signal", lw=2.5, alpha=0.7)

    if len(systolic):
        plt.plot(t[systolic], ppg[systolic], 'ro', label='Systolic peaks')
    if len(diastolic):
        plt.plot(t[diastolic], ppg[diastolic], 'go', label='Diastolic minima')
    if len(notches):
        plt.plot(t[notches], ppg[notches], 'kx', label='Dicrotic notch')

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"PPG waveform – {fname}")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"{os.path.splitext(fname)[0]}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"🖼️  Saved plot → {out_path}")


# =====================================================
# 4️⃣  Batch Processing (with failed-file reporting)
# =====================================================
def process_all(folder="preprocessed_signals",
                out_csv="ppg_time_domain_features.csv",
                plot_each=True):

    results = []
    failed_files = []   # Store failures and error messages

    for fname in os.listdir(folder):
        if not fname.endswith(".npz"):
            continue

        fp = os.path.join(folder, fname)

        try:
            data = np.load(fp, allow_pickle=True)

            # --- Safe load ---
            sig = np.asarray(data["moving_average"]).astype(float).squeeze()
            if sig.ndim != 1:
                raise ValueError(f"moving_average has shape {sig.shape}, expected 1D")

            fs = float(data["fs"])
            sig -= np.mean(sig)

            # Extract features
            feats, spk, dpk, notch = extract_ppg_features(sig, fs)

            feats.update({
                "file": fname,
                "fs_Hz": fs,
                "duration_s": len(sig)/fs
            })

            results.append(feats)

            if plot_each:
                plot_ppg(sig, fs, spk, dpk, notch, fname)

        except Exception as e:
            print(f"⚠️ {fname}: {e}")
            failed_files.append((fname, str(e)))

    # Save results
    if results:
        df = pd.DataFrame(results)
        df.to_csv(out_csv, index=False)
        print("\nSaved feature table:", out_csv)
        print(df)
    else:
        print("⚠️ No signals processed.")

    # Report failed files
    if failed_files:
        print("\n❌ Failed files:")
        for name, err in failed_files:
            print(f"  - {name}: {err}")

        print(f"\n❗ Total failed files: {len(failed_files)}")
    else:
        print("\n🎉 All files processed successfully!")


# =====================================================
# 5️⃣  Run
# =====================================================
if __name__ == "__main__":
    process_all("preprocessed_signals")
