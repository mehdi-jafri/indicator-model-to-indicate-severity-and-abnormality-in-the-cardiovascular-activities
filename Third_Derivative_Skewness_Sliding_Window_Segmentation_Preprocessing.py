'''
import os
import numpy as np
import wfdb
from glob import glob
from scipy.stats import skew
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# Step 1: Moving Average Filter (Eq. 1)
# -------------------------------------------------------------------
def moving_average_filter(signal, N=5):
    """Smooth signal using simple moving average filter."""
    return np.convolve(signal, np.ones(N) / N, mode='same')

# -------------------------------------------------------------------
# Step 2: First, Second, Third Derivatives (Eq. 2–4)
# -------------------------------------------------------------------
def compute_derivatives(q):
    """Compute first, second, and third numerical derivatives."""
    FD = np.gradient(q)                     # d/dt of q(t)
    SD = np.gradient(FD)                    # d/dt of FD
    TD = np.gradient(SD)                    # d/dt of SD
    return FD, SD, TD

# -------------------------------------------------------------------
# Step 3: Skewness Signal Quality Index (Eq. 5)
# -------------------------------------------------------------------
def skewness_sqi(segment):
    """Compute Skewness-based Signal Quality Index within sliding window."""
    mu = np.mean(segment)
    sigma = np.std(segment)
    if sigma == 0:
        return 0
    S_SQI = np.mean((segment - mu) ** 3) / (sigma ** 3)
    return S_SQI

# -------------------------------------------------------------------
# Step 4: Segmentation and Quality Classification
# -------------------------------------------------------------------
def sliding_window_ssqi(signal, fs, window_sec=2.0, overlap=0.5):
    """Sliding window segmentation with SSQI computation."""
    win_len = int(window_sec * fs)
    step = int(win_len * (1 - overlap))
    seg_ssqi = []
    high_quality_segments = []

    for start in range(0, len(signal) - win_len, step):
        end = start + win_len
        segment = signal[start:end]
        ssqi = skewness_sqi(segment)
        seg_ssqi.append(ssqi)
        # Example threshold for classification (adjust as needed)
        if abs(ssqi) < 0.5:
            high_quality_segments.append(segment)

    return np.array(seg_ssqi), high_quality_segments

# -------------------------------------------------------------------
# Step 5: Apply pipeline to PPG records
# -------------------------------------------------------------------
input_folder = "downloaded/P100/p10014354/81739927"
output_folder = "preprocessed_signals"
plot_folder = "preprocessed_plots"

os.makedirs(output_folder, exist_ok=True)
os.makedirs(plot_folder, exist_ok=True)

hea_files = sorted(glob(os.path.join(input_folder, "*.hea")))

for hea_file in hea_files:
    segment_name = os.path.splitext(os.path.basename(hea_file))[0]
    record_path = os.path.join(input_folder, segment_name)
    try:
        # Read record
        signals, fields = wfdb.rdsamp(record_path)
        fs = fields['fs']

        # Identify PPG-like channel
        possible_names = ["PPG", "PLETH", "PULSE", "SPO2", "OXY"]
        ppg_candidates = [
            i for i, s in enumerate(fields['sig_name'])
            if any(name in s.upper() for name in possible_names)
        ]
        if not ppg_candidates:
            print(f"❌ No PPG-like channel found in {segment_name}. Skipping.")
            continue

        ppg_ch = ppg_candidates[0]
        ppg = np.nan_to_num(signals[:, ppg_ch])
        t = np.arange(len(ppg)) / fs

        # Step 1: Moving average filter
        q = moving_average_filter(ppg, N=5)

        # Step 2: Derivatives
        FD, SD, TD = compute_derivatives(q)

        # Step 3: Segmentation – SSQI computation
        ssqi_vals, high_quality_segs = sliding_window_ssqi(
            TD, fs, window_sec=2.0, overlap=0.5
        )

        # Save filtered results
        np.savez(
            os.path.join(output_folder, f"{segment_name}_filtered.npz"),
            fs=fs,
            moving_average=q,
            third_derivative=TD,
            ssqi_values=ssqi_vals,
            high_quality_segments=high_quality_segs,
        )

        # Visualization
        plt.figure(figsize=(10, 6))
        plt.subplot(3, 1, 1)
        plt.plot(t, ppg, label="Raw PPG", color='gray')
        plt.title(f"{segment_name} – Raw PPG Signal")
        plt.subplot(3, 1, 2)
        plt.plot(t, q, label="Filtered (Moving Average)", color='blue')
        plt.title("After Moving Average (Equation 1)")
        plt.subplot(3, 1, 3)
        plt.plot(t, TD, label="Third Derivative", color='red')
        plt.title("Third Derivative (Equation 4)")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_folder, f"{segment_name}_preprocessing.png"))
        plt.close()

        print(
            f"✅ {segment_name}: SSQI computed for {len(ssqi_vals)} segments "
            f"| High-quality segments: {len(high_quality_segs)}"
        )

    except Exception as e:
        print(f"⚠️ Error processing {segment_name}: {e}")
'''
import os
import numpy as np
import wfdb
from glob import glob
from scipy.stats import skew
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# -------------------------------------------------------------------
# Step 1: Moving Average Filter (Eq. 1)
# -------------------------------------------------------------------
def moving_average_filter(signal, N=5):
    """Smooth signal using simple moving average filter."""
    return np.convolve(signal, np.ones(N) / N, mode='same')

# -------------------------------------------------------------------
# Step 2: First, Second, Third Derivatives (Eq. 2–4)
# -------------------------------------------------------------------
def compute_derivatives(q):
    """Compute first, second, and third numerical derivatives."""
    FD = np.gradient(q)                     # d/dt of q(t)
    SD = np.gradient(FD)                    # d/dt of FD
    TD = np.gradient(SD)                    # d/dt of SD
    return FD, SD, TD

# -------------------------------------------------------------------
# Step 3: Skewness Signal Quality Index (Eq. 5)
# -------------------------------------------------------------------
def skewness_sqi(segment):
    """Compute Skewness-based Signal Quality Index within a segment."""
    mu = np.mean(segment)
    sigma = np.std(segment)
    if sigma == 0:
        return 0
    # The formula for skewness is the mean of the cubed standardized values
    S_SQI = np.mean(((segment - mu) / sigma) ** 3)
    return S_SQI

# -------------------------------------------------------------------
# Step 4: Segmentation and Quality Classification (MODIFIED)
# -------------------------------------------------------------------
def sliding_window_ssqi(signal, fs, window_sec=2.0, overlap=0.5):
    """
    Sliding window segmentation with SSQI computation.
    
    Returns:
        ssqi_times (np.array): Time points for the center of each window.
        seg_ssqi (np.array): SSQI value for each window.
        high_quality_segments (list): List of signal segments classified as high quality.
    """
    win_len = int(window_sec * fs)
    step = int(win_len * (1 - overlap))
    
    seg_ssqi = []
    time_points = []
    high_quality_segments = []

    for start in range(0, len(signal) - win_len, step):
        end = start + win_len
        segment = signal[start:end]
        
        # Calculate SSQI for the segment
        ssqi = skewness_sqi(segment)
        seg_ssqi.append(ssqi)
        
        # Store the time point at the center of the window
        center_time = (start + win_len / 2) / fs
        time_points.append(center_time)

        # Example threshold for classification (adjust as needed)
        if abs(ssqi) < 0.5:
            high_quality_segments.append(segment)

    return np.array(time_points), np.array(seg_ssqi), high_quality_segments

# -------------------------------------------------------------------
# Step 5: Apply pipeline to PPG records
# -------------------------------------------------------------------
# --- Use a sample path, please update this to your actual path ---
input_folder = "downloaded/P100/p10014354/81739927" 
output_folder = "preprocessed_signals"
plot_folder = "preprocessed_plots"

# Check if the input folder exists to prevent errors
if not os.path.exists(input_folder):
    print(f"❌ Input folder not found: {input_folder}")
    print("Please update the 'input_folder' variable to the correct path.")
else:
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(plot_folder, exist_ok=True)

    hea_files = sorted(glob(os.path.join(input_folder, "*.hea")))

    for hea_file in hea_files:
        segment_name = os.path.splitext(os.path.basename(hea_file))[0]
        record_path = os.path.join(input_folder, segment_name)
        try:
            # Read record
            signals, fields = wfdb.rdsamp(record_path)
            fs = fields['fs']

            # Identify PPG-like channel
            possible_names = ["PPG", "PLETH", "PULSE", "SPO2", "OXY"]
            ppg_candidates = [
                i for i, s in enumerate(fields['sig_name'])
                if any(name in s.upper() for name in possible_names)
            ]
            if not ppg_candidates:
                print(f"❌ No PPG-like channel found in {segment_name}. Skipping.")
                continue

            ppg_ch = ppg_candidates[0]
            ppg = np.nan_to_num(signals[:, ppg_ch])
            t = np.arange(len(ppg)) / fs

            # Step 1: Moving average filter
            q = moving_average_filter(ppg, N=int(fs*0.04)) # N=5 is too small for higher fs

            # Step 2: Derivatives
            FD, SD, TD = compute_derivatives(q)

            # Step 3 & 4: Segmentation & SSQI computation
            ssqi_times, ssqi_vals, high_quality_segs = sliding_window_ssqi(
                TD, fs, window_sec=2.0, overlap=0.5
            )

            # Save filtered results (MODIFIED)
            np.savez(
                os.path.join(output_folder, f"{segment_name}_filtered.npz"),
                fs=fs,
                moving_average=q,
                third_derivative=TD,
                ssqi_times=ssqi_times, # <-- Storing the time points
                ssqi_values=ssqi_vals,
                high_quality_segments=np.array(high_quality_segs, dtype=object), # Ensure saving works
            )

            # Visualization (MODIFIED)
            fig, axs = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
            
            # Plot 1: Raw PPG
            axs[0].plot(t, ppg, label="Raw PPG", color='gray', linewidth=2.5, alpha=0.7, solid_capstyle='round', solid_joinstyle='round')
            axs[0].set_title(f"Preprocessing and Quality Assessment for {segment_name}", fontsize=14, fontweight='bold')
            axs[0].set_ylabel("Amplitude", fontsize=12, fontweight='bold')
            axs[0].tick_params(axis='both', labelsize=12, width=2)
            axs[0].legend(prop={'weight': 'bold'})
            
            for label in (axs[0].get_xticklabels() + axs[0].get_yticklabels()):
                label.set_fontweight('bold')
           
            # Plot 2: Filtered Signal
            axs[1].plot(t, q, label="Filtered (Moving Average)", color='blue', linewidth=2.5, alpha=0.7, solid_capstyle='round', solid_joinstyle='round')
            axs[1].set_ylabel("Amplitude", fontsize=12, fontweight='bold')
            axs[1].tick_params(axis='both', labelsize=12, width=2)
            axs[1].legend(prop={'weight': 'bold'})


            for label in (axs[1].get_xticklabels() + axs[1].get_yticklabels()):
                label.set_fontweight('bold')


            
            # Plot 3: Third Derivative
            axs[2].plot(t, TD, label="Third Derivative", color='red', linewidth=2.5, alpha=0.7, solid_capstyle='round', solid_joinstyle='round')
            axs[2].set_ylabel("d³/dt³", fontsize=12, fontweight='bold')
            axs[2].tick_params(axis='both', labelsize=12, width=2)
            axs[2].legend(prop={'weight': 'bold'})

            for label in (axs[2].get_xticklabels() + axs[2].get_yticklabels()):
                label.set_fontweight('bold')


            axs[3].plot(ssqi_times, ssqi_vals, 'o-', label="SSQI", markersize=8, color='purple', linewidth=2.5, alpha=0.7, solid_capstyle='round', solid_joinstyle='round')
            
            # Plot 4: SSQI Values
            
            axs[3].axhline(y=0.5, color='green', linestyle='--', label='Quality Threshold', linewidth=1.5)
            axs[3].axhline(y=-0.5, color='green', linestyle='--', linewidth=1.5)
            axs[3].set_title("Skewness SQI (SSQI) per 2-second Segment", fontsize=12, fontweight='bold')
            axs[3].set_xlabel("Time (s)", fontsize=12, fontweight='bold')
            axs[3].set_ylabel("SSQI Value", fontsize=12, fontweight='bold')
            axs[3].tick_params(axis='both', labelsize=12, width=2)
            axs[3].legend(prop={'weight': 'bold'})

            for label in (axs[3].get_xticklabels() + axs[3].get_yticklabels()):
                label.set_fontweight('bold')
            plt.savefig(os.path.join(plot_folder, f"{segment_name}_preprocessing.png"), dpi=300)
            plt.close()

            print(
                f"✅ {segment_name}: SSQI computed for {len(ssqi_vals)} segments "
                f"| High-quality segments: {len(high_quality_segs)}"
            )

        except Exception as e:
            print(f"⚠️ Error processing {segment_name}: {e}")

