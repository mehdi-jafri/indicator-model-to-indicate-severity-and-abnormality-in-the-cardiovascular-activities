import wfdb
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os

os.makedirs("Input_Plots", exist_ok=True)



root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    initialdir=os.getcwd(),  
    title="Select WFDB record file (.dat, .hea, .atr)",
    filetypes=[("WFDB files",  "*.hea "), ("All files", "*.*")]
)

if not file_path:
    raise ValueError("No file selected.")


rec_path = os.path.splitext(file_path)[0]

signals, fields = wfdb.rdsamp(rec_path)
fs = fields['fs']

print(f"Sampling rate: {fs} Hz")
print(f"Total samples: {len(signals)}")
print(f"Duration: {len(signals) / fs:.2f} sec")
print("Available channels:", fields['sig_name'])

possible_names = ["PPG", "PLETH", "PULSE", "SPO2", "OXY"]
ppg_candidates = [i for i, sig_name in enumerate(fields['sig_name'])
                  if any(name in sig_name.upper() for name in possible_names)]

if not ppg_candidates:
    raise ValueError(f"No PPG-like channel found. Available channels: {fields['sig_name']}")

print("Possible PPG channels:", [fields['sig_name'][i] for i in ppg_candidates])

ppg_channel_idx = ppg_candidates[0]
ppg_signal = np.nan_to_num(signals[:, ppg_channel_idx])

time_axis = np.arange(len(ppg_signal)) / fs

plt.figure(figsize=(12, 4))
plt.plot(time_axis[:int(60*fs)], ppg_signal[:int(60*fs)], color='red')
plt.title(f"PPG Waveform ({fields['sig_name'][ppg_channel_idx]}) - First 60 sec", fontsize=12,fontweight='bold')
plt.xlabel("Time (s)", fontsize=10,fontweight='bold')
plt.xticks(fontsize=10,fontweight='bold')
plt.yticks(fontsize=10,fontweight='bold')
plt.ylabel("Amplitude", fontsize=10,fontweight='bold')
plt.grid(True)
plt.show()


plt.figure(figsize=(12, 4))
plt.plot(time_axis, ppg_signal, color='red')
plt.title(f"PPG Waveform ({fields['sig_name'][ppg_channel_idx]}) - Full sec", fontsize=12,fontweight='bold')
plt.xlabel("Time (s)", fontsize=10,fontweight='bold')
plt.ylabel("Amplitude", fontsize=10,fontweight='bold')
plt.grid(True)
plt.xticks(fontsize=10,fontweight='bold')
plt.yticks(fontsize=10,fontweight='bold')
plt.show()


fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharey=True)

for i in range(4):
    start_t = i * 15
    end_t = (i + 1) * 15
    start_idx = int(start_t * fs)
    end_idx = int(end_t * fs)

    axes[i].plot(time_axis[start_idx:end_idx], ppg_signal[start_idx:end_idx], color='red')
    axes[i].set_xlim(start_t, end_t)
    axes[i].set_title(f"PPG Waveform ({start_t}-{end_t} sec)", fontsize=12,fontweight='bold')
    axes[i].set_xlabel("Time (s)", fontsize=10,fontweight='bold')
    axes[i].set_ylabel("Amplitude", fontsize=10,fontweight='bold')
    
    axes[i].grid(True)


plt.tight_layout()
plt.savefig(f"Input_Plots/PPG_{os.path.basename(rec_path)}.png", dpi=300)
plt.show()