

import os
from glob import glob
from collections import Counter
import numpy as np
import scipy.signal as signal
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import wfdb
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score


SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

INPUT_FOLDER = "downloaded/P100/p10014354/81739927"   
OUTPUT_FOLDER = "output_pipeline_force4"
PLOT_FOLDER = os.path.join(OUTPUT_FOLDER, "plots")
MODEL_FOLDER = os.path.join(OUTPUT_FOLDER, "models")
FEATURE_FOLDER = os.path.join(OUTPUT_FOLDER, "features")
PREDICTION_CSV = os.path.join(OUTPUT_FOLDER, "predictions_val.csv")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(FEATURE_FOLDER, exist_ok=True)


LABEL_NAMES = {0: "Asystole", 1: "Bradycardia", 2: "Tachycardia", 3: "Normal"}
ALL_LABELS = sorted(LABEL_NAMES.keys())


def third_derivative_moving_average(x: np.ndarray, window: int = 5) -> np.ndarray:
    if len(x) < 4:
        return x.copy()
    d1 = np.diff(x, n=1, prepend=x[0])
    d2 = np.diff(d1, n=1, prepend=d1[0])
    d3 = np.diff(d2, n=1, prepend=d2[0])
    kernel = np.ones(window) / window
    smooth = np.convolve(d3, kernel, mode="same")
    out = np.cumsum(smooth)
    out = (out - np.mean(out)) / (np.std(out) + 1e-8)
    out = out * (np.std(x) + 1e-8) + np.mean(x)
    return out

def bandpass_filter(x: np.ndarray, fs: float, lowcut=0.5, highcut=8.0, order=4) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = signal.butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return signal.filtfilt(b, a, x)

def compute_ssqi(signal_segment: np.ndarray) -> float:
    if np.std(signal_segment) < 1e-6:
        return -np.inf
    return stats.skew(signal_segment)

def sliding_window_ssqi_segments(ppg: np.ndarray, fs: int,
                                 win_sec: float = 8.0, step_sec: float = 4.0,
                                 ssqi_threshold: float = 0.0):
    win = int(win_sec * fs)
    step = int(step_sec * fs)
    segments = []
    if win <= 0:
        return segments
    for start in range(0, max(1, len(ppg) - win + 1), step):
        seg = ppg[start:start + win]
        ssqi = compute_ssqi(seg)
        if ssqi >= ssqi_threshold:
            segments.append((start, start + win, seg.copy(), ssqi))
    return segments


def estimate_hr_from_ppg(seg: np.ndarray, fs: int, min_peak_distance_sec=0.3) -> float:
    if len(seg) < int(0.5 * fs):
        return 0.0
    seg_f = bandpass_filter(seg, fs=fs, lowcut=0.5, highcut=6.0, order=3)
    distance = int(min_peak_distance_sec * fs)
    peaks, props = signal.find_peaks(seg_f, distance=distance, prominence=max(1e-6, np.std(seg_f)*0.5))
    num_peaks = len(peaks)
    duration_min = len(seg) / fs / 60.0
    if duration_min <= 0:
        return 0.0
    hr = num_peaks / duration_min
    return float(hr)

def hr_to_label(hr: float):
    if hr == 0:
        return 0
    if hr < 50:
        return 1
    if hr > 130:
        return 2
    if 60 <= hr <= 100:
        return 3
    return None

def frequency_features_welch(seg: np.ndarray, fs: int, nperseg=256):
    freqs, psd = signal.welch(seg, fs=fs, nperseg=min(nperseg, len(seg)))
    total_power = np.trapz(psd, freqs)
    centroid = np.sum(freqs * psd) / (np.sum(psd) + 1e-12)
    peak_freq = freqs[np.argmax(psd)]
    return freqs, psd, np.array([total_power, centroid, peak_freq])

class TimeEncoderSimple(nn.Module):
    def __init__(self, in_channels=1, d_model=128):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, d_model, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return x

class FreqEncoderSimple(nn.Module):
    def __init__(self, in_dim, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class CrossAttentionFusionSimple(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
    def forward(self, t_emb, f_emb):
        x = torch.cat([t_emb, f_emb], dim=1)
        return self.fc(x)

class FusionClassifierSimple(nn.Module):
    def __init__(self, d_model=128, psd_len=129, n_classes=4):
        super().__init__()
        self.time_enc = TimeEncoderSimple(d_model=d_model)
        self.freq_enc = FreqEncoderSimple(in_dim=psd_len, out_dim=d_model)
        self.fusion = CrossAttentionFusionSimple(d_model=d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model//2, n_classes)
        )
    def forward(self, seg_batch, psd_batch):
        t_emb = self.time_enc(seg_batch)
        f_emb = self.freq_enc(psd_batch)
        fused = self.fusion(t_emb, f_emb)
        logits = self.classifier(fused)
        return logits

class PPGSegmentDataset(torch.utils.data.Dataset):
    def __init__(self, segs, psds, labels):
        assert len(segs) == len(psds) == len(labels)
        self.segs = segs
        self.psds = psds
        self.labels = labels
        self.fixed_len = max([len(s) for s in segs]) if segs else 256
    def __len__(self):
        return len(self.segs)
    def __getitem__(self, idx):
        seg = self.segs[idx].astype(np.float32)
        if len(seg) < self.fixed_len:
            pad = np.zeros(self.fixed_len - len(seg), dtype=np.float32)
            seg = np.concatenate([seg, pad])
        else:
            seg = seg[:self.fixed_len]
        psd = self.psds[idx].astype(np.float32)
        label = self.labels[idx]
        return torch.from_numpy(seg), torch.from_numpy(psd), torch.tensor(label, dtype=torch.long)


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for segs, psds, labels in loader:
        segs = segs.to(DEVICE)
        psds = psds.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(segs, psds)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * segs.size(0)
    return total_loss / len(loader.dataset)

def eval_model_logits(model, loader):
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for segs, psds, labels in loader:
            segs = segs.to(DEVICE)
            psds = psds.to(DEVICE)
            logits = model(segs, psds)
            all_logits.append(logits.cpu().numpy())
            all_labels.extend(labels.numpy().tolist())
    if all_logits:
        all_logits = np.vstack(all_logits)
    else:
        all_logits = np.zeros((0,4))
    return all_logits, np.array(all_labels)


def plot_ppg_segments(ppg_raw, ppg_filtered, fs, record_name):
    try:
        t = np.arange(len(ppg_raw)) / fs
        plt.figure(figsize=(10, 3.5))
        plt.plot(t, ppg_raw, label="Raw", alpha=0.6)
        plt.plot(t, ppg_filtered, label="Filtered", linewidth=1.2)
        plt.title(f"PPG Raw vs Filtered — {record_name}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(PLOT_FOLDER, f"{record_name}_ppg_raw_filtered.png")
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print("Error plotting PPG for", record_name, ":", e)

def plot_psd(freqs, psd, record_name, idx):
    try:
        plt.figure(figsize=(8, 3.5))
        plt.plot(freqs, psd, linewidth=1.0)
        plt.title(f"PSD of Segment {idx} — {record_name}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("PSD")
        plt.tight_layout()
        save_path = os.path.join(PLOT_FOLDER, f"{record_name}_psd_seg{idx}.png")
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print("Error plotting PSD for", record_name, "segment", idx, ":", e)

def plot_ssqi_distribution(meta):
    try:
        ssqi_values = [m["ssqi"] for m in meta if "ssqi" in m]
        if not ssqi_values:
            return
        plt.figure(figsize=(6,4))
        plt.hist(ssqi_values, bins=40, alpha=0.7)
        plt.title("SSQI Distribution of Segments")
        plt.xlabel("SSQI")
        plt.ylabel("Count")
        plt.tight_layout()
        path = os.path.join(PLOT_FOLDER, "ssqi_distribution.png")
        plt.savefig(path)
        plt.close()
    except Exception as e:
        print("Error plotting SSQI distribution:", e)

def plot_hr_distribution(meta):
    try:
        hrs = [m["hr"] for m in meta if "hr" in m]
        if not hrs:
            return
        plt.figure(figsize=(6,4))
        plt.hist(hrs, bins=50, alpha=0.7)
        plt.title("Estimated HR (bpm) Distribution")
        plt.xlabel("Heart Rate (bpm)")
        plt.ylabel("Count")
        plt.tight_layout()
        save_path = os.path.join(PLOT_FOLDER, "hr_distribution.png")
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print("Error plotting HR distribution:", e)

def plot_confusion_matrix(y_true, y_pred, labels, outpath):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[LABEL_NAMES[l] for l in labels])
    fig, ax = plt.subplots(figsize=(6,6))
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close(fig)

def process_input_folder(input_folder, ssqi_threshold=0.0, win_sec=8.0, step_sec=4.0, target_bins=129):
    all_segments = []
    all_psds = []
    all_labels = []
    meta = []

    hea_files = sorted(glob(os.path.join(input_folder, "*.hea")))
    if not hea_files:
        print("No .hea files found. Check input_folder:", input_folder)
        return [], [], [], []

    for hea_file in tqdm(hea_files, desc="Records"):
        segment_name = os.path.splitext(os.path.basename(hea_file))[0]
        record_path = os.path.join(input_folder, segment_name)
        plotted_one_segment = False
        try:
            signals, fields = wfdb.rdsamp(record_path)
            fs = int(fields['fs'])
            possible_names = ["PPG", "PLETH", "PULSE", "SPO2", "OXY"]
            ppg_candidates = [
                i for i, s in enumerate(fields['sig_name'])
                if any(name in s.upper() for name in possible_names)
            ]
            if not ppg_candidates:
                continue
            ppg_ch = ppg_candidates[0]
            ppg = np.nan_to_num(signals[:, ppg_ch].astype(np.float32))
            ppg = ppg - np.mean(ppg)
            ppg_filtered = third_derivative_moving_average(ppg, window=7)
            ppg_filtered = bandpass_filter(ppg_filtered, fs=fs, lowcut=0.5, highcut=8.0, order=3)
            
            try:
                plot_ppg_segments(ppg, ppg_filtered, fs, segment_name)
            except Exception:
                pass

            segments = sliding_window_ssqi_segments(ppg_filtered, fs=fs,
                                                   win_sec=win_sec, step_sec=step_sec,
                                                   ssqi_threshold=ssqi_threshold)
            if not segments:
                continue

            for seg_idx, (start, end, seg, ssqi_val) in enumerate(segments):
                hr = estimate_hr_from_ppg(seg, fs=fs)
                label = hr_to_label(hr)
                if label is None:
                    continue
                freqs, psd, _ = frequency_features_welch(seg, fs=fs, nperseg=256)
                psd_interp = np.interp(np.linspace(freqs.min(), freqs.max(), target_bins), freqs, psd)
                all_segments.append(seg.astype(np.float32))
                all_psds.append(psd_interp.astype(np.float32))
                all_labels.append(label)
                meta.append({
                    "record": segment_name,
                    "start": int(start),
                    "end": int(end),
                    "ssqi": float(ssqi_val),
                    "hr": float(hr),
                    "label": int(label)
                })
                
                if not plotted_one_segment:
                    try:
                        plot_psd(freqs, psd, segment_name, seg_idx)
                    except Exception:
                        pass
                    plotted_one_segment = True

        except Exception as e:
            print("Error processing", segment_name, ":", e)
            continue

    
    np.save(os.path.join(FEATURE_FOLDER, "meta.npy"), np.array(meta, dtype=object))
    print(f"Extracted {len(all_segments)} labelled segments.")

    try:
        plot_ssqi_distribution(meta)
    except Exception:
        pass
    try:
        plot_hr_distribution(meta)
    except Exception:
        pass

    return all_segments, all_psds, all_labels, meta

def add_synthetic_placeholders(segs, psds, labels, meta, target_bins=129):
    present = set(labels)
    missing = [c for c in ALL_LABELS if c not in present]
    if not missing:
        return segs, psds, labels, meta
    
    if segs:
        lengths = [len(s) for s in segs]
        syn_len = int(np.median(lengths))
    else:
        syn_len = 256

    for m in missing:
        syn_seg = np.zeros(syn_len, dtype=np.float32)
        syn_psd = np.zeros(target_bins, dtype=np.float32)
        segs.append(syn_seg)
        psds.append(syn_psd)
        labels.append(m)
        meta.append({
            "record": "SYNTHETIC",
            "start": 0,
            "end": syn_len,
            "ssqi": 0.0,
            "hr": 0.0 if m == 0 else (30.0 if m == 1 else (140.0 if m == 2 else 80.0)),
            "label": int(m)
        })
        print(f"Added synthetic sample for missing class {m} ({LABEL_NAMES[m]})")
    return segs, psds, labels, meta


def main():
    ssqi_thresh = 0.0
    win_sec = 8.0
    step_sec = 4.0
    target_bins = 129

    segs, psds, labels, meta = process_input_folder(INPUT_FOLDER, ssqi_threshold=ssqi_thresh,
                                                    win_sec=win_sec, step_sec=step_sec,
                                                    target_bins=target_bins)
    
    segs, psds, labels, meta = add_synthetic_placeholders(segs, psds, labels, meta, target_bins=target_bins)

    if len(segs) == 0:
        print("No segments found - exiting.")
        return

    print("Label distribution (after synthetic additions if any):", Counter(labels))

    
    counts = Counter(labels)
    can_stratify = all(counts[c] >= 2 for c in counts)

    if can_stratify:
        stratify_labels = labels
        print("Using stratified train/val split.")
        train_idx, val_idx = train_test_split(range(len(segs)), test_size=0.2, stratify=stratify_labels, random_state=SEED)
    else:
        print("Not enough samples per class for stratify. Using random split.")
        train_idx, val_idx = train_test_split(range(len(segs)), test_size=0.2, random_state=SEED)

    train_segs = [segs[i] for i in train_idx]
    train_psds = [psds[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]

    val_segs = [segs[i] for i in val_idx]
    val_psds = [psds[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    val_meta = [meta[i] for i in val_idx]

    train_ds = PPGSegmentDataset(train_segs, train_psds, train_labels)
    val_ds = PPGSegmentDataset(val_segs, val_psds, val_labels)

    bs = 8
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)

    
    psd_len = len(val_psds[0])
    model = FusionClassifierSimple(d_model=128, psd_len=psd_len, n_classes=4).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    n_epochs = 20    
    best_val_acc = -1.0
    
    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        logits_train, labels_train = eval_model_logits(model, train_loader)
        preds_train = np.argmax(logits_train, axis=1) if logits_train.size else np.array([])
        train_acc = accuracy_score(labels_train, preds_train) if preds_train.size else 0.0

        logits_val, labels_val = eval_model_logits(model, val_loader)
        preds_val = np.argmax(logits_val, axis=1) if logits_val.size else np.array([])
        val_acc = accuracy_score(labels_val, preds_val) if preds_val.size else 0.0

        print(f"Epoch {epoch}/{n_epochs}  Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODEL_FOLDER, "best_demo_model_force4.pth"))


    logits_val, labels_val = eval_model_logits(model, val_loader)
    
    probs_val = torch.softmax(torch.from_numpy(logits_val), dim=1).numpy()
    
    preds_val = np.argmax(probs_val, axis=1)

    
    labels_to_report = ALL_LABELS
    target_names = [LABEL_NAMES[i] for i in labels_to_report]

    print("\nClassification report (validation) — forced 4 classes:")
    
    print(classification_report(labels_val, preds_val, labels=labels_to_report, target_names=target_names, digits=4))

    
    plot_confusion_matrix(labels_val, preds_val, labels=labels_to_report, outpath=os.path.join(PLOT_FOLDER, "confusion_matrix_force4.png"))
    
    print("Saved confusion matrix to:", os.path.join(PLOT_FOLDER, "confusion_matrix_force4.png"))

    
    rows = []
    header = ["record", "start", "end", "true_label", "true_label_name", "pred_label", "pred_label_name", "hr"] + [f"prob_{LABEL_NAMES[i]}" for i in labels_to_report]
    for i, m in enumerate(val_meta):
        row = {
            "record": m["record"],
            "start": m["start"],
            "end": m["end"],
            "true_label": m["label"],
            "true_label_name": LABEL_NAMES[m["label"]],
            "pred_label": int(preds_val[i]),
            "pred_label_name": LABEL_NAMES[int(preds_val[i])],
            "hr": m["hr"]
        }
        probs = probs_val[i].tolist()
        for j, p in enumerate(probs):
            row[f"prob_{LABEL_NAMES[j]}"] = float(p)
        rows.append(row)

    df = pd.DataFrame(rows, columns=header)
    
    df.to_csv(PREDICTION_CSV, index=False)
    
    print("Saved validation predictions to:", PREDICTION_CSV)

    print("\nSample predictions (first 10):")
    
    print(df.head().to_string(index=False))
    
    torch.save(model.state_dict(), os.path.join(MODEL_FOLDER, "final_demo_model_force4.pth"))

    print("\nPipeline finished. Artifacts saved in:", OUTPUT_FOLDER)

if __name__ == "__main__":
    main()
