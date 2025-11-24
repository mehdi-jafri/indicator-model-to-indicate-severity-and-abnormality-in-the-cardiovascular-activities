import numpy as np
import matplotlib.pyplot as plt


data = np.loadtxt("Performance/FPR.txt")

samples = data[:, 0].astype(int)               
fpr = data[:, 1:5]                              

labels = ["CM-APAI", "Residual U-Neet", "TQCPat", "Two-stage framework"]

plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

x = np.arange(len(samples))
width = 0.20

plt.figure(figsize=(12, 6))

for i in range(4):
    plt.bar(x + (i - 1.5) * width, fpr[:, i], width, label=labels[i])

plt.xlabel("Samples", fontweight='bold')
plt.ylabel("False Positive Rate", fontweight='bold')
plt.title("False Positive Rate Comparison", fontweight='bold')

plt.xticks(x, samples, rotation=45, fontweight='bold')

legend = plt.legend()
for t in legend.get_texts():
    t.set_fontweight('bold')

plt.tight_layout()
plt.savefig("false_positive_rate.png", dpi=300, bbox_inches='tight')
plt.show()
