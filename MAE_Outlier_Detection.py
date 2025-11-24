import numpy as np
import matplotlib.pyplot as plt


data = np.loadtxt("Performance/MAE.txt")

samples = data[:, 0]
cm_apai = data[:, 1]
unet = data[:, 2]
tqcp = data[:, 3]
two_stage = data[:, 4]


x = np.arange(len(samples))
width = 0.18

plt.figure(figsize=(12, 6))

plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['legend.fontsize'] = 'medium'

plt.bar(x - 1.5*width, cm_apai, width, label='CM-APAI')
plt.bar(x - 0.5*width, unet, width, label='Residual U-Net')
plt.bar(x + 0.5*width, tqcp, width, label='TQCPat')
plt.bar(x + 1.5*width, two_stage, width, label='Two-stage framework')

plt.xlabel("Samples", fontweight='bold')
plt.ylabel("MAE of Outlier Detection", fontweight='bold')
plt.title("Outlier Detection MAE Comparison", fontweight='bold')

plt.xticks(x, samples.astype(int), rotation=45, fontweight='bold')

legend = plt.legend()
for text in legend.get_texts():
    text.set_fontweight('bold')
plt.tight_layout()

plt.savefig("outlier_detection_mae.png", dpi=300, bbox_inches='tight')

plt.show()
