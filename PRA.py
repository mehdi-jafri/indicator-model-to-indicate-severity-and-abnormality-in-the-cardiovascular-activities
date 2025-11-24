import numpy as np
import matplotlib.pyplot as plt


data = np.loadtxt("Performance/PRA.txt")

samples = data[:, 0].astype(int)

precision = data[:, 1:5]

recall = data[:, 5:9]

accuracy = data[:, 9:13]

labels = ["CM-APAI", "Residual U-Net", "TQCPat", "Two-stage framework"]

plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

def plot_bar(metric_data, metric_name, filename):
    x = np.arange(len(samples))
    width = 0.2

    plt.figure(figsize=(12, 6))

    for i in range(4):
        plt.bar(x + (i - 1.5) * width, metric_data[:, i], width, label=labels[i])

    plt.xlabel("Samples", fontweight='bold')
    plt.ylabel(metric_name, fontweight='bold')
    plt.title(f"{metric_name} Comparison", fontweight='bold')

    plt.xticks(x, samples, rotation=45, fontweight='bold')

    legend = plt.legend()
    for t in legend.get_texts():
        t.set_fontweight('bold')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

plot_bar(precision, "Precision", "precision.png")
plot_bar(recall, "Recall", "recall.png")
plot_bar(accuracy, "Accuracy", "accuracy.png")
