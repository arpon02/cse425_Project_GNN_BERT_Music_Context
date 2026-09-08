"""
Generate publication-quality result comparison charts from results/metrics.json.
Outputs:
1. plots/model_comparison_benchmark.png
2. plots/f1_auc_tradeoff.png
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 11

os.makedirs("plots", exist_ok=True)
os.makedirs("results/plots", exist_ok=True)

with open("results/metrics.json", "r") as f:
    metrics = json.load(f)

models = list(metrics.keys())
macro_f1 = [metrics[m]["Macro-F1"] for m in models]
auc_pr = [metrics[m]["AUC-PR"] for m in models]

# Clean model names for display
clean_labels = [
    "Random Baseline",
    "CNN Mel-Spec (B2)",
    "Task 1: BERT-only",
    "Task 2: GNN-only",
    "Task 3: GNN-BERT",
    "Task 4: Contrastive"
]

# -------------------------------------------------------------
# Figure 1: Comprehensive 3-Panel Benchmark Figure
# -------------------------------------------------------------
fig = plt.figure(figsize=(15, 6), dpi=300)
gs = fig.add_gridspec(1, 3, width_ratios=[2.2, 1.1, 1.0])

# Subplot 1: Macro-F1 & AUC-PR Bar Chart
ax1 = fig.add_subplot(gs[0, 0])
x = np.arange(len(clean_labels))
width = 0.36

rects1 = ax1.bar(x - width/2, macro_f1, width, label="Macro-F1", color="#3b82f6", edgecolor="#1d4ed8", alpha=0.9)
rects2 = ax1.bar(x + width/2, auc_pr, width, label="AUC-PR", color="#10b981", edgecolor="#047857", alpha=0.9)

ax1.set_ylabel("Score (0.0 – 1.0)", fontsize=12, fontweight="bold")
ax1.set_title("(A) Context Understanding Performance (Table 3)", fontsize=13, fontweight="bold", pad=12)
ax1.set_xticks(x)
ax1.set_xticklabels(clean_labels, rotation=25, ha="right", fontsize=10)
ax1.set_ylim(0, 0.75)
ax1.legend(loc="upper left", frameon=True, fontsize=10)
ax1.grid(axis="y", linestyle="--", alpha=0.7)

# Value annotations
for rect in rects1:
    h = rect.get_height()
    ax1.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                 xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
for rect in rects2:
    h = rect.get_height()
    ax1.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                 xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

# Highlight Task 3 winner
ax1.annotate("Best Overall\n(Fusion)", xy=(4, 0.61), xytext=(3.4, 0.68),
             arrowprops=dict(facecolor="#dc2626", shrink=0.08, width=1.5, headwidth=6),
             fontweight="bold", color="#dc2626", fontsize=9.5)

# Subplot 2: Emotion Regression MAE (Lower is better)
ax2 = fig.add_subplot(gs[0, 1])
emotion_models = ["CNN (B2)", "GNN (T2)", "GNN-BERT (T3)"]
emotion_mae = [
    float(metrics["CNN mel-spec"]["MAE (emotion)"]),
    float(metrics["Task 2: GNN-only"]["MAE (emotion)"]),
    float(metrics["Task 3: GNN-BERT"]["MAE (emotion)"])
]
colors_mae = ["#f87171", "#fbbf24", "#10b981"]

bars_mae = ax2.bar(emotion_models, emotion_mae, color=colors_mae, edgecolor="#374151", width=0.55)
ax2.set_ylabel("MAE (Valence & Arousal)", fontsize=11, fontweight="bold")
ax2.set_title("(B) Emotion Error (Lower is Better)", fontsize=12, fontweight="bold", pad=12)
ax2.set_ylim(0, max(emotion_mae) * 1.25)
ax2.grid(axis="y", linestyle="--", alpha=0.7)

for b in bars_mae:
    h = b.get_height()
    label_str = f"{h:.2f}" if h >= 0.05 else f"{h:.3f}"
    ax2.annotate(label_str, xy=(b.get_x() + b.get_width() / 2, h),
                 xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

# Subplot 3: Retrieval Recall@5 (Task 4)
ax3 = fig.add_subplot(gs[0, 2])
retrieval_models = ["Random", "Task 4 (Dual-Enc)"]
retrieval_r5 = [
    float(metrics["Random tags"]["R@5 (retrieval)"]),
    float(metrics["Task 4: Contrastive"]["R@5 (retrieval)"])
]
colors_r5 = ["#9ca3af", "#8b5cf6"]

bars_r5 = ax3.bar(retrieval_models, retrieval_r5, color=colors_r5, edgecolor="#4c1d95", width=0.5)
ax3.set_ylabel("Recall@5 (0.0 – 1.0)", fontsize=11, fontweight="bold")
ax3.set_title("(C) MusicCaps Retrieval R@5", fontsize=12, fontweight="bold", pad=12)
ax3.set_ylim(0, 0.6)
ax3.grid(axis="y", linestyle="--", alpha=0.7)

for b in bars_r5:
    h = b.get_height()
    ax3.annotate(f"{h:.2f}", xy=(b.get_x() + b.get_width() / 2, h),
                 xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.tight_layout()
fig.savefig("results/metrics_barchart.png", bbox_inches="tight")
fig.savefig("plots/metrics_barchart.png", bbox_inches="tight")
fig.savefig("plots/model_comparison_benchmark.png", bbox_inches="tight")
fig.savefig("results/plots/model_comparison_benchmark.png", bbox_inches="tight")
plt.close(fig)

print("Saved results/metrics_barchart.png and plots/metrics_barchart.png successfully!")
