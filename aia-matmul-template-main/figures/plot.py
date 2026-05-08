import matplotlib.pyplot as plt

sizes = [64, 128, 256, 512, 1024, 2048]

data = {
    "Naive":      [3.39, 2.02, 2.23, 2.03, 0.87, 0.25],
    "Reordered":  [27.56, 22.51, 16.18, 24.86, 19.60, 12.65],
    "Tiled":      [23.70, 21.90, 17.33, 19.92, 16.74, 8.51],
    "Parallel":   [0.65, 4.35, 22.84, 41.07, 48.50, 24.83],
}

markers = ["o", "s", "^", "D"]
colors  = ["#e15759", "#4e79a7", "#f28e2b", "#59a14f"]

fig, ax = plt.subplots(figsize=(8, 5))

for (label, values), marker, color in zip(data.items(), markers, colors):
    ax.plot(sizes, values, marker=marker, label=label, color=color, linewidth=2, markersize=6)

ax.set_xscale("log", base=2)
ax.set_xticks(sizes)
ax.set_xticklabels(sizes)
ax.set_xlabel("Matrix size N", fontsize=12)
ax.set_ylabel("Performance (GFLOP/s)", fontsize=12)
ax.set_title("Matrix Multiplication Performance vs. Matrix Size", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, which="both", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("performance.png", dpi=150)
plt.show()
