import matplotlib

matplotlib.use("Agg")  # WSL has no display, so save to a file

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from data.wm811k import load_processed

N_PER_CLASS = 5
OUTPUT = "results/wm811k_examples.png"

d = load_processed()
classes = d["classes"]
# outside wafer, good die, defective die
cmap = ListedColormap(["#1b2733", "#7fb7c9", "#d9534f"])

rng = np.random.default_rng(0)
fig, axes = plt.subplots(
    len(classes), N_PER_CLASS, figsize=(2 * N_PER_CLASS, 2 * len(classes))
)

for c, name in enumerate(classes):
    idx = np.where(d["y"] == c)[0]
    picks = rng.choice(idx, N_PER_CLASS, replace=False)
    for j, i in enumerate(picks):
        ax = axes[c, j]
        ax.imshow(d["maps"][i], cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
        ax.axis("off")
        if j == 0:
            ax.set_title(name, loc="left", fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT, dpi=150)
print(f"saved {OUTPUT}")
