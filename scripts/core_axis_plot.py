"""Row plot: each row is a group of items placed along the 0–100 axis, labels packed into lanes."""
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["figure.dpi"] = 110
import matplotlib.pyplot as plt

BLUE, ORANGE = "#5b57e0", "#e0442a"

def plot_rows(rows, P, bold, out, title, lanes=7):
    fig_h = sum((1.0 + 0.32 * min(lanes, max(3, len(items) // 6))) * 0.62 for items in rows.values())
    fig, axes = plt.subplots(len(rows), 1, figsize=(26, fig_h), sharex=True,
                             gridspec_kw={"height_ratios": [min(lanes, max(3, len(i) // 6)) + 2 for i in rows.values()]})
    for ax, (name, items) in zip(axes, rows.items()):
        L = min(lanes, max(3, len(items) // 6))
        items = sorted(items, key=lambda t: P(t[0]))
        lane_end = [-1e9] * L
        ax.axvline(50, color="#999", lw=0.8, ls="--")
        for w, side in items:
            x = P(w); width = 0.55 * len(w) + 1.2
            lane = min(range(L), key=lambda i: (lane_end[i] > x, lane_end[i]))
            lane_end[lane] = x + width
            right = (x >= 50) == (side == "bouba")
            col = BLUE if side == "bouba" else ORANGE
            ax.plot([x, x], [0, lane + 0.6], color="#ccc", lw=0.5, zorder=1)
            ax.scatter(x, 0, s=14, color=col, zorder=3, marker="o" if right else "X")
            ax.text(x, lane + 0.6, w, fontsize=8.5, color=col, fontweight="bold" if w in bold else "normal",
                    style="normal" if right else "italic", va="bottom", ha="left", zorder=4)
        ax.set_ylim(-0.6, L + 1.2); ax.set_yticks([])
        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=11)
        for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    axes[-1].set_xlim(-3, 112)
    axes[-1].set_xlabel("← kiki 0            position on the axis (divide = 50)            bouba 100 →", fontsize=12)
    fig.suptitle(title + "\nblue = labelled bouba, red = labelled kiki; ✕ and italics = landed on the wrong side", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out); print("saved", out)
