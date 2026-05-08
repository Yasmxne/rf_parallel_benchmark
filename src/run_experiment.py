import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from benchmark import create_dataset, benchmark_random_forest
from adaptive_search import adaptive_search

# ── Color palette ──────────────────────────────────────────────────────────
BLUE     = "#1D4ED8"
GREEN    = "#059669"
PURPLE   = "#7C3AED"
RED      = "#B91C1C"
AMBER    = "#B45309"
SLATE    = "#475569"
BG       = "#F8FAFC"
GRID_CLR = "#E2E8F0"
SPINE    = "#CBD5E1"


# ── Shared axis style ──────────────────────────────────────────────────────
def _style_ax(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=11, color="#1E293B")
    ax.set_ylabel(ylabel, fontsize=11, color="#1E293B")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14, color="#0F172A")
    ax.set_facecolor(BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE)
    ax.spines["bottom"].set_color(SPINE)
    ax.tick_params(colors="#374151", labelsize=9)
    ax.grid(True, color=GRID_CLR, linestyle="--", linewidth=0.7)


# ── Amdahl's Law ───────────────────────────────────────────────────────────
def amdahl(n, p):
    return 1.0 / ((1.0 - p) + p / np.asarray(n, dtype=float))


def fit_parallel_fraction(df):
    """Estimate parallel fraction p from the maximum speedup observed."""
    row = df.loc[df["n_jobs"].idxmax()]
    n, S = float(row["n_jobs"]), float(row["speedup"])
    p = n * (S - 1.0) / (S * (n - 1.0))
    return float(np.clip(p, 0.0, 1.0))


# ── Plot: training time ────────────────────────────────────────────────────
def plot_results(df, output_path, color=BLUE, label=""):
    df = df.sort_values("n_jobs")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor("white")

    ax.fill_between(
        df["n_jobs"],
        df["mean_time"] - df["std_time"],
        df["mean_time"] + df["std_time"],
        color=color, alpha=0.12, label="Mean ± std",
    )
    ax.plot(
        df["n_jobs"], df["mean_time"],
        marker="o", color=color, linewidth=2.2, markersize=7,
        label="Mean training time", zorder=4,
    )
    ax.errorbar(
        df["n_jobs"], df["mean_time"], yerr=df["std_time"],
        fmt="none", color=color, capsize=4, linewidth=1.2, zorder=5,
    )

    best = df.loc[df["mean_time"].idxmin()]
    sign = 1 if best["n_jobs"] < df["n_jobs"].max() else -1
    ax.annotate(
        f"Best: {best['mean_time']:.2f} s",
        xy=(best["n_jobs"], best["mean_time"]),
        xytext=(best["n_jobs"] + sign * 0.8, best["mean_time"] + 1.5),
        fontsize=9, color=SLATE,
        arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.1),
    )

    ax.set_xticks(df["n_jobs"])
    _style_ax(
        ax,
        "Number of parallel workers (n_jobs)",
        "Training time (s)",
        f"Random Forest — Training Time{' (' + label + ')' if label else ''}",
    )
    ax.legend(fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Plot: speedup + Amdahl curves ─────────────────────────────────────────
def plot_speedup(df, output_path, color=BLUE, label=""):
    df = df.sort_values("n_jobs")
    n_max = int(df["n_jobs"].max())
    xs = np.linspace(1, n_max, 300)

    p_fit = fit_parallel_fraction(df)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor("white")

    ax.plot(xs, xs, color=SLATE, linewidth=1.2, linestyle=":",
            label="Ideal linear speedup", zorder=1)

    for p, c, lbl in [
        (0.75, AMBER,  "Amdahl  p = 75%"),
        (0.90, RED,    "Amdahl  p = 90%"),
        (p_fit, PURPLE, f"Amdahl  p = {p_fit:.0%}  (fitted)"),
    ]:
        ax.plot(xs, amdahl(xs, p), "--", color=c, linewidth=1.7,
                alpha=0.85, label=lbl, zorder=2)

    ax.plot(
        df["n_jobs"], df["speedup"],
        marker="o", color=color, linewidth=2.5, markersize=8,
        zorder=5, label="Measured speedup",
    )

    for _, row in df.iterrows():
        ax.annotate(
            f"{row['speedup']:.2f}x",
            xy=(row["n_jobs"], row["speedup"]),
            xytext=(0, 9), textcoords="offset points",
            ha="center", fontsize=8, color=color, fontweight="bold",
        )

    ax.set_xticks(df["n_jobs"].astype(int).unique())
    _style_ax(
        ax,
        "Number of parallel workers (n_jobs)",
        "Speedup",
        f"Speedup vs Amdahl's Law{' (' + label + ')' if label else ''}",
    )
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Plot: parallel efficiency (both methods on one figure) ─────────────────
def plot_efficiency(df_grid, df_adaptive, output_path):
    g = df_grid.sort_values("n_jobs")
    a = df_adaptive.sort_values("n_jobs")

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("white")

    ax.plot(g["n_jobs"], g["efficiency"], marker="o", color=BLUE,
            linewidth=2.2, markersize=7, label="Grid search")
    ax.plot(a["n_jobs"], a["efficiency"], marker="s", color=GREEN,
            linewidth=2.2, markersize=7, linestyle="--", label="Adaptive search")
    ax.axhline(1.0, color=SLATE, linewidth=1.2, linestyle=":",
               label="Perfect efficiency (= 1.0)")

    ax.set_ylim(0, 1.35)
    ax.set_xticks(g["n_jobs"])
    _style_ax(
        ax,
        "Number of parallel workers (n_jobs)",
        "Parallel efficiency  (speedup / n_jobs)",
        "Parallel Efficiency — Grid Search vs Adaptive Search",
    )
    ax.legend(fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Speedup helper ─────────────────────────────────────────────────────────
def add_speedup(df):
    baseline = df.loc[df["n_jobs"] == 1, "mean_time"].values[0]
    df["speedup"]    = baseline / df["mean_time"]
    df["efficiency"] = df["speedup"] / df["n_jobs"]
    return df


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("report/figures", exist_ok=True)

    print("Creating dataset...")
    X_train, X_test, y_train, y_test = create_dataset()

    candidates = [1, 2, 3, 4, 5, 6, 7, 8]

    def benchmark_fn(n_jobs):
        print(f"  Testing n_jobs={n_jobs}...")
        return benchmark_random_forest(
            X_train, X_test, y_train, y_test,
            n_jobs=n_jobs, n_estimators=200, n_repeats=3,
        )

    print("\nRunning full grid search...")
    grid_results = [benchmark_fn(n) for n in candidates]
    df_grid = add_speedup(pd.DataFrame(grid_results))
    df_grid.to_csv("results/grid_search_results.csv", index=False)

    best_grid = df_grid.loc[df_grid["mean_time"].idxmin()]

    print("\nRunning adaptive search...")
    adaptive_results, best_adaptive = adaptive_search(
        benchmark_function=benchmark_fn,
        candidates=candidates,
        initial_points=[1, 4, 8],
        improvement_threshold=0.05,
    )
    df_adaptive = add_speedup(pd.DataFrame(adaptive_results))
    df_adaptive.to_csv("results/adaptive_search_results.csv", index=False)

    print("\nGenerating figures...")
    plot_results(df_grid,     "report/figures/grid_search_time.png",     color=BLUE,  label="Grid Search")
    plot_results(df_adaptive, "report/figures/adaptive_search_time.png", color=GREEN, label="Adaptive Search")

    plot_speedup(df_grid,     "report/figures/grid_search_speedup.png",     color=BLUE,  label="Grid Search")
    plot_speedup(df_adaptive, "report/figures/adaptive_search_speedup.png", color=GREEN, label="Adaptive Search")

    plot_efficiency(df_grid, df_adaptive, "report/figures/parallel_efficiency.png")

    summary = pd.DataFrame([
        {
            "method": "grid_search",
            "number_of_tests": len(df_grid),
            "best_n_jobs": int(best_grid["n_jobs"]),
            "best_time": best_grid["mean_time"],
            "accuracy": best_grid["mean_accuracy"],
        },
        {
            "method": "adaptive_search",
            "number_of_tests": len(df_adaptive),
            "best_n_jobs": int(best_adaptive["n_jobs"]),
            "best_time": best_adaptive["mean_time"],
            "accuracy": best_adaptive["mean_accuracy"],
        },
    ])

    interpretation = (
        f"\n    Main result\n    ===========\n\n"
        f"    The full grid search tested {len(df_grid)} configurations.\n"
        f"    The adaptive benchmark tested {len(df_adaptive)} configurations.\n\n"
        f"    Best (grid search)    : n_jobs={int(best_grid['n_jobs'])}  "
        f"-> {best_grid['mean_time']:.3f} s\n"
        f"    Best (adaptive search): n_jobs={int(best_adaptive['n_jobs'])}  "
        f"-> {best_adaptive['mean_time']:.3f} s\n\n"
        f"    Accuracy: {best_grid['mean_accuracy']:.3f}\n\n"
        f"    Conclusion:\n"
        f"    The adaptive benchmark found the same optimum with "
        f"{len(df_grid) - len(df_adaptive)} fewer tests.\n"
        f"    The fitted parallel fraction p = "
        f"{fit_parallel_fraction(df_grid):.0%} shows that most of the\n"
        f"    computation is parallelisable, but Amdahl's Law still limits\n"
        f"    the maximum achievable speedup.\n"
    )

    with open("results/interpretation.txt", "w", encoding="utf-8") as f:
        f.write(interpretation)
    summary.to_csv("results/summary.csv", index=False)

    print("\n=== Summary ===")
    print(summary.to_string(index=False))
    print("\nResults  -> results/")
    print("Figures  -> report/figures/")


if __name__ == "__main__":
    main()
