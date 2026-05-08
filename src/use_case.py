import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, confusion_matrix

from benchmark import create_dataset

# ── Color palette (shared with run_experiment.py) ──────────────────────────
BLUE     = "#1D4ED8"
GREEN    = "#059669"
ORANGE   = "#EA580C"
SLATE    = "#475569"
BG       = "#F8FAFC"
GRID_CLR = "#E2E8F0"
SPINE    = "#CBD5E1"

FEATURE_NAMES = [f"feature_{i}" for i in range(50)]
CLASS_NAMES   = ["Class 0", "Class 1"]


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


# ── Train models ───────────────────────────────────────────────────────────
def train_models(X_train, X_test, y_train, y_test, n_estimators=200, random_state=42):
    single_tree = DecisionTreeClassifier(max_depth=5, random_state=random_state)
    single_tree.fit(X_train, y_train)
    tree_acc = accuracy_score(y_test, single_tree.predict(X_test))

    forest = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=5,
        random_state=random_state,
        n_jobs=-1,
    )
    forest.fit(X_train, y_train)
    forest_acc = accuracy_score(y_test, forest.predict(X_test))

    return single_tree, forest, tree_acc, forest_acc


# ── Figure 1: decision tree visualization ─────────────────────────────────
def plot_single_tree(tree, output_path):
    fig, ax = plt.subplots(figsize=(22, 9))
    fig.patch.set_facecolor("white")

    plot_tree(
        tree,
        max_depth=3,
        feature_names=FEATURE_NAMES,
        class_names=CLASS_NAMES,
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax,
        impurity=True,
        proportion=False,
    )
    ax.set_title(
        "Decision Tree structure (depth <= 3)  —  one tree extracted from the Random Forest",
        fontsize=13, fontweight="bold", pad=16, color="#0F172A",
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Figure 2: feature importances (horizontal bars) ───────────────────────
def plot_feature_importances(forest, top_n=15, output_path=None):
    importances = forest.feature_importances_
    std = np.std([t.feature_importances_ for t in forest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1][:top_n]
    # Reverse for horizontal chart (highest on top)
    indices = indices[::-1]

    vals   = importances[indices]
    errors = std[indices]
    labels = [FEATURE_NAMES[i] for i in indices]

    # Gradient color: low importance -> amber, high -> blue
    norm_vals = (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)
    cmap = LinearSegmentedColormap.from_list("imp", ["#FCD34D", "#2563EB"])
    colors = [cmap(v) for v in norm_vals]

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("white")

    ax.barh(range(top_n), vals, xerr=errors, color=colors,
            capsize=3, height=0.65, error_kw={"elinewidth": 1, "ecolor": SLATE})

    # Value labels at end of each bar
    for i, (v, e) in enumerate(zip(vals, errors)):
        ax.text(v + e + 0.001, i, f"{v:.3f}", va="center", fontsize=8, color=SLATE)

    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, vals.max() + errors.max() + 0.03)
    _style_ax(
        ax,
        "Mean decrease in impurity (Gini)",
        "",
        f"Top {top_n} Feature Importances — Random Forest ({len(forest.estimators_)} trees)",
    )
    ax.grid(True, axis="x", color=GRID_CLR, linestyle="--", linewidth=0.7)
    ax.grid(False, axis="y")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Figure 3: confusion matrix with % annotations ─────────────────────────
def plot_confusion_matrix(forest, X_test, y_test, output_path):
    y_pred = forest.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    fig.patch.set_facecolor("white")

    cmap = LinearSegmentedColormap.from_list("cm_blue", ["#EFF6FF", "#1D4ED8"])
    im = ax.imshow(cm_pct, cmap=cmap, vmin=0, vmax=1)

    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct   = cm_pct[i, j]
            text_color = "white" if pct > 0.55 else "#1E293B"
            ax.text(j, i, f"{count}\n({pct:.1%})",
                    ha="center", va="center",
                    fontsize=12, fontweight="bold", color=text_color)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASS_NAMES, fontsize=10)
    ax.set_yticklabels(CLASS_NAMES, fontsize=10)
    ax.set_xlabel("Predicted label", fontsize=11, color="#1E293B")
    ax.set_ylabel("True label", fontsize=11, color="#1E293B")
    ax.set_title("Confusion Matrix — Random Forest",
                 fontsize=13, fontweight="bold", pad=14, color="#0F172A")
    ax.tick_params(colors="#374151")
    for spine in ax.spines.values():
        spine.set_color(SPINE)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Figure 4: single tree vs Random Forest accuracy ───────────────────────
def plot_accuracy_comparison(tree_acc, forest_acc, output_path):
    labels = [
        "Single Decision Tree\n(depth = 5)",
        "Random Forest\n(200 trees, depth = 5)",
    ]
    accs   = [tree_acc, forest_acc]
    colors = [ORANGE, BLUE]
    gain   = (forest_acc - tree_acc) * 100

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    fig.patch.set_facecolor("white")

    bars = ax.bar(labels, accs, color=colors, width=0.45, alpha=0.92,
                  edgecolor="white", linewidth=1.5)

    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() - 0.013,
            f"{acc:.3f}",
            ha="center", va="top",
            color="white", fontweight="bold", fontsize=13,
        )

    # Gain annotation
    x0 = bars[0].get_x() + bars[0].get_width()
    x1 = bars[1].get_x()
    y  = max(accs) + 0.008
    ax.annotate(
        "", xy=(x1, y), xytext=(x0, y),
        arrowprops=dict(arrowstyle="<->", color=SLATE, lw=1.3),
    )
    ax.text((x0 + x1) / 2, y + 0.004, f"+{gain:.1f} pp",
            ha="center", fontsize=10, color=SLATE, fontweight="bold")

    ax.set_ylim(min(accs) - 0.05, max(accs) + 0.04)
    _style_ax(ax, "", "Accuracy (test set)",
              "Single Decision Tree vs. Random Forest")
    ax.grid(True, axis="y", color=GRID_CLR, linestyle="--", linewidth=0.7)
    ax.grid(False, axis="x")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def print_tree_text(tree):
    print("\n--- Decision tree structure (depth <= 3, text format) ---")
    print(export_text(tree, feature_names=FEATURE_NAMES, max_depth=3, show_weights=True))


def main():
    os.makedirs("report/figures", exist_ok=True)

    print("Creating dataset (same as benchmark)...")
    X_train, X_test, y_train, y_test = create_dataset()
    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")

    print("\nTraining models...")
    single_tree, forest, tree_acc, forest_acc = train_models(
        X_train, X_test, y_train, y_test
    )
    print(f"  Single tree accuracy : {tree_acc:.4f}")
    print(f"  Random Forest accuracy: {forest_acc:.4f}")

    print_tree_text(single_tree)

    print("\nGenerating figures...")
    plot_single_tree(single_tree,    "report/figures/single_tree.png")
    plot_feature_importances(forest, top_n=15,
                             output_path="report/figures/feature_importances.png")
    plot_confusion_matrix(forest, X_test, y_test,
                          "report/figures/confusion_matrix.png")
    plot_accuracy_comparison(tree_acc, forest_acc,
                             "report/figures/accuracy_comparison.png")

    print("\n=== Use case summary ===")
    print(f"  Dataset           : make_classification (20 000 samples, 50 features)")
    print(f"  Single tree acc   : {tree_acc:.4f}")
    print(f"  Random Forest acc : {forest_acc:.4f}")
    print(f"  Gain              : +{(forest_acc - tree_acc)*100:.2f} percentage points")
    print("\nAll figures saved in report/figures/")


if __name__ == "__main__":
    main()
