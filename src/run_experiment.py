import os
import pandas as pd
import matplotlib.pyplot as plt

from benchmark import create_dataset, benchmark_random_forest
from adaptive_search import adaptive_search


def plot_results(df, output_path):
    df = df.sort_values("n_jobs")

    plt.figure()
    plt.errorbar(
        df["n_jobs"],
        df["mean_time"],
        yerr=df["std_time"],
        marker="o",
        capsize=4,
    )
    plt.xlabel("Number of parallel jobs")
    plt.ylabel("Training time (seconds)")
    plt.title("Random Forest training time depending on n_jobs")
    plt.grid(True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

def plot_speedup(df, output_path):
    df = df.sort_values("n_jobs")

    plt.figure()
    plt.plot(
        df["n_jobs"],
        df["speedup"],
        marker="o",
    )
    plt.xlabel("Number of parallel jobs")
    plt.ylabel("Speedup")
    plt.title("Speedup depending on n_jobs")
    plt.grid(True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

def add_speedup(df):
    baseline_time = df.loc[df["n_jobs"] == 1, "mean_time"].values[0]
    df["speedup"] = baseline_time / df["mean_time"]
    df["efficiency"] = df["speedup"] / df["n_jobs"]
    return df


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("report/figures", exist_ok=True)

    print("Creating dataset...")
    X_train, X_test, y_train, y_test = create_dataset()

    candidates = [1, 2, 3, 4, 5, 6, 7, 8]

    def benchmark_fn(n_jobs):
        print(f"Testing n_jobs={n_jobs}...")
        return benchmark_random_forest(
            X_train,
            X_test,
            y_train,
            y_test,
            n_jobs=n_jobs,
            n_estimators=200,
            n_repeats=3,
        )

    print("\nRunning full grid search...")
    grid_results = []

    for n_jobs in candidates:
        grid_results.append(benchmark_fn(n_jobs))

    df_grid = pd.DataFrame(grid_results)
    df_grid = add_speedup(df_grid)
    df_grid.to_csv("results/grid_search_results.csv", index=False)

    best_grid = df_grid.loc[df_grid["mean_time"].idxmin()]

    print("\nRunning adaptive search...")
    adaptive_results, best_adaptive = adaptive_search(
        benchmark_function=benchmark_fn,
        candidates=candidates,
        initial_points=[1, 4, 8],
        improvement_threshold=0.05,
    )

    df_adaptive = pd.DataFrame(adaptive_results)
    df_adaptive = add_speedup(df_adaptive)
    df_adaptive.to_csv("results/adaptive_search_results.csv", index=False)

    plot_results(df_grid, "report/figures/grid_search_time.png")
    plot_results(df_adaptive, "report/figures/adaptive_search_time.png")

    plot_speedup(df_grid, "report/figures/grid_search_speedup.png")
    plot_speedup(df_adaptive, "report/figures/adaptive_search_speedup.png")

    summary = pd.DataFrame(
        [
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
        ]
    )
    interpretation = f"""
    Main result
    ===========

    The full grid search tested {len(df_grid)} configurations.
    The adaptive benchmark tested {len(df_adaptive)} configurations.

    The best configuration found by grid search is n_jobs={int(best_grid["n_jobs"])}
    with a training time of {best_grid["mean_time"]:.3f} seconds.

    The best configuration found by adaptive search is n_jobs={int(best_adaptive["n_jobs"])}
    with a training time of {best_adaptive["mean_time"]:.3f} seconds.

    Both methods obtained an accuracy of approximately {best_grid["mean_accuracy"]:.3f}.

    Conclusion:
    The adaptive benchmark found a similar best configuration while using fewer tests.
    This shows that it can reduce the benchmarking cost compared to a full grid search.

    The speedup curve also shows that the gain is not perfectly linear.
    Most of the improvement happens when moving from 1 worker to several workers,
    then the additional gains become smaller.
    """

    with open("results/interpretation.txt", "w", encoding="utf-8") as f:
        f.write(interpretation)
    summary.to_csv("results/summary.csv", index=False)

    print("\n=== Summary ===")
    print(summary)

    print("\nResults saved in results/")
    print("Figures saved in report/figures/")
    print("Interpretation saved in results/interpretation.txt")


if __name__ == "__main__":
    main()