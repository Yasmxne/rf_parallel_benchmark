def adaptive_search(
    benchmark_function,
    candidates,
    initial_points=None,
    improvement_threshold=0.05,
):
    """
    Adaptive benchmark strategy.

    Instead of testing all candidates, we first test a few points,
    then evaluate only the neighbours of the best configuration.

    Parameters
    ----------
    benchmark_function : callable
        Function taking n_jobs as input and returning a dict with mean_time.
    candidates : list[int]
        All possible n_jobs values.
    initial_points : list[int]
        First configurations to test.
    improvement_threshold : float
        Minimum relative improvement required to continue.

    Returns
    -------
    results : list[dict]
        Tested configurations and their performances.
    best_result : dict
        Best configuration found.
    """

    candidates = sorted(candidates)

    if initial_points is None:
        initial_points = [candidates[0], candidates[len(candidates) // 2], candidates[-1]]

    tested = set()
    results = []

    for n_jobs in initial_points:
        if n_jobs in candidates and n_jobs not in tested:
            result = benchmark_function(n_jobs)
            results.append(result)
            tested.add(n_jobs)

    best_result = min(results, key=lambda x: x["mean_time"])
    previous_best_time = best_result["mean_time"]

    best_n_jobs = best_result["n_jobs"]
    best_index = candidates.index(best_n_jobs)

    neighbour_indices = [
        best_index - 2,
        best_index - 1,
        best_index + 1,
        best_index + 2,
    ]

    for idx in neighbour_indices:
        if 0 <= idx < len(candidates):
            n_jobs = candidates[idx]

            if n_jobs not in tested:
                result = benchmark_function(n_jobs)
                results.append(result)
                tested.add(n_jobs)

                current_best = min(results, key=lambda x: x["mean_time"])
                improvement = (
                    previous_best_time - current_best["mean_time"]
                ) / previous_best_time

                if improvement > improvement_threshold:
                    best_result = current_best
                    previous_best_time = current_best["mean_time"]

    best_result = min(results, key=lambda x: x["mean_time"])

    return results, best_result