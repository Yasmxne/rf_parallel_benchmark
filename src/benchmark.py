import time
import numpy as np

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def create_dataset(
    n_samples=20_000,
    n_features=50,
    n_informative=30,
    random_state=42,
):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=10,
        n_classes=2,
        random_state=random_state,
    )

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )


def benchmark_random_forest(
    X_train,
    X_test,
    y_train,
    y_test,
    n_jobs,
    n_estimators=100,
    n_repeats=2,
    random_state=42,
):
    times = []
    accuracies = []

    for i in range(n_repeats):
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            random_state=random_state + i,
        )

        start = time.perf_counter()
        model.fit(X_train, y_train)
        end = time.perf_counter()

        y_pred = model.predict(X_test)

        times.append(end - start)
        accuracies.append(accuracy_score(y_test, y_pred))

    return {
        "n_jobs": n_jobs,
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "mean_accuracy": np.mean(accuracies),
        "std_accuracy": np.std(accuracies),
    }