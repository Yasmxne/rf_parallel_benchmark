# Parallélisation des Random Forests : benchmark adaptatif

Ce projet étudie la parallélisation de l'entraînement d'une forêt aléatoire
(`RandomForestClassifier`) et compare deux stratégies pour choisir le nombre de
workers `n_jobs` :

- une **grid search complète**, qui teste toutes les valeurs candidates ;
- une **recherche adaptative**, qui teste quelques points puis raffine autour du
  meilleur résultat observé.

L'objectif est de trouver une bonne configuration de parallélisation sans payer
le coût d'une exploration exhaustive, dans l'esprit du cours de programmation
parallèle.

## Idée générale

Une Random Forest contient plusieurs arbres de décision entraînés sur des
échantillons bootstrap différents. Ces arbres sont indépendants pendant
l'entraînement, ce qui rend l'algorithme naturellement parallélisable.

Cependant, utiliser plus de workers n'est pas toujours plus rapide : au-delà
d'un certain point, l'ordonnancement des tâches, les accès mémoire, les effets de
cache et l'agrégation finale peuvent coûter plus cher que le gain apporté par un
worker supplémentaire.

Le projet mesure donc :

- le temps d'entraînement moyen ;
- l'écart-type du temps ;
- l'accuracy sur le jeu de test ;
- le speedup : `T(n_jobs=1) / T(n_jobs=k)` ;
- l'efficacité parallèle : `speedup / n_jobs`.

## Résultats principaux

Les expériences utilisent un dataset synthétique de 20 000 observations et 50
features, généré avec `make_classification`. Chaque configuration est mesurée
sur une forêt de 200 arbres.

| Méthode | Configurations testées | Meilleur `n_jobs` | Temps moyen | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| Grid search | 8 | 7 | 6.70 s | 94.8% |
| Recherche adaptative | 5 | 6 | 9.15 s | 94.8% |

La grid search trouve le meilleur temps avec `n_jobs=7`. La recherche adaptative
teste 37,5% de configurations en moins et retrouve un optimum proche
(`n_jobs=6`). Les temps peuvent varier d'une exécution à l'autre, ce qui explique
que l'optimum mesuré ne soit pas toujours strictement identique entre les deux
stratégies.

Le meilleur speedup observé est d'environ `4.21x` avec 7 workers, contre un
speedup idéal de `7x`. Cela illustre la limite imposée par l'overhead parallèle
et par la loi d'Amdahl.

## Structure du dépôt

```text
rf_parallel_benchmark/
+-- README.md
+-- requirements.txt
+-- src/
|   +-- benchmark.py          # Dataset et benchmark Random Forest
|   +-- adaptive_search.py    # Recherche adaptative sur n_jobs
|   +-- run_experiment.py     # Lance les benchmarks et génère les figures
|   +-- use_case.py           # Figures explicatives arbre / forêt
+-- results/
|   +-- grid_search_results.csv
|   +-- adaptive_search_results.csv
|   +-- summary.csv
|   +-- interpretation.txt
+-- report/
    +-- RF_Parallel_Benchmark_Report ALUCH Yasmine OUALY Ossama.pdf
    +-- figures/
        +-- grid_search_time.png
        +-- adaptive_search_time.png
        +-- grid_search_speedup.png
        +-- adaptive_search_speedup.png
        +-- parallel_efficiency.png
        +-- single_tree.png
        +-- feature_importances.png
        +-- confusion_matrix.png
        +-- accuracy_comparison.png
```

## Installation

Créer un environnement virtuel puis installer les dépendances :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Les dépendances principales sont :

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`

## Lancer les expériences

Pour exécuter le benchmark principal :

```powershell
python src/run_experiment.py
```

Ce script :

1. génère le dataset ;
2. lance la grid search sur `n_jobs = 1, ..., 8` ;
3. lance la recherche adaptative ;
4. calcule les métriques de speedup et d'efficacité ;
5. sauvegarde les CSV et les figures.

Pour générer les figures explicatives sur les arbres et la Random Forest :

```powershell
python src/use_case.py
```

## Fichiers générés

Les résultats numériques sont sauvegardés dans :

```text
results/grid_search_results.csv
results/adaptive_search_results.csv
results/summary.csv
results/interpretation.txt
```

Les figures sont sauvegardées dans :

```text
report/figures/
```

Le rapport final se trouve dans :

```text
report/RF_Parallel_Benchmark_Report ALUCH Yasmine OUALY Ossama.pdf
```

## Remarques de reproductibilité

Les temps de calcul dépendent fortement de la machine, de la charge CPU au moment
de l'exécution et du backend parallèle utilisé par scikit-learn/joblib. Les
valeurs exactes peuvent donc changer légèrement d'un ordinateur à l'autre.

Le paramètre `n_jobs` modifie la façon dont l'entraînement est exécuté en
parallèle, mais il ne change pas l'algorithme statistique. L'accuracy reste donc
stable lorsque seule la configuration de parallélisation varie.
