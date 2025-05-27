import pandas as pd
import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings(
    "ignore", message=".*Stochastic Optimizer: Maximum iterations.*"
)

# Læs data
data_path = "./data/land_mines.csv"
df = pd.read_csv(data_path)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

split_percents = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def hypertune_on_train(X_train, y_train, cv):
    # 1. KNN
    knn_params = {"n_neighbors": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}
    knn = KNeighborsClassifier()
    gs_knn = GridSearchCV(knn, knn_params, cv=cv, n_jobs=-1)
    gs_knn.fit(X_train, y_train)
    print("  KNN:", gs_knn.best_params_, "| Score:", round(gs_knn.best_score_, 3))

    # 2. DecisionTree
    dt_params = {
        "max_depth": [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, None],
        "min_samples_split": [2, 4, 6, 8, 10],
        "min_samples_leaf": [1, 2, 4, 6, 8],
        "max_features": [None, "sqrt", "log2"],
        "criterion": ["gini", "entropy"],  # eller "gini", "entropy", "log_loss"
        "splitter": ["best", "random"],
    }
    dt = DecisionTreeClassifier(random_state=1)
    gs_dt = GridSearchCV(dt, dt_params, cv=cv, n_jobs=-1)
    gs_dt.fit(X_train, y_train)
    print(
        "  DecisionTree:", gs_dt.best_params_, "| Score:", round(gs_dt.best_score_, 3)
    )

    # 3. RandomForest
    rf_params = {
        "n_estimators": [50, 100, 150, 200, 300, 500],
        "max_depth": [5, 10, 15, 20, 25, 30, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": [None, "sqrt", "log2"],  # Fjern "auto"
    }
    rf = RandomForestClassifier(random_state=1)
    gs_rf = GridSearchCV(rf, rf_params, cv=cv, n_jobs=-1)
    gs_rf.fit(X_train, y_train)
    print(
        "  RandomForest:", gs_rf.best_params_, "| Score:", round(gs_rf.best_score_, 3)
    )

    # 4. NeuralNetwork
    mlp_params = {
        "hidden_layer_sizes": [(30, 10), (50, 20), (100,), (100, 50), (50,)],
        "max_iter": [2000, 3000, 5000, 10000, 20000],  # Udvid
        # "learning_rate_init": [0.001, 0.01],         # Mulig udvidelse
        # "activation": ["relu", "tanh"],             # Mulig udvidelse
        # "early_stopping": [True],  # Mulig udvidelse
    }
    mlp = MLPClassifier(random_state=1)
    gs_mlp = GridSearchCV(mlp, mlp_params, cv=cv, n_jobs=-1)
    gs_mlp.fit(X_train, y_train)
    print("  MLP:", gs_mlp.best_params_, "| Score:", round(gs_mlp.best_score_, 3))

    # 5. SVM
    svm_params = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
    svm = SVC(random_state=1)
    gs_svm = GridSearchCV(svm, svm_params, cv=cv, n_jobs=-1)
    gs_svm.fit(X_train, y_train)
    print("  SVM:", gs_svm.best_params_, "| Score:", round(gs_svm.best_score_, 3))


# --------- Manuelle splits (CV=5 for hypertuning på træningsdata) ----------
print("------ Manuelle Train-Test split ------")
for split in split_percents:
    print(f"=== Test size {int(split*100)}% / Train size {100-int(split*100)}% ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=split, random_state=42, stratify=y
    )
    hypertune_on_train(X_train, y_train, cv=5)  # grid search CV=5

# --------- K-Fold tuning ----------
print("\\n------ K-Fold (kun hypertuning, ikke test) ------")
for k in range(2, 11):
    print(f"=== K-Fold Cross-Validation, K={k} ===")
    hypertune_on_train(X_scaled, y, cv=k)

print("\\n------ Færdig ------")
