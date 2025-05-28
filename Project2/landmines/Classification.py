import pandas as pd
import numpy as np
import sys

sys.stdout = open("output.txt", "w", encoding="utf-8")


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report


from sklearn.exceptions import ConvergenceWarning
import warnings

# Suppress convergence warnings for MLP
warnings.filterwarnings("ignore", category=ConvergenceWarning)


class Classification:
    class Data:
        def __init__(self, path):
            self.df = pd.read_csv(path)
            # Assumes last column is label; adjust if needed
            self.X = self.df.iloc[:, :-1].values
            self.y = self.df.iloc[:, -1].values
            # Simple preprocessing: scale features with StandardScaler
            self.scaler = StandardScaler()
            self.X_scaled = self.scaler.fit_transform(self.X)

        def get_train_test(self, test_size=0.25, random_state=0):
            return train_test_split(
                self.X_scaled,
                self.y,
                test_size=test_size,
                random_state=random_state,
                stratify=self.y,
            )

    def __init__(self, data_path):
        self.data = Classification.Data(data_path)

    def KNN(self, n_neighbors=2):
        X_train, X_test, y_train, y_test = self.data.get_train_test()
        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(f"KNN (k={n_neighbors}) Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        # print(classification_report(y_test, y_pred))
        print(classification_report(y_test, y_pred, zero_division=0))

        return clf

    def NaiveBayes(self):
        X_train, X_test, y_train, y_test = self.data.get_train_test()
        clf = GaussianNB()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(f"Naive Bayes Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        # print(classification_report(y_test, y_pred))
        print(classification_report(y_test, y_pred, zero_division=0))

        return clf

    def DecisionTree(self, max_depth=None):
        X_train, X_test, y_train, y_test = self.data.get_train_test()
        clf = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        # print(classification_report(y_test, y_pred))
        print(classification_report(y_test, y_pred, zero_division=0))

        return clf

    def RandomForest(self, n_estimators=100, max_depth=None, random_state=0):
        X_train, X_test, y_train, y_test = self.data.get_train_test()
        clf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(classification_report(y_test, y_pred, zero_division=0))
        return clf

    # Inside your NeuralNetwork method:
    def NeuralNetwork(self, hidden_layer_sizes=(100,), max_iter=2000):
        X_train, X_test, y_train, y_test = self.data.get_train_test()
        clf = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=0
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(f"Neural Network Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        # print(classification_report(y_test, y_pred))
        print(classification_report(y_test, y_pred, zero_division=0))
        return clf

    def SVM(self, kernel="rbf", C=1.0):
        X_train, X_test, y_train, y_test = self.data.get_train_test()
        clf = SVC(kernel=kernel, C=C, random_state=0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(f"SVM (kernel={kernel}) Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        # print(classification_report(y_test, y_pred))
        print(classification_report(y_test, y_pred, zero_division=0))

        return clf

    def Ensemble(self):
        X_train, X_test, y_train, y_test = self.data.get_train_test()
        estimators = [
            ("knn", KNeighborsClassifier()),
            ("nb", GaussianNB()),
            ("dt", DecisionTreeClassifier(random_state=0)),
        ]
        clf = VotingClassifier(estimators=estimators, voting="hard")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(f"Ensemble (Voting) Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        # print(classification_report(y_test, y_pred))
        print(classification_report(y_test, y_pred, zero_division=0))

        return clf

    def eval_test_train_splits(
        classifier_obj, split_percents=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    ):
        methods = [
            ("KNN", lambda: classifier_obj.KNN(n_neighbors=5)),
            ("NaiveBayes", classifier_obj.NaiveBayes),
            ("DecisionTree", lambda: classifier_obj.DecisionTree(max_depth=5)),
            (
                "RandomForest",
                lambda: classifier_obj.RandomForest(n_estimators=100, max_depth=10),
            ),
            (
                "NeuralNetwork",
                lambda: classifier_obj.NeuralNetwork(
                    hidden_layer_sizes=(30, 10), max_iter=1000
                ),
            ),
            ("SVM", lambda: classifier_obj.SVM(kernel="rbf", C=1.0)),
            ("Ensemble", classifier_obj.Ensemble),
        ]
        for split in split_percents:
            print(
                f"=== Test size {int(split*100)}% / Train size {100-int(split*100)}% ==="
            )
            # (You have to set the split on the Data instance)
            classifier_obj.data.get_train_test = (
                lambda test_size=split, random_state=0: train_test_split(
                    classifier_obj.data.X_scaled,
                    classifier_obj.data.y,
                    test_size=split,
                    random_state=0,
                    stratify=classifier_obj.data.y,
                )
            )

            # if split == 0.1 save the test data to a csv file
            if split == 0.1:
                X_train, X_test, y_train, y_test = classifier_obj.data.get_train_test(
                    test_size=split, random_state=0
                )
                test_df = pd.DataFrame(X_test)
                test_df["label"] = y_test
                test_df.to_csv("test_data.csv", index=False)
                print("Test data saved to 'test_data.csv'")

            for name, method in methods:
                print(f"{name}:")
                method()

    def eval_kfold_cv(classifier_obj, k_list=[2, 3, 4, 5, 6, 7, 8, 9, 10]):
        methods = [
            ("KNN", KNeighborsClassifier(n_neighbors=2)),
            ("NaiveBayes", GaussianNB()),
            ("DecisionTree", DecisionTreeClassifier(max_depth=5, random_state=0)),
            (
                "RandomForest",
                RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0),
            ),
            (
                "NeuralNetwork",
                MLPClassifier(
                    hidden_layer_sizes=(30, 10), max_iter=1000, random_state=0
                ),
            ),
            ("SVM", SVC(kernel="rbf", C=1.0, random_state=0)),
            (
                "Ensemble",
                VotingClassifier(
                    estimators=[
                        ("knn", KNeighborsClassifier()),
                        ("nb", GaussianNB()),
                        ("dt", DecisionTreeClassifier(random_state=0)),
                    ],
                    voting="hard",
                ),
            ),
        ]
        X = classifier_obj.data.X_scaled
        y = classifier_obj.data.y
        for k in k_list:
            print(f"=== K-Fold Cross Validation, K={k} ===")
            cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
            for name, model in methods:
                scores = cross_val_score(model, X, y, cv=cv)
                print(
                    f"{name}: Mean accuracy: {scores.mean():.4f} (Std: {scores.std():.4f})"
                )

    def eval_kfold_cv_reports(classifier_obj, k_list=[2, 3, 4, 5, 6, 7, 8, 9, 10]):
        methods = [
            ("KNN", KNeighborsClassifier(n_neighbors=5)),
            ("NaiveBayes", GaussianNB()),
            ("DecisionTree", DecisionTreeClassifier(max_depth=5, random_state=0)),
            (
                "RandomForest",
                RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0),
            ),
            (
                "NeuralNetwork",
                MLPClassifier(
                    hidden_layer_sizes=(30, 10), max_iter=1000, random_state=0
                ),
            ),
            ("SVM", SVC(kernel="rbf", C=1.0, random_state=0)),
            (
                "Ensemble",
                VotingClassifier(
                    estimators=[
                        ("knn", KNeighborsClassifier()),
                        ("nb", GaussianNB()),
                        ("dt", DecisionTreeClassifier(random_state=0)),
                    ],
                    voting="hard",
                ),
            ),
        ]
        X = classifier_obj.data.X_scaled
        y = classifier_obj.data.y
        for k in k_list:
            print(f"=== K-Fold Classification Report, K={k} ===")
            cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
            for name, model in methods:
                y_pred = cross_val_predict(model, X, y, cv=cv)
                print(f"{name} classification report (K={k}):")
                print(classification_report(y, y_pred, zero_division=0))

    def opsummering(output_file="output.txt"):
        """
        Udtrækker en top-liste over accuracy fra hver metode for både split og K-fold, og printer til terminalen.
        """
        with open(output_file, encoding="utf-8") as f:
            lines = f.readlines()

        accuracies = []
        section = ""
        for line in lines:
            if (
                "Test size" in line
                or "K-Fold Cross Validation" in line
                or "K-Fold Classification Report" in line
            ):
                section = line.strip()
            if "Accuracy:" in line:
                parts = line.split("Accuracy:")
                if len(parts) > 1:
                    acc = parts[1].strip().replace("\\\\n", "").replace("\\n", "")
                    accuracies.append((section, acc))
            if "Mean accuracy:" in line:
                name, rest = line.split(":", 1)
                mean = (
                    rest.strip()
                    .split("(Std")[0]
                    .replace("Mean accuracy", "")
                    .replace(":", "")
                    .strip()
                )
                accuracies.append((section + " | " + name, mean))

        print("=== Opsummeret accuracies ===")
        for sec, acc in accuracies:
            print(f"{sec}: {acc}")

    import matplotlib.pyplot as plt

    def visualisering(output_file="output.txt"):
        """
        Finder alle linjer med 'Mean accuracy' fra K-Fold Cross Validation i output.txt og visualiserer dem.
        """
        import re
        import matplotlib.pyplot as plt

        # Gem data for plotting
        data = {}
        current_K = None
        with open(output_file, encoding="utf-8") as f:
            for line in f:
                kmatch = re.search(r"K-Fold Cross Validation, K=(\\d+)", line)
                if kmatch:
                    current_K = int(kmatch.group(1))
                    continue
                m = re.search(
                    r"^(\\w+): Mean accuracy: ([0-9.]+) \\(Std: ([0-9.]+)\\)", line
                )
                if m:
                    method = m.group(1)
                    mean = float(m.group(2))
                    std = float(m.group(3))
                    if method not in data:
                        data[method] = {"K": [], "mean": [], "std": []}
                    data[method]["K"].append(current_K)
                    data[method]["mean"].append(mean)
                    data[method]["std"].append(std)

        # Plot
        plt.figure(figsize=(12, 6))
        markers = ["o", "v", "s", "D", "^", "<", ">", "p"]
        for idx, (method, vals) in enumerate(data.items()):
            plt.errorbar(
                vals["K"],
                vals["mean"],
                yerr=vals["std"],
                label=method,
                marker=markers[idx % len(markers)],
                capsize=4,
            )
        plt.title(
            "Mean accuracy for each method at different K (K-Fold Cross Validation)"
        )
        plt.xlabel("K (number of folds)")
        plt.ylabel("Mean Accuracy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def visualisering_fra_accuracy(output_file="output.txt"):
        import re
        import matplotlib.pyplot as plt

        # Find accuracy for test split og k-fold (kun classification reports)
        pattern_split = re.compile(r"=== Test size (\\d+)% / Train size \\d+% ===")
        pattern_kfold = re.compile(r"=== K-Fold Classification Report, K=(\\d+) ===")
        pattern_name = re.compile(
            r"^(KNN|NaiveBayes|DecisionTree|RandomForest|NeuralNetwork|SVM|Ensemble)",
            re.I,
        )
        pattern_acc = re.compile(r"Accuracy:\\s*([0-9.]+)")

        data_split = {}  # {alg: ([splits], [accuracies])}
        data_kfold = {}  # {alg: ([K], [accuracies])}

        current_section = None
        split_pct = None
        k = None
        alg = None
        with open(output_file, encoding="utf-8") as f:
            for line in f:
                if m := pattern_split.search(line):
                    split_pct = int(m.group(1))
                    k = None
                    alg = None
                elif m := pattern_kfold.search(line):
                    k = int(m.group(1))
                    split_pct = None
                    alg = None
                elif pattern_name.match(line):
                    alg = pattern_name.match(line).group(1)
                elif m := pattern_acc.search(line):
                    acc = float(m.group(1))
                    if split_pct is not None and alg:
                        data_split.setdefault(alg, ([], []))[0].append(split_pct)
                        data_split[alg][1].append(acc)
                    elif k is not None and alg:
                        data_kfold.setdefault(alg, ([], []))[0].append(k)
                        data_kfold[alg][1].append(acc)

        # Plot for K-Fold
        if data_kfold:
            plt.figure(figsize=(10, 6))
            for alg, (ks, accs) in data_kfold.items():
                plt.plot(ks, accs, marker="o", label=alg)
            plt.title("Accuracy per method over K (K-Fold Classification Report)")
            plt.xlabel("K (number of folds)")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        # Plot for Split
        if data_split:
            plt.figure(figsize=(10, 6))
            for alg, (splits, accs) in data_split.items():
                plt.plot(splits, accs, marker="o", label=alg)
            plt.title("Accuracy per method at different test sizes")
            plt.xlabel("Test size (%)")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    def opsummering_avanceret(output_file="output.txt", out_summary="summary.txt"):
        """
        Udtrækker test-split/K-split, algoritmenavn, accuracy og hele classification_report
        og skriver det tydeligt til filen summary.txt samt printer til terminalen.
        """
        with open(output_file, encoding="utf-8") as f:
            lines = f.readlines()

        summary = []
        sec = None
        alg = None
        buf = []
        for line in lines:
            # Find start på ny split/K-section
            if "Test size" in line or "K-Fold Classification Report" in line:
                sec = line.strip()
                alg = None  # reset
            # Find algoritme
            elif line.startswith("\\n") or line.strip() == "":
                continue  # ignorér blanke linjer
            elif any(
                [
                    line.lstrip().startswith(a)
                    for a in [
                        "KNN",
                        "Naive Bayes",
                        "Decision Tree",
                        "Random Forest",
                        "Neural Network",
                        "SVM",
                        "Ensemble",
                    ]
                ]
            ):
                alg = line.strip().replace(":", "")
                buf = [sec + " / " + alg]  # start ny blok
            # Find accuracy
            elif "Accuracy:" in line:
                acc = line.strip().replace("\\\\n", "")
                buf.append(acc)
            # Udtræk classification_report (afsluttes typisk af en linje uden tal)
            elif (
                (any(c.isdigit() for c in line) and "." in line)
                or "macro avg" in line
                or "weighted avg" in line
                or "accuracy" in line
                or "support" in line
            ):
                buf.append(line.rstrip())
            # Hvis vi har samlet en blok for denne algoritme/sektion
            elif buf and (line.strip() == "" or "---" in line):
                # skriv hele blokken
                summary.append("\n".join(buf) + "\n")
                buf = []
        # Indsæt eventuelt sidste blok
        if buf:
            summary.append("\n".join(buf) + "\n")

        # Gemmer til ny fil og printer til skærm
        with open(out_summary, "w", encoding="utf-8") as outf:
            for part in summary:
                print(part)
                outf.write(part + "\\n")
        print(f"Opsummering gemt i {out_summary}")

    # Kald fra dit main-flow:
    # opsummering_avanveret("output.txt")

    # plot the original data from csv file where the last column is the label and the rest are features. The label is either 0,1,2,3,4 and each datapoint should be plotted in a different color based on the label.
    def plot_data_from_csv(data_path):
        import matplotlib.pyplot as plt

        df = pd.read_csv(data_path)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        # Plot each class in a different color
        unique_labels = np.unique(y)
        colors = plt.cm.get_cmap("viridis", len(unique_labels))

        for label in unique_labels:
            plt.scatter(
                X[y == label, 0],
                X[y == label, 1],
                color=colors(label),
                label=f"Class {label}",
            )

        plt.title("Data Points by Class")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.show()

    # plot the original data from csv file where the last column is the label and the rest are features. The label is either 0,1,2,3,4 and each datapoint should be plotted in a different color based on the label.

    # plot the classified data from neural network (best model) each datapoint should be plotted in a different color based on the predicted label.
    def plot_classified_data_from_nn(data_path, model):
        import matplotlib.pyplot as plt
        from sklearn.neural_network import MLPClassifier

        df = pd.read_csv(data_path)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Predict using the model
        y_pred = model.predict(X_scaled)

        # Plot each predicted class in a different color
        unique_labels = np.unique(y_pred)
        colors = plt.cm.get_cmap("viridis", len(unique_labels))

        for label in unique_labels:
            plt.scatter(
                X_scaled[y_pred == label, 0],
                X_scaled[y_pred == label, 1],
                color=colors(label),
                label=f"Predicted Class {label}",
            )

        plt.title("Classified Data Points by Neural Network")
        plt.xlabel("Feature 1 (scaled)")
        plt.ylabel("Feature 2 (scaled)")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    # Change path as needed; assumes script run from project root
    data_path = "./data/land_mines.csv"
    clf = Classification(data_path)

    # Test/train split evaluering:
    Classification.eval_test_train_splits(
        clf, split_percents=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    )

    # K-Fold Cross-Validation evaluering:
    Classification.eval_kfold_cv(clf, k_list=range(2, 11))
    Classification.eval_kfold_cv_reports(clf, k_list=range(2, 11))

    # print("KNN:")
    # clf.KNN(n_neighbors=5)
    # print("Naive Bayes:")
    # clf.NaiveBayes()
    # print("Decision Tree:")
    # clf.DecisionTree(max_depth=5)
    # print("Random Forest:")
    # clf.RandomForest(n_estimators=100, max_depth=10)
    # print("Neural Network:")
    # clf.NeuralNetwork(hidden_layer_sizes=(30, 10), max_iter=1000)
    # print("SVM:")
    # clf.SVM(kernel="rbf", C=1.0)
    # print("Ensemble (Voting):")
    # clf.Ensemble()

    print("KNN:")
    clf.KNN(n_neighbors=2)
    print("Naive Bayes:")
    clf.NaiveBayes()
    print("Decision Tree:")
    clf.DecisionTree(max_depth=20)
    print("Random Forest:")
    clf.RandomForest(n_estimators=200, max_depth=20)
    print("Neural Network:")
    clf.NeuralNetwork(hidden_layer_sizes=(100, 50), max_iter=20000)
    print("SVM:")
    clf.SVM(kernel="rbf", C=10)
    print("Ensemble (Voting):")
    clf.Ensemble()

    # print output created
    print("Output written to output.txt")

    # save the output of the Neural Network classifier for later use in a csv file

    # Kald opsummering:
    # Classification.opsummering("output.txt")
    # Classification.opsummering_avanceret("output.txt")

    # Kald visualisering:
    # Classification.visualisering("output.txt")

    # Classification.visualisering_fra_accuracy("output.txt")

    # Classification.plot_data_from_csv(data_path)

    # Classification.plot_classified_data_from_nn(data_path)

    # genrate a confusion matrix for the best model
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    y_true = clf.data.y
    y_pred = clf.NeuralNetwork(hidden_layer_sizes=(100, 50), max_iter=20000).predict(
        clf.data.X_scaled
    )
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=np.unique(y_true),
        yticklabels=np.unique(y_true),
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()
    """
    import pandas as pd
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.neural_network import MLPClassifier

    # Indlæs testdata
    test_df = pd.read_csv("test_data.csv")
    X_test = test_df.drop(columns="label").values
    y_test = test_df["label"].values

    # Indlæs/fit scaler præcis som på træningsdata! Allerede gjort i din pipeline hvis du bruger clf.data.scaler

    # Gen-træn modellen på træningsdata fra eksperimentet
    # (Alternativt: Hvis du har gemt modellen, brug den samme, ellers re-trænes den)
    X_train, _, y_train, _ = clf.data.get_train_test(test_size=0.1, random_state=0)
    nn = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=20000, random_state=0)
    nn.fit(X_train, y_train)

    # Forudsig på testdata
    y_pred = nn.predict(X_test)

    # Lav confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=np.unique(y_test),
        yticklabels=np.unique(y_test),
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (10% test data, Neural Network)")
    plt.show()
