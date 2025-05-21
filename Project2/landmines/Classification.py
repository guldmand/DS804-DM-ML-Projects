import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report


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

    def KNN(self, n_neighbors=5):
        X_train, X_test, y_train, y_test = self.data.get_train_test()
        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(
            f"KNN (k={n_neighbors}) Accuracy: {accuracy_score(y_test, y_pred):.4f}\\n"
        )
        # print(classification_report(y_test, y_pred))
        print(classification_report(y_test, y_pred, zero_division=0))

        return clf

    def NaiveBayes(self):
        X_train, X_test, y_train, y_test = self.data.get_train_test()
        clf = GaussianNB()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(f"Naive Bayes Accuracy: {accuracy_score(y_test, y_pred):.4f}\\n")
        # print(classification_report(y_test, y_pred))
        print(classification_report(y_test, y_pred, zero_division=0))

        return clf

    def DecisionTree(self, max_depth=None):
        X_train, X_test, y_train, y_test = self.data.get_train_test()
        clf = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred):.4f}\\n")
        # print(classification_report(y_test, y_pred))
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
        print(f"Neural Network Accuracy: {accuracy_score(y_test, y_pred):.4f}\\n")
        # print(classification_report(y_test, y_pred))
        print(classification_report(y_test, y_pred, zero_division=0))
        return clf

    def SVM(self, kernel="rbf", C=1.0):
        X_train, X_test, y_train, y_test = self.data.get_train_test()
        clf = SVC(kernel=kernel, C=C, random_state=0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(
            f"SVM (kernel={kernel}) Accuracy: {accuracy_score(y_test, y_pred):.4f}\\n"
        )
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
        print(f"Ensemble (Voting) Accuracy: {accuracy_score(y_test, y_pred):.4f}\\n")
        # print(classification_report(y_test, y_pred))
        print(classification_report(y_test, y_pred, zero_division=0))

        return clf


if __name__ == "__main__":
    # Change path as needed; assumes script run from project root
    data_path = "./data/land_mines.csv"
    clf = Classification(data_path)

    print("KNN:\\n")
    clf.KNN(n_neighbors=5)
    print("\\nNaive Bayes:\\n")
    clf.NaiveBayes()
    print("\\nDecision Tree:\\n")
    clf.DecisionTree(max_depth=5)
    print("\\nNeural Network:\\n")
    clf.NeuralNetwork(hidden_layer_sizes=(30, 10), max_iter=1000)
    print("\\nSVM:\\n")
    clf.SVM(kernel="rbf", C=1.0)
    print("\\nEnsemble (Voting):\\n")
    clf.Ensemble()
