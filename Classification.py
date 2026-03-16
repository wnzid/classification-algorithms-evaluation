import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

#loading dataset_039
df = pd.read_csv("dataset_039.csv")

#feature targer split
X = df.drop(columns=["target"])
y = df["target"]

#test train split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#standardizing
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#selecting models with parameters
models = {
    "Decision Tree": DecisionTreeClassifier(
        criterion="entropy",
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        criterion="entropy",
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    ),
    "SVM": SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        random_state=42
    ),
    "KNN": KNeighborsClassifier(
        n_neighbors=7,
        weights="distance",
        p=2
    )
}

#training and results
for name, model in models.items():
    if name in ["SVM", "KNN"]:
        model.fit(X_train_scaled, y_train)
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

    print(f"\n{name}")
    print("Train Accuracy:", accuracy_score(y_train, y_train_pred))
    print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
    print("Precision:", precision_score(y_test, y_test_pred, average="weighted"))
    print("Recall:", recall_score(y_test, y_test_pred, average="weighted"))
    print("F1-score:", f1_score(y_test, y_test_pred, average="weighted"))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))