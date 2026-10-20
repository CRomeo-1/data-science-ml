import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# classify tumors as malignant or benign from 30 features
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
)

# scale features (important for logistic regression and KNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=0),
}

best_name, best_acc = None, 0
for name, model in models.items():
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{name:<22} accuracy: {acc:.2%}")
    if acc > best_acc:
        best_name, best_acc = name, acc

best = models[best_name]
print(f"\nBest model: {best_name}\n")
print(classification_report(y_test, best.predict(X_test), target_names=data.target_names))

ConfusionMatrixDisplay.from_estimator(best, X_test, y_test, display_labels=data.target_names, cmap="Blues")
plt.title(f"Confusion matrix - {best_name}")
plt.savefig("confusion_matrix.png")
print("saved confusion_matrix.png")
