"""
Task 3: Iris Flower Classification
-----------------------------------
Goal: Train a machine learning model to classify Iris flowers into
setosa, versicolor, or virginica based on sepal and petal measurements.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

RANDOM_STATE = 42

# ------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

print("Dataset shape:", df.shape)
print("\nClass distribution:\n", df["species"].value_counts())
print("\nSummary statistics:\n", df.describe())

# ------------------------------------------------------------------
# 2. Exploratory Data Analysis
# ------------------------------------------------------------------
sns.pairplot(df, hue="species", corner=True)
plt.savefig("iris_pairplot.png", dpi=120, bbox_inches="tight")
plt.close()

plt.figure(figsize=(6, 5))
sns.heatmap(df.drop(columns="species").corr(), annot=True, cmap="viridis")
plt.title("Feature Correlation")
plt.savefig("iris_correlation.png", dpi=120, bbox_inches="tight")
plt.close()

# ------------------------------------------------------------------
# 3. Train / test split
# ------------------------------------------------------------------
X = df.drop(columns="species")
y = iris.target  # numeric labels for modeling

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------------
# 4. Train multiple models and compare
# ------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Support Vector Machine": SVC(kernel="linear"),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    cv_scores = cross_val_score(model, scaler.fit_transform(X), y, cv=5)
    results[name] = {"test_accuracy": acc, "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std()}
    print(f"\n{name}")
    print(f"  Test accuracy: {acc:.4f}")
    print(f"  5-fold CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ------------------------------------------------------------------
# 5. Pick best model, show detailed report
# ------------------------------------------------------------------
best_name = max(results, key=lambda k: results[k]["cv_mean"])
best_model = models[best_name]
print(f"\nBest model: {best_name}")

preds = best_model.predict(X_test_scaled)
print("\nClassification report:\n", classification_report(y_test, preds, target_names=iris.target_names))

cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix - {best_name}")
plt.savefig("iris_confusion_matrix.png", dpi=120, bbox_inches="tight")
plt.close()

# Save summary results to CSV
results_df = pd.DataFrame(results).T
results_df.to_csv("model_comparison.csv")
print("\nSaved: iris_pairplot.png, iris_correlation.png, iris_confusion_matrix.png, model_comparison.csv")