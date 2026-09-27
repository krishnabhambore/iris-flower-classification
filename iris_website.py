"""
Task 3: Iris Flower Classification — Web Report Generator
------------------------------------------------------------
Trains models on the Iris dataset, builds an HTML report with
embedded charts, and opens it directly in Chrome.
"""

import base64
import io
import webbrowser
import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # render without a GUI window
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


def fig_to_base64(fig):
    """Convert a matplotlib figure to a base64 string for embedding in HTML."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

# ------------------------------------------------------------------
# 2. EDA charts (kept in memory as base64, nothing saved to disk)
# ------------------------------------------------------------------
pairplot = sns.pairplot(df, hue="species", corner=True)
pairplot_b64 = fig_to_base64(pairplot.fig)

fig_corr, ax_corr = plt.subplots(figsize=(5, 4))
sns.heatmap(df.drop(columns="species").corr(), annot=True, cmap="viridis", ax=ax_corr)
ax_corr.set_title("Feature Correlation")
corr_b64 = fig_to_base64(fig_corr)

# ------------------------------------------------------------------
# 3. Train / test split
# ------------------------------------------------------------------
X = df.drop(columns="species")
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------------
# 4. Train & compare models
# ------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Support Vector Machine": SVC(kernel="linear"),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
}

rows = []
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    cv_scores = cross_val_score(model, scaler.fit_transform(X), y, cv=5)
    rows.append((name, acc, cv_scores.mean(), cv_scores.std()))

results_df = pd.DataFrame(rows, columns=["Model", "Test Accuracy", "CV Mean", "CV Std"])
best_row = results_df.loc[results_df["CV Mean"].idxmax()]
best_name = best_row["Model"]
best_model = models[best_name]

preds = best_model.predict(X_test_scaled)
report_text = classification_report(y_test, preds, target_names=iris.target_names)
cm = confusion_matrix(y_test, preds)

fig_cm, ax_cm = plt.subplots(figsize=(4.5, 3.8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=iris.target_names, yticklabels=iris.target_names, ax=ax_cm)
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")
ax_cm.set_title(f"Confusion Matrix — {best_name}")
cm_b64 = fig_to_base64(fig_cm)

# ------------------------------------------------------------------
# 5. Build HTML report
# ------------------------------------------------------------------
table_rows = "".join(
    f"<tr><td>{r['Model']}{' 🏆' if r['Model'] == best_name else ''}</td>"
    f"<td>{r['Test Accuracy']*100:.1f}%</td>"
    f"<td>{r['CV Mean']*100:.1f}% ± {r['CV Std']*100:.1f}%</td></tr>"
    for _, r in results_df.iterrows()
)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Iris Flower Classification Report</title>
<style>
body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 900px;
        margin: 40px auto; padding: 0 16px; color: #1c1c24; background: #f7f7fb; }}
h1 {{ margin-bottom: 4px; }}
.sub {{ color: #6b6b76; margin-top: 0; }}
.card {{ background: #fff; border: 1px solid #e6e6ee; border-radius: 12px;
         padding: 20px; margin-bottom: 24px; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid #e6e6ee; }}
th {{ color: #6b6b76; }}
img {{ max-width: 100%; border-radius: 8px; }}
pre {{ background: #f0f0f5; padding: 12px; border-radius: 8px; overflow-x: auto; }}
</style>
</head>
<body>
<h1>🌸 Iris Flower Classification</h1>
<p class="sub">Task 3 — generated by Python (matplotlib + scikit-learn)</p>

<div class="card">
<h2>Model Comparison</h2>
<table>
<tr><th>Model</th><th>Test Accuracy</th><th>5-Fold CV Accuracy</th></tr>
{table_rows}
</table>
</div>

<div class="card">
<h2>Pairwise Feature Relationships</h2>
<img src="data:image/png;base64,{pairplot_b64}">
</div>

<div class="card">
<h2>Feature Correlation</h2>
<img src="data:image/png;base64,{corr_b64}">
</div>

<div class="card">
<h2>Best Model: {best_name} — Confusion Matrix</h2>
<img src="data:image/png;base64,{cm_b64}">
<pre>{report_text}</pre>
</div>

</body>
</html>
"""

output_path = os.path.join(os.getcwd(), "iris_report.html")
with open(output_path, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Report saved to: {output_path}")

# ------------------------------------------------------------------
# 6. Open directly in Chrome
# ------------------------------------------------------------------
chrome_paths = [
    "C:/Program Files/Google/Chrome/Application/chrome.exe",
    "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/usr/bin/google-chrome",
]
chrome_path = next((p for p in chrome_paths if os.path.exists(p)), None)

if chrome_path:
    webbrowser.register("chrome", None, webbrowser.BackgroundBrowser(chrome_path))
    webbrowser.get("chrome").open("file://" + output_path)
else:
    print("Chrome not found at common install paths — opening with default browser instead.")
    webbrowser.open("file://" + output_path)
