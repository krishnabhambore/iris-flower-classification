# 🌸 Iris Flower Classification

A machine learning project that classifies Iris flowers into three species — **setosa**, **versicolor**, and **virginica** — based on their sepal and petal measurements.

## 📊 Dataset

The classic Iris dataset: 150 samples, 4 features (sepal length, sepal width, petal length, petal width), evenly split across 3 species (50 each).

## 🤖 Models Trained

Five classification models were trained and compared using an 80/20 train-test split and 5-fold cross-validation:

| Model | Test Accuracy | 5-Fold CV Accuracy |
|---|---|---|
| Logistic Regression | 93.3% | 96.0% ± 3.9% |
| K-Nearest Neighbors | 93.3% | 96.0% ± 2.5% |
| Decision Tree | 93.3% | 95.3% ± 3.4% |
| **Support Vector Machine** 🏆 | **100%** | 96.7% ± 3.0% |
| Random Forest | 90.0% | 96.7% ± 2.1% |

**Best model:** Support Vector Machine (linear kernel), with perfect precision, recall, and F1-score on the test set.

## 📁 Files

- `iris_classification.py` — main script: loads data, runs EDA, trains/evaluates all 5 models
- `iris_website.py` — generates an HTML report (with embedded charts) and opens it in Chrome
- `iris_report.html` — the generated report
- `iris_pairplot.png` — pairwise feature relationships by species
- `iris_correlation.png` — feature correlation heatmap
- `iris_confusion_matrix.png` — confusion matrix for the best model (SVM)
- `model_comparison.csv` — accuracy comparison table for all models

## ▶️ How to Run

```bash
pip install scikit-learn pandas matplotlib seaborn
python iris_classification.py    # runs the analysis, saves plots + CSV
python iris_website.py           # builds and opens the HTML report
```

## 🔑 Key Insight

Petal length and petal width are far more useful for distinguishing species than sepal measurements — setosa is linearly separable from the other two species using petal size alone, which is why most models achieve 90%+ accuracy with minimal tuning.
