# Student Risk Prediction — Final Project

Machine Learning–Based Early Risk Prediction for Student Academic Performance
Using Behavioral and Engagement Data.

Team: Simran Mogadala, Kritika Joshi, Mary Neha Reddy Thumma, Khadija Warraich

## Languages Used

- Python
- Libraries Used: Pandas, Numpy

## Quickstart

```bash
pip install scikit-learn pandas numpy matplotlib seaborn xgboost shap joblib
python src/01_generate_dataset.py   # writes data/student-mat.csv
python src/02_eda.py                # figures + outputs/eda_summary.txt
python src/03_preprocessing.py      # fits + saves the preprocessor
python src/04_modeling.py           # trains all 5 models with CV tuning
python src/05_fairness.py           # fairness audit
python src/06_interpretability.py   # SHAP + feature importance
```

To use the real UCI Student Performance dataset, download student.zip from the
UCI ML Repository (Student Performance, dataset 320), extract student-mat.csv
into `data/`, and re-run from script 02. Every script downstream of dataset
generation is dataset-agnostic.

## Project layout

```
src/                source code (run in numbered order)
data/               student-mat.csv (UCI-schema)
outputs/            CSVs and text summaries
figures/            all PNGs used in the report
models/             pickled trained models + preprocessor
build_report.js     generates the final Word report
```

## Key result

| Model               | Accuracy | Recall | F1   | ROC-AUC | MCC  |
|---------------------|----------|--------|------|---------|------|
| Gradient Boosting   | 0.85     | 0.38   | 0.46 | 0.78    | 0.39 |
| Logistic Regression | 0.76     | 0.62   | 0.46 | 0.81    | 0.33 |
| XGBoost             | 0.83     | 0.38   | 0.43 | 0.79    | 0.33 |
| K-Nearest Neighbors | 0.85     | 0.24   | 0.35 | 0.68    | 0.33 |
| Random Forest       | 0.82     | 0.15   | 0.22 | 0.79    | 0.17 |

For an early-warning use case, **Logistic Regression** is the recommended
deployment model — best recall, best ROC-AUC, best fairness profile, and
fully interpretable.
