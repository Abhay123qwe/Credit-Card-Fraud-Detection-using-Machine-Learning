## Credit Card Fraud Detection using Machine Learning
## Overview

Credit card fraud detection is a critical real-world machine learning problem characterized by extreme class imbalance, where fraudulent transactions represent a very small fraction of all transactions.
This project applies supervised and unsupervised machine learning techniques to accurately detect fraudulent credit card transactions while minimizing false positives.

The focus is on precision, recall, and robust evaluation, rather than accuracy, to reflect real-world fraud detection requirements.

## 📊 Dataset
- Source: Kaggle – Credit Card Fraud Dataset
- Total Transactions: 284,807
- Fraudulent Transactions: 492 (≈ 0.172%)
- Features:
    - 28 anonymized PCA components (V1–V28)
    - Time, Amount
    - Target variable: Class (0 = Normal, 1 = Fraud)

⚠️ The dataset is highly imbalanced, making metric selection and model tuning critical.

## 🧠 Machine Learning Workflow

## 1️⃣ Exploratory Data Analysis (EDA)
- Analyzed class imbalance and feature distributions
- Studied transaction amount and fraud patterns
- Identified challenges related to skewed data

📁 Notebook: EDA.ipynb

## 2️⃣ Baseline Model
- Logistic Regression used as a baseline
- Evaluated using precision and recall
- Demonstrated why accuracy is misleading for imbalanced datasets

📁 Notebook: baseline_logistic_regression.ipynb

## 3️⃣ Hyperparameter Tuning
- Applied GridSearchCV
- Used custom scoring metrics to balance precision and recall
- Tuned class_weight to improve fraud detection sensitivity

📁 Notebook: logistic_regression_gridsearch.ipynb

## 4️⃣ Tree-Based Model
- Random Forest Classifier
- Captured non-linear relationships
- Compared performance against logistic regression

📁 Notebook: random_forest.ipynb

## 5️⃣ Unsupervised Anomaly Detection
- Isolation Forest
- Explored fraud detection without labeled data
- Highlighted limitations of unsupervised methods in this domain

📁 Notebook: isolation_forest.ipynb

## 6️⃣ Model Comparison
- Compared all models using:
    - Precision
    - Recall
    - F1-score
    - ROC-AUC
- Identified trade-offs between models

📁 Notebook: model_comparison.ipynb

## In model_comparison_Results
| Model                          | Precision |  Recall  |    F1    | ROC-AUC  |
| ------------------------------ | --------- |  ------  |    --    | -------  |
| Logistic Regression (Baseline) |  0.060976 | 0.918367 | 0.114358 | 0.972083 |
| Logistic Regression (Tuned)    |  0.715517 | 0.846939 | 0.775701 | 0.974023 |
| Random Forest                  |  0.961039 | 0.755102 | 0.845714 | 0.957189 |
| Isolation Forest               |  0.242647 | 0.336735 | 0.282051 | 0.953403 |


## 📈 Evaluation Metrics
- Due to extreme class imbalance, the following metrics were prioritized:
    - Recall – Minimize missed fraud cases
    - Precision – Reduce false fraud alerts
    - F1-score
    - ROC-AUC
Accuracy was intentionally avoided as a primary metric.

## 🏆 Key Findings
- Tuned Logistic Regression provided the best balance between recall and precision
- Random Forest captured non-linear patterns but required careful tuning to avoid overfitting
- Isolation Forest was less effective than supervised methods for this dataset
- Class weighting significantly improved fraud detection performance

## 🛠 Technologies Used
- Python
- NumPy, Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## 🚀 Future Improvements
- Apply SMOTE or other resampling techniques
- Experiment with XGBoost / LightGBM
- Add Precision–Recall curve comparisons
- Deploy the model using FastAPI or Flask

## 📌 Conclusion
This project demonstrates a complete end-to-end fraud detection pipeline, from data exploration to model tuning and comparison.
It highlights the importance of proper metric selection, handling class imbalance, and model interpretability in real-world machine learning applications.

## 👤 Author
- Abhay Singh
- Machine Learning Enthusiast
