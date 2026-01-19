

# Credit Card Fraud Detection Using Machine Learning

## Overview

This project focuses on building a robust machine learning system to detect fraudulent credit card transactions in a **highly imbalanced dataset**. The primary goal is to maximize fraud detection while controlling false positives, using industry-relevant evaluation metrics such as **ROC-AUC, recall, and precision**.

Multiple models were developed, evaluated, and compared to identify the most effective approach for real-world fraud detection scenarios.



## Dataset

* **Source**: Credit Card Transactions Dataset
* **Total Transactions**: 284,807
* **Fraudulent Transactions**: 492 (0.17%)
* **Target Variable**:

  * `0` → Legitimate transaction
  * `1` → Fraudulent transaction
* **Features**:

  * `V1–V28`: PCA-transformed variables for confidentiality
  * `Time`: Seconds elapsed since first transaction
  * `Amount`: Transaction value

**Data Quality**:

* No missing values
* 1,081 duplicate rows identified

---

## Data Preparation

* Scaled `Amount` and `Time` using **RobustScaler** to handle outliers
* Dropped original unscaled features after transformation
* Stratified train-test split (80/20) to preserve class distribution

---

## Class Imbalance Strategy

Given the extreme imbalance in the target variable:

* **SMOTE** was applied for Logistic Regression
* **Class weighting** used for Random Forest
* **Scale_pos_weight** applied in XGBoost

This ensured fair learning without bias toward the majority class.

---

## Models and Approach

### Logistic Regression

* Trained on SMOTE-balanced data
* Custom probability threshold applied to improve fraud recall

**Outcome**:

* High recall for fraud detection
* Lower precision due to increased false positives

---

### Random Forest Classifier

* Used `class_weight='balanced'`
* Trained on original imbalanced data

**Outcome**:

* Strong balance between precision and recall
* Robust performance with minimal tuning
* ROC-AUC ≈ 0.96

---

### XGBoost Classifier

* Implemented with imbalance-aware weighting
* Hyperparameter tuning via GridSearchCV
* Optimized using ROC-AUC scoring

**Best Parameters**:

* max_depth: 4
* learning_rate: 0.01
* n_estimators: 200
* subsample: 0.7
* colsample_bytree: 0.7

**Best ROC-AUC**: 0.98

---

## Model Evaluation

Models were evaluated using:

* Classification report
* Confusion matrix
* ROC curve
* ROC-AUC score

Accuracy was intentionally not used as the primary metric due to class imbalance.

---

## Results Summary

| Model                       | Fraud Recall | Fraud Precision | ROC-AUC |
| --------------------------- | ------------ | --------------- | ------- |
| Logistic Regression (SMOTE) | Very High    | Low             | ~0.94   |
| Random Forest               | High         | High            | ~0.96   |
| XGBoost (Tuned)             | High         | Moderate–High   | ~0.98   |

---

## Key Insights

* Handling class imbalance is essential for fraud detection systems
* ROC-AUC and recall are more reliable than accuracy for rare-event prediction
* Ensemble and boosting models outperform linear models in this domain
* Threshold tuning has a significant impact on business outcomes

---

## Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Imbalanced-learn
* XGBoost
* Matplotlib, Seaborn

---

## Future Enhancements

* Cost-based evaluation aligned with financial loss
* Threshold optimization using precision-recall curves
* Model explainability using SHAP
* End-to-end ML pipeline for production deployment

---

## Author

**Jai Kishan Kokkiligadda**
Data Science and Analytics

