### Credit Card Fraud Detection using Machine Learning

- Handling Imbalanced Datasets with Multiple Resampling Techniques

# 📌 Overview

Credit card fraud is a major challenge in financial systems due to the extremely imbalanced nature of fraud vs. non-fraud transactions. This project focuses on:

Handling severely imbalanced datasets

Evaluating multiple machine learning algorithms

Testing models on datasets processed through various resampling methods

Comparing performance for fraud detection

The main goal is to build a consistent and reusable workflow that applies multiple models to multiple resampled datasets using a single unified function.

## 🎯 Objectives

Handle data imbalance using:

- 1.StandardScaler

- Random Oversampling

- 3.Random Undersampling

- 4.SMOTE

Train and evaluate multiple ML models

Compare results across all resampling techniques

Maintain modularity by using separate files for each model

Provide scalable, reproducible code for fraud detection experiments

 

## 🧠 Machine Learning Algorithms Used

- This project evaluates six different machine learning models:

- 1.Logistic Regression

- 2. RandomForestClassifier

- 3.DecisionTreeClassifier

- 4.XGBClassifier

- 5.K-Nearest Neighbors (KNN)

- 6.Gaussian Naive Bayes

Each model is trained separately on datasets generated after different resampling techniques.

## ⚙️ Resampling Techniques Applied

The experiment uses multiple dataset versions created using:

# Technique	Description
StandardScaler	Normalizes numerical features
Oversampling	Increase minority class samples
Undersampling	Reduce majority class samples
SMOTE	Synthetic Minority Oversampling Technique

Every model receives each of these datasets for training & testing.

# 🛠️ How It Works

Preprocess the original dataset

Generate transformed datasets

scaled

oversampled

undersampled

SMOTE

For each dataset, trains each ML model via a single function to ensure consistency

Saves evaluation metrics and model files for comparison

## 📊 Evaluation Metrics

Common fraud detection metrics:

- Precision

- Recall

- F1-Score

- Confusion Matrix

- ROC-AUC

## Accuracy (used carefully due to imbalance)
 

## 📈 Results

The project compares how each model performs under different resampling strategies.
This helps identify which combinations work best for detecting rare fraudulent transactions.

##v🧪 Key Insights

Fraud detection performance improves drastically with SMOTE and oversampling

Tree-based models like Random Forest and XGBoost generally perform better

Logistic Regression is stable and interpretable

Undersampling reduces training time but may lose important patterns

##📜 Conclusion

This project provides a systematic approach to dealing with imbalanced datasets for fraud detection.
It highlights the importance of resampling methods and model comparison while maintaining clean, modular code using separate model scripts.

# Dataset Link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
