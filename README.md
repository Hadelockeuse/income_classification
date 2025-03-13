# Income Classification Project

## Overview
This project focuses on classifying individuals' income levels using machine learning techniques. The goal is to predict whether a person earns more or less than $50,000 per year based on various demographic and employment-related features.

## Dataset
The dataset contains features such as age, education level, employment type, working hours, and more. Some key aspects of data preprocessing include:
- Handling missing values (~6%) using one-hot encoding and creating NaN indicators.
- Normalizing numerical features for better performance in certain models.
- Addressing class imbalance (~75% earn <= $50K, ~25% earn > $50K).
- Feature selection methods were applied to reduce dimensionality.

## Models used
To tackle this binary classification problem, several machine learning models were considered:
- **Linear support Vector Machine (SVM)** - A linear classification approach.
- **Random forest classifier** - A non-linear approach leveraging ensemble learning.

## Feature engineering
Feature selection was performed using both:
- **Forward sequential feature selection**: Starts with no features and adds them one by one based on performance.
- **Backward sequential feature selection**: Starts with all features and removes the least important ones iteratively.

### Features kept after selection
- **SVM (forward selection)**: 17 features including age, schooling, gains, losses, working time, and employment type.
- **SVM (backward selection)**: 96 features kept; 4 features removed.
- **Random Forest (forward selection)**: 8 features including age, schooling, marital status, and employment area.
- **Random Forest (backward selection)**: 95 features kept; 5 features removed.

## Model evaluation
Performance was assessed using:
- **Accuracy**
- **ROC AUC score** (Alternative measure of risk)
- **Stratified K-Fold Cross-Validation** to maintain class balance across folds.
- **Confusion matrix** to analyze classification performance.

Best model: **Random forest classifier with backward feature selection**
- Achieved highest test accuracy and ROC AUC scores across multiple folds.

## Hyperparameter tuning
Hyperparameters were tuned using **GridSearchCV** with custom refitting:
- **SVM best parameters**: 
  - `C = 0.1`, `penalty = l1`, `loss = squared_hinge`, `dual = False`, `max_iter = 1,000,000`
- **Random forest best parameters**:
  - `criterion = entropy`, `max_depth = None`, `max_features = sqrt`, `min_samples_leaf = 3`, `n_estimators = 500`

## Findings
- **Random forest with backward selection** provided the best overall results. It achieved 88.6% accuracy and 91.1% ROC-AUC.

## Future improvements
- Experiment with different balancing techniques.
- Try additional classifiers Neural Networks.
- Implement a more robust nested cross-validation approach.
