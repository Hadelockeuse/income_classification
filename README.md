# Income Classification Project

## Overview
This project focuses on classifying individuals' income levels using machine learning techniques. The goal is to predict whether a person earns more or less than $50,000 per year based on various demographic and employment-related features.

## Dataset
The dataset contains features such as age, education level, employment type, working hours, and more. Some key aspects of data preprocessing include:
- Handling missing values (~6%) using one-hot encoding and creating NaN indicators.
- Normalizing numerical features for better performance in certain models.
- Addressing class imbalance (~75% earn <= $50K, ~25% earn > $50K).
- Feature selection methods were applied to reduce dimensionality.

## Models Used
To tackle this binary classification problem, several machine learning models were considered:
- **Linear Support Vector Machine (SVM)** - A linear classification approach.
- **Random Forest Classifier** - A non-linear approach leveraging ensemble learning.

## Feature Engineering
Feature selection was performed using both:
- **Forward Sequential Feature Selection**: Starts with no features and adds them one by one based on performance.
- **Backward Sequential Feature Selection**: Starts with all features and removes the least important ones iteratively.

### Features Kept After Selection
- **SVM (Forward Selection)**: 17 features including age, schooling, gains, losses, working time, and employment type.
- **SVM (Backward Selection)**: 96 features kept; 4 features removed.
- **Random Forest (Forward Selection)**: 8 features including age, schooling, marital status, and employment area.
- **Random Forest (Backward Selection)**: 95 features kept; 5 features removed.

## Model Evaluation
Performance was assessed using:
- **Accuracy**
- **ROC AUC Score** (Alternative measure of risk)
- **Stratified K-Fold Cross-Validation** to maintain class balance across folds.
- **Confusion Matrix** to analyze classification performance.

Best model: **Random Forest Classifier with Backward Feature Selection**
- Achieved highest test accuracy and ROC AUC scores across multiple folds.

## Hyperparameter Tuning
Hyperparameters were tuned using **GridSearchCV** with custom refitting:
- **SVM Best Parameters**: 
  - `C = 0.1`, `penalty = l1`, `loss = squared_hinge`, `dual = False`, `max_iter = 1,000,000`
- **Random Forest Best Parameters**:
  - `criterion = entropy`, `max_depth = None`, `max_features = sqrt`, `min_samples_leaf = 3`, `n_estimators = 500`

## Findings
- **Class Imbalance Issue**: Model predictions tend to favor the majority class (<=50K income), potentially affecting precision for the minority class.
- **Feature Importance**: Employment type and marital status were crucial predictors.
- **Random Forest with Backward Selection** provided the best overall results.

## Future Improvements
- Experiment with different balancing techniques (e.g., SMOTE, weighted loss functions).
- Try additional classifiers like Gradient Boosting or Neural Networks.
- Implement a more robust nested cross-validation approach.
