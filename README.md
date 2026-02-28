# Titanic Survival Prediction (Machine Learning Project)

A supervised machine learning project that predicts passenger survival on the Titanic using structured preprocessing pipelines and hyperparameter tuning.



# Project Overview:

This project builds a complete end-to-end ML workflow using:

- Data preprocessing with `ColumnTransformer`
- Feature scaling and imputation
- One-hot encoding for categorical features
- Model training with `RandomForestClassifier`
- Hyperparameter tuning using `GridSearchCV`
- Stratified cross-validation
- Model evaluation using classification metrics
- Visualization of confusion matrix and feature importance

The objective is to predict whether a passenger survived the Titanic disaster based on demographic and ticket-related features.



# Dataset:

The dataset is the built-in Titanic dataset from Seaborn.

# Features Used:
- `pclass` – Ticket class
- `sex` – Gender
- `age` – Age in years
- `sibsp` – Number of siblings/spouses aboard
- `parch` – Number of parents/children aboard
- `fare` – Passenger fare
- `class` – Ticket class (categorical)
- `who` – Man, woman, or child
- `adult_male` – Boolean indicator
- `alone` – Whether passenger traveled alone

Target:
- `survived` (0 = No, 1 = Yes)



# Technologies Used:

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

# Machine Learning Pipeline

# Data Preprocessing
- Median imputation for numerical features
- Most frequent imputation for categorical features
- Standard scaling for numerical features
- One-hot encoding for categorical variables

# Model
- RandomForestClassifier

# Hyperparameter Tuning
Grid search over:
- `n_estimators`: [50, 100]
- `max_depth`: [None, 10, 20]
- `min_samples_split`: [2, 5]

Cross-validation:
- Stratified 5-fold CV



# Results:

Test Accuracy: **81.56%**

Classification Report:

- Precision (Non-survivors): 0.83
- Precision (Survivors): 0.79
- Weighted F1-score: 0.81

---

# Visualizations:

* Confusion Matrix  
* Feature Importance (Random Forest)

These plots are saved automatically when running the script.

---

