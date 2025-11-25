# Credit Risk Scoring Model (XGBoost)

This repository contains the Jupyter Notebook code for training a credit risk classification model using XGBoost. The goal is to predict the probability of a borrower defaulting on a loan (loan_status).

The core development is performed in the credit_risk_analysis.ipynb Colab notebook.

 ## Getting Started

### Prerequisites

Dataset: This project requires the credit_risk_dataset.csv file. You must upload this file to your Colab environment's file directory to run the notebook successfully.

Environment: A Google Colab environment or a local Python environment with the following packages installed:

pandas

numpy

scikit-learn

xgboost

matplotlib

seaborn

joblib

### Running the Analysis

Open the credit_risk_analysis.ipynb file in Google Colab.

Upload credit_risk_dataset.csv to the Colab environment.

Run the notebook cells sequentially.

## 📊 Analysis and Modeling Overview

### 1. Exploratory Data Analysis (EDA)

The notebook performs key visualizations to understand the relationship between features and the target variable (loan_status):

Target Distribution: Shows the imbalance between defaulted (1) and repaid (0) loans.

Numerical Features: Histograms and boxplots compare distributions of features like person_age, person_income, and the derived feature loan_percent_income across loan statuses.

Categorical Features: Count plots show how default rates vary across person_home_ownership and loan_intent.

### 2. Preprocessing Pipeline

To ensure consistent data transformation during both training and inference, a Scikit-learn Pipeline is constructed:

Missing Value Imputation: Missing numerical values (e.g., person_emp_length, loan_int_rate) are imputed using the median of the training data.

One-Hot Encoding: Categorical features are converted into numerical binary columns using OneHotEncoder.

Feature Engineering: The critical feature loan_percent_income is created prior to the pipeline.

### 3. Model Training

The classification task is handled by XGBoost (XGBClassifier), a highly effective gradient boosting framework.

Parameters: The model is initialized with 200 estimators, a maximum depth of 5, and a learning rate of 0.05.

Pipeline Integration: The preprocessor and the classifier are combined into a single full_pipeline to manage all steps efficiently.

### 4. Evaluation and Results

The model's performance is measured using the Area Under the ROC Curve (AUC), which is crucial for binary classification tasks, especially with imbalanced data.

Evaluation: The AUC score is calculated on the held-out test set.

Visualization: A Receiver Operating Characteristic (ROC) curve is plotted to visually represent the model's performance across different classification thresholds.

### 5. Feature Importance

The importance of each processed feature in making predictions is visualized, allowing for interpretation of the model's decisions. The top 15 features by Gain are plotted, typically highlighting factors like income, loan amount, and credit history length as primary drivers of risk.

## 💾 Model Artifact

The final trained Scikit-learn Pipeline object is serialized (saved) to disk using the joblib library:

Filename: credit_risk_model.joblib

Purpose: This single file contains the entire trained system (preprocessor + XGBoost model) and is the file required for deployment to prediction services like Streamlit or cloud endpoints.
