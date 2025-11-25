# ==============================================================================
# COLAB NOTEBOOK: Credit Risk Analysis and XGBoost Modeling
#
# This script is designed to be run in a Google Colab environment.
# It performs data loading, exploratory data analysis (EDA), preprocessing,
# model training (XGBoost), evaluation, and model saving.
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. Setup and Library Installation
# ------------------------------------------------------------------------------
# Install necessary libraries if not already available in the Colab environment
# !pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve

# XGBoost import
import xgboost as xgb

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['figure.dpi'] = 100

print("Setup complete. Ready to load data.")


# ------------------------------------------------------------------------------
# 2. Data Loading and Initial Inspection
# ------------------------------------------------------------------------------
# NOTE: Upload 'credit_risk_dataset.csv' to your Colab session's files before running this cell.

# Load the dataset
try:
    df = pd.read_csv('credit_risk_dataset.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("ERROR: Please upload 'credit_risk_dataset.csv' to the Colab environment and try again.")
    exit()

print("\n--- Initial Data Check ---")
print(f"Shape: {df.shape}")
print("\nFirst 5 Rows:")
print(df.head())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nData Types:")
print(df.dtypes)


# ------------------------------------------------------------------------------
# 3. Data Cleaning and Feature Engineering
# ------------------------------------------------------------------------------
print("\n--- Data Cleaning and Feature Engineering ---")

# 3.1. Target Variable Check
print(f"Target Variable (loan_status) Distribution:\n{df['loan_status'].value_counts(normalize=True)}")

# 3.2. Handle Outliers and Missing Values
# Capping Age outliers (e.g., ages above 80 are suspicious/rare)
df['person_age'] = np.where(df['person_age'] > 80, 80, df['person_age'])

# Derived Feature: Loan Percentage Income (Crucial feature)
df['loan_percent_income'] = df['loan_amnt'] / df['person_income']

# 3.3. Handle Missing Values
# employment length has some missing values, which will be handled by the pipeline.
# loan_int_rate also has missing values, which will be handled by the pipeline.


# ------------------------------------------------------------------------------
# 4. Exploratory Data Analysis (EDA)
# ------------------------------------------------------------------------------
print("\n--- Exploratory Data Analysis (EDA) ---")

# 4.1. Univariate Distribution of Target Variable
plt.figure(figsize=(6, 4))
sns.countplot(x='loan_status', data=df)
plt.title('Loan Status Distribution (0=Repaid, 1=Default)')
plt.show()

# 4.2. Key Numerical Features vs. Target (Loan Status)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Age
sns.histplot(data=df, x='person_age', hue='loan_status', kde=True, ax=axes[0])
axes[0].set_title('Age vs. Loan Status')

# Income
sns.boxplot(data=df, y='person_income', x='loan_status', ax=axes[1], showfliers=False) # Exclude extreme outliers
axes[1].set_title('Income vs. Loan Status')

# Loan Percent Income (The derived feature)
sns.histplot(data=df, x='loan_percent_income', hue='loan_status', kde=True, ax=axes[2])
axes[2].set_title('Loan % Income vs. Loan Status')

plt.tight_layout()
plt.show()

# 4.3. Key Categorical Features vs. Target (Loan Status)
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Home Ownership
sns.countplot(data=df, x='person_home_ownership', hue='loan_status', ax=axes[0])
axes[0].set_title('Home Ownership vs. Loan Status')

# Loan Intent
sns.countplot(data=df, x='loan_intent', hue='loan_status', ax=axes[1])
axes[1].tick_params(axis='x', rotation=45)
axes[1].set_title('Loan Intent vs. Loan Status')

plt.tight_layout()
plt.show()


# ------------------------------------------------------------------------------
# 5. Pipeline Definition and Model Training
# ------------------------------------------------------------------------------
print("\n--- Pipeline Definition and Training ---")

# Define features and target
X = df.drop('loan_status', axis=1)
y = df['loan_status']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Identify feature types
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()

# Define Preprocessing Steps
numerical_transformer = Pipeline(steps=[
    # Impute missing numerical values with the median of the training set
    ('imputer', SimpleImputer(strategy='median')),
])

categorical_transformer = Pipeline(steps=[
    # One-Hot Encode categorical features
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create the ColumnTransformer to apply transformations to the correct columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

# Define the Classifier
classifier = xgb.XGBClassifier(
    objective='binary:logistic',
    use_label_encoder=False,
    eval_metric='logloss',
    n_estimators=200, # Increased for better performance
    max_depth=5,
    learning_rate=0.05,
    random_state=42
)

# Build the Full Pipeline
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', classifier)
])

# Train the Model
print(f"Training XGBoost Pipeline on {len(X_train)} samples...")
full_pipeline.fit(X_train, y_train)
print("Training complete.")


# ------------------------------------------------------------------------------
# 6. Model Evaluation
# ------------------------------------------------------------------------------
print("\n--- Model Evaluation ---")

# Predict probabilities on the test set
y_pred_proba = full_pipeline.predict_proba(X_test)[:, 1]

# Calculate AUC Score
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC Score on Test Set: {auc_score:.4f}")

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# ------------------------------------------------------------------------------
# 7. Feature Importance and Model Saving
# ------------------------------------------------------------------------------
print("\n--- Feature Importance and Saving ---")

# 7.1. Extract Feature Names and Importance
# Get the trained XGBoost model from the pipeline
xgb_booster = full_pipeline.named_steps['classifier']

# Get the feature names after preprocessing (crucial for XGBoost)
feature_names = full_pipeline.named_steps['preprocessor'].get_feature_names_out()

# Create a DataFrame for importance
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': xgb_booster.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Plot top 15 features
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
plt.title('Top 15 XGBoost Feature Importance (Gain)')
plt.show()

# 7.2. Model Saving
model_output_path = 'credit_risk_model.joblib'
joblib.dump(full_pipeline, model_output_path)

print(f"\n✅ SUCCESS: Complete Pipeline saved as '{model_output_path}'")
print("\nNext step: Download 'credit_risk_model.joblib' for deployment.")
