# ETL Pipeline Script using Pandas and Scikit-Learn
# This script automates the Extract, Transform, Load (ETL) process.
# - Extract: Load data from a CSV file (e.g., sample dataset like Iris).
# - Transform: Preprocess (handle missing values, encode categoricals, scale features) using Pandas and Scikit-Learn.
# - Load: Save the transformed data to a new CSV file.
#
# Assumptions:
# - Input file: 'input_data.csv' (replace with your file path). For demo, we'll use a sample from Scikit-Learn.
# - Output file: 'transformed_data.csv'
# - Requires: pip install pandas scikit-learn (if not installed)
#
# Run this script in a Python environment or Jupyter Notebook.

import pandas as pd
from sklearn.datasets import load_iris  # For sample data; replace with your own data loading
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Step 1: Extract - Load the data
# For demonstration, using Iris dataset. In production, replace with pd.read_csv('your_file.csv')
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['target'] = iris.target  # Adding a categorical target for encoding demo

print("Extracted Data Sample:")
print(data.head())

# Step 2: Transform - Preprocessing and Transformation
# Define numerical and categorical features
numerical_features = iris.feature_names  # All features are numerical except the target we added
categorical_features = ['target']  # Treating target as categorical for demo

# Create preprocessing pipelines
numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values (if any)
    ('scaler', StandardScaler())  # Scale numerical features
])

categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing categoricals
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categoricals
])

# Combine into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

# Apply the transformations
transformed_data = preprocessor.fit_transform(data)

# Convert back to DataFrame for readability (optional, as transformed_data is a numpy array)
# Get column names after transformation
num_cols = numerical_features
cat_cols = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
all_cols = list(num_cols) + list(cat_cols)
transformed_df = pd.DataFrame(transformed_data, columns=all_cols)

print("\nTransformed Data Sample:")
print(transformed_df.head())

# Step 3: Load - Save the transformed data
transformed_df.to_csv('transformed_data.csv', index=False)
print("\nData loaded to 'transformed_data.csv' successfully.")

# Additional: If loading to a database (e.g., SQLite), uncomment below
# import sqlite3
# conn = sqlite3.connect('etl_database.db')
# transformed_df.to_sql('transformed_table', conn, if_exists='replace', index=False)
# conn.close()
# print("Data loaded to SQLite database successfully.")
