import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split  # Import the train_test_split function
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load your dataset
file_path = "euphoria.csv"
df = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())

# Check for missing values
missing_values = df.isnull().sum()

# Display the count of missing values for each column
print("\nMissing Values Summary:")
print(missing_values)

# Handle missing values
# Fill numerical missing values with the median
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

# Fill categorical missing values with the mode
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Check again for missing values after handling them
print("\nMissing Values After Imputation:")
print(df.isnull().sum())

# Basic EDA
print("\nDataset Overview:")
print(df.describe())  # Summary statistics for numerical data
print("\nColumns in the Dataset:")
print(df.columns)

# Correlation Heatmap for numerical features
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Count plots for top 5 categorical features
for col in categorical_cols[:5]:
    plt.figure(figsize=(10, 4))
    sns.countplot(data=df, x=col, order=df[col].value_counts().iloc[:10].index)
    plt.title(f"Distribution of {col}")
    plt.xticks(rotation=45)
    plt.show()

# Boxplot for numerical columns to identify outliers
plt.figure(figsize=(15, 6))
sns.boxplot(data=df[numerical_cols])
plt.title("Boxplot for Numerical Features")
plt.xticks(rotation=45)
plt.show()

# Save the cleaned dataset
cleaned_file_path = "cleaned_euphoria.csv"
df.to_csv(cleaned_file_path, index=False)
print(f"\nCleaned dataset saved to {cleaned_file_path}")
