import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load your dataset
file_path = "euphoria.csv"
df = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing Values Summary:")
print(missing_values)

# Display dataset info and column types
print("\nDataset Information:")
df.info()

# Display basic statistics of the dataset
print("\nDataset Overview:")
print(df.describe(include="all"))

# --- HANDLE MISSING VALUES ---

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

# --- OUTLIER DETECTION AND REMOVAL ---

# Z-Score Method for outlier detection
z_scores = np.abs(stats.zscore(df[numerical_cols]))
outliers = (z_scores > 3).any(axis=1)
print(f"\nNumber of Outliers Detected: {outliers.sum()}")

# Remove rows with outliers
df = df[~outliers]

# --- EDA VISUALIZATIONS ---

# Correlation Heatmap for numerical features
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Distribution plots for numerical features
for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde=True, bins=30, color='blue')
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

# Count plots for categorical features
for col in categorical_cols:
    plt.figure(figsize=(10, 4))
    sns.countplot(data=df, x=col, order=df[col].value_counts().index)
    plt.title(f"Count Plot for {col}")
    plt.xticks(rotation=45)
    plt.show()

# Boxplots for detecting spread and potential outliers
for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df, x=col)
    plt.title(f"Boxplot for {col}")
    plt.show()

# --- ENCODE CATEGORICAL FEATURES ---

# Label Encoding for categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# --- FEATURE SCALING ---

# Standardize numerical features
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# --- SPLIT DATASET INTO TRAINING AND TEST SETS ---

# Example target column (you may replace with the actual target column in your dataset)
target_column = "happiness_index"  # Replace with your target column
features = df.drop(columns=[target_column])
target = df[target_column]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# --- SAVE THE PROCESSED DATASET ---

# Save the cleaned dataset for further analysis
cleaned_file_path = "cleaned_euphoria.csv"
df.to_csv(cleaned_file_path, index=False)
print(f"\nCleaned dataset saved to {cleaned_file_path}")

