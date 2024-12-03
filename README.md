**Artificial Intelligence and Machine Learning Project: Euphoria Dataset**

*Project Overview*

This project explores a dataset, euphoria.csv, which contains features related to islands in a virtual world. Each island's characteristics include happiness levels, amenities, and geographical information. The project's objective is to leverage machine learning techniques to predict the happiness index of the islands, analyze model performance, and identify the best model for prediction.

**Step 1**: Dataset Details

The dataset includes features like:
- Happiness levels
- Island size
- Amenities
- Geographical coordinates

**Key Libraries Used**

The following libraries are used in this project:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
```


Based on the provided content and the style of the linked README template, here's how you can incorporate the provided information into a structured README file:

## EDA Analysis and Data Preprocessing

This section details the steps taken to understand the dataset, summarize its main characteristics, and visualize it to derive insights.

 1. **Loading the Dataset to Have an Initial Overview**
Objective: Display the column information, including their names and corresponding data types, to understand the dataset's structure.
Steps:
- Print the names and data types of the columns to give a detailed overview.
- Display the first five rows of the dataset to provide a preview of its content.

```python
print("\nColumn Names and Data Types:")
print(df.dtypes)
print("\nFirst 5 Rows of the Dataset:")
print(df.head())
```

**Output:**
Column Names and Data Types:

`referral_friends`: float64
`water_sources`: float64
`shelters`: float64
`fauna_friendly`: object
`island_size`                float64
`creation_time`              float64
`region`                      object
`happiness_metric`            object
`features`                    object
`happiness_index`            float64
`loyalty_score`              float64
`total_refunds_requested`    float64
`trade_goods`                 object
`x_coordinate`               float64
`avg_time_in_euphoria`       float64
`y_coordinate`               float64
`island_id`                  float64
`entry_fee`                   object
`nearest_city`                object
`dtype`: object

2. Evaluation:

- Numerical Variables:
Continuous data represented as float64.
Examples: referral_friends, water_sources, loyalty_score, etc.

- Categorical Variables:
Non-numerical data represented as object.
Examples: fauna_friendly, region, happiness_metric.

- Observations:
Certain columns like fauna_friendly appear to be multi-value categorical data.
Missing values (NaN) exist in several rows, requiring preprocessing.
The dataset is a mix of numerical and categorical features, suitable for different preprocessing techniques like encoding and normalization.

3. Showing the dimension of the dataset: We can see that the dataset has 19 columns and 99492 rows.

 **2. Dropping Unnecessary Columns**
Objective: Remove columns that are not relevant to predicting or understanding the happiness index.
Steps:
- Drop columns like creation_time, entry_fee, nearest_city, and trade_goods as they are either irrelevant or do not contribute to model building.
- Save the cleaned dataset for further processing.
```python
columns_to_drop = ['creation_time', 'entry_fee', 'nearest_city', 'trade_goods']
df_cleaned = df.drop(columns=columns_to_drop)
cleaned_file_path = "cleaned_euphoria.csv"
df_cleaned.to_csv(cleaned_file_path, index=False)
print(f"Columns dropped: {columns_to_drop}")
```
**Output:**

Columns dropped: ['creation_time', 'entry_fee', 'nearest_city', 'trade_goods']

 **3. Visualizing Missing Data**

This step involves identifying and visualizing the missing data in the dataset to handle them effectively.

**Objective:**
- To check for missing data and visualize it for a better understanding of the dataset's structure and issues.

**Steps**:
1. Load the cleaned dataset from the file `cleaned_euphoria.csv`.
2. Compute the summary of missing values for each column using `isnull().sum()`.
3. Visualize the missing data using the `msno.matrix()` function for a graphical representation of missing values.

**Code**:
```python
file_path = "cleaned_euphoria.csv"
data_cleaned = pd.read_csv(file_path)

print("\nMissing Values Summary:")
print(data_cleaned.isnull().sum())
msno.matrix(data_cleaned)
plt.title("Missing Values Matrix")
plt.show()
```

**Graphical Representation:**

The graphical visualization below highlights missing data for each feature:
![image](https://github.com/user-attachments/assets/b7aa3fe4-bf2e-4b05-89ea-b454b1002b2f)

Next, there are notable levels of missing data in several columns:

- `Referral_friends, Water_sources, and Shelters`: These columns have missing data, which should be addressed during data cleaning even though their level of missing data is not severe.
- `Region and Happiness_metric`: These columns also have missing data, but the amount is relatively moderate.
- `Happiness_index and Loyalty_score`: While these columns exhibit missing data, the percentage is low enough to use imputation strategies without significant risk of data distortion.
- `Fauna_friendly`: This column has a high percentage of missing values and requires specific handling.

Addressing these missing values is crucial to ensure the integrity of the analysis and modeling steps.


 **4. Dropping Columns with Too Much Missing Data**

- **Objective**: Determine which columns to drop due to excessive missing values.
- **Steps**:
  1. Calculate the percentage of missing values for each column.
  2. Apply a threshold: Columns with more than 50% missing values are dropped.

**Code**:
```python
missing_percentage = (data_cleaned.isnull().sum() / len(data_cleaned)) * 100
print("\nPercentage of Missing Values per Column:")
print(missing_percentage)
```

- Dropping columns with more than 50% missing values:
```python
data_cleaned = data_cleaned.drop(columns=['fauna_friendly'])
data_cleaned.to_csv("cleaned_euphoria.csv", index=False)
```
 **5.Histogram for numerical features**

-We create histograms for all numerical columns in the dataset to visualize the distribution of each feature, to help us identify patterns such as skewness, normality, or outliers. 

**The code:**
```python
numerical_data_cleaned = data_cleaned.select_dtypes(include=['float64', 'int64'])
numerical_data_cleaned.hist(bins=20, figsize=(15, 10), edgecolor='black')
plt.suptitle("Histograms for Numerical Features", fontsize=16)
plt.show()
```
Loyalty score and average time in euphoria does not provide us with any relevant information in this dataset so we can remove them.

```python
data_cleaned = data_cleaned.drop(columns=['loyalty_score', 'avg_time_in_euphoria'])
data_cleaned.to_csv("cleaned_euphoria.csv", index=False)
```

This highlights the range of values for each numerical feature.
![image](https://github.com/user-attachments/assets/59cfc1df-50eb-4814-abf8-d579c2007c49)

 **6. Value Counts for Categorical Features**

We analyze categorical columns (`region`, `happiness_metric`, `features`) to understand their distributions and identify imbalances.

---

**Code:**
```python
file_path = "cleaned_euphoria.csv"
data_cleaned = pd.read_csv(file_path)
categorical_features = ['region', 'happiness_metric', 'features']
df[categorical_features] = df[categorical_features].astype('category')

for col in categorical_features:
    print(f"\nValue Counts for {col}:")
    print(df[col].value_counts())

    if col in ['region', 'happiness_metric']:
        plt.figure(figsize=(8, 4))
        sns.countplot(data=df, x=col, order=df[col].value_counts().index, hue=col, dodge=False, legend=False)
        plt.title(f"Distribution of {col}")
        plt.ylabel("Count")
        plt.xlabel(col.capitalize())
        plt.xticks(rotation=45)
        plt.show()
```

**Output Summary:**

*Analysis:*
States like `TX, CA, and VA` have the highest counts.
A significant imbalance exists across states.

*Visualization:*
![image](https://github.com/user-attachments/assets/0f8ac6d5-af31-44a6-82a7-019961afc1e2)

2. Happiness Metric:
The column is heavily skewed towards Monthly.
Weekly has only 2 occurrences, making this column unfit for analysis.

*Visualization:*
![image](https://github.com/user-attachments/assets/84da3663-fd30-4aea-aa96-8dec8418b7ce)

3. Features
Analysis:

High variability in combinations, e.g., `Parking, Gym,Pool`.
Further aggregation or preprocessing is required.

Key Findings:
   `happiness_metric` column shows significant skew and low utility for analysis.
   The features column has high variability, requiring preprocessing or aggregation.
Imbalances in categorical distributions, especially in region, need consideration in modeling.

**Analysis of the `happiness_metric` Column**

**Problem Description**
The `happiness_metric` column shows extreme imbalance:
- **"Monthly"** is overwhelmingly dominant with **89,564 instances**.
- **"Weekly"** appears only **2 times**.

This imbalance indicates:
1. Potential data entry issues.
2. Underrepresentation of the "Weekly" metric in the dataset.

**Steps Taken:**
1. **Visualize the distribution:**
   - Created a histogram to check for skewness in the `happiness_metric` column.

2. **Code:**
 ```python
   plt.figure(figsize=(8, 5))
   sns.histplot(data_cleaned['happiness_metric'], kde=True, color='blue', bins=30)
   plt.title("Distribution of Happiness Metric", fontsize=14)
   plt.xlabel("Happiness Metric")
   plt.ylabel("Frequency")
   plt.show()
```
**Key Findings:**

The histogram clearly demonstrates the overwhelming dominance of "Monthly."
The extreme skew makes the column unsuitable for numerical or categorical analysis.
*Decision:*
- The column is dropped from the dataset because:
- It does not differentiate between most islands.
- It provides little information for predictive modeling.
- 
Code for Dropping the Column:
```python
data_cleaned = data_cleaned.drop(columns=['happiness_metric'])
updated_file_path = "cleaned_euphoria.csv"
data_cleaned.to_csv(updated_file_path, index=False)
```

**7. Encoding Categorical Features**

**Summary:**
Categorical features are encoded to ensure compatibility with machine learning models. Special handling is applied to the `features` column to preserve meaningful information by grouping amenities into broader categories before encoding.

**Steps:**
1. Group amenities into categories like Basic Amenities, Luxury Features, Connectivity Features, Appliances, and Building Features.
2. Encode the grouped features into binary columns.
3. Drop the original `features` column.
4. Use Label Encoding for the remaining categorical columns.

```python
df_cleaned = pd.read_csv("cleaned_euphoria.csv")
feature_groups = {
    'Basic Amenities': ['Parking', 'Pool', 'Patio/Deck', 'Storage', 'Elevator'],
    'Luxury Features': ['Gym', 'Clubhouse', 'Tennis', 'Fireplace', 'Hot Tub', 'View'],
    'Connectivity Features': ['Internet Access', 'Cable or Satellite', 'TV'],
    'Appliances': ['Dishwasher', 'Refrigerator', 'Washer Dryer', 'Garbage Disposal'],
    'Building Features': ['Gated', 'Doorman', 'Wood Floors']
}

def encode_features_group(row, group_features):
    if isinstance(row, str):
        return any(feature in row for feature in group_features)
    return False

for group_name, features in feature_groups.items():
    df_cleaned[group_name] = df_cleaned['features'].apply(lambda x: encode_features_group(x, features))
df_cleaned.drop(columns=['features'], inplace=True)
df_cleaned.to_csv("cleaned_euphoria.csv", index=False)
print(df_cleaned.head())
```
```python
df_cleaned = pd.read_csv("cleaned_euphoria.csv")
categorical_features = df_cleaned.select_dtypes(include=['object']).columns
label_encoder = LabelEncoder()

for col in categorical_features:
    df_cleaned[col] = label_encoder.fit_transform(df_cleaned[col])
df_cleaned.to_csv("cleaned_euphoria.csv", index=False)
print(df_cleaned.head())
```

**Key Findings:**
The dataset now contains numerical values for previously categorical columns.
Features like `region and grouped amenities` have been successfully encoded.
Missing values in other columns (e.g., `shelters`, `features`, `x_coordinate`) need separate handling.

**8. Handling Missing Values**

We used the KNN Imputation technique to fill in missing values based on the similarity of data points. Scaling was applied before imputation to ensure consistent distance calculations, and scaling was reversed post-imputation.

**Code:**
```python
from sklearn.impute import KNNImputer

df_cleaned = pd.read_csv("cleaned_euphoria.csv")
numerical_cols = df_cleaned.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_cleaned[numerical_cols])

knn_imputer = KNNImputer(n_neighbors=5)
imputed_data = knn_imputer.fit_transform(scaled_data)

imputed_data = scaler.inverse_transform(imputed_data)
df_cleaned[numerical_cols] = imputed_data

print("\nMissing Values After Imputation:")
print(df_cleaned.isnull().sum())

df_cleaned.to_csv("cleaned_euphoria.csv", index=False)
```

**Analysis:**

Missing values in numerical columns were successfully filled using KNN Imputation.
All numerical columns now have 0 missing values, ensuring a clean dataset for further analysis.

**9. Detecting and Removing Outliers**

-The Interquartile Range (IQR) is used to detect outliers by measuring the spread of the middle 50% of the data, specifically between the 25th percentile (Q1) and the 75th percentile (Q3). Points outside this range, beyond 1.5 IQR below Q1 or above Q3, are marked as outliers.
-Boxplots are used to visualize the spread and identify potential outliers in numerical features.

```python
df_cleaned = pd.read_csv("cleaned_euphoria.csv")
numerical_cols = df_cleaned.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Remove 'island_id' if present in numerical columns
if 'island_id' in numerical_cols:
    numerical_cols.remove('island_id')

Q1 = df_cleaned[numerical_cols].quantile(0.25)
Q3 = df_cleaned[numerical_cols].quantile(0.75)
IQR = Q3 - Q1

outliers = ((df_cleaned[numerical_cols] < (Q1 - 1.5 * IQR)) | 
            (df_cleaned[numerical_cols] > (Q3 + 1.5 * IQR)))

print(outliers)
df_cleaned.to_csv("cleaned_euphoria.csv", index=False)
```

Summary:

- Outliers were identified based on IQR. The "island_id" column, which is not predictive, was removed to improve model performance.
- After outlier detection, most columns show no extreme outliers.

**Pair Plot Visualization**

-Pair plots are used to visualize the relationships between features, detect patterns, and identify outliers (if any remain). We focused on features with top positive and negative correlations with the happiness_index.
df_cleaned = pd.read_csv("cleaned_euphoria.csv")
numerical_cols = df_cleaned.select_dtypes(include=['float64', 'int64']).columns

**Calculate correlation matrix**
```python
correlation_matrix = df_cleaned[numerical_cols].corr()
target_col = 'happiness_index'

top_positive = correlation_matrix[target_col].sort_values(ascending=False).index[1:4]
top_negative = correlation_matrix[target_col].sort_values().index[:3]

important_features = list(top_positive) + list(top_negative)
important_features.append(target_col)

sns.pairplot(df_cleaned[important_features], height=2.5, aspect=1, plot_kws={'alpha': 0.7})
plt.subplots_adjust(hspace=0.3, wspace=0.3)
plt.show()
```
![image](https://github.com/user-attachments/assets/2916a894-d08a-49d5-9fa8-c8bb93e02405)

**Analysis:**

- Removing outliers clarified the relationships between features.
`happiness_inde`x correlates positively with `island_size`, implying larger islands provide amenities contributing to happiness.
- `total_refunds_requested` shows variability but little correlation with `happiness_index`.

**10. Correlation Heatmap**

-A heatmap was generated to visualize correlations between numerical features to identify strongly correlated features that might influence the target variable or require feature engineering.

```python
df_cleaned = pd.read_csv("cleaned_euphoria.csv")
correlation_matrix = df_cleaned.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(
    correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1,
    linewidths=0.5, square=True, cbar_kws={"shrink": 0.8}
)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()
```
![image](https://github.com/user-attachments/assets/fc43c0a5-1ba2-44d1-a752-5c652838b2bb)


**Analysis:**

-The heatmap reveals the following key insights:

- `referral_friends` and `total_refunds_requested` have a correlation of 0 with the target variable `happiness_index`.
- Columns with no significant correlation to the target variable are not useful for predictive analysis and should be dropped.

```python
df_cleaned = pd.read_csv("cleaned_euphoria.csv")
columns_to_drop = ['referral_friends', 'total_refunds_requested']
df_cleaned.drop(columns=columns_to_drop, inplace=True)
df_cleaned.to_csv("cleaned_euphoria.csv", index=False)
```
**Outcome:**

Dropped `referral_friends` and `total_refunds_requested` due to their lack of correlation with the target variable, improving the dataset for predictive modeling.

**11. Distribution Plots**

**Objective:**
- Visualize the spread of numerical column values to understand their distribution patterns.

```python
df_cleaned = pd.read_csv("cleaned_euphoria.csv")
numerical_cols = df_cleaned.select_dtypes(include=['float64', 'int64']).columns
n_cols = 3
n_rows = int(np.ceil(len(numerical_cols) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
axes = axes.flatten()
for i, col in enumerate(numerical_cols):
    sns.histplot(df_cleaned[col], kde=True, bins=30, color='blue', ax=axes[i])
    axes[i].set_title(f"Distribution of {col}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Frequency")
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.show()
```

**Insights:**
- *Skewness*: Features such as water_sources, shelters, and island_size show significant skewness, suggesting limited resources on most islands.
- *Clustering*: region and x_coordinate distributions suggest geographical patterns.
- *Target Association*: happiness_index distribution indicates a relationship with resource availability.
  
*Summary of Data Cleaning Process:*
1. *Handled Missing Values:* Missing data was imputed effectively using KNN Imputer.
20
2.  *Dropped Columns:* Removed irrelevant or redundant columns like island_id and others with low correlation to happiness_index.
3. *Encoded Categorical Data:* Converted features like region into numerical format using LabelEncoder.
4. *Analyzed Distributions:* Examined numerical feature spreads and identified skewness or clustering trends.
5. *Prepared Data:* The dataset is now ready for modeling and further analysis.
