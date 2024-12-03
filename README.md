**Artificial Intelligence and Machine Learning Project: Euphoria Dataset**

*Team Members: Uendi Caka, Neda Lozanovska, Hedera Pema*

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


## EDA Analysis and Data Preprocessing

 1. **Loading the Dataset to Have an Initial Overview**
Objective: Display the column information, including their names and corresponding data types, to understand the dataset's structure.

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
Examples: `referral_friends`, `water_sources`, `loyalty_score`, etc.

- Categorical Variables:
Non-numerical data represented as object.
Examples: `fauna_friendly`, `region`, `happiness_metri`c.

- Observations:
Certain columns like `fauna_friendly` appear to be multi-value categorical data.
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

- `Referral_friends`, `Water_sources`, and `Shelters`: These columns have missing data, which should be addressed during data cleaning even though their level of missing data is not severe.
- `Region and Happiness_metric`: These columns also have missing data, but the amount is relatively moderate.
- `Happiness_index and Loyalty_score`: While these columns exhibit missing data, the percentage is low enough to use imputation strategies without significant risk of data distortion.
- `Fauna_friendly`: This column has a high percentage of missing values and requires specific handling.


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
- "Monthly" is overwhelmingly dominant with 89,564 instances.
- "Weekly" appears only 2 times.

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
![image](https://github.com/user-attachments/assets/b0eec9d5-afea-4d9b-96a1-16293145239e)


**Insights:**
- *Skewness*: Features such as water_sources, shelters, and island_size show significant skewness, suggesting limited resources on most islands.
- *Clustering*: region and x_coordinate distributions suggest geographical patterns.
- *Target Association*: happiness_index distribution indicates a relationship with resource availability.
  
**Summary of Data Cleaning Process:**
1. *Handled Missing Values:* Missing data was imputed effectively using KNN Imputer.
2.  *Dropped Columns:* Removed irrelevant or redundant columns like island_id and others with low correlation to happiness_index.
3. *Encoded Categorical Data:* Converted features like region into numerical format using LabelEncoder.
4. *Analyzed Distributions:* Examined numerical feature spreads and identified skewness or clustering trends.
5. *Prepared Data:* The dataset is now ready for modeling and further analysis.

## Defining the Problem
**Objective:**
- Use regression to predict the happiness_index based on island features like amenities, island size, and geographical coordinates.Understand how these features impact happiness and predict happiness for new islands.

**1. Splitting the Dataset into Train and Test Sets**
**Purpose:** Divide the dataset for training and evaluation.

```python
file_path = "cleaned_euphoria.csv"
cleaned_euphoria = pd.read_csv(file_path)
X = cleaned_euphoria.drop('happiness_index', axis=1)
y = cleaned_euphoria['happiness_index']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training data shape:", X_train.shape)
print("Test data shape:", X_test.shape)
```

**Output:**
1. Training Data: 71,564 rows (samples) × 11 columns (features).
2. Test Data: 17,891 rows (samples) × 11 columns (features).

**Key Insight:**
The dataset was split into training (80%) and testing (20%) sets. Both contain features and the target variable, preparing them for model training and validation.

**2. Creating a Validation Set from the Training Data**
**Objective:** Avoid overfitting by splitting the training data into a new training set and a validation set (20% of the training set).
**Purpose:** Tune hyperparameters and evaluate the model before final testing.

```python
X_train_new, X_val, y_train_new, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print("New Training data shape:", `X_train_new.shape`)
print("Validation data shape:", `X_val.shape`)
```
 **Output:**

- New Training Data: 57,251 rows (samples) × 11 columns (features).
- Validation Data: 14,313 rows (samples) × 11 columns (features).

 **Key Insight:**
- Training and Validation Split: An 80/20 split of the original training data.
- Training Data (`X_train_new`, `y_train_new`) → Used for training the model.
- Validation Data (`X_val`,`y_val`) → Used for hyperparameter tuning and early evaluation.

  **2. Feature Scaling**  
Numerical features are standardized to ensure consistent scaling across datasets, reducing bias from varying feature ranges.

```python
numerical_cols = X_train.select_dtypes(include=['float64', 'int64']).columns

X_train_numerical = X_train[numerical_cols]
X_test_numerical = X_test[numerical_cols]
X_val_numerical = X_val[numerical_cols]

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_numerical)
X_test_scaled = scaler.transform(X_test_numerical)
X_val_scaled = scaler.transform(X_val_numerical)

scaling_results = pd.DataFrame({
    "Dataset": ["Training", "Test", "Validation"],
    "Mean": [X_train_scaled.mean(axis=0).mean(),
             X_test_scaled.mean(axis=0).mean(),
             X_val_scaled.mean(axis=0).mean()],
    "Std": [X_train_scaled.std(axis=0).mean(),
            X_test_scaled.std(axis=0).mean(),
            X_val_scaled.std(axis=0).mean()]
})

print(scaling_results)
```
**Summary:**
The training data is scaled correctly with a mean close to 0 and a standard deviation of 1. The test and validation datasets show similar scaling, ensuring uniformity and preventing bias. Proper scaling prepares data for effective model training and evaluation.

## Selecting models
**Objective**:
> We use regression to predict the "happiness index" of the islands, as it is a continuous variable that quantifies the well-being of each island.

- Linear Regression: Establishes a simple linear relationship between features and the happiness index.
- Random Forest: Captures non-linear relationships and feature interactions for deeper insights.
- Support Vector Regression (SVR): Handles high-dimensional data and non-linear relationships effectively.
- Model Strategy: Train, evaluate, and compare all models on the cleaned dataset.
- Hyperparameter Tuning: Use cross-validation to optimize the best-performing model.
- Final Comparison: Compare model performance before and after tuning.

**Defining the models**

- We define three different regression models for our machine learning task.

```python
lr = LinearRegression()
svr = SVR()
rf = RandomForestRegressor(random_state=42)
```
**Evaluating the initial model performances**

-We trained three models and evaluated them using `Mean Squared Error (MSE)` and `R-squared (R²)` scores on both training and validation sets. Below are the key results:

**Linear Regression**
- Training R²: 0.2126, Validation R²: 0.2011
- Training MSE: 629291.1379, Validation MSE: 682708.1118
- Shows underfitting and performs poorly on unseen data.

**Support Vector Regression (SVR)**
- Training R²: 0.2296, Validation R²: 0.2191
- Training MSE: 615721.6336, Validation MSE: 667321.3802
- Slightly better than Linear Regression but still underfitting.

**Random Forest Regressor**
- Training R²: 0.9565, Validation R²: 0.9512
- Training MSE: 34797.5485, Validation MSE: 41722.8119
- Excellent performance with high accuracy and generalization.

**Bar Plots for R² and MSE scores**

```python
models = ['Linear Regression', 'SVR', 'Random Forest']
train_r2 = [0.2126, 0.2296, 0.9565]
val_r2 = [0.2011, 0.2191, 0.9512]

x = np.arange(len(models))
width = 0.35

plt.bar(x - width/2, train_r2, width, label='Training R²', color='blue')
plt.bar(x + width/2, val_r2, width, label='Validation R²', color='orange')

plt.xlabel('Models')
plt.ylabel('R² Score')
plt.title('R² Scores for Models')
plt.xticks(x, models)
plt.legend()
plt.show()
```
![image](https://github.com/user-attachments/assets/75fce38f-b415-4779-b470-bd9fff5428af)


```python
train_mse = [629291.1379, 615721.6336, 34797.5485]
val_mse = [682708.1118, 667321.3802, 41722.8119]

plt.bar(x - width/2, train_mse, width, label='Training MSE', color='green')
plt.bar(x + width/2, val_mse, width, label='Validation MSE', color='red')

plt.xlabel('Models')
plt.ylabel('MSE')
plt.title('MSE Scores for Models')
plt.xticks(x, models)
plt.legend()
plt.show()
```
![image](https://github.com/user-attachments/assets/921dc618-7a0c-4ba3-b549-dd60f20aab0d)

**Analysis**

Random Forest Regressor is the clear winner with high R² and low MSE. It captures data complexities without overfitting and generalizes well.

**Hyperparameter Tuning**

-We'll use `RandomizedSearchCV` to test various combinations of hyperparameters for the Random Forest model.

1. **n_estimators** is the number of decision trees in the forest.

2. **max_depth** is the maximum depth of each tree, it controls how complex each tree can become.

3. **min_samples_split** is the minimum number of samples required to split an internal node.

4. **min_samples_leaf** is the minimum number of samples required to be at a leaf node.

5. **max_features** is the number of features to consider when looking for the best split.

6. **bootstrap**, whether to use bootstrapping when sampling data for training.

- Samples hyperparameter combinations, which reduces the time and computational cost compared to trying all combinations in `GridSearchCV`.

```python
from sklearn.model_selection import RandomizedSearchCV

rf = RandomForestRegressor(random_state=42)

param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt'],
    'bootstrap': [True, False]
}

random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=10,
    scoring='neg_mean_squared_error',
    cv=3,
    n_jobs=-1,
    verbose=0,
    random_state=42
)

random_search.fit(X_train_scaled, y_train)

print("Best parameters found: ", random_search.best_params_)
print("Best CV MSE score: ", -random_search.best_score_)
```
**Conclusion:**
Best hyperparameters for Random Forest:

- n_estimators: 100 (uses 100 trees in the forest).
- min_samples_split: 5 (minimum samples to split a node).
- min_samples_leaf: 2 (minimum samples at a leaf node).
- max_features: 'sqrt' (uses the square root of total features for splitting).
- max_depth: None (trees grow until all leaves are pure or less than min_samples_split).
- bootstrap: False (does not use bootstrap sampling).
- Best CV MSE: 242250.28 (indicates moderate performance).

## Training the Model
```python
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

best_rf = RandomForestRegressor(
    n_estimators=100,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    max_depth=None,
    bootstrap=False,
    random_state=42
)

best_rf.fit(X_train_scaled, y_train)

y_test_pred = best_rf.predict(X_test_scaled)

test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)
print(f"Test MSE: {test_mse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")
print(f"Test MAE: {test_mae:.4f}")
print(f"Test R²: {test_r2:.4f}")

```

- **Test MSE**: 206,967.1458  
  Indicates a moderate error, suggesting the model predictions are somewhat off but typical for complex tasks.  

- **Test RMSE**: 454.9364  
  Shows that, on average, predictions deviate by ~455 units, which is reasonable for regression tasks.  

- **Test MAE**: 212.7307  
  Suggests most predictions are off by ~213 units, demonstrating a moderate level of accuracy.  

- **Test R²**: 0.7491  
  The model explains ~75% of the variance in the target variable, indicating a strong fit and good prediction capabilities.  

**Conclusion**:  
The Random Forest model performs well with good predictive accuracy and effectively fits the test data, making it suitable for predicting the happiness index.

**Retraining the Model with Optimal Hyperparameters**

We retrained the Random Forest model using the best hyperparameters obtained from tuning to improve prediction accuracy.

**Code:**

```python
# Retrain the model with optimal hyperparameters
best_rf = RandomForestRegressor(
    n_estimators=100,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    max_depth=None,
    bootstrap=False,
    random_state=42
)

# Fit the model on the training data
best_rf.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred_test = best_rf.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

print(f"Test MSE: {mse:.4f}")
print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"Test R²: {r2:.4f}")
```

![image](https://github.com/user-attachments/assets/9334200f-4ce6-4d03-8da3-4002f9d04f68)

**Analysis:**

The tuned Random Forest model shows strong predictive performance.
An R² score of 0.7491 indicates that the model explains approximately 75% of the variance in the happiness index.
Lower error metrics (MSE, RMSE, MAE) reflect improved accuracy over initial models.

**Conclusion:**

Hyperparameter tuning significantly enhanced the model's ability to predict the happiness index.
The final model effectively captures the relationships between island features and happiness levels.


**Residual Plot**
**Description:** Displays residuals (difference between actual and predicted values) to assess model fit and identify patterns or biases.

```python
y_pred = best_rf.predict(X_test_scaled)

residuals = y_test - y_pred

plt.figure(figsize=(10, 6))
plt.scatter(range(len(residuals)), residuals, color='blue', alpha=0.6)
plt.axhline(y=0, color='black', linestyle='--')
plt.title('Residuals')
plt.xlabel('Index')
plt.ylabel('Residuals')
plt.show()
**Observation:** Residuals are mostly centered around 0, indicating a decent model fit, but some outliers suggest occasional prediction errors.
```
![image](https://github.com/user-attachments/assets/15d4fd91-26ef-493e-9627-1f9171bc0266)


**Observation:** Residuals are mostly centered around 0, indicating a decent model fit, but some outliers suggest occasional prediction errors.

**Histogram of Residuals**
**Description:** Shows the distribution of residuals to check for normality or skewness.

```python
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, color='blue', bins=20)
plt.title('Distribution of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()
```

![image](https://github.com/user-attachments/assets/6cb8bcf9-c3a8-4dd9-aac8-9c5ec7496fd1)

**Observation:** Residuals are centered around 0, with a slight positive skew, indicating small biases in prediction.


**Residuals vs Predicted Values Scatter Plot**

**Code:** The provided code plots residuals (errors) against the predicted values.
```python
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, color='blue', alpha=0.6)
plt.axhline(y=0, color='black', linestyle='--')
plt.title('Residuals vs Predicted Values')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()
```

![image](https://github.com/user-attachments/assets/fdf107e9-e622-4fb1-8e12-07fe7ac53e24)

**Graph Explanation:** The residuals are mostly centered around zero but show some spread, especially for higher predicted values. This indicates potential heteroscedasticity (variance of residuals increases with predicted values), which might impact the model's assumptions.


**Feature Importance**
**Code:** The code generates a bar plot of feature importance as determined by the Random Forest model.

```python
importances = best_rf.feature_importances_

indices = np.argsort(importances)[::-1]

features = X_train.columns

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.barh(features[indices], importances[indices], align="center")
plt.xlabel("Relative Importance")
plt.ylabel("Feature")
plt.show()
```


![image](https://github.com/user-attachments/assets/22d8d3b2-323c-4b2e-89ee-1828cda41603)

**Graph Explanation:**
The feature y_coordinate is the most influential in predicting the target (happiness_index), followed by island_size.
Shelters is the least important, likely indicating that its variation has little impact on happiness levels.
This information helps prioritize features for analysis or further engineering.

**Comparing the performance of the initial model versus the trained model**

- A comparison of model performance metrics at different stages of training, validation, and testing. It evaluates the Mean Squared Error (MSE) and \( R^2 \) (R-squared) values for the Random Forest model.

```python
initial_training_mse = 34797.5485
initial_training_r2 = 0.9565
initial_validation_mse = 41722.8119
initial_validation_r2 = 0.9512
tuned_test_mse = 206967.1458
tuned_test_r2 = 0.7491

metrics = ['MSE', 'R^2']
initial_values = [initial_training_mse, initial_training_r2]
validation_values = [initial_validation_mse, initial_validation_r2]
tuned_values = [tuned_test_mse, tuned_test_r2]

x = np.arange(len(metrics))
width = 0.25

plt.figure(figsize=(10, 6))
plt.bar(x - width, initial_values, width, label='Initial (Training)', color='skyblue')
plt.bar(x, validation_values, width, label='Initial (Validation)', color='orange')
plt.bar(x + width, tuned_values, width, label='Tuned (Test)', color='green')

plt.xlabel('Evaluation Metric')
plt.ylabel('Metric Value')
plt.title('Comparison of Model Performance')
plt.xticks(x, metrics)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()

for i in range(len(metrics)):
    plt.text(x[i] - width, initial_values[i] + 0.02 * max(tuned_values), f"{initial_values[i]:.2f}", ha='center')
    plt.text(x[i], validation_values[i] + 0.02 * max(tuned_values), f"{validation_values[i]:.2f}", ha='center')
    plt.text(x[i] + width, tuned_values[i] + 0.02 * max(tuned_values), f"{tuned_values[i]:.2f}", ha='center')

plt.tight_layout()
plt.show()
```
![image](https://github.com/user-attachments/assets/24787000-2cae-48d1-9477-1c6356e2b5d4)

*Observations*
- MSE (Mean Squared Error): Initially, the model performed well on training and validation data (low MSE).
- R^2: Initial training and validation are high indicating a good fit to the data. The tuned models R^2 on the test set is lower , showing reduced predictive power.

**The subplots**

```python
stages = ['Initial (Training)', 'Initial (Validation)', 'Tuned (Test)']
mse_values = [initial_training_mse, initial_validation_mse, tuned_test_mse]
r2_values = [initial_training_r2, initial_validation_r2, tuned_test_r2]

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(stages, mse_values, marker='o', linestyle='-', color='blue', label='MSE')
plt.title('MSE Comparison')
plt.xlabel('Stage')
plt.ylabel('Mean Squared Error')
plt.grid(axis='y', linestyle='--', alpha=0.7)
for i, val in enumerate(mse_values):
    plt.text(i, val + 0.02 * max(mse_values), f"{val:.2f}", ha='center')
plt.xticks(rotation=15)

plt.subplot(1, 2, 2)
plt.plot(stages, r2_values, marker='o', linestyle='-', color='green', label='R^2')
plt.title('R^2 Comparison')
plt.xlabel('Stage')
plt.ylabel('R^2 Score')
plt.grid(axis='y', linestyle='--', alpha=0.7)
for i, val in enumerate(r2_values):
    plt.text(i, val + 0.02 * max(r2_values), f"{val:.4f}", ha='center')
plt.xticks(rotation=15)

plt.tight_layout()
plt.show()
```

![image](https://github.com/user-attachments/assets/2329c17f-814e-4e85-bd3e-3022bdfe1ce5)

**Observations from Subplots**
- MSE Comparison: MSE shows a significant increase from the initial model to the tuned test stage due to generalization.
- Comparison: R^2 shows a drop from the initial model to the tuned test stage, reflecting the reduced fit on unseen data.

## Results
**1. Model Performance:**

> - Random Forest's strength consists of High R^2 (0.9565 for training, 0.9512 for validation) and low MSE (34,797.55 for training, 41,722.81 for validation) during initial evaluation highlighted its suitability.
> -  After hyperparameter tuning, the Random Forest model’s R^2 on the test set decreased to 0.7491, with a significant MSE increase to 206,967.15. While this highlights a reduction in overfitting, it also shows limitations in generalization.

**2. Insights into Predictors:**
> - Spatial features (y_coordinate and x_coordinate) and physical attributes (island_size) contributed the most to predicting happiness.
> - Shelter availability and regional characteristics had negligible impact, possibly due to uniformity or poor representation in the dataset.

**3. Statistical Limitations:**

> - Heteroscedasticity in residuals for higher predictions suggested variance issues.
> - The difference in performance metrics (MSE and R^2) between training/validation and testing phases pointed to model complexity and possible dataset limitations.

**Final Recommendations:** 
> - We need to improve metrics including focusing on reducing test set MSE (currently 206,967.15) and increasing R^2 (currently 0.7491).
> - We need to add Add additional predictors, such as socioeconomic or environmental variables, to address the unexplained variance and improve model generalization.
> - We need to explore techniques like boosting (e.g., XGBoost or Gradient Boosting) to compare performance with Random Forest.

## Conclusion

The primary takeaway from this work is the effectiveness of the Random Forest Regressor in predicting the happiness index of islands. By leveraging its capability to capture complex, non-linear relationships between features, it provided robust performance and interpretability. Compared to other models like Linear Regression and SVR, Random Forest achieved significantly higher accuracy and generalization, making it the optimal choice. Feature importance analysis revealed that geographic factors, such as `y_coordinate` and `island_size`, play a crucial role in determining happiness levels, offering actionable insights for policymakers and planners.

However, some questions remain unanswered, such as whether additional features, like cultural or economic variables, could further improve predictions. The observed variance in residuals for extreme values also suggests room for refining the model to handle outliers better. Future work could focus on incorporating richer datasets, exploring ensemble techniques like boosting, or integrating interpretability tools to enhance model transparency further. Expanding the scope of the study to include cross-regional or temporal analysis could also provide deeper insights into the drivers of happiness.
  
