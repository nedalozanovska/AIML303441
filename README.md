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


This highlights the range of values for each numerical feature.
![image](https://github.com/user-attachments/assets/59cfc1df-50eb-4814-abf8-d579c2007c49)



