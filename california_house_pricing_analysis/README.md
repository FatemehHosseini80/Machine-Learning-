# California Housing: Regression, Classification & Model Interpretability

This project presents a complete machine learning workflow using the **California Housing dataset**. The project explores both **regression** and **binary classification** tasks and demonstrates the complete pipeline from data exploration and preprocessing to model training, hyperparameter tuning, evaluation, learning curves, and model interpretability using **SHAP**.

The project is designed as a practical machine learning study covering important concepts such as:

* Exploratory Data Analysis (EDA)
* Data preprocessing
* Outlier handling
* Feature scaling
* Train/validation/test splitting
* Regression
* Binary classification
* Baseline models
* Hyperparameter tuning
* Learning curves
* Model evaluation
* Feature importance and model interpretability
* SHAP explanations

---

## 📌 Project Overview

The California Housing dataset contains information about housing districts in California collected from the **1990 U.S. Census**.

The dataset contains:

* **20,640 instances**
* **8 numerical features**
* **1 continuous target variable**

The target variable, `MedHouseValue`, represents the median house value for a California district, expressed in units of **$100,000**.

The project approaches this dataset from two different machine learning perspectives:

### 1. Regression

The original target variable is used to predict the median house value.

### 2. Binary Classification

The continuous target is transformed into two classes based on the median value of the training target:

* `0`: below the training median
* `1`: greater than or equal to the training median

This allows the same dataset to be studied as a binary classification problem.

---

## 📊 Dataset

The dataset is loaded directly using Scikit-learn:

```python
from sklearn.datasets import fetch_california_housing

california = fetch_california_housing()
```

The dataset contains the following features:

| Feature      | Description                              |
| ------------ | ---------------------------------------- |
| `MedInc`     | Median income in the block group         |
| `HouseAge`   | Median house age in the block group      |
| `AveRooms`   | Average number of rooms per household    |
| `AveBedrms`  | Average number of bedrooms per household |
| `Population` | Block group population                   |
| `AveOccup`   | Average number of household members      |
| `Latitude`   | Geographic latitude                      |
| `Longitude`  | Geographic longitude                     |

The project verifies the dataset dimensions as:

```text
X: (20640, 8)
y: (20640,)
```

## and constructs a Pandas DataFrame containing the features and target.

# 🔎 Exploratory Data Analysis

The first stage of the project focuses on understanding the dataset before model development.

The analysis includes:

* Dataset inspection
* Feature inspection
* Descriptive statistics
* Missing-value analysis
* Target distribution visualization
* Boxplots for numerical features
* Outlier investigation

The dataset contains no missing values in the analyzed columns.

The project also visualizes the distribution of the target variable:

```python
plt.hist(
    x=california_df["MedHouseValue"],
    bins=50
)
```

This provides an initial understanding of the distribution of house values.

---

# 🧹 Data Preprocessing

A preprocessing pipeline is implemented to make the data preparation consistent across different models.

## Outlier Handling

Boxplots are used to inspect numerical variables for potential outliers.

The project specifically applies outlier handling to:

* `AveRooms`
* `AveBedrms`

The **IQR (Interquartile Range)** method is used:

```text
IQR = Q3 - Q1

Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

Values outside these boundaries are replaced with the median of the corresponding column.

This process is implemented as a reusable preprocessing function.

---

## Feature Scaling

Two preprocessing pipelines are used depending on the model.

### Models that do not require scaling

These models use outlier handling without standardization.

```python
preprocess_noscaler
```

### Models that benefit from scaling

These models use:

1. Outlier handling
2. `StandardScaler`

```python
preprocess_withscaler
```

This separation makes it possible to apply appropriate preprocessing to different machine learning algorithms.

---

# ✂️ Train / Validation / Test Split

The dataset is divided into three subsets.

The split strategy is:

```text
Training:   60%
Validation: 20%
Test:       20%
```

The first split reserves 40% of the data, which is then divided equally into validation and test sets.

```python
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42
)
```

A fixed `random_state=42` is used to make the experiments reproducible.

---

# 📈 Part I — Regression

In the regression task, the objective is to predict the continuous `MedHouseValue` target.

## Regression Evaluation Metrics

The following metrics are used:

### RMSE — Root Mean Squared Error

Measures the square root of the average squared prediction error.

Lower values indicate better performance.

### MAE — Mean Absolute Error

Measures the average absolute difference between predictions and actual values.

Lower values indicate better performance.

### R² — R-Squared

Measures the proportion of target variance explained by the model.

Higher values indicate better performance.

The project implements a reusable evaluation function for these metrics.

---

## Models

The regression experiments include:

* Linear Regression
* Ridge Regression
* Lasso Regression
* Random Forest Regressor

---

## 1. Linear Regression

Linear Regression is used as the primary baseline model.

The model is combined with the preprocessing pipeline using Scikit-learn's `Pipeline`.

The validation results reported by the notebook are:

| Metric | Validation |
| ------ | ---------: |
| RMSE   |    0.73017 |
| MAE    |    0.53656 |
| R²     |    0.59157 |

## The implementation also includes a learning-curve analysis to study training and validation error as the training set grows.

## 2. Ridge Regression

Ridge Regression is evaluated as a regularized linear model.

The initial configuration uses:

```python
Ridge(alpha=100)
```

The project then evaluates different values of `alpha` to identify a better configuration.

The best validation result reported for the hyperparameter search uses:

```text
alpha = 10
Validation RMSE = 0.73016
```

The corresponding test evaluation is:

| Metric |    Test |
| ------ | ------: |
| RMSE   | 0.72983 |
| MAE    | 0.53495 |
| R²     | 0.61152 |

---

## 3. Lasso Regression

Lasso Regression is used to introduce L1 regularization.

Several values of `alpha` are evaluated:

```python
alphas = [
    0.001,
    0.01,
    0.1,
    1,
    10,
    100,
    1000
]
```

The best configuration reported by the experiment is:

```text
alpha = 0.001
```

with:

```text
Validation RMSE = 0.73018
Train RMSE = 0.72135
```

The final test performance is:

| Metric |    Test |
| ------ | ------: |
| RMSE   | 0.72984 |
| MAE    | 0.53496 |
| R²     | 0.61152 |

---

# 🌲 4. Random Forest Regression

Random Forest is used to capture nonlinear relationships between the input features and house values.

The baseline model uses:

```python
RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    random_state=42
)
```

The project performs hyperparameter experiments over:

### Number of Trees

```python
n_estimators = [
    50,
    100,
    200,
    500,
    800
]
```

The best validation configuration reported is:

```text
n_estimators = 800
Validation RMSE = 0.51802
Train RMSE = 0.18916
```

### Maximum Tree Depth

The following values are evaluated:

```python
max_depth = [
    5,
    10,
    20,
    None
]
```

The notebook also evaluates the resulting model on the test set. One reported Random Forest regression configuration achieves:

| Metric | Validation |    Test |
| ------ | ---------: | ------: |
| RMSE   |    0.55494 | 0.55933 |
| MAE    |    0.37705 | 0.37906 |
| R²     |    0.76408 | 0.77183 |

Overall, the Random Forest models substantially improve upon the linear regression-based approaches in terms of regression performance.

---

# 📊 Regression Learning Curves

Learning curves are used to analyze model behavior as the amount of training data increases.

For regression, the project evaluates:

* Training error
* Validation error

over multiple training-set sizes.

This helps investigate:

* Overfitting
* Underfitting
* Generalization behavior
* Whether additional training data may improve performance

The learning-curve implementation evaluates ten different training-set fractions.

---

# 🏷️ Part II — Binary Classification

The continuous house-value target is converted into a binary classification problem.

The threshold is calculated from the **median of the training target**:

```python
threshold_cls = np.median(y_reg_train)
```

Then:

```python
y_cls = (y_reg >= threshold_cls).astype(int)
```

Therefore:

```text
Class 0 → House value below the training median
Class 1 → House value greater than or equal to the training median
```

---

# 📏 Classification Evaluation

The classification models are evaluated using:

## Macro F1 Score

Macro F1 calculates the F1 score independently for each class and then averages the results.

This is useful for evaluating performance across both classes.

## ROC-AUC

ROC-AUC measures the model's ability to distinguish between the two classes based on predicted probabilities.

The project implements both metrics in a reusable evaluation function.

---

# 🤖 Classification Models

The classification experiments include:

* Logistic Regression
* Random Forest Classifier
* Gradient Boosting Classifier

---

## 1. Logistic Regression

Logistic Regression is used as the classification baseline.

The reported baseline performance is:

| Metric   | Validation |
| -------- | ---------: |
| Macro F1 |    0.48543 |
| ROC-AUC  |    0.48116 |

This provides a useful baseline for comparison with nonlinear ensemble models.

---

# 🌲 2. Random Forest Classifier

Random Forest is used to model nonlinear relationships in the classification task.

The baseline configuration uses:

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42
)
```

The baseline results are:

| Metric   | Validation |
| -------- | ---------: |
| Macro F1 |    0.88905 |
| ROC-AUC  |    1.00000 |

---

## Random Forest Hyperparameter Tuning

The project evaluates different values of `max_depth`.

The best reported configuration is:

```text
max_depth = 20
Validation Macro F1 = 0.89099
```

The project also evaluates different values of `n_estimators`:

```python
n_estimators = [
    50,
    100,
    200,
    500,
    800
]
```

One of the experiments reports:

```text
n_estimators = 800
Validation Macro F1 = 0.89196
Train Macro F1 = 1.00000
```

The final reported test performance for one Random Forest configuration is:

| Metric   | Validation |    Test |
| -------- | ---------: | ------: |
| Macro F1 |    0.87935 | 0.87958 |
| ROC-AUC  |    1.00000 | 1.00000 |

---

# 🚀 3. Gradient Boosting Classifier

Gradient Boosting is also investigated as a nonlinear ensemble classification model.

The notebook evaluates the model using the same classification metrics and performs hyperparameter experiments.

One reported Gradient Boosting configuration achieves:

| Metric   | Validation |    Test |
| -------- | ---------: | ------: |
| Macro F1 |    0.90625 | 0.89749 |
| ROC-AUC  |    1.00000 | 1.00000 |

Based on the reported test results, Gradient Boosting provides the strongest classification performance among the reported final experiments.

---

# 📈 Classification Learning Curves

Learning curves are also generated for the classification models.

The project evaluates training and validation Macro F1 scores across ten different training-set sizes.

This analysis helps determine:

* Whether the model is overfitting
* Whether more training data may help
* How training performance changes with dataset size
* How well the model generalizes to unseen data

---

# 🔍 Model Interpretability with SHAP

In addition to predictive performance, the project investigates **why the models make their predictions**.

The interpretability section uses **SHAP (SHapley Additive exPlanations)**.

SHAP is applied to tree-based models using:

```python
shap.TreeExplainer(...)
```

For the regression analysis, SHAP is applied to a Random Forest Regressor.

The project generates:

* SHAP summary plots
* Beeswarm plots
* Individual force plots

For example:

```python
shap.plots.force(explanation[190])
```

is used to inspect the contribution of individual features for a particular prediction.

SHAP is also applied to the Gradient Boosting classification model.

---

# 🧠 What SHAP Provides

The SHAP analysis allows the project to move beyond simply asking:

> "How accurate is the model?"

and investigate:

> "Which features are influencing the model's predictions, and in which direction?"

This makes the machine learning workflow more interpretable and provides insight into the relationship between housing characteristics and model predictions.

---

# 🏗️ Project Workflow

The complete workflow can be summarized as:

```text
California Housing Dataset
          │
          ▼
   Data Inspection
          │
          ▼
  Exploratory Analysis
          │
          ├── Descriptive Statistics
          ├── Missing Values
          ├── Target Distribution
          └── Outlier Analysis
          │
          ▼
    Data Preprocessing
          │
          ├── Outlier Handling
          └── Feature Scaling
          │
          ▼
 Train / Validation / Test Split
          │
          ├─────────────────────┐
          ▼                     ▼
      Regression           Classification
          │                     │
          ├─ Linear             ├─ Logistic Regression
          ├─ Ridge              ├─ Random Forest
          ├─ Lasso              └─ Gradient Boosting
          └─ Random Forest
          │                     │
          ▼                     ▼
     Hyperparameter        Hyperparameter
        Tuning                Tuning
          │                     │
          ▼                     ▼
    Learning Curves       Learning Curves
          │                     │
          └──────────┬──────────┘
                     ▼
                Evaluation
                     │
                     ▼
               SHAP Analysis
                     │
                     ▼
             Model Interpretation
```

---

# 📊 Main Results

## Regression

The reported results demonstrate that tree-based ensemble methods outperform the linear models on this dataset.

| Model             |   Test RMSE |    Test MAE |     Test R² |
| ----------------- | ----------: | ----------: | ----------: |
| Linear Regression |           — |           — |           — |
| Ridge             |     0.72983 |     0.53495 |     0.61152 |
| Lasso             |     0.72984 |     0.53496 |     0.61152 |
| Random Forest     | **0.55933** | **0.37906** | **0.77183** |

The Random Forest result reported in the notebook provides the strongest regression performance among the listed final results.

> **Note:** The notebook contains multiple experiments/configurations for some models, so the table focuses on the clearly reported final test metrics rather than combining every intermediate experiment.

---

## Classification

| Model               | Test Macro F1 | Test ROC-AUC |
| ------------------- | ------------: | -----------: |
| Logistic Regression |             — |            — |
| Random Forest       |       0.87958 |      1.00000 |
| Gradient Boosting   |   **0.89749** |      1.00000 |

The reported Gradient Boosting experiment achieves the highest test Macro F1 among these final classification results.



