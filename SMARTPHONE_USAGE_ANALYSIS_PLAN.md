# Smartphone Usage and Addiction Analysis Plan

Dataset:

`Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv`

Main working notebook:

`Analysis.ipynb`

This is a simple classroom-style analysis plan. The team should work in one notebook, with each member completing one phase and leaving clear notes so the next person can continue. No analytics has been run while preparing this plan.

## 1. Project Goal

The goal is to analyze smartphone usage behavior and understand how usage patterns relate to addiction indicators in the dataset.

The final notebook should answer:

1. What does the dataset contain?
2. Are there missing, duplicate, invalid, or inconsistent values?
3. What preprocessing is needed before analysis and modeling?
4. What patterns are visible from descriptive analytics and EDA?
5. Can `addicted_label` be predicted using non-leaking features?
6. Which variables seem most related to smartphone addiction indicators?

Important wording:

The team should avoid saying one variable "causes" another. Use wording such as "associated with", "related to", or "predictive of".

## 2. Dataset Overview

The CSV has 7,500 rows and 16 columns.

| Column | Role | Notes |
|---|---|---|
| `transaction_id` | Identifier | Do not use as a model feature. |
| `user_id` | Identifier | Do not use as a model feature. Check whether it is unique. |
| `age` | Numeric feature | User age. |
| `gender` | Binary categorical feature | For this project, treat valid values as `Male` and `Female` only. Report any other values before deciding how to handle them. |
| `daily_screen_time_hours` | Numeric feature | Daily screen time. |
| `social_media_hours` | Numeric feature | Social media usage. |
| `gaming_hours` | Numeric feature | Gaming usage. |
| `work_study_hours` | Numeric feature | Work or study usage. |
| `sleep_hours` | Numeric feature | Sleep duration. |
| `notifications_per_day` | Numeric feature | Daily notifications. |
| `app_opens_per_day` | Numeric feature | Daily app opens. |
| `weekend_screen_time` | Numeric feature | Weekend screen time. Confirm interpretation before making strong claims. |
| `stress_level` | Categorical feature | Example values may include `Low`, `Medium`, `High`. |
| `academic_work_impact` | Categorical feature | Example values may include `Yes`, `No`. |
| `addiction_level` | Addiction severity label | Useful for EDA, but likely leaks information when predicting `addicted_label`. |
| `addicted_label` | Main prediction target | Binary target, likely `0` or `1`. |

## 3. Notebook Structure

Use only `Analysis.ipynb`.

Each person should create their section using these headings:

```text
# Smartphone Usage and Addiction Analysis

## 0. Setup and Data Loading
## 1. Cleaning and Data Quality - Person 1
## 2. Preprocessing and Feature Preparation - Person 2
## 3. Descriptive Analytics and EDA - Person 3
## 4. Predictive Analytics - Person 4
## 5. Final Summary and Limitations - All
```

Each section should contain:

1. Markdown explaining what the person is doing.
2. Code cells for that phase.
3. Short written observations after important outputs.
4. A small handoff note at the end for the next member.

The notebook should run from top to bottom.

## 4. Simple Team Split

| Member | Notebook Section | Main Responsibility |
|---|---|---|
| Person 1 | Cleaning and Data Quality | Load data, inspect structure, check data quality, prepare cleaned dataframe. |
| Person 2 | Preprocessing and Feature Preparation | Prepare variables for EDA and modeling, handle missing values, encode categories, avoid leakage. |
| Person 3 | Descriptive Analytics and EDA | Summarize the dataset and create charts to find patterns. |
| Person 4 | Predictive Analytics | Build simple classification models and evaluate prediction of `addicted_label`. |

## 5. Shared Notebook Rules

All members should follow these rules:

1. Do not edit the original CSV file.
2. Use clear variable names.
3. Add Markdown notes before and after important code blocks.
4. Do not delete another member's work.
5. Keep the notebook simple and readable.
6. Use the same dataframe names where possible so the next person can continue.
7. If a decision is based on an assumption, write it clearly in Markdown.
8. Restart and run all cells before final submission.
9. Use `random_state=42` for train/test splits and models where possible.
10. Do not use `transaction_id`, `user_id`, or `addiction_level` as features when predicting `addicted_label`.
11. Treat `gender` as a binary variable for this project: valid cleaned values should be `Male` and `Female`.
12. Do not silently convert values such as `Other`, blanks, or unclear labels into `Male` or `Female`.

Recommended common dataframe names:

| Name | Meaning |
|---|---|
| `df_raw` | Original loaded dataframe. |
| `df_clean` | Cleaned dataframe after Person 1. |
| `df_prepared` | Preprocessed dataframe after Person 2. |
| `X` | Feature matrix for modeling. |
| `y` | Target variable, `addicted_label`. |

## 6. Section 0: Setup and Data Loading

This can be started by Person 1 and reused by everyone.

Recommended steps:

1. Import libraries.
2. Load the CSV.
3. Display the first few rows.
4. Show row and column count.
5. Show column names.

Suggested libraries:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

Optional modeling libraries for Person 4:

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
```

Expected output:

1. `df_raw` exists.
2. The dataset loads without error.
3. Everyone can see the basic shape and columns.

## 7. Phase 1: Cleaning and Data Quality - Person 1

### Goal

Check whether the dataset is clean enough for analysis and create `df_clean`.

### Tasks

Person 1 should complete these tasks in the notebook:

1. Display dataset shape.
2. Display column names.
3. Check data types using `df.info()`.
4. Check missing values.
5. Check duplicate rows.
6. Check duplicate `transaction_id`.
7. Check duplicate `user_id`.
8. Check unique values for categorical columns.
9. Check basic numeric ranges.
10. Clean `gender` as a binary field with valid values `Male` and `Female`.
11. Fix simple formatting issues if found.
12. Create `df_clean`.

### Columns to Check Carefully

Numeric columns:

```text
age
daily_screen_time_hours
social_media_hours
gaming_hours
work_study_hours
sleep_hours
notifications_per_day
app_opens_per_day
weekend_screen_time
addicted_label
```

Categorical columns:

```text
gender
stress_level
academic_work_impact
addiction_level
```

Identifier columns:

```text
transaction_id
user_id
```

### Cleaning Guidance

Use simple cleaning only:

1. Strip spaces from column names and categorical values if needed.
2. Convert numeric columns to numeric types if needed.
3. For `gender`, standardize obvious variants only:
   - `male`, `Male`, `M` should become `Male`.
   - `female`, `Female`, `F` should become `Female`.
4. For this project, values outside `Male` and `Female` should be treated as invalid or out-of-scope for the binary gender analysis.
5. If values such as `Other`, blank, or unclear labels appear in `gender`, report how many rows are affected.
6. Do not force `Other` or unclear gender values into `Male` or `Female`.
7. If non-binary or invalid gender values exist, the simple recommended action is to remove those rows from `df_clean` and document the number removed, because the project requirement is binary gender.
8. Do not remove other rows unless there is a clear reason.
9. If missing values exist, report them first. Person 2 can decide how to handle them.
10. If invalid values exist, write a note explaining them.

### Handoff to Person 2

At the end of the section, Person 1 should write:

1. Number of rows and columns after cleaning.
2. Whether missing values exist.
3. Whether duplicate rows or duplicate IDs exist.
4. The final unique values in `gender`.
5. Number of rows removed or flagged because `gender` was not `Male` or `Female`.
6. Any unusual values found.
7. Confirmation that `df_clean` is ready.

## 8. Phase 2: Preprocessing and Feature Preparation - Person 2

### Goal

Prepare the data so Person 3 can do EDA and Person 4 can do predictive modeling.

### Tasks

Person 2 should complete these tasks:

1. Start from `df_clean`.
2. Decide how to handle missing values.
3. Confirm which columns are numeric and categorical.
4. Transform binary `gender` into a modeling-ready numeric feature.
5. Create simple useful features if appropriate.
6. Prepare a dataframe for EDA called `df_prepared`.
7. Prepare feature columns and target column for modeling.
8. Clearly list which columns should be excluded from modeling.

### Missing Value Handling

If missing values exist:

1. For numeric columns, use median imputation or explain another simple choice.
2. For categorical columns, use mode imputation or an `Unknown` category.
3. Do not impute the target `addicted_label`.
4. For `gender`, do not create an `Unknown` category if the project is using binary gender only. Person 1 should already have removed or flagged non-`Male`/`Female` rows.
5. Explain all choices in Markdown.

### Gender Transformation

For this project, transform `gender` as a binary feature after cleaning:

| Cleaned Value | Suggested Encoded Value |
|---|---|
| `Female` | `0` |
| `Male` | `1` |

Recommended new column:

```text
gender_binary
```

Guidance:

1. Keep the original cleaned `gender` column for EDA labels.
2. Use `gender_binary` for modeling.
3. Confirm there are no remaining gender values outside `Male` and `Female` before encoding.
4. Document the encoding choice in Markdown so Person 4 knows what `0` and `1` mean.

### Feature Preparation

Possible simple engineered features:

| Feature | Formula | Why It May Help |
|---|---|---|
| `usage_component_total_hours` | `social_media_hours + gaming_hours + work_study_hours` | Shows total measured activity categories. |
| `notifications_per_screen_hour` | `notifications_per_day / daily_screen_time_hours` | Shows notification intensity. |
| `app_opens_per_screen_hour` | `app_opens_per_day / daily_screen_time_hours` | Shows app-opening intensity. |
| `sleep_below_7_hours` | `sleep_hours < 7` | Simple low-sleep flag. |

Important:

Handle division by zero if using ratios.

### Modeling Feature Rules

When predicting `addicted_label`, exclude:

```text
transaction_id
user_id
addicted_label
addiction_level
gender
```

Reason:

1. `transaction_id` and `user_id` are identifiers.
2. `addicted_label` is the target.
3. `addiction_level` likely contains very similar information to the target and can cause leakage.
4. Use `gender_binary` for modeling instead of the original text `gender` column.

### Handoff to Person 3 and Person 4

At the end of the section, Person 2 should write:

1. What preprocessing was done.
2. What new features were created.
3. How `gender` was encoded.
4. What columns are excluded from modeling.
5. Confirmation that `df_prepared` is ready.
6. Recommended feature list for Person 4.

## 9. Phase 3: Descriptive Analytics and EDA - Person 3

### Goal

Understand the dataset through summary statistics, grouped analysis, and visual exploration.

### Descriptive Analytics Tasks

Person 3 should complete:

1. Overall row and column count.
2. Summary statistics for numeric columns.
3. Counts for categorical columns.
4. Distribution of `addicted_label`.
5. Distribution of `addiction_level`.
6. Distribution of binary `gender`.
7. Average usage values by `addicted_label`.
8. Average usage values by `gender`.
9. Average usage values by `stress_level`.
10. Average usage values by `academic_work_impact`.

Recommended summary columns:

```text
daily_screen_time_hours
social_media_hours
gaming_hours
work_study_hours
sleep_hours
notifications_per_day
app_opens_per_day
weekend_screen_time
```

### EDA Chart Tasks

Create clear charts for:

1. Distribution of `daily_screen_time_hours`.
2. Distribution of `sleep_hours`.
3. Count of `addicted_label`.
4. Count of `addiction_level`.
5. Screen time by `addicted_label`.
6. Sleep hours by `addicted_label`.
7. Notifications per day by `addicted_label`.
8. App opens per day by `addicted_label`.
9. Stress level vs `addicted_label`.
10. Academic/work impact vs `addicted_label`.
11. Gender vs `addicted_label`.
12. Correlation heatmap for numeric variables.

### Questions to Answer

Person 3 should write short answers to:

1. Is the target variable balanced or imbalanced?
2. Do addicted and non-addicted groups appear different in screen time?
3. Do addicted and non-addicted groups appear different in sleep hours?
4. Are stress level and academic/work impact related to addiction labels?
5. Which variables seem most useful for modeling?

### Handoff to Person 4

At the end of the section, Person 3 should write:

1. Three to five main EDA observations.
2. Any outliers that may affect modeling.
3. Whether the target class looks balanced.
4. Which features seem important based on EDA.

## 10. Phase 4: Predictive Analytics - Person 4

### Goal

Build simple models to predict `addicted_label` and evaluate them clearly.

### Target

```text
addicted_label
```

### Features

Use the feature list prepared by Person 2.

Do not use:

```text
transaction_id
user_id
addiction_level
addicted_label
gender
```

Use `gender_binary` instead of the original text `gender` column.

### Recommended Modeling Steps

1. Set `X` as the feature dataframe.
2. Set `y` as `df_prepared["addicted_label"]`.
3. Split data into training and testing sets.
4. Use `stratify=y` if the class balance is uneven.
5. Build at least one baseline model.
6. Build all suitable classification models from the allowed course list.
7. Compare model performance before tuning.
8. Select the most suitable model based on performance and explainability.
9. Tune the selected model's hyperparameters.
10. Evaluate the tuned model.
11. Show confusion matrix for the final selected model.
12. Write a short interpretation of results.

Recommended split:

```text
80% training
20% testing
random_state=42
```

Allowed course models:

| Model | Use for This Project? | Notes |
|---|---|---|
| Majority class baseline | Yes, as a benchmark | Not from the course list, but useful as a simple comparison point. |
| Logistic regression | Yes | Best simple starting model for binary `addicted_label`. |
| Decision tree | Yes | Easy to explain and allowed by the course list. |
| Naive Bayes | Yes | Simple classification model. Can be compared with logistic regression and decision tree. |
| KNN | Yes | Allowed, but scale numeric features before using it. |
| Neural network | Optional | Allowed, but use only if time permits because it is harder to explain. |
| Linear regression | Not recommended for main target | `addicted_label` is classification, not a continuous numeric target. Use only if the instructor specifically asks for a regression comparison. |
| Regression | Not recommended for main target | Use only for a separate numeric prediction question, such as predicting `daily_screen_time_hours`. |

Recommended model comparison workflow for Person 4:

1. Build a majority class baseline first.
2. Train all suitable allowed classification models:
   - Logistic regression.
   - Decision tree.
   - Naive Bayes.
   - KNN.
   - Neural network, if time permits.
3. Compare their first-pass performance using the same train/test split.
4. Choose the most suitable model based on performance, interpretability, and assignment requirements.
5. Tune only the selected model, instead of tuning every model.
6. Report both before-tuning and after-tuning performance for the selected model.

Simple hyperparameter tuning examples:

| Model | Hyperparameters to Try |
|---|---|
| Logistic regression | `C`, `max_iter`, solver if needed. |
| Decision tree | `max_depth`, `min_samples_split`, `min_samples_leaf`, criterion. |
| Naive Bayes | `var_smoothing`. |
| KNN | `n_neighbors`, `weights`, distance metric. |
| Neural network | hidden layer size, activation, learning rate, `max_iter`. |

### Recommended Metrics

Report:

1. Accuracy.
2. Precision.
3. Recall.
4. F1-score.
5. Confusion matrix.
6. Classification report.

Metric guidance:

Accuracy alone may be misleading if there are many more `0` labels than `1` labels, or the reverse. Use precision, recall, and F1-score to explain performance better.

### Interpretation

Person 4 should answer:

1. Which model performed best?
2. Did the model beat the baseline?
3. Which features seem important?
4. Did the model make more false positives or false negatives?
5. What are the limitations of the model?

### Handoff to Final Summary

At the end of the section, Person 4 should write:

1. Best model name.
2. Key metrics.
3. Main modeling conclusion.
4. Important limitations.

## 11. Final Summary and Limitations - All

All team members should contribute a short final section at the bottom of `Analysis.ipynb`.

Include:

1. Dataset summary.
2. Cleaning summary.
3. Preprocessing summary.
4. Main EDA findings.
5. Main predictive modeling findings.
6. Limitations.
7. Possible future improvements.

Suggested final limitation points:

1. The dataset is observational, so the team should not claim causation.
2. `weekend_screen_time` needs careful interpretation.
3. `addiction_level` should not be used to predict `addicted_label` in the main model because of leakage risk.
4. Model results depend on the features available in the dataset.
5. The model should not be treated as a medical or clinical diagnosis tool.

## 12. Final Notebook Checklist

Before submission, check:

1. The notebook opens correctly.
2. The CSV path works.
3. All cells run from top to bottom.
4. Each person has a clearly marked section.
5. Each section has Markdown explanation.
6. `df_clean` is created by Person 1.
7. `df_prepared` is created by Person 2.
8. Person 3 includes summary tables and charts.
9. Person 4 includes model metrics and confusion matrix.
10. The final summary is written in simple, clear language.
11. The notebook does not include unsupported causal claims.
12. The original CSV remains unchanged.

## 13. Senior Data Engineer Notes

Keep the work simple, but keep the handoff clean. The most important thing is that each member leaves the notebook in a state where the next person can continue without guessing.

The key risk is leakage. For the main prediction task, do not use `addiction_level` to predict `addicted_label`. That would make the model look better than it really is.

The second key risk is unclear assumptions. If a value or column meaning is uncertain, write the uncertainty in Markdown instead of pretending it is known.
