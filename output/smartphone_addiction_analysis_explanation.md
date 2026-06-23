# Smartphone Usage and Addiction Analysis Explanation

This document explains the updated notebook in plain language: what was done, why each decision was made, how the decisions are backed by data, and how the final decision tree model should be interpreted.

## Executive Summary

The analysis predicts `addicted_label` using non-leaking behavioral usage variables from `Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv`. The corrected analysis keeps all 7,500 rows and fixes an important parsing issue: the value `None` in `addiction_level` is a real category, not missing data.

The final model is a tuned decision tree that excludes identifiers, gender, `addiction_level`, and aggregate/ratio engineered features. It achieved about 93.53% accuracy, 95.48% F1-score, and 91.48% balanced accuracy on the held-out test set. These scores should be interpreted carefully: `addicted_label` is a deterministic binarization of `addiction_level`, so the model is reconstructing a rule-defined target from behavioral features rather than diagnosing real-world smartphone addiction.

## 1. Analysis Objective

The project goal is to predict whether a row belongs to `addicted_label = 1` or `addicted_label = 0` using behavioral variables such as screen time, social-media hours, gaming hours, sleep hours, notifications, app opens, stress level, and academic/work impact.

The analysis is predictive and educational. It is not clinical. It does not prove that any variable causes addiction, and it should not be used as a medical or diagnostic tool.

| Target class | Count | Share |
| --- | ---: | ---: |
| `addicted_label = 0` | 2,192 | 29.23% |
| `addicted_label = 1` | 5,308 | 70.77% |

Main decisions:

- Use `addicted_label` as the binary target.
- Use behavioral usage variables as model inputs.
- Exclude identifiers and answer-key variables.
- Exclude gender from modeling after testing its weak target association.
- Use stratified splitting and class-aware metrics because the target is imbalanced.

## 2. Corrected Data Loading and Data Quality

The CSV must be loaded with:

```python
pd.read_csv(
    "Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv",
    keep_default_na=False,
    na_values=[""]
)
```

This matters because pandas normally treats the literal text `None` as missing. In this dataset, `None` is a valid `addiction_level` category. If the CSV is loaded with pandas defaults, 819 real `None` rows are incorrectly converted into `NaN`.

| Check | Result | Decision |
| --- | ---: | --- |
| Rows and columns | 7,500 rows, 16 columns | Keep full dataset. |
| Duplicate rows | 0 | No row removal needed. |
| Duplicate `transaction_id` | 0 | Clean identifier, but excluded from modeling. |
| Duplicate `user_id` | 0 | Clean identifier, but excluded from modeling. |
| Actual missing values | 0 | No missing-value imputation needed. |
| `addiction_level = "None"` | 819 rows | Treat as real lowest-severity category. |

## 3. Addiction Level and Target Definition

The corrected loading shows that `addiction_level` is complete and has four levels: `None`, `Mild`, `Moderate`, and `Severe`.

| `addiction_level` | `addicted_label = 0` | `addicted_label = 1` | Interpretation |
| --- | ---: | ---: | --- |
| `None` | 819 | 0 | Always label 0 |
| `Mild` | 1,373 | 0 | Always label 0 |
| `Moderate` | 0 | 2,874 | Always label 1 |
| `Severe` | 0 | 2,434 | Always label 1 |

This means `addicted_label` is a deterministic binarization of `addiction_level`:

- `None` and `Mild` become `addicted_label = 0`.
- `Moderate` and `Severe` become `addicted_label = 1`.

Therefore, `addiction_level` is not an independent predictor. It is effectively the answer key for the target and must be excluded from the model.

## 4. Gender Decision

Gender is retained for descriptive checks but excluded from modeling. The target rates are close across Female, Male, and Other, and Cramer's V is only about 0.0232, indicating a very weak categorical association.

| Gender | Label 0 | Label 1 | Label 1 share |
| --- | ---: | ---: | ---: |
| Female | 733 | 1,728 | 70.22% |
| Male | 709 | 1,844 | 72.23% |
| Other | 750 | 1,736 | 69.83% |

The project keeps all rows and avoids forcing `Other` into `Male` or `Female`. Excluding gender also keeps the model focused on behavioral usage variables instead of demographics.

## 5. Preprocessing and Feature Preparation

The model feature matrix excludes:

- `transaction_id` and `user_id`, because identifiers can encourage memorization and do not represent behavior.
- `addicted_label`, because it is the target.
- `addiction_level` and `addiction_level_display`, because they define or directly proxy the target.
- `gender`, because it showed weak association with the target and is not needed for predictive performance.

The notebook creates diagnostic engineered fields:

- `usage_component_total_hours`
- `notifications_per_screen_hour`
- `app_opens_per_screen_hour`
- `sleep_below_7_hours`

The final recommended model excludes the aggregate and ratio engineered fields because later diagnostics show they are not clean behavioral measurements.

## 6. Exploratory Data Analysis

The strongest descriptive differences are in screen-time-related variables. Users with `addicted_label = 1` have higher average daily screen time, weekend screen time, and social-media hours than users with `addicted_label = 0`.

| Variable | Mean for label 0 | Mean for label 1 | Difference |
| --- | ---: | ---: | ---: |
| Daily screen time | 5.16 hours | 8.47 hours | +3.31 hours |
| Weekend screen time | 6.89 hours | 10.21 hours | +3.32 hours |
| Social-media hours | 2.28 hours | 3.72 hours | +1.44 hours |

Other variables, such as sleep hours, notifications per day, app opens per day, stress level, academic/work impact, and gender, show weaker separation.

## 7. Effect Sizes

Because the dataset has 7,500 rows, p-values can become significant even for tiny differences. The updated notebook adds rank-biserial effect sizes to show which variables have practical separation between label groups.

| Feature | Rank-biserial effect size |
| --- | ---: |
| `daily_screen_time_hours` | +0.731 |
| `weekend_screen_time` | +0.712 |
| `social_media_hours` | +0.527 |
| `sleep_hours` | +0.045 |
| `gaming_hours` | +0.011 |

The practical signal is concentrated in daily screen time, weekend screen time, and social-media hours. Most other numeric variables have weak practical separation.

## 8. Collinearity and Engineered Feature Checks

`daily_screen_time_hours` and `weekend_screen_time` are highly correlated:

| Check | Value |
| --- | ---: |
| Correlation between daily and weekend screen time | 0.964 |

This means they should be interpreted as overlapping screen-time information, not as two independent behavioral drivers.

The notebook also checks whether component-hour fields behave like clean parts of daily screen time. They do not.

| Consistency check | Count | Share |
| --- | ---: | ---: |
| `social_media_hours + gaming_hours + work_study_hours` exceeds total daily screen time | 4,553 | 60.71% |
| `sleep_hours + daily_screen_time_hours + work_study_hours` exceeds 24 hours | 156 | 2.08% |

Because the component fields are not mutually exclusive parts of total screen time, `usage_component_total_hours` and the per-screen-hour ratios are treated as diagnostics only. They are excluded from the final recommended model.

## 9. Modeling Workflow

The notebook uses an 80/20 train-test split with stratification so the train and test sets preserve the target class proportions. Model-family selection is done with 5-fold cross-validation inside the training set, then the selected model is evaluated on the held-out test set.

Models compared:

- Majority-class baseline
- Logistic regression
- Decision tree
- Naive Bayes
- KNN
- Neural network

The decision tree performed best in training-only cross-validation and remains explainable, so it was selected for tuning.

## 10. Recommended Final Model

The final recommendation is the simpler tuned decision tree without `usage_component_total_hours`, `notifications_per_screen_hour`, or `app_opens_per_screen_hour`.

| Metric | Value | Meaning |
| --- | ---: | --- |
| Accuracy | 93.53% | Overall share of correct predictions. |
| Precision for label 1 | 94.55% | Of predicted label 1 rows, how many were truly label 1. |
| Recall for label 1 | 96.42% | Of actual label 1 rows, how many were found. |
| Recall for label 0 | 86.53% | Of actual label 0 rows, how many were found. |
| Balanced accuracy | 91.48% | Average recall across both classes. |
| F1-score for label 1 | 95.48% | Balance of precision and recall for label 1. |
| Macro-F1 | 92.07% | Average F1 across both classes. |

Confusion matrix:

|  | Predicted 0 | Predicted 1 |
| --- | ---: | ---: |
| Actual 0 | 379 | 59 |
| Actual 1 | 38 | 1,024 |

The model is strong compared with the baseline, but the interpretation must be careful. Since the target is rule-defined from `addiction_level`, the model is learning to reconstruct that binary rule from behavioral variables in this dataset.

## 11. Decision Log

| Decision | Why it was made | Effect on analysis |
| --- | --- | --- |
| Load CSV with `keep_default_na=False` | Preserves `None` as a real category. | Removes fake missing-data narrative. |
| Keep all gender categories | `Other` is observed, not missing. | Preserves all 7,500 rows. |
| Exclude gender from modeling | Cramer's V is only about 0.0232. | Keeps model focused on behavior. |
| Exclude `addiction_level` | It deterministically defines `addicted_label`. | Prevents target leakage. |
| Add effect sizes | Large samples can make tiny differences significant. | Separates practical signal from statistical noise. |
| Flag daily/weekend collinearity | Correlation is 0.964. | Avoids treating near-duplicate features as independent drivers. |
| Exclude aggregate/ratio engineered fields from final model | Component-hour arithmetic is inconsistent. | Improves interpretability and defensibility. |
| Use training-only CV | Avoids choosing the model based on test data. | Makes held-out reporting cleaner. |

## 12. Limitations

- The dataset appears synthetic or rule-defined, so the findings may not generalize to real populations.
- The target is a deterministic binarization of `addiction_level`, not an independently validated clinical diagnosis.
- The model should not be used as a medical or clinical tool.
- The analysis shows association and prediction, not causation.
- `daily_screen_time_hours` and `weekend_screen_time` are highly collinear.
- Component-hour fields do not behave as clean subparts of total screen time.
- Feature importance values are specific to the selected decision tree and should not be treated as causal importance.

## 13. Technical Term Glossary

| Term | Meaning in this analysis |
| --- | --- |
| `addicted_label` | The binary target being predicted. |
| `addiction_level` | A four-level severity label that defines the binary target. |
| Baseline model | A simple comparison model, here always predicting the majority class. |
| Balanced accuracy | Average recall across both classes. |
| Class imbalance | One target class is much more common than another. |
| Collinearity | When two input variables carry highly overlapping information. |
| Confusion matrix | Counts true positives, true negatives, false positives, and false negatives. |
| Cramer's V | Association strength between categorical variables, from 0 to 1. |
| Cross-validation | Repeated training-data splits used to estimate model performance. |
| Decision tree | A model that predicts through if-then split rules. |
| Effect size | A measure of practical difference or separation between groups. |
| F1-score | A metric balancing precision and recall. |
| Feature | An input variable used by the model. |
| Feature importance | A model-specific score showing which inputs contributed most. |
| Hold-out test set | Data kept aside for final performance reporting. |
| Leakage | When answer-like information is accidentally included as input. |
| Macro-F1 | F1 averaged across classes equally. |
| One-hot encoding | Turning category labels into 0/1 columns. |
| Precision | Of predicted positives, the share that were truly positive. |
| Rank-biserial correlation | An effect size for comparing two groups on a numeric or ranked variable. |
| Recall | Of true positives, the share the model found. |
| Stratified split | A split that preserves class proportions. |
| Target proxy | A variable that indirectly or directly gives away the target. |
