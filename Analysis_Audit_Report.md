# Data Analysis Audit Report: Smartphone Usage and Addiction

This document provides a comprehensive audit of the `Analysis.ipynb` project. The analysis demonstrates strong methodological discipline — particularly in leakage prevention and model evaluation — but several important gaps and structural issues must be addressed before the findings can be considered reliable or deployable.

---

## 1. Strengths

### 1.1 Target Proxy Detection and Leakage Prevention
Recognizing that `addiction_level` deterministically maps to `addicted_label` (Mild/Not recorded → 0, Moderate/Severe → 1) and excluding it from modeling was the single most important decision in this project. Including it would have produced a model with 100% artificial accuracy that is functionally useless for prediction.

### 1.2 Imbalanced Data Handling
The target is 70.77% class 1 / 29.23% class 0. The team correctly avoided raw accuracy as the primary metric, used `stratify=y` in the train/test split, and selected models using `macro_f1` during cross-validation. This ensures the minority class is not hidden behind majority-class performance.

### 1.3 Strict Train/Test Separation
Model family selection, hyperparameter tuning, and feature-set comparison were all performed using 5-fold cross-validation exclusively on the training data. The held-out 20% test set was reserved for final reporting only. This is the gold standard for preventing data dredging.

### 1.4 Parsimony Principle
The "Engineered Feature Sensitivity Check" was well-executed. When removing the aggregate and ratio features (`usage_component_total_hours`, `notifications_per_screen_hour`, `app_opens_per_screen_hour`) produced the same CV macro-F1, the simpler model was chosen. Simpler models are easier to explain, maintain, and are less prone to overfitting.

### 1.5 Outlier Restraint
The IQR range check was performed but no records were blindly removed. In behavioral data, extreme users are often the most informative cohort, not statistical errors to be deleted.

### 1.6 Causal vs. Associative Language
Throughout the notebook, the team consistently used "predictive of" and "associated with" rather than "causes," and explicitly warned that the model should not be used as a clinical diagnostic tool. This semantic rigor is important and should be maintained in all stakeholder-facing materials.

---

## 2. Critical Issues

### 2.1 The Dataset is Synthetic with Deterministic Targets

This is the most fundamental issue and affects the validity of all downstream conclusions.

**Evidence:**

| Check | Result |
|-------|--------|
| Skewness of all 8 numeric usage features | Near 0.0 (range: −0.02 to +0.02). Real behavioral data is right-skewed; uniform distributions indicate synthetic generation. |
| Unlimited-depth Decision Tree on `addicted_label` | 100.00% training accuracy (435 leaves with just 2 features) |
| Unlimited-depth Decision Tree on `addiction_level` | 100.00% training accuracy |
| Sufficient rule: `screen_time > 8.0` OR `social_media > 4.0` | Captures 4,841 of 5,308 class-1 cases with 0 false positives |
| Missing `addiction_level` predictability | 89.75% accuracy using simple splits — missingness occurs when `screen_time ≤ 6.01` and `social_media ≤ 4.01` |

**Implications:**

- The model has reverse-engineered the data generator's hardcoded threshold rules, not learned real-world behavioral patterns.
- The 0 false positives in the confusion matrix is an artifact of the deterministic target — in real behavioral data, this is virtually impossible.
- All performance metrics (94.13% accuracy, 93.27% macro-F1) reflect the model's ability to reproduce the simulator's logic, not genuine predictive power.
- The model cannot generalize to real populations because it was trained on data with zero natural noise.

**Required action:** The final report must explicitly state that the dataset is synthetic, the target was generated deterministically, and the results cannot be generalized to real-world smartphone addiction scenarios.

---

### 2.2 Data Integrity Violations

The analysis performed IQR outlier checks but never validated logical consistency between time-based columns. Programmatic verification reveals:

| Violation | Count | Percentage |
|-----------|-------|------------|
| `sleep_hours + daily_screen_time_hours + work_study_hours` > 24 | 156 rows | 2.08% |
| Maximum sum of above | 26.46 hours | — |
| `social_media_hours > daily_screen_time_hours` (part exceeds whole) | 685 rows | 9.13% |
| `work_study_hours > daily_screen_time_hours` (part exceeds whole) | 683 rows | 9.11% |
| `gaming_hours > daily_screen_time_hours` (part exceeds whole) | 111 rows | 1.48% |
| Sum of 3 components exceeds total screen time | 4,553 rows | 60.71% |

These violations confirm the synthetic origin (the generator did not enforce logical constraints) and raise questions about the physical meaning of the features. The analysis should have flagged these contradictions during the data quality phase and discussed their impact on feature interpretation.

---

### 2.3 Methodological Data Leakage in Preprocessing

The team applied `pd.get_dummies` and median imputation to the entire `df_prepared` dataframe before `train_test_split`.

- **Imputation leakage:** Computing the median on the full dataset (including the test portion) leaks distributional information from the held-out set into training. Imputation statistics must be fit only on `X_train`.
- **Encoding leakage:** `pd.get_dummies` applied before the split means the column structure is determined by the full dataset, not just the training data. In production, unseen categories would cause a shape mismatch crash.

**Mitigating factor:** The raw dataset has zero missing numeric values, so no actual median imputation occurred. However, the structural vulnerability remains.

**Fix:** Use `sklearn.impute.SimpleImputer` and `sklearn.preprocessing.OneHotEncoder` inside a `sklearn.pipeline.Pipeline`, fit only on training data, then transform test data.

---

## 3. Missing Analysis Components

### 3.1 No ROC-AUC Evaluation

ROC-AUC is the standard metric for binary classification model comparison, especially with imbalanced targets. The project reports accuracy, precision, recall, F1, and balanced accuracy, but entirely omits the ROC curve and AUC score. Without AUC, it is impossible to assess the model's discrimination ability across different classification thresholds or compare models at a threshold-independent level.

**Recommendation:** Plot ROC curves for all candidate models and report AUC scores alongside the existing metrics.

---

### 3.2 No Formal Statistical Hypothesis Testing

The EDA phase relies on descriptive summaries, grouped averages, and visualizations. The chi-square test was used for gender but no formal hypothesis tests (independent t-tests or Mann-Whitney U tests) were conducted to determine whether the observed differences between addicted and non-addicted groups are statistically significant for each numeric variable.

This is a notable gap for an academic data analytics project. Without hypothesis testing, the EDA conclusions are based on visual inspection and descriptive statistics alone, which cannot establish whether observed differences are real or due to sampling variation.

**Recommendation:** For each key numeric feature, run a two-sample test comparing `addicted_label = 0` vs `addicted_label = 1`, report p-values, and integrate results into the EDA narrative.

---

### 3.3 No Permutation Feature Importance

The analysis reports decision tree feature importance, which is known to be biased toward high-cardinality features and is specific to the trained model. Permutation importance provides a more robust, model-agnostic assessment by measuring the drop in performance when a feature is randomly shuffled.

**Recommendation:** Compute `sklearn.inspection.permutation_importance` on the test set for the final model and compare with the tree-based importance ranking.

---

### 3.4 No Repeated Cross-Validation or Stability Check

The entire pipeline uses a single `random_state=42` split and a single 5-fold CV run. Results could be sensitive to this particular partition. Repeated stratified k-fold CV or testing with different random seeds would confirm whether the reported metrics are stable or a fortunate split.

**Recommendation:** Use `RepeatedStratifiedKFold` (e.g., 5 folds × 3 repeats) for the final model and report mean ± std of all metrics.

---

### 3.5 No Age Subgroup Analysis

The dataset's age range is 18–35 (mean 26.57, std 5.20). While the narrow range itself limits generalizability (the model cannot speak to teenagers or older adults), the analysis never explored whether usage patterns or addiction rates differ by age bracket within this range. Age is included as a model feature but never examined descriptively beyond the correlation heatmap.

**Recommendation:** Create age bins (e.g., 18–22, 23–27, 28–35) and report usage statistics and addiction rates per group. This is a natural demographic dimension that the EDA should cover.

---

### 3.6 No Unsupervised Profiling or Clustering

The analysis is entirely focused on the binary `addicted_label`. No clustering or segmentation was performed to discover user types beyond the provided label. Unsupervised analysis (e.g., k-means or hierarchical clustering on usage features) could reveal subgroups that the binary label oversimplifies — for instance, users who are heavy on social media but light on gaming, or users with high screen time but low notification frequency.

**Recommendation:** Run a clustering analysis on the behavioral features, describe the discovered segments, and compare cluster assignments against `addicted_label` to assess whether the binary label captures meaningful behavioral diversity.

---

### 3.7 No Logistic Regression Coefficient Interpretation

Logistic regression was included as a candidate model but its coefficients were never interpreted. Reporting standardized coefficients would provide a direct measure of each feature's direction and magnitude of association with the target — information that is more interpretable than tree feature importance for stakeholder communication.

**Recommendation:** Fit a standardized logistic regression on the final feature set, report coefficients with their signs and magnitudes, and discuss which features increase vs. decrease the probability of the addicted label.

---

## 4. Minor Issues

### 4.1 Dummy Variable Trap
`pd.get_dummies` was called with `drop_first=False`, creating perfect multicollinearity. This has no practical impact on the decision tree or the L2-regularized logistic regression, but `drop_first=True` remains the correct standard for linear model interpretability.

### 4.2 Weekend Screen Time Ambiguity
The analysis notes that `weekend_screen_time` needs careful interpretation but never clarifies what it represents — average daily hours on weekends? Total weekend hours? A single weekend day? This ambiguity affects feature interpretation and should be resolved by referencing the data source documentation.

---

## 5. Summary of Required Actions

| Priority | Action |
|----------|--------|
| **Critical** | Explicitly state in the final report that the dataset is synthetic with deterministic targets; all performance metrics reflect reverse-engineering of the generator's rules, not real-world predictive power |
| **Critical** | Add 24-hour day and component-vs-total validation checks to the data quality phase; discuss the physical contradictions and their implications for feature meaning |
| **High** | Add ROC-AUC curves and AUC scores for all candidate models |
| **High** | Add formal hypothesis tests (t-tests or Mann-Whitney U) for key numeric variables between addicted and non-addicted groups |
| **High** | Fix preprocessing leakage by moving imputation and encoding inside sklearn Pipelines fit only on training data |
| **Medium** | Compute permutation feature importance on the test set for the final model |
| **Medium** | Use `RepeatedStratifiedKFold` to validate metric stability |
| **Medium** | Add age subgroup analysis (bins: 18–22, 23–27, 28–35) |
| **Medium** | Report and interpret logistic regression coefficients |
| **Low** | Add clustering/segmentation analysis to discover user types beyond the binary label |
| **Low** | Use `drop_first=True` in `pd.get_dummies` |
| **Low** | Clarify the definition of `weekend_screen_time` |

---

## 6. Overall Assessment

The analysis demonstrates strong methodological instincts — leakage prevention, imbalanced data handling, strict train/test separation, and parsimony. These are the hardest things to get right in a data science project and the team handled them well.

However, two fundamental problems undermine confidence in the results: (1) the dataset is synthetic with deterministic targets, meaning the model's impressive performance reflects rule reproduction rather than genuine prediction; and (2) several standard analysis components are missing, particularly ROC-AUC, hypothesis testing, and temporal consistency validation. These gaps prevent the analysis from reaching the completeness expected of an academic data analytics project.

The process is sound. The conclusions require heavy qualification. Addressing the items above — especially the synthetic-data disclosure and the missing evaluation components — will significantly strengthen the final deliverable.
