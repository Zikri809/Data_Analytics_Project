# Senior Analysis Validation Review

## Overall Assessment: Needs revision before final submission

The notebook is directionally strong and executes successfully from top to bottom when run into `tmp/validation_run.ipynb`. The main leakage rule is followed, the target is stratified, a baseline is included, and the EDA conclusions mostly use appropriate association/prediction language.

The analysis should still be revised before submission because several decisions need tighter evidence, caveats, or methodology fixes. The largest issues are the missing final summary, under-scoped binary-gender filtering, deterministic target-proxy behavior in `addiction_level`, and model selection using the held-out test set before reporting final test metrics.

Sources reviewed:

- `Analysis.ipynb`
- `SMARTPHONE_USAGE_ANALYSIS_PLAN.md`
- `Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv`
- Smartphone semantic-layer guidance
- Independent pandas/sklearn recomputation from the CSV
- Temporary top-to-bottom execution copy: `tmp/validation_run.ipynb`

## What Is Correct And Evidence-Backed

1. **Dataset loading and basic profile are correct.**
   - Raw CSV has 7,500 rows and 16 columns.
   - There are 0 duplicate rows, 0 duplicate `transaction_id`, and 0 duplicate `user_id`.
   - The only raw missingness is `addiction_level` with 819 missing values.

2. **Binary-gender cleaning follows the assignment rule.**
   - Raw gender counts: `Male = 2,553`, `Female = 2,461`, `Other = 2,486`.
   - The notebook keeps `Male` and `Female`, leaving 5,014 rows.
   - This is plan-compliant, but it must be described as an assignment scope decision, not a general data-quality truth.

3. **Main leakage exclusions are correct.**
   - The model excludes `transaction_id`, `user_id`, `addicted_label`, `addiction_level`, and original text `gender`.
   - This is the most important modeling decision and it is handled correctly.

4. **The EDA direction is supported by the data.**
   - In the prepared data, `addicted_label = 1` is 71.24% of rows and `0` is 28.76%.
   - Mean daily screen time is higher for label `1`: 8.46 hours vs 5.12.
   - Mean weekend screen time is higher for label `1`: 10.21 vs 6.87.
   - Mean social media hours is higher for label `1`: 3.69 vs 2.27.
   - Stress level, academic/work impact, and gender show weak descriptive separation.

5. **The final selected model genuinely beats the baseline.**
   - Majority baseline test F1 for class `1`: 0.8324, with 0 true negatives.
   - Tuned decision tree test metrics independently reproduced: accuracy 0.9322, balanced accuracy 0.9245, F1 0.9520.
   - Confusion matrix independently reproduced: `[[261, 27], [41, 674]]`.

## High-Priority Issues

### 1. Required final summary section is missing

The plan requires `## 5. Final Summary and Limitations - All`, but the notebook currently ends with `### Person 4 Handoff to Final Summary`.

Impact: The notebook is incomplete against the stated project checklist even though the analytical work is mostly present.

Fix:

- Add the final section after Person 4.
- Include dataset summary, cleaning summary, preprocessing summary, main EDA findings, model findings, limitations, and future improvements.
- State that findings apply to the retained `Male/Female` project subset.
- State that the dataset is observational, not clinical/diagnostic, and not causal.
- State that `weekend_screen_time` needs careful interpretation.

### 2. The binary-gender exclusion removes 33.15% of the data and is under-scoped in the narrative

The notebook removes 2,486 `Other` rows, reducing the dataset from 7,500 to 5,014 rows. That is a large population exclusion.

The removed `Other` rows have 69.83% `addicted_label = 1`, which is close to the retained subset's 71.24%, so the target balance is not dramatically changed. Still, downstream claims no longer describe the full raw dataset.

Impact: Any broad wording like "users in the dataset" can overgeneralize. The correct population is the binary-gender project subset.

Fix:

- Reword "invalid gender" as "excluded from this binary-gender project scope."
- Add a short sensitivity note: removed `Other` rows had similar target balance, but all final EDA/modeling results are based only on retained `Male/Female` rows.
- Use "prepared subset" or "retained project subset" in the final summary.

### 3. `addiction_level` is a deterministic target proxy, not just an ordinary categorical variable

After cleaning and imputation:

| addiction_level | addicted_label = 0 | addicted_label = 1 |
| --- | ---: | ---: |
| Unknown | 547 | 0 |
| Mild | 895 | 0 |
| Moderate | 0 | 1,962 |
| Severe | 0 | 1,610 |

This means `addiction_level` perfectly maps to the target in the prepared subset. The notebook correctly excludes it from modeling, but the EDA should frame it as a data-definition/leakage warning rather than as a normal explanatory category.

Impact: Readers may mistake `addiction_level` as evidence explaining addiction, when it is effectively a label proxy.

Fix:

- Add the cross-tab above in the EDA or preprocessing section.
- Strengthen wording from "may leak target information" to "behaves as a deterministic target proxy in this dataset."
- Keep `Unknown` only for descriptive data-quality accounting, not substantive behavioral interpretation.

### 4. Model family selection uses the held-out test set

The notebook trains multiple candidate models, evaluates them on `X_test`, selects the decision tree because it has the strongest test-set F1, tunes the tree on training folds, and then reports performance on the same `X_test`.

Impact: The final test metrics are likely optimistic because the test set influenced model-family selection.

Fix:

- Use training-only cross-validation to select the model family.
- Tune the selected family inside training data.
- Evaluate once on the untouched test set at the end.
- Alternatively use train/validation/test split: train for fitting, validation for model selection/tuning, test for final reporting.

Important nuance: an independent 5-fold CV on the training data still ranked the decision tree first, so this issue does not invalidate the chosen model. It mainly affects how confidently the final test score should be presented.

### 5. Positive-class F1 is not enough for an imbalanced target

The target is 71.24% class `1`. Optimizing only default binary F1 emphasizes the majority positive class. The majority baseline already gets class-1 F1 of 0.8324 while never identifying class `0`.

Impact: A model can look strong while under-serving the minority class.

Fix:

- Add macro-F1, balanced accuracy, class-0 recall/specificity, and class-level precision/recall to model comparison.
- Tune with a metric aligned to the project objective. If both labels matter, use macro-F1 or balanced accuracy.
- Keep the confusion matrix interpretation, because it is already useful.

## Medium-Priority Issues

### 6. `usage_component_total_hours` is semantically questionable

The engineered feature `social_media_hours + gaming_hours + work_study_hours` exceeds `daily_screen_time_hours` in 3,020 of 5,014 rows, or 60.23% of the prepared data. For label `0`, this happens in 81.76% of rows.

Impact: The feature cannot safely be interpreted as total screen time unless the component measures are known to be non-overlapping and measured on the same basis.

Fix:

- Rename it to `sum_reported_component_hours`.
- Add a caveat that components may overlap or use different reporting logic.
- Include a sensitivity model without this aggregate. In my independent rerun, removing aggregate/ratio features barely changed the tuned-tree result: F1 0.9498 vs 0.9520, so the model does not depend heavily on it.

### 7. Ratio features are mechanically safe but analytically easy to overread

There are 0 zero `daily_screen_time_hours` values, so the division-by-zero guard is safe but unused. The ratio features correlate moderately negatively with the target, around -0.329, largely because the denominator is daily screen time, one of the strongest target-related variables.

Impact: These ratios may reflect transformed screen-time effects rather than independent behavioral intensity.

Fix:

- Keep the features if useful, but call them derived intensity measures.
- Do not describe them as independent evidence of notification/app-opening behavior.
- Consider excluding them from a simple final model if interpretability matters more than a tiny metric gain.

### 8. EDA chart wording sometimes points to evidence from later charts

The distribution chart cell shows ungrouped histograms for daily screen time and sleep. The markdown below says screen-time variables have clearer separation by `addicted_label`, but that separation is supported by later grouped means/boxplots, not the ungrouped histogram itself.

Impact: The reader may think the displayed chart proves a grouped claim it does not directly show.

Fix:

- Move that sentence to the comparison-chart section, or add `hue='addicted_label'` or faceted histograms.
- For categorical comparisons, use normalized stacked bars because the target class is imbalanced.

### 9. Preprocessing should be inside a train-only pipeline for stronger methodology

The current one-hot encoding is applied before train/test split. For this particular dataset it is low risk because categories are simple and known, but best practice is to fit encoders/scalers/imputers only on training folds.

Impact: This is more of a methodology weakness than a likely source of material leakage here.

Fix:

- Use `ColumnTransformer` and `Pipeline`.
- Use `OneHotEncoder(handle_unknown='ignore')`.
- Put imputation, encoding, scaling, model fitting, and tuning inside cross-validation.

### 10. Outlier claim is not fully proven

The notebook says there are no obvious extreme outlier problems based on distribution charts. Numeric ranges look plausible and there are no negative values, but no IQR or rule-based outlier check is shown.

Impact: The statement is probably fine, but it is not fully backed by a visible check.

Fix:

- Add a compact min/max/IQR table for numeric fields.
- Phrase as "no obvious invalid numeric ranges were found" unless an explicit outlier test is added.

## Low-Priority Issues

### 11. `df_raw` is lightly mutated during cleaning checks

Cell 3 strips column names and categorical strings directly on `df_raw` before `df_clean` is created.

Impact: Low. The transformations are harmless here, but `df_raw` no longer represents the exact loaded file after cell 3.

Fix:

- Keep `df_raw` untouched after loading.
- Create `df_working = df_raw.copy()` before stripping and standardizing.

### 12. Feature importance should be treated as model-specific

The notebook already says this, which is good. The caveat could be stronger because tree impurity importances can be unstable and biased when features are correlated.

Impact: Readers may overinterpret `social_media_hours` and `daily_screen_time_hours` as globally "most important" rather than important to this tuned tree.

Fix:

- Add permutation importance on the test set or cross-validation folds.
- Consider grouped/drop-column importance for correlated screen-time feature families.

## Suggested Fix Order

1. Add the missing final summary and limitations section.
2. Reword gender filtering as scope exclusion and carry that scope through all final claims.
3. Add the `addiction_level` by `addicted_label` cross-tab and call it a deterministic target proxy.
4. Redo model selection with training-only CV or add a clear caveat that first-pass model selection used the test set.
5. Add macro-F1, balanced accuracy, and class-0 recall to model comparison.
6. Rename/caveat `usage_component_total_hours` and ratio features.
7. Add a compact outlier/range check.
8. Move preprocessing into a pipeline if time permits.

## Recommended Final Summary Skeleton

```markdown
## 5. Final Summary and Limitations - All

This analysis used the smartphone usage dataset to examine patterns associated with `addicted_label` and to test whether the label can be predicted using non-leaking features. After applying the project's binary-gender scope rule, the prepared analysis used 5,014 retained `Male/Female` rows from the original 7,500 rows.

Cleaning found no duplicate rows, no duplicate `transaction_id`, and no duplicate `user_id`. The main cleaning decision was excluding 2,486 `Other` gender rows because the assignment scope required binary gender. This limits generalization: all downstream EDA and modeling results apply to the retained project subset, not the full raw dataset.

Preprocessing encoded gender as `Female = 0` and `Male = 1`, filled missing `addiction_level` as `Unknown` for descriptive tracking, and created engineered usage features. `addiction_level` was excluded from modeling because it behaves as a deterministic target proxy: `Unknown` and `Mild` align with label `0`, while `Moderate` and `Severe` align with label `1`.

The strongest descriptive differences were in daily screen time, weekend screen time, and social media hours. In the prepared subset, users with `addicted_label = 1` averaged 8.46 daily screen-time hours versus 5.12 for label `0`, and 3.69 social-media hours versus 2.27. Stress level, academic/work impact, gender, sleep hours, notifications, and app opens showed weaker separation.

For predictive modeling, the notebook excluded identifiers, the target, original text gender, and `addiction_level`. The tuned decision tree performed much better than the majority-class baseline, with about 93.22% accuracy and 95.20% class-1 F1 on the held-out test set. These metrics should be interpreted as predictive performance for this dataset split, not as proof of causation or clinical diagnosis.

Important limitations: the dataset is observational, the source and real-world representativeness are not verified, `weekend_screen_time` needs careful interpretation, the binary-gender scope excludes one-third of the raw rows, and the test set was used during first-pass model selection. Future improvements should use training-only cross-validation for model selection, report macro-F1 and balanced accuracy, add sensitivity checks for engineered features, and validate whether the component-hour fields are overlapping.
```

