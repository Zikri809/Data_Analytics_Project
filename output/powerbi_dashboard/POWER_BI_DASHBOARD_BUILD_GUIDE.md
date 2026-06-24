# Smartphone Usage And Addiction Power BI Dashboard

This folder contains a Power BI-ready dashboard package for the classroom project. The dashboard is designed to satisfy the technical rubric areas: data preparation, visualization/dashboard design, descriptive analytics, and predictive analytics.

## Files

- `smartphone_usage_powerbi_model.csv`: cleaned and enriched dashboard fact table.
- `dashboard_kpis.csv`: headline KPI values for quick validation.
- `usage_by_addiction_status.csv`: grouped usage comparison for addicted vs non-addicted labels.
- `usage_by_age_group.csv`, `usage_by_gender.csv`, `usage_by_stress_level.csv`, `usage_by_screen_time_band.csv`: supporting dashboard breakdowns.
- `final_model_feature_importance.csv`: final decision-tree feature importance from `Analysis.ipynb`.
- `final_model_confusion_matrix.csv`: final model confusion matrix from `Analysis.ipynb`.
- `power_query_import.m`: optional Advanced Editor query for importing the main model table.
- `powerbi_measures.dax`: measures to add in Power BI.
- `smartphone_dashboard_theme.json`: optional Power BI theme.

## Build Steps In Power BI Desktop

1. Open Power BI Desktop.
2. Choose **Get Data > Text/CSV**.
3. Import `smartphone_usage_powerbi_model.csv`.
4. Import the supporting CSVs in this folder.
5. In **Transform Data**, confirm that `addiction_level_display = None` is text, not blank.
6. Apply changes.
7. Import the theme from **View > Themes > Browse for themes** and select `smartphone_dashboard_theme.json`.
8. Create the measures from `powerbi_measures.dax` on the main table.

## Page 1: Executive Overview

Purpose: show the story in the first screen.

Recommended visuals:

- Cards: `Rows Analyzed`, `Addicted Label Share`, `Average Daily Screen Time`, `Average Social Media Hours`.
- Clustered bar chart: average daily screen time, weekend screen time, and social media hours by `addiction_status`.
- Donut chart: row count by `addiction_status`.
- Slicers: `age_group`, `gender`, `stress_level`, `screen_time_band`.

Key message: the addicted-label group has much higher daily screen time, weekend screen time, and social media hours.

## Page 2: Usage Drivers

Purpose: support the descriptive analytics mark.

Recommended visuals:

- Bar chart: addicted-label share by `screen_time_band`.
- Bar chart: average screen time metrics by `age_group`.
- Bar chart: average screen time metrics by `gender`.
- Matrix: `age_group`, `gender`, row count, addicted-label share, average daily screen time, average social media hours, average sleep hours.

Key message: screen-time-related fields explain most of the visible separation; gender and stress level should be described cautiously.

## Page 3: Data Quality And Caveats

Purpose: show real-world analytical thinking.

Recommended visuals:

- Cards: `Component Sum Issue Share`, `Day Load Over 24h Share`.
- Bar chart: count of component-hour issues by `addiction_status`.
- Table: `data_dictionary.csv`.
- Text box: "The target appears rule-defined/synthetic. Treat model scores as reconstruction of the dataset label, not clinical diagnosis."

Key message: the dashboard is transparent about data quality issues and target-definition limitations.

## Page 4: Predictive Model

Purpose: satisfy the predictive analytics part of the rubric.

Recommended visuals:

- Cards: `Final Model Accuracy`, `Final Model F1 Score`, `Final Model Balanced Accuracy`, `Final Model Macro F1`.
- Matrix or heatmap-style table: `final_model_confusion_matrix.csv`.
- Bar chart: `final_model_feature_importance.csv`, sorted descending by `importance`.
- Text box: "Final model: tuned decision tree without aggregate/ratio engineered features."

Key message: the model performs strongly but mainly reconstructs a deterministic/rule-defined target using screen-time and social-media variables.

## Validation Checklist

- Row count in Power BI is `7,500`.
- Addicted-label share is `70.77%`.
- Average daily screen time is `7.50h`.
- Component sum issue share is `60.71%`.
- Day-load-over-24h issue share is `2.08%`.
- Final model metrics match `Analysis.ipynb`: accuracy `93.53%`, F1 `95.48%`, macro-F1 `92.07%`.

## Important Presentation Caveat

Do not say the dashboard proves smartphone addiction causes or is caused by any feature. Say the features are associated with, related to, or predictive of the dataset label.
