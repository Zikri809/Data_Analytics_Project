from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv"
OUT = ROOT / "output" / "powerbi_dashboard"


def pct(series):
    return (series.mean() * 100).round(2)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SOURCE, keep_default_na=False, na_values=[""])
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    df["addiction_level_display"] = df["addiction_level"].replace("", "Missing/blank")
    df["addiction_status"] = np.where(
        df["addicted_label"].eq(1),
        "1 - Addicted label",
        "0 - Not addicted label",
    )
    df["age_group"] = pd.cut(
        df["age"],
        bins=[17, 22, 27, 35],
        labels=["18-22", "23-27", "28-35"],
        include_lowest=True,
    ).astype(str)
    df["screen_time_band"] = pd.cut(
        df["daily_screen_time_hours"],
        bins=[0, 6, 8, 10, 24],
        labels=["Up to 6h", "6-8h", "8-10h", "10h+"],
        include_lowest=True,
    ).astype(str)
    df["usage_component_total_hours"] = (
        df["social_media_hours"] + df["gaming_hours"] + df["work_study_hours"]
    )
    df["component_sum_exceeds_daily_screen_time"] = (
        df["usage_component_total_hours"] > df["daily_screen_time_hours"]
    )
    df["day_load_hours"] = (
        df["sleep_hours"] + df["daily_screen_time_hours"] + df["work_study_hours"]
    )
    df["day_load_exceeds_24_hours"] = df["day_load_hours"] > 24
    df["weekend_to_daily_screen_time_ratio"] = (
        df["weekend_screen_time"] / df["daily_screen_time_hours"]
    ).round(4)
    df["low_sleep_flag"] = df["sleep_hours"] < 7

    model_cols = [
        "transaction_id",
        "user_id",
        "age",
        "age_group",
        "gender",
        "daily_screen_time_hours",
        "screen_time_band",
        "social_media_hours",
        "gaming_hours",
        "work_study_hours",
        "sleep_hours",
        "low_sleep_flag",
        "notifications_per_day",
        "app_opens_per_day",
        "weekend_screen_time",
        "weekend_to_daily_screen_time_ratio",
        "stress_level",
        "academic_work_impact",
        "addiction_level_display",
        "addicted_label",
        "addiction_status",
        "usage_component_total_hours",
        "component_sum_exceeds_daily_screen_time",
        "day_load_hours",
        "day_load_exceeds_24_hours",
    ]
    df[model_cols].to_csv(OUT / "smartphone_usage_powerbi_model.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "metric": "Rows analyzed",
                "value": len(df),
                "display_value": f"{len(df):,}",
                "note": "Full source CSV after corrected loading.",
            },
            {
                "metric": "Addicted-label share",
                "value": pct(df["addicted_label"]),
                "display_value": f"{pct(df['addicted_label']):.2f}%",
                "note": "Share of rows where addicted_label = 1.",
            },
            {
                "metric": "Average daily screen time",
                "value": round(df["daily_screen_time_hours"].mean(), 2),
                "display_value": f"{df['daily_screen_time_hours'].mean():.2f}h",
                "note": "Mean daily_screen_time_hours.",
            },
            {
                "metric": "Average social media hours",
                "value": round(df["social_media_hours"].mean(), 2),
                "display_value": f"{df['social_media_hours'].mean():.2f}h",
                "note": "Mean social_media_hours.",
            },
            {
                "metric": "Component sum exceeds daily screen time",
                "value": pct(df["component_sum_exceeds_daily_screen_time"]),
                "display_value": f"{pct(df['component_sum_exceeds_daily_screen_time']):.2f}%",
                "note": "Rows where social + gaming + work/study exceeds daily screen time.",
            },
            {
                "metric": "Day load exceeds 24 hours",
                "value": pct(df["day_load_exceeds_24_hours"]),
                "display_value": f"{pct(df['day_load_exceeds_24_hours']):.2f}%",
                "note": "Rows where sleep + screen time + work/study exceeds 24 hours.",
            },
            {
                "metric": "Final model accuracy",
                "value": 93.53,
                "display_value": "93.53%",
                "note": "Recommended tuned decision tree from Analysis.ipynb.",
            },
            {
                "metric": "Final model F1 score",
                "value": 95.48,
                "display_value": "95.48%",
                "note": "Positive-class F1 from Analysis.ipynb.",
            },
        ]
    )
    summary.to_csv(OUT / "dashboard_kpis.csv", index=False)

    usage_cols = [
        "daily_screen_time_hours",
        "weekend_screen_time",
        "social_media_hours",
        "gaming_hours",
        "work_study_hours",
        "sleep_hours",
        "notifications_per_day",
        "app_opens_per_day",
    ]
    by_addiction = (
        df.groupby(["addiction_status"], observed=True)
        .agg(
            rows=("transaction_id", "count"),
            addicted_rate=("addicted_label", "mean"),
            **{f"avg_{col}": (col, "mean") for col in usage_cols},
        )
        .reset_index()
    )
    by_addiction["addicted_rate"] = (by_addiction["addicted_rate"] * 100).round(2)
    for col in by_addiction.select_dtypes(include="number").columns:
        by_addiction[col] = by_addiction[col].round(2)
    by_addiction.to_csv(OUT / "usage_by_addiction_status.csv", index=False)

    def grouped_table(dim):
        grouped = (
            df.groupby(dim, observed=True)
            .agg(
                rows=("transaction_id", "count"),
                addicted_rate=("addicted_label", "mean"),
                avg_daily_screen_time_hours=("daily_screen_time_hours", "mean"),
                avg_weekend_screen_time=("weekend_screen_time", "mean"),
                avg_social_media_hours=("social_media_hours", "mean"),
                avg_sleep_hours=("sleep_hours", "mean"),
            )
            .reset_index()
        )
        grouped["addicted_rate"] = (grouped["addicted_rate"] * 100).round(2)
        for col in grouped.select_dtypes(include="number").columns:
            grouped[col] = grouped[col].round(2)
        return grouped

    grouped_table("age_group").to_csv(OUT / "usage_by_age_group.csv", index=False)
    grouped_table("gender").to_csv(OUT / "usage_by_gender.csv", index=False)
    grouped_table("stress_level").to_csv(OUT / "usage_by_stress_level.csv", index=False)
    grouped_table("screen_time_band").to_csv(OUT / "usage_by_screen_time_band.csv", index=False)

    feature_importance = pd.DataFrame(
        [
            {"feature": "daily_screen_time_hours", "importance": 0.5712},
            {"feature": "social_media_hours", "importance": 0.4253},
            {"feature": "sleep_hours", "importance": 0.0034},
            {"feature": "age", "importance": 0.0},
            {"feature": "gaming_hours", "importance": 0.0},
            {"feature": "work_study_hours", "importance": 0.0},
            {"feature": "notifications_per_day", "importance": 0.0},
            {"feature": "app_opens_per_day", "importance": 0.0},
            {"feature": "weekend_screen_time", "importance": 0.0},
            {"feature": "low_sleep_flag", "importance": 0.0},
        ]
    )
    feature_importance.to_csv(OUT / "final_model_feature_importance.csv", index=False)

    confusion = pd.DataFrame(
        [
            {"actual": "0 - Not addicted label", "predicted": "0 - Not addicted label", "count": 379},
            {"actual": "0 - Not addicted label", "predicted": "1 - Addicted label", "count": 59},
            {"actual": "1 - Addicted label", "predicted": "0 - Not addicted label", "count": 38},
            {"actual": "1 - Addicted label", "predicted": "1 - Addicted label", "count": 1024},
        ]
    )
    confusion.to_csv(OUT / "final_model_confusion_matrix.csv", index=False)

    data_dictionary = pd.DataFrame(
        [
            {"field": col, "description": desc}
            for col, desc in {
                "addicted_label": "Binary target from source data; 1 means addicted label, 0 means not addicted label.",
                "addiction_level_display": "Original severity category; None is a real category, not missing data.",
                "addiction_status": "Dashboard-friendly label derived from addicted_label.",
                "age_group": "Age binned as 18-22, 23-27, and 28-35.",
                "screen_time_band": "Daily screen time grouped for filtering and comparison.",
                "usage_component_total_hours": "social_media_hours + gaming_hours + work_study_hours.",
                "component_sum_exceeds_daily_screen_time": "True when component hours exceed daily_screen_time_hours.",
                "day_load_exceeds_24_hours": "True when sleep + daily screen time + work/study exceeds 24 hours.",
                "weekend_to_daily_screen_time_ratio": "weekend_screen_time divided by daily_screen_time_hours.",
                "low_sleep_flag": "True when sleep_hours is below 7.",
            }.items()
        ]
    )
    data_dictionary.to_csv(OUT / "data_dictionary.csv", index=False)


if __name__ == "__main__":
    main()
