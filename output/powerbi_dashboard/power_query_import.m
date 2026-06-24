let
    Source = Csv.Document(
        File.Contents("C:\Degree\A252\Data Analytics\Data_Analytics_Project\output\powerbi_dashboard\smartphone_usage_powerbi_model.csv"),
        [Delimiter=",", Columns=25, Encoding=65001, QuoteStyle=QuoteStyle.Csv]
    ),
    PromotedHeaders = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    ChangedType = Table.TransformColumnTypes(
        PromotedHeaders,
        {
            {"transaction_id", type text},
            {"user_id", type text},
            {"age", Int64.Type},
            {"age_group", type text},
            {"gender", type text},
            {"daily_screen_time_hours", type number},
            {"screen_time_band", type text},
            {"social_media_hours", type number},
            {"gaming_hours", type number},
            {"work_study_hours", type number},
            {"sleep_hours", type number},
            {"low_sleep_flag", type logical},
            {"notifications_per_day", Int64.Type},
            {"app_opens_per_day", Int64.Type},
            {"weekend_screen_time", type number},
            {"weekend_to_daily_screen_time_ratio", type number},
            {"stress_level", type text},
            {"academic_work_impact", type text},
            {"addiction_level_display", type text},
            {"addicted_label", Int64.Type},
            {"addiction_status", type text},
            {"usage_component_total_hours", type number},
            {"component_sum_exceeds_daily_screen_time", type logical},
            {"day_load_hours", type number},
            {"day_load_exceeds_24_hours", type logical}
        }
    )
in
    ChangedType
