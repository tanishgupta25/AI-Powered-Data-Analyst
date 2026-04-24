import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px

from utils import (
    aggregate_for_display,
    build_categorical_summary,
    build_dataset_overview,
    build_key_insights,
    build_numeric_summary,
    CleaningSummary,
    create_query_chart,
    DatasetProfile,
    create_default_chart,
    detect_aggregation,
    generate_full_report_charts,
    analyze_relationships,
    pick_groupby_column,
    pick_relevant_date_column,
    pick_relevant_numeric_column,
    perform_automated_cleaning,
    ChartArtifact,
)

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

try:
    from pandasai import SmartDataframe
    from pandasai.llm.openai import OpenAI as PandasAIOpenAI
except ImportError:  # pragma: no cover
    SmartDataframe = None
    PandasAIOpenAI = None


@dataclass
class AnalysisResult:
    title: str
    result_text: str
    explanation: str
    report_markdown: str
    chart: Optional[Any] = None
    table: Optional[pd.DataFrame] = None
    confidence: float = 0.9
    warnings: Optional[List[str]] = None


@dataclass
class FullReportResult:
    cleaned_df: pd.DataFrame
    cleaned_profile: DatasetProfile
    dataset_overview: List[str]
    cleaning_summary: CleaningSummary
    numeric_summary: pd.DataFrame
    categorical_summary: pd.DataFrame
    correlation_matrix: pd.DataFrame
    relationship_lines: List[str]
    charts: List[ChartArtifact]
    insights: List[str]
    conclusion: str
    report_markdown: str
    action_log: List[str]


class AIAnalystEngine:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.client = OpenAI(api_key=self.api_key) if self.api_key and OpenAI else None

    @property
    def llm_available(self) -> bool:
        return self.client is not None

    @property
    def pandasai_available(self) -> bool:
        return bool(self.api_key and SmartDataframe and PandasAIOpenAI)

    def answer_query(self, df: pd.DataFrame, profile: DatasetProfile, query: str) -> AnalysisResult:
        query_lower = query.lower().strip()
        warnings: List[str] = []

        if not query_lower:
            return AnalysisResult(
                title="Empty Query",
                result_text="Please enter a question about your data.",
                explanation="The app needs a query to analyze the dataset.",
                report_markdown="## Please ask a question\nTry examples like `Total sales` or `Generate report`.",
                confidence=1.0,
                warnings=[],
            )

        if self._is_report_query(query_lower):
            result = self._generate_full_report(df, profile, query)
        elif "correlation" in query_lower:
            result = self._correlation_analysis(df, query)
        elif any(term in query_lower for term in ["count", "how many", "number of", "frequency"]):
            result = self._count_analysis(df, profile, query)
        elif any(term in query_lower for term in ["top", "highest", "best"]):
            result = self._top_n_analysis(df, query)
        elif any(term in query_lower for term in ["trend", "month-wise", "growth", "last 3 months", "compare"]):
            result = self._trend_analysis(df, query)
        elif any(term in query_lower for term in ["wise", " by ", "breakdown"]):
            result = self._grouped_analysis(df, query)
        elif any(term in query_lower for term in ["average", "mean", "avg"]):
            result = self._average_analysis(df, profile, query)
        elif any(term in query_lower for term in ["total", "sum"]):
            result = self._total_analysis(df, profile, query)
        elif any(term in query_lower for term in ["plot", "graph", "chart", "visualize", "distribution"]):
            result = self._visual_analysis(df, profile, query)
        else:
            result = self._fallback_analysis(df, profile, query)
            warnings.append("The query was interpreted using a best-effort fallback. Please verify the result.")

        result.warnings = (result.warnings or []) + warnings
        return result

    def _is_report_query(self, query_lower: str) -> bool:
        report_terms = ["generate report", "full analysis", "sales report", "report", "analysis summary"]
        return any(term in query_lower for term in report_terms)

    def _total_analysis(self, df: pd.DataFrame, profile: DatasetProfile, query: str) -> AnalysisResult:
        measure = pick_relevant_numeric_column(df, query)
        if not measure:
            return self._clarification_result("I couldn't find a numeric column for this total. Try mentioning the measure, like `Total sales`.")

        total_value = df[measure].sum()
        explanation = f"The total {measure.replace('_', ' ')} across the current filtered dataset is {total_value:,.2f}."
        report = (
            f"## Result\n{explanation}\n\n"
            f"## Explanation\n- Metric used: `{measure}`\n- Aggregation: sum across {len(df):,} rows\n"
        )
        return AnalysisResult(
            title=f"Total {measure.replace('_', ' ').title()}",
            result_text=f"{total_value:,.2f}",
            explanation=explanation,
            report_markdown=report,
            chart=create_query_chart(df, profile, query, measure=measure),
            confidence=0.98,
        )

    def _average_analysis(self, df: pd.DataFrame, profile: DatasetProfile, query: str) -> AnalysisResult:
        measure = pick_relevant_numeric_column(df, query)
        if not measure:
            return self._clarification_result("I couldn't find a numeric column for this average. Try `Average profit` or `Average sales`.")

        avg_value = df[measure].mean()
        explanation = f"The average {measure.replace('_', ' ')} is {avg_value:,.2f}."
        report = (
            f"## Result\n{explanation}\n\n"
            f"## Explanation\n- Metric used: `{measure}`\n- Aggregation: mean across {len(df):,} rows\n"
        )
        return AnalysisResult(
            title=f"Average {measure.replace('_', ' ').title()}",
            result_text=f"{avg_value:,.2f}",
            explanation=explanation,
            report_markdown=report,
            chart=create_query_chart(df, profile, query, measure=measure),
            confidence=0.98,
        )

    def _top_n_analysis(self, df: pd.DataFrame, query: str) -> AnalysisResult:
        group_col = pick_groupby_column(df, query)
        measure = pick_relevant_numeric_column(df, query)
        top_n = self._extract_number(query) or 5
        aggregation = detect_aggregation(query, measure)

        if not group_col:
            return self._clarification_result("I couldn't identify what to rank. Try `Top 5 products` or `Top 5 regions by sales`.")

        if measure and group_col != measure and aggregation != "count":
            table, value_col = aggregate_for_display(df, group_col, measure, aggregation)
            table = table.head(top_n)
        else:
            table, value_col = aggregate_for_display(df, group_col, measure, "count")
            table = table.head(top_n)

        chart = px.bar(table, x=group_col, y=value_col, title=f"Top {top_n} {group_col.replace('_', ' ').title()}")
        explanation = f"These are the top {top_n} {group_col.replace('_', ' ')} based on {value_col.replace('_', ' ')}."
        report = (
            f"## Result\n{explanation}\n\n"
            f"## Top {top_n}\n{table.to_markdown(index=False)}\n"
        )
        return AnalysisResult(
            title=f"Top {top_n} {group_col.replace('_', ' ').title()}",
            result_text=table.to_markdown(index=False),
            explanation=explanation,
            report_markdown=report,
            chart=chart,
            table=table,
            confidence=0.95,
        )

    def _count_analysis(self, df: pd.DataFrame, profile: DatasetProfile, query: str) -> AnalysisResult:
        group_col = pick_groupby_column(df, query)

        if group_col and any(term in query.lower() for term in ["by", "wise", "distribution", "frequency"]):
            table, value_col = aggregate_for_display(df, group_col, None, "count")
            chart = px.bar(table.head(15), x=group_col, y=value_col, title=f"Count by {group_col.replace('_', ' ').title()}")
            explanation = f"This shows the number of records in each {group_col.replace('_', ' ')} group."
            return AnalysisResult(
                title=f"Count by {group_col.replace('_', ' ').title()}",
                result_text=table.head(15).to_markdown(index=False),
                explanation=explanation,
                report_markdown=f"## Count Analysis\n{explanation}\n\n{table.head(20).to_markdown(index=False)}",
                chart=chart,
                table=table,
                confidence=0.95,
            )

        total_count = len(df)
        explanation = f"The current filtered dataset contains {total_count:,} records."
        return AnalysisResult(
            title="Record Count",
            result_text=f"{total_count:,}",
            explanation=explanation,
            report_markdown=f"## Count Analysis\n{explanation}",
            chart=create_query_chart(df, profile, "count distribution"),
            confidence=0.98,
        )

    def _grouped_analysis(self, df: pd.DataFrame, query: str) -> AnalysisResult:
        group_col = pick_groupby_column(df, query)
        measure = pick_relevant_numeric_column(df, query)
        if not group_col or not measure:
            return self._clarification_result("I need both a category and a numeric measure. Try `Region wise sales`.")

        aggregation = detect_aggregation(query, measure)
        table, value_col = aggregate_for_display(df, group_col, measure, aggregation)
        chart = px.bar(table.head(15), x=group_col, y=value_col, title=f"{value_col.replace('_', ' ').title()} by {group_col.replace('_', ' ').title()}")
        explanation = f"This shows the {aggregation} {measure.replace('_', ' ')} by {group_col.replace('_', ' ')}."
        report = (
            f"## Breakdown\n{explanation}\n\n"
            f"{table.head(20).to_markdown(index=False)}\n"
        )
        return AnalysisResult(
            title=f"{group_col.replace('_', ' ').title()} Wise {measure.replace('_', ' ').title()}",
            result_text=table.head(10).to_markdown(index=False),
            explanation=explanation,
            report_markdown=report,
            chart=chart,
            table=table,
            confidence=0.94,
        )

    def _trend_analysis(self, df: pd.DataFrame, query: str) -> AnalysisResult:
        date_col = pick_relevant_date_column(df, query)
        measure = pick_relevant_numeric_column(df, query)
        if not date_col or not measure:
            return self._clarification_result("I need a date column and a numeric metric for trend analysis. Try `Show sales trend by date`.")

        freq = "M"
        if "week" in query.lower():
            freq = "W"
        elif "day" in query.lower():
            freq = "D"
        aggregation = detect_aggregation(query, measure)

        trend_df = (
            df.assign(period=df[date_col].dt.to_period(freq).dt.to_timestamp())
            .groupby("period", as_index=False)[measure]
            .agg(aggregation)
            .sort_values("period")
        )
        if "last 3 months" in query.lower() and len(trend_df) > 3:
            trend_df = trend_df.tail(3)

        chart = px.line(trend_df, x="period", y=measure, markers=True, title=f"{measure.replace('_', ' ').title()} Trend")
        growth_text = "Not enough periods to calculate growth."
        if len(trend_df) >= 2 and trend_df[measure].iloc[0] != 0:
            growth_pct = ((trend_df[measure].iloc[-1] - trend_df[measure].iloc[0]) / abs(trend_df[measure].iloc[0])) * 100
            growth_text = f"Overall change across the selected periods is {growth_pct:,.2f}%."

        explanation = f"The trend shows how {aggregation} {measure.replace('_', ' ')} changes over time. {growth_text}"
        report = (
            f"## Trend Analysis\n{explanation}\n\n"
            f"{trend_df.to_markdown(index=False)}\n"
        )
        return AnalysisResult(
            title=f"{measure.replace('_', ' ').title()} Trend",
            result_text=growth_text,
            explanation=explanation,
            report_markdown=report,
            chart=chart,
            table=trend_df,
            confidence=0.96,
        )

    def _correlation_analysis(self, df: pd.DataFrame, query: str) -> AnalysisResult:
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        referenced_cols = [col for col in numeric_cols if col in query.lower()]
        if len(referenced_cols) >= 2:
            col_x, col_y = referenced_cols[:2]
        elif len(numeric_cols) >= 2:
            col_x, col_y = numeric_cols[:2]
        else:
            return self._clarification_result("I need at least two numeric columns for correlation analysis.")

        corr = df[[col_x, col_y]].corr().iloc[0, 1]
        chart = px.scatter(df, x=col_x, y=col_y, trendline="ols", title=f"Correlation: {col_x} vs {col_y}")
        strength = "strong" if abs(corr) >= 0.7 else "moderate" if abs(corr) >= 0.4 else "weak"
        direction = "positive" if corr >= 0 else "negative"
        explanation = f"The correlation between {col_x} and {col_y} is {corr:,.2f}, indicating a {strength} {direction} relationship."
        report = (
            f"## Correlation Analysis\n{explanation}\n\n"
            f"- Column 1: `{col_x}`\n- Column 2: `{col_y}`\n- Correlation coefficient: `{corr:,.2f}`\n"
        )
        return AnalysisResult(
            title="Correlation Analysis",
            result_text=f"{corr:,.2f}",
            explanation=explanation,
            report_markdown=report,
            chart=chart,
            confidence=0.95,
        )

    def _visual_analysis(self, df: pd.DataFrame, profile: DatasetProfile, query: str) -> AnalysisResult:
        chart = create_query_chart(df, profile, query)
        explanation = "The chart was generated automatically using the most relevant date/category and numeric columns."
        report = "## Visualization\nA chart was generated automatically based on the dataset structure."
        return AnalysisResult(
            title="Automatic Visualization",
            result_text="Chart generated successfully.",
            explanation=explanation,
            report_markdown=report,
            chart=chart,
            confidence=0.85,
            warnings=["Please confirm the selected columns match your intent."],
        )

    def _fallback_analysis(self, df: pd.DataFrame, profile: DatasetProfile, query: str) -> AnalysisResult:
        pandasai_response = self._try_pandasai(df, query)
        if pandasai_response:
            return pandasai_response

        summary_lines = [
            f"- Rows analyzed: {len(df):,}",
            f"- Columns analyzed: {len(df.columns)}",
            f"- Numeric fields: {', '.join(profile.numeric_columns[:5]) or 'None'}",
            f"- Category fields: {', '.join(profile.categorical_columns[:5]) or 'None'}",
        ]
        explanation = (
            "I could not map the query to a deterministic analysis pattern, so I returned a dataset-oriented summary and suggestions."
        )
        report = "## Dataset Summary\n" + "\n".join(summary_lines) + "\n\n## Suggestions\n- Try `Top 5 products`\n- Try `Generate full analysis report`"
        return AnalysisResult(
            title="Best-Effort Summary",
            result_text="I could not confidently interpret the exact request.",
            explanation=explanation,
            report_markdown=report,
            chart=create_default_chart(df, profile),
            confidence=0.55,
        )

    def _generate_full_report(self, df: pd.DataFrame, profile: DatasetProfile, query: str) -> AnalysisResult:
        measure = pick_relevant_numeric_column(df, query or "sales revenue profit")
        dimension = pick_groupby_column(df, query or "product region category")
        date_col = pick_relevant_date_column(df, query or "date month")
        aggregation = detect_aggregation(query, measure)

        lines = [
            "## Dataset Summary",
            f"- Total rows: {profile.rows:,}",
            f"- Total columns: {profile.columns}",
            f"- Numeric columns: {', '.join(profile.numeric_columns[:6]) or 'None'}",
            f"- Categorical columns: {', '.join(profile.categorical_columns[:6]) or 'None'}",
            f"- Date columns: {', '.join(profile.date_columns[:3]) or 'None'}",
            "",
        ]

        chart = None
        key_insight = "No numeric metric available for quantitative analysis."
        if measure:
            lines.extend(
                [
                    "## Important Metrics",
                    f"- Total {measure}: {df[measure].sum():,.2f}",
                    f"- Average {measure}: {df[measure].mean():,.2f}",
                    f"- Maximum {measure}: {df[measure].max():,.2f}",
                    f"- Minimum {measure}: {df[measure].min():,.2f}",
                    "",
                ]
            )
            key_insight = (
                f"The dataset is centered around `{measure}`, with a total of {df[measure].sum():,.2f} and average of {df[measure].mean():,.2f}."
            )

        if dimension and measure:
            ranked, value_col = aggregate_for_display(df, dimension, measure, aggregation)
            if not ranked.empty:
                top_row = ranked.iloc[0]
                bottom_row = ranked.iloc[-1]
                lines.extend(
                    [
                        "## Top And Bottom Performers",
                        f"- Best performer: {top_row[dimension]} ({top_row[value_col]:,.2f})",
                        f"- Lowest performer: {bottom_row[dimension]} ({bottom_row[value_col]:,.2f})",
                        "",
                    ]
                )
                chart = px.bar(ranked.head(10), x=dimension, y=value_col, title=f"Top {dimension.replace('_', ' ').title()} by {value_col.replace('_', ' ').title()}")

        if date_col and measure:
            trend_df = (
                df.assign(period=df[date_col].dt.to_period("M").dt.to_timestamp())
                .groupby("period", as_index=False)[measure]
                .sum()
                .sort_values("period")
            )
            if len(trend_df) >= 2:
                growth_pct = ((trend_df[measure].iloc[-1] - trend_df[measure].iloc[0]) / max(abs(trend_df[measure].iloc[0]), 1e-9)) * 100
                trend_statement = f"{measure.replace('_', ' ').title()} changed by {growth_pct:,.2f}% from the first to the latest period."
            else:
                trend_statement = "Not enough date periods to calculate a meaningful trend."
            lines.extend(["## Trend Analysis", f"- {trend_statement}", ""])
            if chart is None:
                chart = px.line(trend_df, x="period", y=measure, markers=True, title=f"{measure.replace('_', ' ').title()} Trend")

        lines.extend(
            [
                "## Key Insights",
                f"- {key_insight}",
                "- Missing values were cleaned before analysis for a stable result.",
                "- Results reflect the currently selected columns and active filters.",
                "",
                "## Final Conclusion",
                "This report provides a quick business-friendly snapshot of the dataset, highlighting scale, performance drivers, and trends that deserve follow-up.",
            ]
        )

        if self.llm_available:
            llm_text = self._enhance_report_with_llm(df, profile, query, "\n".join(lines))
            if llm_text:
                lines.append("")
                lines.append("## AI Narrative")
                lines.append(llm_text)

        report_markdown = "\n".join(lines)
        return AnalysisResult(
            title="Detailed Analysis Report",
            result_text="Detailed report generated successfully.",
            explanation="A structured report was generated with summary, metrics, performers, trends, and a final conclusion.",
            report_markdown=report_markdown,
            chart=chart,
            confidence=0.92 if self.llm_available else 0.97,
            warnings=[] if self.llm_available else ["LLM enhancement is not active. The report is based on deterministic analytics only."],
        )

    def generate_full_dataset_report(self, df: pd.DataFrame, profile: DatasetProfile) -> FullReportResult:
        cleaned_df, cleaning_summary = perform_automated_cleaning(df)
        cleaned_profile = DatasetProfile(
            rows=len(cleaned_df),
            columns=len(cleaned_df.columns),
            numeric_columns=[col for col in cleaned_df.columns if pd.api.types.is_numeric_dtype(cleaned_df[col])],
            categorical_columns=[
                col for col in cleaned_df.columns
                if not pd.api.types.is_numeric_dtype(cleaned_df[col]) and not pd.api.types.is_datetime64_any_dtype(cleaned_df[col])
            ],
            date_columns=[col for col in cleaned_df.columns if pd.api.types.is_datetime64_any_dtype(cleaned_df[col])],
            missing_cells=int(cleaned_df.isna().sum().sum()),
            column_labels={col: col.replace("_", " ").title() for col in cleaned_df.columns},
        )

        dataset_overview = build_dataset_overview(cleaned_df, cleaned_profile)
        numeric_summary = build_numeric_summary(cleaned_df, cleaned_profile.numeric_columns)
        categorical_summary = build_categorical_summary(cleaned_df, cleaned_profile.categorical_columns[:10])
        correlation_matrix, relationship_lines = analyze_relationships(cleaned_df, cleaned_profile.numeric_columns)
        charts = generate_full_report_charts(cleaned_df, cleaned_profile, correlation_matrix)
        insights = build_key_insights(
            cleaned_df,
            cleaned_profile,
            categorical_summary,
            correlation_matrix,
            relationship_lines,
            cleaning_summary,
        )

        action_log = [
            "Read the uploaded dataset with Pandas and identified row count, column count, column names, and data types.",
            *cleaning_summary.cleaning_actions,
            "Generated statistical summaries for numeric columns.",
            "Generated frequency summaries for categorical columns.",
            "Generated correlation analysis to detect relationships between numeric columns.",
        ]
        action_log.extend([chart.description for chart in charts])

        conclusion_parts = []
        if insights:
            conclusion_parts.append(insights[0])
        if relationship_lines:
            conclusion_parts.append(relationship_lines[0])
        if cleaning_summary.total_missing_handled or cleaning_summary.duplicate_rows_removed or cleaning_summary.total_outliers_handled:
            conclusion_parts.append(
                f"Cleaning improved analysis reliability by handling {cleaning_summary.total_missing_handled} missing values, "
                f"removing {cleaning_summary.duplicate_rows_removed} duplicate rows, and stabilizing {cleaning_summary.total_outliers_handled} outliers."
            )
        conclusion = " ".join(conclusion_parts) or "The dataset was profiled successfully and the main structure, patterns, and relationships were summarized."

        report_lines = ["## Dataset Overview"]
        report_lines.extend([f"- {line}" for line in dataset_overview])
        report_lines.extend(
            [
                "",
                "## Data Cleaning Summary",
                f"- {cleaning_summary.total_missing_handled} missing values handled",
                f"- {cleaning_summary.duplicate_rows_removed} duplicate rows removed",
                f"- {cleaning_summary.total_outliers_handled} outliers handled",
            ]
        )
        report_lines.extend([f"- {line}" for line in cleaning_summary.cleaning_actions])

        report_lines.append("")
        report_lines.append("## Key Statistics")
        if not numeric_summary.empty:
            report_lines.append(numeric_summary.head(10).to_markdown(index=False))
        else:
            report_lines.append("- No numeric columns available for statistical summary.")

        report_lines.append("")
        report_lines.append("## Categorical Frequency Summary")
        if not categorical_summary.empty:
            report_lines.append(categorical_summary.head(10).to_markdown(index=False))
        else:
            report_lines.append("- No categorical columns available for frequency analysis.")

        report_lines.append("")
        report_lines.append("## Relationships Found")
        report_lines.extend([f"- {line}" for line in relationship_lines])

        report_lines.append("")
        report_lines.append("## Key Insights")
        report_lines.extend([f"- {line}" for line in insights] or ["- No standout insights were detected automatically."])

        report_lines.append("")
        report_lines.append("## Visualizations")
        report_lines.extend([f"- {chart.description}" for chart in charts])

        report_lines.append("")
        report_lines.append("## Final Conclusion")
        report_lines.append(conclusion)

        return FullReportResult(
            cleaned_df=cleaned_df,
            cleaned_profile=cleaned_profile,
            dataset_overview=dataset_overview,
            cleaning_summary=cleaning_summary,
            numeric_summary=numeric_summary,
            categorical_summary=categorical_summary,
            correlation_matrix=correlation_matrix,
            relationship_lines=relationship_lines,
            charts=charts,
            insights=insights,
            conclusion=conclusion,
            report_markdown="\n".join(report_lines),
            action_log=action_log,
        )

    def _enhance_report_with_llm(self, df: pd.DataFrame, profile: DatasetProfile, query: str, base_report: str) -> Optional[str]:
        if not self.llm_available:
            return None

        sample = df.head(12).to_dict(orient="records")
        prompt = {
            "user_query": query,
            "dataset_overview": {
                "rows": profile.rows,
                "columns": profile.columns,
                "numeric_columns": profile.numeric_columns,
                "categorical_columns": profile.categorical_columns,
                "date_columns": profile.date_columns,
            },
            "sample_rows": sample,
            "base_report": base_report,
            "instruction": (
                "Write a short business-friendly narrative in simple English/Hinglish. "
                "Do not invent metrics. Only elaborate on the provided report."
            ),
        }

        try:
            response = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": "You are a careful business data analyst. Stay grounded in the provided facts."},
                    {"role": "user", "content": json.dumps(prompt, default=str)},
                ],
                max_output_tokens=300,
            )
            return response.output_text.strip()
        except Exception:
            return None

    def _try_pandasai(self, df: pd.DataFrame, query: str) -> Optional[AnalysisResult]:
        if not self.pandasai_available:
            return None

        try:
            smart_df = SmartDataframe(
                df,
                config={
                    "llm": PandasAIOpenAI(api_token=self.api_key, model=self.model),
                    "save_charts": False,
                    "verbose": False,
                },
            )
            response = smart_df.chat(
                f"Answer this business query in a concise way and do not modify the dataframe: {query}"
            )
            explanation = f"PandasAI interpreted the question and returned: {response}"
            report = f"## PandasAI Result\n{response}"
            return AnalysisResult(
                title="PandasAI Analysis",
                result_text=str(response),
                explanation=explanation,
                report_markdown=report,
                confidence=0.7,
                warnings=["This result came from LLM interpretation. Please validate it against the data preview."],
            )
        except Exception:
            return None

    def _clarification_result(self, message: str) -> AnalysisResult:
        return AnalysisResult(
            title="Need Clarification",
            result_text=message,
            explanation=message,
            report_markdown=f"## Clarification Needed\n{message}",
            confidence=0.4,
            warnings=["The query appears ambiguous. Try naming the metric or category explicitly."],
        )

    def _extract_number(self, text: str) -> Optional[int]:
        match = re.search(r"\b(\d+)\b", text)
        return int(match.group(1)) if match else None
