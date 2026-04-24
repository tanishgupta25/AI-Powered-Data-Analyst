import io
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


MAX_ROWS = 5000
PREVIEW_ROWS = 10


@dataclass
class DatasetProfile:
    rows: int
    columns: int
    numeric_columns: List[str]
    categorical_columns: List[str]
    date_columns: List[str]
    missing_cells: int
    column_labels: Dict[str, str]


@dataclass
class ChartArtifact:
    section: str
    title: str
    figure: object
    description: str
    observation: str


@dataclass
class CleaningSummary:
    total_missing_handled: int
    missing_actions: List[str]
    duplicate_rows_removed: int
    outlier_actions: List[str]
    total_outliers_handled: int
    cleaning_actions: List[str]


def clean_column_name(name: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "column"


def tokenize_query(query: str) -> List[str]:
    return [token for token in re.split(r"[^a-z0-9]+", query.lower()) if token]


def _matching_aliases(query_tokens: List[str], aliases: List[str]) -> int:
    score = 0
    joined_query = " ".join(query_tokens)
    for alias in aliases:
        alias_tokens = alias.split()
        if len(alias_tokens) == 1 and alias in query_tokens:
            score += 6
        elif len(alias_tokens) > 1 and alias in joined_query:
            score += 8
    return score


def _looks_like_date_column(column_name: str, series: pd.Series) -> bool:
    lowered = column_name.lower()
    if any(keyword in lowered for keyword in ["date", "time", "month", "year", "day", "week"]):
        return True
    sample = series.dropna().astype(str).head(25)
    if sample.empty:
        return False
    date_like_pattern = r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$|^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$|^[A-Za-z]{3,9}\s+\d{4}$"
    return sample.str.match(date_like_pattern).mean() >= 0.6


def _column_score(column: str, query: str, aliases: List[str]) -> int:
    query_tokens = tokenize_query(query)
    column_tokens = column.split("_")
    score = 0
    joined_query = " ".join(query_tokens)
    normalized_column = " ".join(column_tokens)

    if normalized_column in joined_query:
        score += 12

    score += _matching_aliases(query_tokens, aliases)

    overlapping_tokens = set(query_tokens).intersection(column_tokens)
    score += len(overlapping_tokens) * 4
    return score


def detect_aggregation(query: str, measure: Optional[str] = None) -> str:
    query_lower = query.lower()
    average_terms = ["average", "avg", "mean", "typical", "per", "by category"]
    count_terms = ["count", "how many", "number of", "frequency", "distribution", "patients", "records"]
    min_terms = ["minimum", "min", "lowest", "smallest"]
    max_terms = ["maximum", "max", "highest", "largest", "top", "best"]
    sum_terms = ["total", "sum", "overall", "combined"]
    additive_measure_terms = ["sales", "revenue", "profit", "amount", "cost", "quantity", "qty", "units"]

    if any(term in query_lower for term in count_terms):
        return "count"
    if any(term in query_lower for term in average_terms):
        return "mean"
    if any(term in query_lower for term in min_terms):
        return "min"
    if any(term in query_lower for term in max_terms):
        return "max"
    if any(term in query_lower for term in sum_terms):
        return "sum"
    if measure and any(term in measure for term in additive_measure_terms):
        return "sum"
    return "mean"


def aggregate_for_display(df: pd.DataFrame, group_col: str, measure: Optional[str], aggregation: str) -> Tuple[pd.DataFrame, str]:
    if aggregation == "count" or not measure:
        table = df.groupby(group_col, as_index=False).size().rename(columns={"size": "count"})
        return table.sort_values("count", ascending=False), "count"

    aggregated = (
        df.groupby(group_col, as_index=False)[measure]
        .agg(aggregation)
        .sort_values(measure, ascending=False)
    )
    return aggregated, measure


@st.cache_data(show_spinner=False)
def load_dataset(file_bytes: bytes, file_name: str, max_rows: int = MAX_ROWS) -> pd.DataFrame:
    buffer = io.BytesIO(file_bytes)
    lower_name = file_name.lower()
    if lower_name.endswith(".csv"):
        df = pd.read_csv(buffer)
    elif lower_name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(buffer)
    else:
        raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")

    if len(df) > max_rows:
        df = df.head(max_rows).copy()

    return df


def standardize_dataframe(df: pd.DataFrame, apply_basic_cleaning: bool = True) -> Tuple[pd.DataFrame, Dict[str, str]]:
    column_map: Dict[str, str] = {}
    renamed_columns = []

    for column in df.columns:
        cleaned = clean_column_name(column)
        original_cleaned = cleaned
        suffix = 1
        while cleaned in renamed_columns:
            suffix += 1
            cleaned = f"{original_cleaned}_{suffix}"
        column_map[cleaned] = str(column)
        renamed_columns.append(cleaned)

    cleaned_df = df.copy()
    cleaned_df.columns = renamed_columns

    for column in cleaned_df.columns:
        if cleaned_df[column].dtype == "object":
            cleaned_df[column] = cleaned_df[column].replace(r"^\s*$", pd.NA, regex=True)

    # Numeric coercion
    for column in cleaned_df.columns:
        if cleaned_df[column].dtype == "object":
            stripped = cleaned_df[column].astype(str).str.replace(",", "", regex=False)
            numeric_version = pd.to_numeric(stripped, errors="coerce")
            if numeric_version.notna().mean() >= 0.8:
                cleaned_df[column] = numeric_version

    # Date coercion
    for column in cleaned_df.columns:
        if not pd.api.types.is_numeric_dtype(cleaned_df[column]) and _looks_like_date_column(column, cleaned_df[column]):
            parsed = pd.to_datetime(cleaned_df[column], errors="coerce")
            if parsed.notna().mean() >= 0.8:
                cleaned_df[column] = parsed

    if apply_basic_cleaning:
        # Fill missing values conservatively for the interactive query flow.
        for column in cleaned_df.columns:
            if pd.api.types.is_numeric_dtype(cleaned_df[column]):
                cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].median())
            elif pd.api.types.is_datetime64_any_dtype(cleaned_df[column]):
                cleaned_df[column] = cleaned_df[column].ffill().bfill()
            else:
                mode_series = cleaned_df[column].mode(dropna=True)
                fill_value = mode_series.iloc[0] if not mode_series.empty else "unknown"
                cleaned_df[column] = cleaned_df[column].fillna(fill_value)

    return cleaned_df, column_map


def detect_column_types(df: pd.DataFrame) -> DatasetProfile:
    numeric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    date_columns = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    categorical_columns = [col for col in df.columns if col not in numeric_columns + date_columns]
    return DatasetProfile(
        rows=len(df),
        columns=len(df.columns),
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        date_columns=date_columns,
        missing_cells=int(df.isna().sum().sum()),
        column_labels={col: col.replace("_", " ").title() for col in df.columns},
    )


def build_filter_options(df: pd.DataFrame, profile: DatasetProfile) -> Dict[str, Dict]:
    filter_options: Dict[str, Dict] = {}

    for column in profile.date_columns[:3]:
        filter_options[column] = {
            "type": "date",
            "min": df[column].min(),
            "max": df[column].max(),
        }

    for column in profile.categorical_columns[:4]:
        unique_values = df[column].dropna().astype(str).unique().tolist()
        if 1 < len(unique_values) <= 50:
            filter_options[column] = {
                "type": "category",
                "values": sorted(unique_values),
            }

    return filter_options


def apply_filters(df: pd.DataFrame, active_filters: Dict[str, Dict]) -> pd.DataFrame:
    filtered_df = df.copy()
    for column, filter_value in active_filters.items():
        if column not in filtered_df.columns:
            continue
        if filter_value.get("type") == "date":
            start = pd.to_datetime(filter_value.get("start"))
            end = pd.to_datetime(filter_value.get("end"))
            filtered_df = filtered_df[
                (filtered_df[column] >= start) & (filtered_df[column] <= end)
            ]
        elif filter_value.get("type") == "category":
            selected = filter_value.get("selected", [])
            if selected:
                filtered_df = filtered_df[filtered_df[column].astype(str).isin(selected)]
    return filtered_df


def pick_relevant_numeric_column(df: pd.DataFrame, query: str) -> Optional[str]:
    metric_aliases = {
        "sales": ["sales", "sale", "revenue", "turnover"],
        "profit": ["profit", "margin", "earnings"],
        "amount": ["amount", "value", "total"],
        "price": ["price", "rate"],
        "cost": ["cost", "expense", "spend"],
        "quantity": ["quantity", "qty", "units", "volume"],
        "discount": ["discount", "off"],
    }

    numeric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    if not numeric_columns:
        return None

    scored_columns = []
    for column in numeric_columns:
        aliases = []
        for key, values in metric_aliases.items():
            if key in column:
                aliases.extend(values)
        score = _column_score(column, query, aliases)
        scored_columns.append((score, column))

    scored_columns.sort(reverse=True)
    best_score, best_column = scored_columns[0]
    return best_column if best_score > 0 else numeric_columns[0]


def pick_relevant_date_column(df: pd.DataFrame, query: str) -> Optional[str]:
    date_columns = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    if not date_columns:
        return None

    date_aliases = ["date", "day", "month", "quarter", "year", "time", "period", "timeline"]
    scored_columns = [(_column_score(column, query, date_aliases), column) for column in date_columns]
    scored_columns.sort(reverse=True)
    best_score, best_column = scored_columns[0]
    if best_score > 0:
        return best_column

    for preferred in ["date", "order_date", "month", "created_at"]:
        if preferred in df.columns and pd.api.types.is_datetime64_any_dtype(df[preferred]):
            return preferred

    return date_columns[0]


def pick_groupby_column(df: pd.DataFrame, query: str) -> Optional[str]:
    dimension_aliases = {
        "product": ["product", "item", "sku"],
        "category": ["category", "type", "class"],
        "region": ["region", "area", "zone", "territory"],
        "segment": ["segment", "channel"],
        "customer": ["customer", "client", "buyer"],
        "city": ["city", "town"],
        "state": ["state", "province"],
    }

    candidate_columns = [
        col
        for col in df.columns
        if not pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_datetime64_any_dtype(df[col])
    ]
    if not candidate_columns:
        return None

    scored_columns = []
    for column in candidate_columns:
        aliases = []
        for key, values in dimension_aliases.items():
            if key in column:
                aliases.extend(values)
        score = _column_score(column, query, aliases)
        scored_columns.append((score, column))

    scored_columns.sort(reverse=True)
    best_score, best_column = scored_columns[0]
    return best_column if best_score > 0 else candidate_columns[0]


def generate_smart_suggestions(df: pd.DataFrame, profile: DatasetProfile) -> List[str]:
    measure = pick_relevant_numeric_column(df, "sales profit amount") or "numeric column"
    dimension = pick_groupby_column(df, "region product category") or "category"
    date_col = pick_relevant_date_column(df, "date month")

    suggestions = [
        f"Total {measure}",
        f"Top 5 {dimension}",
        f"{dimension} wise {measure}",
        f"Average {measure}",
        "Generate full analysis report",
    ]
    if date_col:
        suggestions.insert(2, f"Show {measure} trend by {date_col}")
        suggestions.append("Compare last 3 months")
    if len(profile.numeric_columns) >= 2:
        suggestions.append(f"Correlation between {profile.numeric_columns[0]} and {profile.numeric_columns[1]}")
    return suggestions[:7]


def generate_auto_summary(df: pd.DataFrame, profile: DatasetProfile) -> Dict[str, str]:
    summary = {
        "rows": f"{profile.rows:,}",
        "columns": f"{profile.columns}",
        "missing_cells": f"{profile.missing_cells:,}",
        "numeric_columns": ", ".join(profile.numeric_columns[:5]) or "None",
        "categorical_columns": ", ".join(profile.categorical_columns[:5]) or "None",
    }
    measure = pick_relevant_numeric_column(df, "sales revenue profit amount")
    if measure:
        summary["total_metric"] = f"{measure}: {df[measure].sum():,.2f}"
        summary["average_metric"] = f"{measure}: {df[measure].mean():,.2f}"
    return summary


def perform_automated_cleaning(df: pd.DataFrame) -> Tuple[pd.DataFrame, CleaningSummary]:
    cleaned_df = df.copy()
    missing_actions: List[str] = []
    outlier_actions: List[str] = []
    cleaning_actions: List[str] = []
    total_missing_handled = 0
    total_outliers_handled = 0

    duplicate_rows_removed = int(cleaned_df.duplicated().sum())
    if duplicate_rows_removed:
        cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
        cleaning_actions.append(f"Dropped {duplicate_rows_removed} duplicate rows.")
    else:
        cleaning_actions.append("No duplicate rows were found.")

    for column in cleaned_df.columns:
        missing_count = int(cleaned_df[column].isna().sum())
        if missing_count == 0:
            continue

        if pd.api.types.is_numeric_dtype(cleaned_df[column]):
            fill_value = cleaned_df[column].median()
            cleaned_df[column] = cleaned_df[column].fillna(fill_value)
            action = f"Removed {missing_count} missing values from '{column}' by filling with median ({fill_value:,.2f})."
        elif pd.api.types.is_datetime64_any_dtype(cleaned_df[column]):
            cleaned_df[column] = cleaned_df[column].ffill().bfill()
            action = f"Removed {missing_count} missing values from '{column}' using forward/backward fill."
        else:
            mode_series = cleaned_df[column].mode(dropna=True)
            fill_value = mode_series.iloc[0] if not mode_series.empty else "unknown"
            cleaned_df[column] = cleaned_df[column].fillna(fill_value)
            action = f"Removed {missing_count} missing values from '{column}' by filling with mode ('{fill_value}')."

        total_missing_handled += missing_count
        missing_actions.append(action)

    if not missing_actions:
        cleaning_actions.append("No missing values were found.")
    else:
        cleaning_actions.extend(missing_actions)

    numeric_columns = [col for col in cleaned_df.columns if pd.api.types.is_numeric_dtype(cleaned_df[col])]
    for column in numeric_columns:
        q1 = cleaned_df[column].quantile(0.25)
        q3 = cleaned_df[column].quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_mask = (cleaned_df[column] < lower_bound) | (cleaned_df[column] > upper_bound)
        outlier_count = int(outlier_mask.sum())
        if outlier_count == 0:
            continue

        cleaned_df[column] = cleaned_df[column].astype(float)
        cleaned_df.loc[outlier_mask, column] = cleaned_df.loc[outlier_mask, column].clip(lower_bound, upper_bound)
        total_outliers_handled += outlier_count
        outlier_actions.append(
            f"Detected and handled {outlier_count} outliers in '{column}' using IQR clipping."
        )

    if not outlier_actions:
        cleaning_actions.append("No significant numeric outliers were detected.")
    else:
        cleaning_actions.extend(outlier_actions)

    return cleaned_df, CleaningSummary(
        total_missing_handled=total_missing_handled,
        missing_actions=missing_actions,
        duplicate_rows_removed=duplicate_rows_removed,
        outlier_actions=outlier_actions,
        total_outliers_handled=total_outliers_handled,
        cleaning_actions=cleaning_actions,
    )


def build_dataset_overview(df: pd.DataFrame, profile: DatasetProfile) -> List[str]:
    return [
        f"Total rows: {profile.rows:,}",
        f"Total columns: {profile.columns}",
        f"Column names: {', '.join(df.columns)}",
        f"Numeric columns: {', '.join(profile.numeric_columns) or 'None'}",
        f"Categorical columns: {', '.join(profile.categorical_columns) or 'None'}",
        f"Datetime columns: {', '.join(profile.date_columns) or 'None'}",
    ]


def build_numeric_summary(df: pd.DataFrame, numeric_columns: List[str]) -> pd.DataFrame:
    if not numeric_columns:
        return pd.DataFrame()
    summary = df[numeric_columns].describe().T.reset_index().rename(columns={"index": "column"})
    summary["median"] = df[numeric_columns].median().values
    return summary[["column", "count", "mean", "median", "std", "min", "25%", "50%", "75%", "max"]]


def build_categorical_summary(df: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:
    rows = []
    for column in categorical_columns:
        top_values = df[column].astype(str).value_counts().head(3)
        rows.append(
            {
                "column": column,
                "unique_values": int(df[column].nunique(dropna=True)),
                "top_values": ", ".join([f"{idx} ({val})" for idx, val in top_values.items()]) or "N/A",
            }
        )
    return pd.DataFrame(rows)


def analyze_relationships(df: pd.DataFrame, numeric_columns: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    if len(numeric_columns) < 2:
        return pd.DataFrame(), ["Not enough numeric columns to compute correlations."]

    corr_df = df[numeric_columns].corr().round(3)
    relationship_lines: List[str] = []
    seen_pairs = set()
    for column_a in corr_df.columns:
        for column_b in corr_df.columns:
            if column_a == column_b or (column_b, column_a) in seen_pairs:
                continue
            seen_pairs.add((column_a, column_b))
            value = corr_df.loc[column_a, column_b]
            if abs(value) >= 0.7:
                direction = "positively" if value > 0 else "negatively"
                relationship_lines.append(
                    f"{column_a.replace('_', ' ').title()} and {column_b.replace('_', ' ').title()} are strongly {direction} correlated ({value:,.2f})."
                )
            elif abs(value) >= 0.4:
                direction = "positively" if value > 0 else "negatively"
                relationship_lines.append(
                    f"{column_a.replace('_', ' ').title()} and {column_b.replace('_', ' ').title()} show a moderate {direction} relationship ({value:,.2f})."
                )

    if not relationship_lines:
        relationship_lines.append("No strong numeric relationships were detected.")
    return corr_df, relationship_lines[:8]


def build_key_insights(
    df: pd.DataFrame,
    profile: DatasetProfile,
    categorical_summary: pd.DataFrame,
    corr_df: pd.DataFrame,
    relationship_lines: List[str],
    cleaning_summary: CleaningSummary,
) -> List[str]:
    insights: List[str] = []

    if profile.numeric_columns:
        numeric_summary = build_numeric_summary(df, profile.numeric_columns)
        highest_variation = numeric_summary.sort_values("std", ascending=False).iloc[0]
        insights.append(
            f"'{highest_variation['column']}' shows the highest variability, which may deserve deeper investigation."
        )

    if not categorical_summary.empty:
        first_row = categorical_summary.iloc[0]
        insights.append(
            f"'{first_row['column']}' has {int(first_row['unique_values'])} unique groups; top values are {first_row['top_values']}."
        )

    if relationship_lines and "No strong numeric relationships" not in relationship_lines[0]:
        insights.append(relationship_lines[0])

    if cleaning_summary.total_outliers_handled:
        insights.append(
            f"The analysis stabilized {cleaning_summary.total_outliers_handled} outlier values before generating metrics and charts."
        )

    if profile.date_columns and profile.numeric_columns:
        date_col = profile.date_columns[0]
        measure = profile.numeric_columns[0]
        trend_df = (
            df.assign(period=df[date_col].dt.to_period("M").dt.to_timestamp())
            .groupby("period", as_index=False)[measure]
            .mean()
            .sort_values("period")
        )
        if len(trend_df) >= 2:
            change_pct = ((trend_df[measure].iloc[-1] - trend_df[measure].iloc[0]) / max(abs(trend_df[measure].iloc[0]), 1e-9)) * 100
            direction = "grew" if change_pct >= 0 else "declined"
            insights.append(
                f"{measure.replace('_', ' ').title()} {direction} by {abs(change_pct):,.2f}% across the available time periods."
            )

    return insights[:8]


def generate_full_report_charts(df: pd.DataFrame, profile: DatasetProfile, corr_df: pd.DataFrame) -> List[ChartArtifact]:
    charts: List[ChartArtifact] = []

    if profile.date_columns and profile.numeric_columns:
        date_col = profile.date_columns[0]
        measure = profile.numeric_columns[0]
        trend_df = (
            df.assign(period=df[date_col].dt.to_period("M").dt.to_timestamp())
            .groupby("period", as_index=False)[measure]
            .mean()
            .sort_values("period")
        )
        if not trend_df.empty:
            charts.append(
                ChartArtifact(
                    section="Trend Analysis",
                    title=f"{measure.replace('_', ' ').title()} Trend",
                    figure=px.line(trend_df, x="period", y=measure, markers=True, title=f"{measure.replace('_', ' ').title()} Trend"),
                    description="Generated a line chart to show how a primary numeric metric changes over time.",
                    observation=(
                        f"This line chart tracks the average {measure.replace('_', ' ')} over time and helps reveal growth or decline patterns."
                    ),
                )
            )

    for group_col in profile.categorical_columns[:2]:
        if profile.numeric_columns:
            measure = profile.numeric_columns[0]
            bar_df = (
                df.groupby(group_col, as_index=False)[measure]
                .mean()
                .sort_values(measure, ascending=False)
                .head(10)
            )
            if not bar_df.empty:
                top_label = str(bar_df.iloc[0][group_col])
                charts.append(
                    ChartArtifact(
                        section="Category Analysis",
                        title=f"{measure.replace('_', ' ').title()} by {group_col.replace('_', ' ').title()}",
                        figure=px.bar(
                            bar_df,
                            x=group_col,
                            y=measure,
                            title=f"{measure.replace('_', ' ').title()} by {group_col.replace('_', ' ').title()}",
                            labels={group_col: group_col.replace("_", " ").title(), measure: measure.replace("_", " ").title()},
                        ),
                        description="Generated a bar chart for category-wise comparison.",
                        observation=f"This bar chart compares average {measure.replace('_', ' ')} across {group_col.replace('_', ' ')} groups; the top group is {top_label}.",
                    )
                )

            pie_df = (
                df[group_col]
                .astype(str)
                .value_counts(normalize=True)
                .head(6)
                .reset_index()
            )
            pie_df.columns = [group_col, "percentage"]
            if not pie_df.empty:
                charts.append(
                    ChartArtifact(
                        section="Category Analysis",
                        title=f"{group_col.replace('_', ' ').title()} Share",
                        figure=px.pie(
                            pie_df,
                            names=group_col,
                            values="percentage",
                            title=f"{group_col.replace('_', ' ').title()} Share",
                        ),
                        description="Generated a pie chart to show percentage distribution of top categories.",
                        observation=f"This pie chart shows the share of the top {group_col.replace('_', ' ')} groups in the dataset.",
                    )
                )
        else:
            count_df = df[group_col].astype(str).value_counts().head(10).reset_index()
            count_df.columns = [group_col, "count"]
            if not count_df.empty:
                charts.append(
                    ChartArtifact(
                        section="Category Analysis",
                        title=f"{group_col.replace('_', ' ').title()} Frequency",
                        figure=px.bar(
                            count_df,
                            x=group_col,
                            y="count",
                            title=f"{group_col.replace('_', ' ').title()} Frequency",
                            labels={group_col: group_col.replace("_", " ").title(), "count": "Count"},
                        ),
                        description="Generated a bar chart to show categorical frequency.",
                        observation=f"This bar chart shows which {group_col.replace('_', ' ')} values appear most often.",
                    )
                )

    for column in profile.numeric_columns[:3]:
        series = df[column].dropna()
        if series.empty:
            continue
        skew_direction = "right-skewed" if series.mean() > series.median() else "left-skewed" if series.mean() < series.median() else "fairly symmetric"
        charts.append(
            ChartArtifact(
                section="Distribution Analysis",
                title=f"{column.replace('_', ' ').title()} Distribution",
                figure=px.histogram(df, x=column, nbins=25, title=f"{column.replace('_', ' ').title()} Distribution"),
                description=f"Generated a histogram to inspect the distribution of '{column}'.",
                observation=f"This histogram shows that {column.replace('_', ' ')} appears {skew_direction}.",
            )
        )
        charts.append(
            ChartArtifact(
                section="Distribution Analysis",
                title=f"{column.replace('_', ' ').title()} Outlier Check",
                figure=px.box(df, y=column, title=f"{column.replace('_', ' ').title()} Outlier Check"),
                description=f"Generated a boxplot to inspect outliers in '{column}'.",
                observation=f"This boxplot helps detect unusual values and spread in {column.replace('_', ' ')}.",
            )
        )

    if len(profile.numeric_columns) >= 2:
        numeric_pairs = []
        for i, x_col in enumerate(profile.numeric_columns[:3]):
            for y_col in profile.numeric_columns[i + 1 : 3]:
                numeric_pairs.append((x_col, y_col))
        for x_col, y_col in numeric_pairs[:2]:
            corr_value = df[[x_col, y_col]].corr().iloc[0, 1]
            charts.append(
                ChartArtifact(
                    section="Relationship Analysis",
                    title=f"{x_col.replace('_', ' ').title()} vs {y_col.replace('_', ' ').title()}",
                    figure=px.scatter(
                        df,
                        x=x_col,
                        y=y_col,
                        trendline="ols",
                        title=f"{x_col.replace('_', ' ').title()} vs {y_col.replace('_', ' ').title()}",
                        labels={x_col: x_col.replace("_", " ").title(), y_col: y_col.replace("_", " ").title()},
                    ),
                    description="Generated a scatter plot to inspect relationships between two numeric columns.",
                    observation=f"This scatter plot compares {x_col.replace('_', ' ')} and {y_col.replace('_', ' ')}; the correlation is {corr_value:,.2f}.",
                )
            )

    if not corr_df.empty:
        heatmap = go.Figure(
            data=go.Heatmap(
                z=corr_df.values,
                x=corr_df.columns,
                y=corr_df.index,
                colorscale="RdBu",
                zmin=-1,
                zmax=1,
                colorbar=dict(title="Correlation"),
            )
        )
        heatmap.update_layout(title="Correlation Heatmap")
        charts.append(
            ChartArtifact(
                section="Relationship Analysis",
                title="Correlation Heatmap",
                figure=heatmap,
                description="Generated a heatmap to analyze correlations between numeric columns.",
                observation="This heatmap summarizes positive and negative relationships across all numeric variables.",
            )
        )

    if len(charts) < 5:
        for column in profile.numeric_columns[3:5]:
            charts.append(
                ChartArtifact(
                    section="Distribution Analysis",
                    title=f"{column.replace('_', ' ').title()} Distribution",
                    figure=px.histogram(df, x=column, nbins=20, title=f"{column.replace('_', ' ').title()} Distribution"),
                    description=f"Generated an additional histogram for '{column}' to improve coverage.",
                    observation=f"This histogram expands the distribution review for {column.replace('_', ' ')}.",
                )
            )

    return charts


def create_default_chart(df: pd.DataFrame, profile: DatasetProfile):
    measure = pick_relevant_numeric_column(df, "sales revenue")
    date_col = pick_relevant_date_column(df, "date month")
    dimension = pick_groupby_column(df, "product region category")

    if date_col and measure:
        chart_df = (
            df.assign(_period=df[date_col].dt.to_period("M").dt.to_timestamp())
            .groupby("_period", as_index=False)[measure]
            .sum()
        )
        return px.line(chart_df, x="_period", y=measure, title=f"{measure.replace('_', ' ').title()} Trend")

    if dimension and measure:
        chart_df = (
            df.groupby(dimension, as_index=False)[measure]
            .sum()
            .sort_values(measure, ascending=False)
            .head(10)
        )
        return px.bar(chart_df, x=dimension, y=measure, title=f"Top {dimension.replace('_', ' ').title()} by {measure.replace('_', ' ').title()}")

    return None


def create_query_chart(
    df: pd.DataFrame,
    profile: DatasetProfile,
    query: str,
    measure: Optional[str] = None,
    dimension: Optional[str] = None,
    date_col: Optional[str] = None,
):
    query_lower = query.lower()
    measure = measure or pick_relevant_numeric_column(df, query)
    dimension = dimension or pick_groupby_column(df, query)
    date_col = date_col or pick_relevant_date_column(df, query)
    aggregation = detect_aggregation(query, measure)

    if any(term in query_lower for term in ["correlation", "relationship", "scatter"]):
        numeric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        if len(numeric_columns) >= 2:
            x_col = measure or numeric_columns[0]
            y_col = next((col for col in numeric_columns if col != x_col), numeric_columns[1])
            return px.scatter(df, x=x_col, y=y_col, title=f"{x_col.replace('_', ' ').title()} vs {y_col.replace('_', ' ').title()}")

    if not measure and aggregation != "count":
        return create_default_chart(df, profile)

    if any(term in query_lower for term in ["trend", "growth", "month", "last 3 months", "compare", "timeline"]):
        if date_col:
            trend_df = (
                df.assign(period=df[date_col].dt.to_period("M").dt.to_timestamp())
                .groupby("period", as_index=False)
                .agg({measure: aggregation if aggregation != "count" else "size"})
                .sort_values("period")
            )
            if aggregation == "count":
                trend_df = trend_df.rename(columns={measure: "count"})
                y_col = "count"
            else:
                y_col = measure
            if "last 3 months" in query_lower and len(trend_df) > 3:
                trend_df = trend_df.tail(3)
            title_metric = y_col.replace("_", " ").title()
            return px.line(trend_df, x="period", y=y_col, markers=True, title=f"{title_metric} Trend")

    if any(term in query_lower for term in ["distribution", "share", "pie"]):
        if dimension:
            pie_df, value_col = aggregate_for_display(df, dimension, measure, aggregation)
            pie_df = pie_df.head(8)
            return px.pie(pie_df, names=dimension, values=value_col, title=f"{value_col.replace('_', ' ').title()} Distribution")

    if dimension:
        bar_df, value_col = aggregate_for_display(df, dimension, measure, aggregation)
        bar_df = bar_df.head(10)
        return px.bar(bar_df, x=dimension, y=value_col, title=f"{value_col.replace('_', ' ').title()} by {dimension.replace('_', ' ').title()}")

    if date_col:
        date_df = (
            df.assign(period=df[date_col].dt.to_period("M").dt.to_timestamp())
            .groupby("period", as_index=False)
            .agg({measure: aggregation if aggregation != "count" else "size"})
        )
        if aggregation == "count":
            date_df = date_df.rename(columns={measure: "count"})
            y_col = "count"
        else:
            y_col = measure
        return px.line(date_df, x="period", y=y_col, markers=True, title=f"{y_col.replace('_', ' ').title()} Trend")

    return create_default_chart(df, profile)


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def markdown_to_pdf_bytes(title: str, markdown_text: str) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, title=title)
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Spacer(1, 12)]

    for raw_line in markdown_text.splitlines():
        line = raw_line.strip()
        if not line:
            story.append(Spacer(1, 8))
            continue
        if line.startswith("### "):
            story.append(Paragraph(line[4:], styles["Heading3"]))
        elif line.startswith("## "):
            story.append(Paragraph(line[3:], styles["Heading2"]))
        elif line.startswith("# "):
            story.append(Paragraph(line[2:], styles["Heading1"]))
        elif line.startswith("- "):
            story.append(Paragraph(f"&bull; {line[2:]}", styles["BodyText"]))
        else:
            story.append(Paragraph(line, styles["BodyText"]))
        story.append(Spacer(1, 6))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()
