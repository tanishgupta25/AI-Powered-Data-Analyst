import os
from datetime import datetime

import pandas as pd
import streamlit as st

from ai_module import AIAnalystEngine
from utils import (
    PREVIEW_ROWS,
    apply_filters,
    build_filter_options,
    create_default_chart,
    dataframe_to_csv_bytes,
    detect_column_types,
    generate_auto_summary,
    generate_smart_suggestions,
    load_dataset,
    markdown_to_pdf_bytes,
    standardize_dataframe,
)


st.set_page_config(
    page_title="AI-Powered Data Analyst",
    page_icon=":bar_chart:",
    layout="wide",
)


def initialize_state():
    defaults = {
        "chat_history": [],
        "last_report": "",
        "last_processed_csv": b"",
        "selected_columns": [],
        "active_file_name": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar(df: pd.DataFrame, filter_options):
    st.sidebar.header("Workspace")
    st.sidebar.caption("Upload data, narrow the columns, and apply filters before asking questions.")

    valid_default_columns = [
        column for column in st.session_state.get("selected_columns", []) if column in df.columns
    ]
    if not valid_default_columns:
        valid_default_columns = df.columns.tolist()

    selected_columns = st.sidebar.multiselect(
        "Select columns for analysis",
        options=df.columns.tolist(),
        default=valid_default_columns,
        help="Use this to focus the AI on only the fields you care about.",
    )
    st.session_state["selected_columns"] = selected_columns or df.columns.tolist()

    active_filters = {}
    st.sidebar.subheader("Filters")
    for column, config in filter_options.items():
        if config["type"] == "date":
            start_default = config["min"].date() if pd.notna(config["min"]) else datetime.today().date()
            end_default = config["max"].date() if pd.notna(config["max"]) else datetime.today().date()
            date_range = st.sidebar.date_input(
                f"{column.replace('_', ' ').title()} range",
                value=(start_default, end_default),
            )
            if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
                active_filters[column] = {
                    "type": "date",
                    "start": date_range[0],
                    "end": date_range[1],
                }
        elif config["type"] == "category":
            selected = st.sidebar.multiselect(
                f"{column.replace('_', ' ').title()}",
                options=config["values"],
                default=config["values"],
            )
            active_filters[column] = {
                "type": "category",
                "selected": selected,
            }

    return st.session_state["selected_columns"], active_filters


def render_dataset_overview(df: pd.DataFrame, profile):
    summary = generate_auto_summary(df, profile)
    metric_cols = st.columns(4)
    metric_cols[0].metric("Rows", summary["rows"])
    metric_cols[1].metric("Columns", summary["columns"])
    metric_cols[2].metric("Missing Cells", summary["missing_cells"])
    metric_cols[3].metric("Primary Metric", summary.get("total_metric", "Not detected"))

    with st.expander("Preview and detected schema", expanded=True):
        st.dataframe(df.head(PREVIEW_ROWS), use_container_width=True)
        st.write("**Columns:**", ", ".join(df.columns))
        st.write("**Numeric:**", ", ".join(profile.numeric_columns) or "None")
        st.write("**Categorical:**", ", ".join(profile.categorical_columns) or "None")
        st.write("**Date:**", ", ".join(profile.date_columns) or "None")


def render_suggestions(suggestions):
    st.write("**Smart suggestions**")
    cols = st.columns(min(len(suggestions), 3) or 1)
    for idx, suggestion in enumerate(suggestions):
        if cols[idx % len(cols)].button(suggestion, use_container_width=True):
            st.session_state["query_input"] = suggestion


def render_chat_history():
    if not st.session_state["chat_history"]:
        return
    st.subheader("Chat History")
    for item in reversed(st.session_state["chat_history"][-6:]):
        with st.container(border=True):
            st.markdown(f"**Query:** {item['query']}")
            st.markdown(f"**Result:** {item['result_text']}")
            st.markdown(f"**Explanation:** {item['explanation']}")
            if item.get("warnings"):
                for warning in item["warnings"]:
                    if "LLM enhancement is not active" not in warning:
                        st.warning(warning)


def render_full_report(full_report):
    st.subheader("Full Automated Analysis Report")

    with st.container(border=True):
        st.markdown("### 1. Dataset Overview")
        for line in full_report.dataset_overview:
            st.markdown(f"- {line}")

    with st.container(border=True):
        st.markdown("### 2. Data Cleaning Summary")
        st.markdown(f"- {full_report.cleaning_summary.total_missing_handled} missing values handled")
        st.markdown(f"- {full_report.cleaning_summary.duplicate_rows_removed} duplicate rows removed")
        st.markdown(f"- {full_report.cleaning_summary.total_outliers_handled} outliers handled")
        for line in full_report.cleaning_summary.cleaning_actions:
            st.markdown(f"- {line}")

    with st.container(border=True):
        st.markdown("### 3. Statistical Summary")
        if not full_report.numeric_summary.empty:
            st.dataframe(full_report.numeric_summary, use_container_width=True)
        else:
            st.info("No numeric columns available for statistical analysis.")

        st.markdown("### 4. Categorical Frequency")
        if not full_report.categorical_summary.empty:
            st.dataframe(full_report.categorical_summary, use_container_width=True)
        else:
            st.info("No categorical columns available for frequency analysis.")

    with st.container(border=True):
        st.markdown("### 5. Relationships Found")
        for line in full_report.relationship_lines:
            st.markdown(f"- {line}")
        if not full_report.correlation_matrix.empty:
            st.dataframe(full_report.correlation_matrix, use_container_width=True)

    with st.container(border=True):
        st.markdown("### 6. Visualizations")
        section_order = [
            "Distribution Analysis",
            "Relationship Analysis",
            "Category Analysis",
            "Trend Analysis",
        ]
        grouped_charts = {section: [] for section in section_order}
        for chart in full_report.charts:
            grouped_charts.setdefault(chart.section, []).append(chart)

        chart_index = 0
        for section in section_order:
            section_charts = grouped_charts.get(section, [])
            if not section_charts:
                continue
            st.markdown(f"#### {section}")
            for chart in section_charts:
                st.markdown(f"**{chart.title}**")
                st.caption(chart.description)
                st.plotly_chart(
                    chart.figure,
                    use_container_width=True,
                    key=f"full_report_chart_{chart_index}_{chart.title.lower().replace(' ', '_')}",
                )
                st.markdown(f"- What it shows: {chart.description}")
                st.markdown(f"- Key observation: {chart.observation}")
                chart_index += 1

    with st.container(border=True):
        st.markdown("### 7. Key Insights")
        for line in full_report.insights:
            st.markdown(f"- {line}")

        st.markdown("### 8. Final Conclusion")
        st.write(full_report.conclusion)

        st.markdown("### 9. Explanation Of Actions")
        for line in full_report.action_log:
            st.markdown(f"- {line}")

    st.subheader("Structured Report Output")
    st.markdown(full_report.report_markdown)


def main():
    initialize_state()
    st.title("AI-Powered Data Analyst")
    st.caption("Upload CSV/Excel data, ask questions in English or Hinglish, and get results, charts, and business-ready reports.")

    with st.sidebar:
        uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
        max_rows = st.slider("Rows to analyze", min_value=1000, max_value=5000, value=3000, step=500)
        st.info("For performance, the app analyzes only the first selected rows from large files.")
        st.caption("Set `OPENAI_API_KEY` in your environment to enable LLM narrative enhancement.")

    if not uploaded_file:
        st.info("Upload a dataset to begin. A sample file is available in `data/sample_sales_data.csv` for testing.")
        return

    try:
        if st.session_state.get("active_file_name") != uploaded_file.name:
            st.session_state["selected_columns"] = []
            st.session_state["chat_history"] = []
            st.session_state["last_report"] = ""
            st.session_state["last_processed_csv"] = b""
            st.session_state["active_file_name"] = uploaded_file.name

        raw_df = load_dataset(uploaded_file.getvalue(), uploaded_file.name, max_rows=max_rows)
        cleaned_df, _column_map = standardize_dataframe(raw_df)
        report_base_df, _ = standardize_dataframe(raw_df, apply_basic_cleaning=False)
        profile = detect_column_types(cleaned_df)
        filter_options = build_filter_options(cleaned_df, profile)
        selected_columns, active_filters = render_sidebar(cleaned_df, filter_options)

        scoped_df = cleaned_df[selected_columns].copy() if selected_columns else cleaned_df.copy()
        filtered_df = apply_filters(scoped_df, active_filters)
        filtered_profile = detect_column_types(filtered_df)

        report_scoped_df = report_base_df[selected_columns].copy() if selected_columns else report_base_df.copy()
        report_filtered_df = apply_filters(report_scoped_df, active_filters)
        report_filtered_profile = detect_column_types(report_filtered_df)
    except Exception as exc:
        st.error(f"Could not load the dataset: {exc}")
        return

    if filtered_df.empty:
        st.warning("The current filters removed all rows. Please widen the filter selection.")
        return

    if report_filtered_df.empty:
        st.warning("The current filters removed all rows from the report dataset. Please widen the filter selection.")
        return

    render_dataset_overview(filtered_df, filtered_profile)

    overview_col, chart_col = st.columns([1.2, 1])
    with overview_col:
        render_suggestions(generate_smart_suggestions(filtered_df, filtered_profile))
    with chart_col:
        default_chart = create_default_chart(filtered_df, filtered_profile)
        if default_chart:
            st.plotly_chart(default_chart, use_container_width=True, key="overview_default_chart")

    st.subheader("Ask Your Question")
    query = st.text_input(
        "Type your question",
        key="query_input",
        placeholder="Examples: Total sales batao, Top 5 products, Show sales trend, Generate report",
    )

    action_cols = st.columns(2)
    run_analysis = action_cols[0].button("Analyze", type="primary", use_container_width=True)
    run_full_report = action_cols[1].button("Generate Full Report", use_container_width=True)
    engine = AIAnalystEngine(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    if run_full_report:
        with st.spinner("Running complete automated analysis..."):
            full_report = engine.generate_full_dataset_report(report_filtered_df, report_filtered_profile)

        render_full_report(full_report)
        st.session_state["last_report"] = full_report.report_markdown
        st.session_state["last_processed_csv"] = dataframe_to_csv_bytes(full_report.cleaned_df)

    if run_analysis:
        with st.spinner("Analyzing your data..."):
            result = engine.answer_query(filtered_df, filtered_profile, query)

        st.subheader(result.title)
        st.markdown(f"**Result:** {result.result_text}")
        st.markdown(f"**Explanation:** {result.explanation}")

        if result.warnings:
            for warning in result.warnings:
                if "LLM enhancement is not active" not in warning:
                    st.warning(warning)
        if result.confidence < 0.65:
            st.warning("Confidence is low for this answer. Please refine the query or verify the result from the table/chart.")
        else:
            st.success(f"Confidence score: {result.confidence:.0%}")

        if result.table is not None:
            st.dataframe(result.table, use_container_width=True)
        if result.chart is not None:
            chart_key = f"result_chart_{len(st.session_state['chat_history'])}_{query.strip().lower().replace(' ', '_') or 'query'}"
            st.plotly_chart(result.chart, use_container_width=True, key=chart_key)

        st.subheader("Structured Report")
        st.markdown(result.report_markdown)

        st.session_state["last_report"] = result.report_markdown
        st.session_state["last_processed_csv"] = dataframe_to_csv_bytes(filtered_df)
        st.session_state["chat_history"].append(
            {
                "query": query,
                "result_text": result.result_text,
                "explanation": result.explanation,
                "warnings": result.warnings or [],
            }
        )

    render_chat_history()

    if st.session_state["last_report"]:
        st.subheader("Downloads")
        pdf_bytes = markdown_to_pdf_bytes("AI Data Analyst Report", st.session_state["last_report"])
        download_cols = st.columns(2)
        download_cols[0].download_button(
            "Download Processed CSV",
            data=st.session_state["last_processed_csv"],
            file_name="processed_data.csv",
            mime="text/csv",
            use_container_width=True,
        )
        download_cols[1].download_button(
            "Download PDF Report",
            data=pdf_bytes,
            file_name="analysis_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
