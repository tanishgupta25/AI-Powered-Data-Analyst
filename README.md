# AI-Powered Data Analyst

A complete Streamlit web app for non-technical users to upload CSV/Excel files, ask questions in natural language, and receive data outputs, charts, insights, and structured business reports.

## Features

- Upload CSV or Excel files
- Auto-clean column names and detect data types
- Handle medium datasets with row limiting and caching
- Column selection and sidebar filters
- Hinglish + English style query support
- Smart query suggestions based on the uploaded dataset
- Automatic summaries and chart generation
- Deterministic analytics for totals, averages, top performers, grouped analysis, trends, comparisons, and correlations
- Optional OpenAI narrative enhancement using `OPENAI_API_KEY`
- Optional PandasAI fallback for free-form query interpretation
- Chat history
- Download processed CSV and PDF report
- Graceful error handling and low-confidence warnings

## Project Structure

- `app.py`: Main Streamlit UI
- `utils.py`: Data loading, cleaning, profiling, filtering, export helpers
- `ai_module.py`: Query interpretation, analytics engine, optional LLM/PandasAI integration
- `data/sample_sales_data.csv`: Sample dataset for testing
- `requirements.txt`: Python dependencies

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set environment variables:

Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
$env:OPENAI_MODEL="gpt-4o-mini"
```

Use `.env.example` as a starting point. Optional `.env` values if you use your own loader:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

4. Run the app:

```bash
streamlit run app.py
```

## How It Works

Architecture:

`User -> Upload CSV/Excel -> Streamlit UI -> Pandas -> Optional PandasAI/OpenAI -> Insights + Charts + Report -> Display`

The app always performs a deterministic Pandas-based analysis first for reliability. If `OPENAI_API_KEY` is available, it adds a short business-friendly AI narrative. If `pandasai` is installed and configured, the app can also try PandasAI for fallback interpretation of very open-ended queries.

## Example Queries

- `Total sales batao`
- `Average profit`
- `Top 5 products`
- `Region wise sales`
- `Show sales trend`
- `Compare last 3 months`
- `Correlation between sales and profit`
- `Generate full analysis report`

## Notes

- Large files are limited to the first 1,000 to 5,000 rows for performance.
- Filters and column selection affect every answer and report.
- If the app shows a low-confidence warning, refine the query by naming a metric or category clearly.
- The app works best when the dataset has at least one numeric column, and even better if it also contains a date column.

## Sample Data

Use `data/sample_sales_data.csv` to test the full flow quickly.
