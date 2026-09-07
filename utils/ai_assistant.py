import os
import re
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional
from utils.rag_engine import DatasetRAGEngine

class AIAssistantEngine:
    """
    Intelligent Conversational AI Assistant with RAG knowledge grounding,
    safe text-to-query execution, dynamic Plotly chart rendering, and
    optional Google Gemini / LLM integration.
    """
    def __init__(self, df: pd.DataFrame, col_types: dict, rag_engine: Optional[DatasetRAGEngine] = None, api_key: Optional[str] = None):
        self.df = df
        self.col_types = col_types
        self.rag_engine = rag_engine
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")

    def process_message(self, user_message: str, chat_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Processes a user question and returns:
        - 'text': Markdown formatted assistant response
        - 'chart': Optional Plotly figure to display
        - 'table': Optional DataFrame preview
        - 'rag_citations': List of knowledge chunks referenced
        - 'code_snippet': Optional Python/SQL code generated
        """
        if not user_message or not user_message.strip():
            return {
                'text': "Please enter a question or query regarding your dataset.",
                'chart': None,
                'table': None,
                'rag_citations': [],
                'code_snippet': None
            }

        user_query_clean = user_message.strip()
        
        # Check if Google Gemini API key is configured and can be invoked
        if self.api_key and len(self.api_key) > 10:
            llm_res = self._try_gemini_generation(user_query_clean, chat_history)
            if llm_res:
                return llm_res

        # Otherwise execute Local RAG + Deterministic Analytical Engine
        return self._local_intelligent_dispatch(user_query_clean)

    def _local_intelligent_dispatch(self, query: str) -> Dict[str, Any]:
        q_lower = query.lower()
        numeric_cols = self.col_types.get('numeric', [])
        categorical_cols = self.col_types.get('categorical', [])
        date_cols = self.col_types.get('date', [])
        all_cols = list(self.df.columns)

        # Retrieve RAG context chunks
        rag_citations = []
        if self.rag_engine:
            rag_citations = self.rag_engine.retrieve_context(query, top_k=3)

        matched_num_col = self._match_column(q_lower, numeric_cols)
        matched_cat_col = self._match_column(q_lower, categorical_cols)

        if not matched_num_col and numeric_cols:
            matched_num_col = numeric_cols[0]
        if not matched_cat_col and categorical_cols:
            matched_cat_col = categorical_cols[0]

        # 1. INTENT: Chart / Plot Generation Request
        if any(kw in q_lower for kw in ['plot', 'chart', 'graph', 'visualize', 'histogram', 'scatter', 'bar', 'distribution']):
            fig = None
            explanation = ""
            
            if 'scatter' in q_lower and len(numeric_cols) >= 2:
                col_x = numeric_cols[0]
                col_y = numeric_cols[1]
                fig = px.scatter(
                    self.df, x=col_x, y=col_y,
                    color=matched_cat_col if matched_cat_col else None,
                    title=f"Scatter Analysis: {col_x} vs {col_y}",
                    template="plotly_white"
                )
                explanation = f"Generated interactive scatter plot correlating `{col_x}` against `{col_y}`."
            elif any(kw in q_lower for kw in ['trend', 'line', 'time', 'over time']) and date_cols:
                d_col = date_cols[0]
                df_sorted = self.df.sort_values(by=d_col)
                fig = px.line(
                    df_sorted, x=d_col, y=matched_num_col,
                    title=f"Temporal Trend: {matched_num_col} over {d_col}",
                    template="plotly_white"
                )
                explanation = f"Rendered chronological time-series trajectory of `{matched_num_col}`."
            elif matched_cat_col and matched_num_col:
                agg_df = self.df.groupby(matched_cat_col)[matched_num_col].sum().reset_index().sort_values(by=matched_num_col, ascending=False).head(10)
                fig = px.bar(
                    agg_df, x=matched_cat_col, y=matched_num_col,
                    color=matched_num_col,
                    color_continuous_scale="Blues",
                    title=f"Top {matched_cat_col} by {matched_num_col}",
                    template="plotly_white"
                )
                explanation = f"Plotted ranked aggregation of `{matched_num_col}` across top levels of `{matched_cat_col}`."
            elif matched_num_col:
                fig = px.histogram(
                    self.df, x=matched_num_col, nbins=30,
                    title=f"Distribution of {matched_num_col}",
                    template="plotly_white",
                    color_discrete_sequence=["#0284C7"]
                )
                explanation = f"Generated 30-bin frequency distribution histogram for `{matched_num_col}`."

            if fig:
                fig.update_layout(margin=dict(l=30, r=30, t=50, b=30), height=380)
                return {
                    'text': f"### Dynamic Visualization Generated\n{explanation}\n\nYou can interact with, zoom, and inspect data points on the plot below.",
                    'chart': fig,
                    'table': None,
                    'rag_citations': rag_citations,
                    'code_snippet': f"# Python / Plotly Code:\nimport plotly.express as px\nfig = px.bar(df.groupby('{matched_cat_col}')['{matched_num_col}'].sum().reset_index(), x='{matched_cat_col}', y='{matched_num_col}')\nfig.show()"
                }

        # 2. INTENT: Top / Rank Grouping
        if any(kw in q_lower for kw in ['top', 'highest', 'max', 'best', 'leading', 'rank']) and matched_cat_col and matched_num_col:
            grouped = self.df.groupby(matched_cat_col)[matched_num_col].agg(['sum', 'mean', 'count']).reset_index()
            grouped = grouped.sort_values(by='sum', ascending=False).head(10)
            grouped.columns = [matched_cat_col, f"Total_{matched_num_col}", f"Avg_{matched_num_col}", "Record_Count"]
            
            top_entity = grouped.iloc[0][matched_cat_col]
            top_val = grouped.iloc[0][f"Total_{matched_num_col}"]
            
            summary_text = (
                f"### Top Ranking Analysis: `{matched_cat_col}` by `{matched_num_col}`\n"
                f"The highest performing cohort is **{top_entity}** with a total of **{top_val:,.2f}** "
                f"across **{grouped.iloc[0]['Record_Count']:,} records** (average: `{grouped.iloc[0][f'Avg_{matched_num_col}']:,.2f}`).\n\n"
                f"Here is the ranked top 10 breakdown table:"
            )
            return {
                'text': summary_text,
                'chart': None,
                'table': grouped,
                'rag_citations': rag_citations,
                'code_snippet': f"top_df = df.groupby('{matched_cat_col}')['{matched_num_col}'].agg(['sum', 'mean', 'count']).reset_index().sort_values(by='sum', ascending=False).head(10)"
            }

        # 3. INTENT: Statistical Summary / Average / Total
        if any(kw in q_lower for kw in ['average', 'mean', 'avg', 'total', 'sum', 'stats', 'metrics', 'describe']):
            if matched_num_col:
                series = self.df[matched_num_col].dropna()
                mean_v = series.mean()
                total_v = series.sum()
                med_v = series.median()
                min_v = series.min()
                max_v = series.max()
                std_v = series.std()
                
                stats_df = pd.DataFrame([{
                    "Metric": matched_num_col,
                    "Total Sum": f"{total_v:,.2f}",
                    "Mean": f"{mean_v:,.2f}",
                    "Median": f"{med_v:,.2f}",
                    "Min": f"{min_v:,.2f}",
                    "Max": f"{max_v:,.2f}",
                    "Std Dev": f"{std_v:,.2f}",
                    "Valid Rows": f"{len(series):,}"
                }])
                
                return {
                    'text': f"### Statistical Summary for `{matched_num_col}`\n"
                            f"• **Sum**: `{total_v:,.2f}`\n"
                            f"• **Mean (Average)**: `{mean_v:,.2f}`\n"
                            f"• **Median**: `{med_v:,.2f}`\n"
                            f"• **Range**: `[{min_v:,.2f} to {max_v:,.2f}]`\n"
                            f"• **Standard Deviation**: `{std_v:,.2f}` across **{len(series):,} records**.",
                    'chart': None,
                    'table': stats_df,
                    'rag_citations': rag_citations,
                    'code_snippet': f"df['{matched_num_col}'].describe()"
                }

        # 4. INTENT: Strategic Recommendations / Action Plan
        if any(kw in q_lower for kw in ['recommend', 'strategy', 'action', 'insight', 'improve', 'advice']):
            rec_text = (
                f"### Strategic AI Recommendations Grounded in Telemetry\n\n"
                f"1. **Cohort-Targeted Optimization**: Focus resources on the primary revenue-driving categories while auditing underperforming segments.\n"
                f"2. **Continuous Monitoring on Covariates**: Capitalize on observed coupling in `{matched_num_col}` to model downstream fiscal impact.\n"
                f"3. **Anomaly Boundary Triggers**: Establish real-time alerts for the top 5% extreme variance outliers to mitigate operational risk.\n"
                f"4. **Data Hygiene Continuity**: Preserve standard nanosecond datetime normalization and defensive median imputation."
            )
            return {
                'text': rec_text,
                'chart': None,
                'table': None,
                'rag_citations': rag_citations,
                'code_snippet': None
            }

        # 5. INTENT: Python / SQL Code Request
        if any(kw in q_lower for kw in ['sql', 'code', 'python', 'pandas', 'query']):
            code_str = (
                f"# Python Pandas Query Snippet\n"
                f"import pandas as pd\n\n"
                f"# Group by '{matched_cat_col}' and aggregate '{matched_num_col}'\n"
                f"summary_table = df.groupby('{matched_cat_col}')['{matched_num_col}'].agg([\n"
                f"    ('Total', 'sum'),\n"
                f"    ('Average', 'mean'),\n"
                f"    ('Count', 'count')\n"
                f"]).reset_index().sort_values(by='Total', ascending=False)\n\n"
                f"print(summary_table.head(10))\n\n"
                f"-- Equivalent ANSI SQL Query\n"
                f"SELECT {matched_cat_col},\n"
                f"       SUM({matched_num_col}) AS total_{matched_num_col.lower()},\n"
                f"       AVG({matched_num_col}) AS avg_{matched_num_col.lower()},\n"
                f"       COUNT(*) AS record_count\n"
                f"FROM telemetry_table\n"
                f"GROUP BY {matched_cat_col}\n"
                f"ORDER BY total_{matched_num_col.lower()} DESC\n"
                f"LIMIT 10;"
            )
            return {
                'text': f"### Generated Production Query Code (Pandas & SQL)\nHere is the optimized snippet to extract this metric from your telemetry pipeline:",
                'chart': None,
                'table': None,
                'rag_citations': rag_citations,
                'code_snippet': code_str
            }

        # General RAG Grounded Answer Fallback
        c_info = rag_citations[0]['chunk']['content'] if rag_citations else "Dataset verified with structured schemas."
        return {
            'text': f"### Telemetry Query Synthesis\n"
                    f"**Analysis on `{query}`**:\n\n"
                    f"The dataset contains **{len(self.df):,} rows** and **{len(self.df.columns)} features**. "
                    f"Relevant telemetry context extracted from RAG index:\n\n"
                    f"> *\"{c_info}\"*\n\n"
                    f"You can ask for specific metrics (e.g. *'average {matched_num_col}'*), grouped rankings (e.g. *'top {matched_cat_col} by {matched_num_col}'*), or dynamic charts (e.g. *'plot distribution of {matched_num_col}'*).",
            'chart': None,
            'table': None,
            'rag_citations': rag_citations,
            'code_snippet': None
        }

    def _match_column(self, query_lower: str, candidate_cols: list) -> Optional[str]:
        for c in candidate_cols:
            c_clean = str(c).lower().replace('_', ' ')
            if c_clean in query_lower:
                return c
            for word in c_clean.split():
                if len(word) > 2 and word in query_lower:
                    return c
        return None

    def _try_gemini_generation(self, user_query: str, chat_history: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            
            # Formulate RAG context
            rag_chunks_text = ""
            rag_citations = []
            if self.rag_engine:
                rag_citations = self.rag_engine.retrieve_context(user_query, top_k=4)
                rag_chunks_text = "\n\n".join([f"[{c['chunk']['category']}] {c['chunk']['title']}:\n{c['chunk']['content']}" for c in rag_citations])

            schema_summary = f"Columns: {list(self.df.columns)}\nTypes: {self.col_types.get('details', {})}\nSample Data:\n{self.df.head(3).to_dict(orient='records')}"
            
            system_prompt = (
                "You are an expert Data Scientist and Enterprise Telemetry Analyst. "
                "Answer the user query accurately based on the provided dataset context and RAG knowledge. "
                "Always format your response cleanly with clear markdown headers, bullet points, and actionable takeaways.\n\n"
                f"### RAG KNOWLEDGE BASE:\n{rag_chunks_text}\n\n"
                f"### DATASET SCHEMA & SAMPLE:\n{schema_summary}\n"
            )

            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(f"{system_prompt}\n\nUser Question: {user_query}")
            
            if response and response.text:
                return {
                    'text': response.text,
                    'chart': None,
                    'table': None,
                    'rag_citations': rag_citations,
                    'code_snippet': None
                }
        except Exception as e:
            # If API fails or quota exceeded, fall back gracefully to local engine
            pass
        return None
