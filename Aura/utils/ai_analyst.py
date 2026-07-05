import io
import json
import urllib.request
import urllib.error

class AIAnalyst:
    """
    Dedicated client for formatting context prompts and interacting with
    the local Ollama API or other configured LLMs.
    """
    def __init__(self, base_url="http://localhost:11434", model="llama3.2"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def query_ollama(self, prompt: str, images: list = None) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        if images:
            payload["images"] = images
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"}
        )
        try:
            # Short timeout to avoid blocking main execution threads if Ollama is busy/down
            with urllib.request.urlopen(req, timeout=10) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                return res_data.get("response", "").strip()
        except urllib.error.URLError as e:
            return f"Ollama not reachable: {str(e)}"
        except Exception as e:
            return f"Analysis generation error: {str(e)}"

    def generate_tabular_summary(self, target_col, best_model, metrics_dict, leaderboard, images: list = None) -> str:
        prompt = (
            f"You are Aura's AI Analyst. Summarize these machine learning training results:\n"
            f"Target Column: {target_col}\n"
            f"Best Model: {best_model}\n"
            f"Best Model Metrics: {json.dumps(metrics_dict, indent=2)}\n"
            f"Leaderboard: {json.dumps(leaderboard, indent=2)}\n"
            f"Explain the performance of the best model, interpret metrics, and provide 2-3 specific recommendations."
        )
        return self.query_ollama(prompt, images=images)

    def generate_dataset_context(self, df, target_col: str) -> str:
        """
        Builds a rich, token-efficient snapshot of `df` for injection into
        the AI system prompt. Covers dtypes, missing %, stats, top values,
        5 sample rows, and date range detection. Stays under ~3 000 tokens.
        """
        try:
            import pandas as pd
            import numpy as np

            lines = []
            lines.append(f"## Dataset Snapshot ({len(df):,} rows × {len(df.columns)} columns)")
            lines.append(f"Target column: `{target_col}`\n")

            # --- Column profiles ---
            lines.append("### Column Profiles")
            lines.append("| Column | Type | Non-null % | Min | Max | Mean/Top |".rstrip())
            lines.append("|--------|------|-----------|-----|-----|----------|".rstrip())

            for col in df.columns:
                dtype = str(df[col].dtype)
                non_null_pct = round(100 * df[col].notna().mean(), 1)
                col_display = col[:30]
                if pd.api.types.is_numeric_dtype(df[col]):
                    mn  = df[col].min()
                    mx  = df[col].max()
                    avg = df[col].mean()
                    mn_str  = f"{mn:.4g}" if mn is not None and mn == mn else "N/A"
                    mx_str  = f"{mx:.4g}" if mx is not None and mx == mx else "N/A"
                    avg_str = f"{avg:.4g}" if avg is not None and avg == avg else "N/A"
                    lines.append(f"| {col_display} | {dtype} | {non_null_pct}% | {mn_str} | {mx_str} | mean={avg_str} |")
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    mn_str = str(df[col].min())[:10]
                    mx_str = str(df[col].max())[:10]
                    lines.append(f"| {col_display} | datetime | {non_null_pct}% | {mn_str} | {mx_str} | — |")
                else:
                    top = df[col].dropna().mode()
                    top_val = str(top.iloc[0])[:40] if len(top) > 0 else "N/A"
                    n_unique = df[col].nunique()
                    lines.append(f"| {col_display} | {dtype} | {non_null_pct}% | — | — | top={top_val!r} ({n_unique} unique) |")

            # --- Top value counts for categorical cols (max 3 cols, 5 vals each) ---
            cat_cols = [c for c in df.columns if df[c].dtype == object and c != target_col][:3]
            if cat_cols:
                lines.append("\n### Top Categories")
                for col in cat_cols:
                    vc = df[col].value_counts().head(5)
                    vals = ", ".join([f"{v!r}: {cnt}" for v, cnt in vc.items()])
                    lines.append(f"- **{col}**: {vals}")

            # --- Date range (auto-detect) ---
            dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
            if not dt_cols:
                # Try parsing object columns that look like dates
                for c in df.select_dtypes(include='object').columns[:5]:
                    try:
                        parsed = pd.to_datetime(df[c], infer_datetime_format=True, errors='coerce')
                        if parsed.notna().mean() > 0.8:
                            dt_cols.append(c)
                            df = df.copy()
                            df[c] = parsed
                    except Exception:
                        pass
            if dt_cols:
                col = dt_cols[0]
                lines.append(f"\n### Date Range (column `{col}`)")
                lines.append(f"From **{df[col].min()}** to **{df[col].max()}**")

            # --- 5 sample rows as markdown table ---
            lines.append("\n### Sample Rows (first 5)")
            sample = df.head(5)
            headers = list(sample.columns)
            lines.append("| " + " | ".join(str(h)[:20] for h in headers) + " |")
            lines.append("|" + "---|" * len(headers))
            for _, row in sample.iterrows():
                cells = [str(row[c])[:30].replace("|", "╎") for c in headers]
                lines.append("| " + " | ".join(cells) + " |")

            return "\n".join(lines)
        except Exception as e:
            return f"(Dataset context generation failed: {e})"

    def generate_timeseries_summary(self, target_col, stationarity_info, best_model, metrics_dict) -> str:
        prompt = (
            f"You are Aura's AI Analyst. Summarize these time series forecasting results:\n"
            f"Target Column: {target_col}\n"
            f"Stationarity (ADF test): {stationarity_info}\n"
            f"Best Forecasting Model: {best_model}\n"
            f"Best Model Metrics: {json.dumps(metrics_dict, indent=2)}\n"
            f"Explain whether the data is stationary, interpret the forecasting R2/RMSE, and give recommendations."
        )
        return self.query_ollama(prompt)
