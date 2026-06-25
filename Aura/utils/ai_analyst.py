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

    def query_ollama(self, prompt: str) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
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

    def generate_tabular_summary(self, target_col, best_model, metrics_dict, leaderboard) -> str:
        prompt = (
            f"You are Aura's AI Analyst. Summarize these machine learning training results:\n"
            f"Target Column: {target_col}\n"
            f"Best Model: {best_model}\n"
            f"Best Model Metrics: {json.dumps(metrics_dict, indent=2)}\n"
            f"Leaderboard: {json.dumps(leaderboard, indent=2)}\n"
            f"Explain the performance of the best model, interpret metrics, and provide 2-3 specific recommendations."
        )
        return self.query_ollama(prompt)

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
