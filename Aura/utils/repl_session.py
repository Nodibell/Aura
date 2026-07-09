"""
repl_session.py — Persistent Python REPL sandbox for Aura's agentic AI Analyst.

Architecture
────────────
• A single shared `_namespace` dict acts as the REPL's global scope.
• `df` is loaded into it on `reset(file_path)`.
• `execute(code)` runs arbitrary user/AI-generated code inside the namespace,
  captures stdout/stderr, captures matplotlib figures as base64 PNG, and
  returns a structured dict.
• `llm_query(chunk, question, model, base_url)` is injected into the namespace
  so the AI can call it from its generated code for recursive sub-LM processing.
• Security: a lightweight denylist blocks the most dangerous stdlib calls
  (os.system, subprocess, open in write mode, __import__ overrides).
  This is a defence-in-depth layer — Aura already runs on the user's own machine.

Used by server.py via the /repl/exec and /repl/reset endpoints.
"""

from __future__ import annotations

import io
import sys
import json
import traceback
import contextlib
import re
import base64
import urllib.request
import urllib.error
from typing import Any

# ---------------------------------------------------------------------------
# Shared REPL state
# ---------------------------------------------------------------------------

_namespace: dict[str, Any] = {}
_lineage_states: list[dict[str, Any]] = []

_BLOCKED = re.compile(
    r"\b(os\.system|subprocess\.|__import__\s*\(|open\s*\([^)]*['\"]w['\"])\b"
)


def _make_llm_query(base_url: str, model: str):
    """
    Returns a callable `llm_query(chunk, question)` bound to `base_url`/`model`.
    This is injected into the REPL namespace so the AI can use it from code.
    """
    def llm_query(chunk: str, question: str, timeout: int = 30) -> str:
        """
        Perform a recursive sub-LLM call to Ollama to process a text chunk.

        Args:
            chunk:    The text fragment to analyse (keep short, < 2000 chars).
            question: The question to ask about this chunk.
            timeout:  Max seconds to wait for Ollama (default 30).

        Returns:
            The LLM's response as a string, or an error message.
        """
        prompt = (
            f"You are a data analysis sub-agent. Answer concisely.\n\n"
            f"TEXT CHUNK:\n{chunk}\n\n"
            f"QUESTION: {question}\n\n"
            f"Answer in 1-3 sentences:"
        )
        payload = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("response", "").strip()
        except urllib.error.URLError as e:
            return f"[llm_query error: Ollama not reachable — {e}]"
        except Exception as e:
            return f"[llm_query error: {e}]"
    return llm_query


def reset(file_path: str,
          ollama_base_url: str = "http://localhost:11434",
          ollama_model: str = "llama3.2",
          cleaning_actions: str = None) -> dict:
    """
    Load `file_path` into `df` inside the REPL namespace.
    Clears all previous session variables.
    """
    global _namespace
    _namespace = {}

    if file_path.startswith("http://") or file_path.startswith("https://"):
        try:
            from utils.loader import download_dataset
            file_path = download_dataset(file_path)
        except Exception as e:
            return {"status": "error", "error": f"Failed to download remote dataset: {str(e)}"}

    try:
        import pandas as pd
        import numpy as np

        # Smart loader: support CSV, TSV, Parquet, JSON, Excel
        fp = file_path.lower()
        if fp.endswith(".parquet"):
            df = pd.read_parquet(file_path)
        elif fp.endswith(".json") or fp.endswith(".jsonl"):
            df = pd.read_json(file_path, lines=fp.endswith(".jsonl"))
        elif fp.endswith((".xls", ".xlsx")):
            df = pd.read_excel(file_path)
        elif fp.endswith(".tsv"):
            df = pd.read_csv(file_path, sep="\t")
        else:
            df = pd.read_csv(file_path)

        _namespace.update({
            "df": df,
            "pd": pd,
            "np": np,
            "llm_query": _make_llm_query(ollama_base_url, ollama_model),
            "_file_path": file_path,
        })
        # Optionally import matplotlib non-interactively
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            _namespace["plt"] = plt
            _namespace["matplotlib"] = matplotlib
        except ImportError:
            pass

        # Save initial state lineage
        import pyarrow as pa
        _lineage_states.clear()
        initial_table = pa.Table.from_pandas(df)
        _lineage_states.append({
            "id": 0,
            "description": "Initial Load",
            "shape": df.shape,
            "table": initial_table
        })

        if cleaning_actions:
            try:
                import json
                from utils.cleaning import StatefulCleaner
                actions = json.loads(cleaning_actions)
                if actions:
                    cleaner = StatefulCleaner(actions)
                    cleaner.fit(df)
                    df = cleaner.transform(df, is_training=True)
                    _namespace["df"] = df
                    
                    cleaned_table = pa.Table.from_pandas(df)
                    _lineage_states.append({
                        "id": len(_lineage_states),
                        "description": f"Applied {len(actions)} Cleaning Action(s)",
                        "shape": df.shape,
                        "table": cleaned_table
                    })
            except Exception as clean_err:
                import sys
                sys.stderr.write(f"Warning: Failed to fit/apply cleaning actions in REPL: {str(clean_err)}\n")

        return {"status": "ok", "rows": len(df), "cols": len(df.columns)}
    except Exception as e:
        return {"status": "error", "error": str(e)}



def execute(code: str) -> dict:
    """
    Execute `code` in the shared REPL namespace.

    Returns:
        {
          "stdout": str,
          "error":  str | None,
          "figures": [base64_png_str, ...]  # matplotlib figures auto-captured
        }
    """
    # --- Security denylist ---
    if _BLOCKED.search(code):
        return {
            "stdout": "",
            "error": "SecurityError: code contains a blocked operation "
                     "(os.system, subprocess, open-for-write, __import__).",
            "figures": []
        }

    # --- Close any leftover matplotlib figures before execution ---
    try:
        plt = _namespace.get("plt")
        if plt:
            plt.close("all")
    except Exception:
        pass

    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    figures: list[str] = []

    try:
        with contextlib.redirect_stdout(stdout_capture), \
             contextlib.redirect_stderr(stderr_capture):
            exec(compile(code, "<repl>", "exec"), _namespace)  # noqa: S102

        # --- Capture matplotlib figures ---
        try:
            plt = _namespace.get("plt")
            if plt:
                for fig_num in plt.get_fignums():
                    fig = plt.figure(fig_num)
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=96, bbox_inches="tight")
                    buf.seek(0)
                    figures.append(base64.b64encode(buf.read()).decode("utf-8"))
                plt.close("all")
        except Exception:
            pass

        # --- Track DataFrame Lineage mutations ---
        try:
            import pandas as pd
            import pyarrow as pa
            df = _namespace.get("df")
            if df is not None and isinstance(df, pd.DataFrame):
                # Ensure column names are strictly string typed for Arrow
                df.columns = df.columns.astype(str)
                current_table = pa.Table.from_pandas(df)
                last_state = _lineage_states[-1] if _lineage_states else None
                if last_state is None or not current_table.equals(last_state["table"]):
                    # Deduce description
                    desc = code.split("\n")[0].strip()
                    if not desc:
                        desc = "Executed Python code"
                    elif len(desc) > 50:
                        desc = desc[:47] + "..."
                        
                    new_id = len(_lineage_states)
                    _lineage_states.append({
                        "id": new_id,
                        "description": desc,
                        "shape": df.shape,
                        "table": current_table
                    })
        except Exception as snap_err:
            sys.stderr.write(f"Warning: Lineage snapshot failed: {snap_err}\n")

        out = stdout_capture.getvalue()
        err = stderr_capture.getvalue()
        return {
            "stdout": out[:8000],   # cap to avoid flooding the context window
            "error": err[:2000] if err.strip() else None,
            "figures": figures
        }


    except Exception:
        tb = traceback.format_exc()
        return {
            "stdout": stdout_capture.getvalue(),
            "error": tb[:2000],
            "figures": []
        }



def get_lineage() -> list[dict]:
    """Returns a list of state nodes describing the cleaning/transformation lineage."""
    return [
        {
            "id": node["id"],
            "description": node["description"],
            "shape": f"{node['shape'][0]} rows × {node['shape'][1]} cols"
        }
        for node in _lineage_states
    ]


def rollback(state_id: int) -> dict:
    """Restores the DataFrame to a previous state ID and discards subsequent states."""
    global _namespace, _lineage_states
    if state_id < 0 or state_id >= len(_lineage_states):
        return {"status": "error", "error": f"Invalid state ID {state_id}"}
        
    node = _lineage_states[state_id]
    df = node["table"].to_pandas()
    _namespace["df"] = df
    
    # Prune subsequent states (branch rollback)
    _lineage_states = _lineage_states[:state_id + 1]
    
    return {
        "status": "ok",
        "active_state": state_id,
        "rows": len(df),
        "cols": len(df.columns)
    }


# ---------------------------------------------------------------------------
# Self-test (python utils/repl_session.py --selftest)
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse, os, tempfile

    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        # Create a tiny temp CSV
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                         delete=False) as f:
            f.write("a,b,c\n1,2,3\n4,5,6\n7,8,9\n")
            tmp_path = f.name

        print("=== reset() ===")
        res = reset(tmp_path)
        print(res)
        assert res["status"] == "ok", f"reset failed: {res}"

        print("\n=== execute: df.head() ===")
        res = execute("print(df.head())")
        print(res)
        assert res["error"] is None

        print("\n=== execute: security block ===")
        res = execute("import os; os.system('echo hacked')")
        print(res)
        assert res["error"] and "SecurityError" in res["error"]

        print("\n=== execute: matplotlib figure ===")
        if "plt" in _namespace:
            res = execute("plt.figure(); plt.plot([1,2,3]); plt.title('test')")
            print(f"figures captured: {len(res['figures'])}")
            assert len(res["figures"]) == 1
        else:
            print("matplotlib not installed, skipping figure capture self-test.")

        print("\n=== execute: lineage check ===")
        # Mutate the dataframe
        res = execute("df = df.iloc[:-1]")
        print("Lineage after execution:", get_lineage())
        assert len(get_lineage()) == 2
        assert get_lineage()[1]["description"] == "df = df.iloc[:-1]"

        print("\n=== execute: rollback check ===")
        res = rollback(0)
        print("Rollback output:", res)
        assert res["status"] == "ok"
        assert len(get_lineage()) == 1
        assert len(_namespace["df"]) == 3

        os.unlink(tmp_path)
        print("\n✅ All self-tests passed.")

