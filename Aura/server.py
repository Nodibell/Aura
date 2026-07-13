import os
import sys
import json
import asyncio
import subprocess
import argparse
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request

from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="Aura Local API Server", version="0.8.1")

# Resolve paths relative to this script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
ANALYZE_PY = os.path.join(SCRIPT_DIR, "analyze.py")
QUERY_DB_PY = os.path.join(SCRIPT_DIR, "query_db.py")

# ── Persistent REPL session (loaded once per server lifecycle) ──
import importlib.util as _ilu
repl_path = os.path.join(SCRIPT_DIR, "utils", "repl_session.py")
if not os.path.exists(repl_path):
    repl_path = os.path.join(SCRIPT_DIR, "repl_session.py")

_REPL_SPEC = _ilu.spec_from_file_location("repl_session", repl_path)
_repl: Optional[Any] = None
if _REPL_SPEC and _REPL_SPEC.loader:
    _repl = _ilu.module_from_spec(_REPL_SPEC)
    _REPL_SPEC.loader.exec_module(_repl)



class AnalyzeRequest(BaseModel):
    file_path: str
    target_col: Optional[str] = None
    dataset_type: str = "tabular"
    task_type_override: str = "auto"
    time_col: Optional[str] = None
    exclude_cols: Optional[str] = None
    test_file_path: Optional[str] = None
    val_file_path: Optional[str] = None
    model_export_path: Optional[str] = None
    code_export_path: Optional[str] = None
    notebook_export_path: Optional[str] = None   # Phase 16
    smart_sample: bool = False
    cleaning_actions: Optional[str] = None
    feature_selection: bool = False
    column_type_overrides: Optional[str] = None
    time_range_start: Optional[str] = None
    time_range_end: Optional[str] = None
    active_model: Optional[str] = None


class PreviewRequest(BaseModel):
    file_path: str
    dataset_type: str = "tabular"
    cleaning_actions: Optional[str] = None

class PredictRequest(BaseModel):
    model_path: str
    input_data: Optional[Dict[str, Any]] = None
    input_file_path: Optional[str] = None
    output_csv_path: Optional[str] = None

class MergeRequest(BaseModel):
    file1: str
    file2: str
    key1: str
    key2: str
    join_type: str = "inner"
    output_merge_path: str

class QueryDbRequest(BaseModel):
    db_type: str
    query: str
    conn_params: Dict[str, str]
    output_csv: str

class REPLExecRequest(BaseModel):
    code: str

class REPLResetRequest(BaseModel):
    file_path: str
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    cleaning_actions: Optional[str] = None
class REPLRollbackRequest(BaseModel):
    state_id: int



@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.8.1"}


async def run_subprocess_stream(args, original_file_path=None):
    """
    Runs a subprocess asynchronously and yields progress events and final stdout output.
    """
    # Disable Metal validation and enable fallback for PyTorch
    env = os.environ.copy()
    env["MTL_DEBUG_LAYER"] = "0"
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    if original_file_path:
        env["AURA_ORIGINAL_FILE_PATH"] = original_file_path

    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env
    )

    stdout_data = bytearray()

    async def read_stderr():
        while True:
            line_bytes = await proc.stderr.readline()
            if not line_bytes:
                break
            line = line_bytes.decode("utf-8").strip()
            if not line:
                continue
            
            if line.startswith("PROGRESS: "):
                try:
                    parts = line[len("PROGRESS: "):].split(":", 1)
                    if len(parts) == 2:
                        fraction = float(parts[0].strip())
                        message = parts[1].strip()
                        yield {"type": "progress", "progress": fraction, "message": message}
                except Exception:
                    pass
            else:
                yield {"type": "log", "message": line}

    async def read_stdout():
        data = await proc.stdout.read()
        stdout_data.extend(data)

    # Gather stderr streams
    stderr_generator = read_stderr()

    # Launch stdout reader as task
    stdout_task = asyncio.create_task(read_stdout())

    try:
        async for event in stderr_generator:
            yield f"data: {json.dumps(event)}\n\n"
    except asyncio.CancelledError:
        proc.terminate()
        await proc.wait()
        raise

    await stdout_task
    await proc.wait()

    if proc.returncode != 0:
        err_msg = stdout_data.decode("utf-8") if stdout_data else "Process execution failed."
        yield f"data: {json.dumps({'type': 'error', 'error': f'Subprocess returned non-zero code {proc.returncode}. Detail: {err_msg}'})}\n\n"
        return

    try:
        result_json = json.loads(stdout_data.decode("utf-8"))
        yield f"data: {json.dumps({'type': 'result', 'data': result_json})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'error': f'Failed to parse JSON result: {str(e)}'})}\n\n"

@app.post("/analyze")
async def analyze_endpoint(req: AnalyzeRequest):
    python_exe = sys.executable
    args = [python_exe, ANALYZE_PY, req.file_path, "--dataset-type", req.dataset_type, "--task-type", req.task_type_override]
    
    if req.target_col:
        args += ["--target", req.target_col]
    if req.time_col:
        args += ["--time-col", req.time_col]
    if req.exclude_cols:
        args += ["--exclude-cols", req.exclude_cols]
    if req.test_file_path:
        args += ["--test-file", req.test_file_path]
    if req.val_file_path:
        args += ["--val-file", req.val_file_path]
    if req.model_export_path:
        args += ["--model-export-path", req.model_export_path]
    if req.code_export_path:
        args += ["--code-export-path", req.code_export_path]
    if req.notebook_export_path:
        args += ["--notebook-export-path", req.notebook_export_path]

    if req.smart_sample:
        args.append("--smart-sample")
    if req.cleaning_actions:
        args += ["--cleaning-actions", req.cleaning_actions]
    if req.feature_selection:
        args.append("--feature-selection")
    if req.column_type_overrides:
        args += ["--column-type-overrides", req.column_type_overrides]
    if req.time_range_start:
        args += ["--time-range-start", req.time_range_start]
    if req.time_range_end:
        args += ["--time-range-end", req.time_range_end]
    if req.active_model:
        args += ["--active-model", req.active_model]
 
    return StreamingResponse(run_subprocess_stream(args, original_file_path=req.file_path), media_type="text/event-stream")

@app.post("/analyze/arrow")
async def analyze_arrow_endpoint(request: Request):
    """
    Accepts raw binary stream containing:
    [4 bytes JSON len (big-endian)] + [JSON config bytes] + [Arrow IPC Table bytes]
    Saves the Arrow table to a temp parquet file and executes the pipeline.
    """
    try:
        body = await request.body()
        if len(body) < 4:
            raise HTTPException(status_code=400, detail="Binary payload too small (missing header).")
        
        json_len = int.from_bytes(body[:4], byteorder="big")
        if len(body) < 4 + json_len:
            raise HTTPException(status_code=400, detail="Binary payload truncated (JSON config missing).")
            
        json_bytes = body[4 : 4 + json_len]
        arrow_bytes = body[4 + json_len :]
        
        req_dict = json.loads(json_bytes.decode("utf-8"))
        original_path = req_dict.get("file_path")
        req = AnalyzeRequest(**req_dict)
        
        import pyarrow as pa
        import tempfile
        
        # Load Arrow IPC Table
        reader = pa.ipc.open_stream(arrow_bytes)
        table = reader.read_all()
        df = table.to_pandas()
        import hashlib
        
        # Save DataFrame to a unique persistent file in aura_cache
        cache_dir = os.path.join(tempfile.gettempdir(), "aura_cache")
        os.makedirs(cache_dir, exist_ok=True)
        hash_str = hashlib.sha256(arrow_bytes).hexdigest()
        temp_file_path = os.path.join(cache_dir, f"arrow_{hash_str[:16]}.parquet")
        
        if not os.path.exists(temp_file_path):
            df.to_parquet(temp_file_path)
            
        # Re-route file path to the temporary file
        req.file_path = temp_file_path
        
        # Assemble arguments
        python_exe = sys.executable
        args = [python_exe, ANALYZE_PY, req.file_path, "--dataset-type", req.dataset_type, "--task-type", req.task_type_override]
        
        if req.target_col:
            args += ["--target", req.target_col]
        if req.time_col:
            args += ["--time-col", req.time_col]
        if req.exclude_cols:
            args += ["--exclude-cols", req.exclude_cols]
        if req.test_file_path:
            args += ["--test-file", req.test_file_path]
        if req.val_file_path:
            args += ["--val-file", req.val_file_path]
        if req.model_export_path:
            args += ["--model-export-path", req.model_export_path]
        if req.code_export_path:
            args += ["--code-export-path", req.code_export_path]
        if req.notebook_export_path:
            args += ["--notebook-export-path", req.notebook_export_path]
        if req.smart_sample:
            args.append("--smart-sample")
        if req.cleaning_actions:
            args += ["--cleaning-actions", req.cleaning_actions]
        if req.feature_selection:
            args.append("--feature-selection")
        if req.column_type_overrides:
            args += ["--column-type-overrides", req.column_type_overrides]
        if req.time_range_start:
            args += ["--time-range-start", req.time_range_start]
        if req.time_range_end:
            args += ["--time-range-end", req.time_range_end]
        if req.active_model:
            args += ["--active-model", req.active_model]
            
        return StreamingResponse(run_subprocess_stream(args, original_file_path=original_path), media_type="text/event-stream")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Arrow IPC analyze failed: {str(e)}")


async def run_subprocess_simple(args):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env
    )
    stdout, stderr = await proc.communicate()
    
    if proc.returncode != 0:
        err_msg = stderr.decode("utf-8") or stdout.decode("utf-8") or "Execution failed."
        raise HTTPException(status_code=500, detail=f"Process exited with {proc.returncode}: {err_msg}")
        
    try:
        return json.loads(stdout.decode("utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse result JSON: {str(e)}")

@app.post("/preview")
async def preview_endpoint(req: PreviewRequest):
    python_exe = sys.executable
    args = [python_exe, ANALYZE_PY, req.file_path, "--dataset-type", req.dataset_type, "--preview"]
    if req.cleaning_actions:
        args += ["--cleaning-actions", req.cleaning_actions]
    return StreamingResponse(run_subprocess_stream(args), media_type="text/event-stream")

@app.post("/predict")
async def predict_endpoint(req: PredictRequest):
    python_exe = sys.executable
    args = [
        python_exe, ANALYZE_PY,
        "--predict",
        "--model-path", req.model_path
    ]
    if req.input_file_path:
        args += ["--input-file-path", req.input_file_path]
        if req.output_csv_path:
            args += ["--output-csv-path", req.output_csv_path]
    elif req.input_data:
        args += ["--input-data", json.dumps(req.input_data)]
    else:
        raise HTTPException(status_code=400, detail="Either input_data or input_file_path must be provided.")
        
    return await run_subprocess_simple(args)

@app.post("/merge")
async def merge_endpoint(req: MergeRequest):
    python_exe = sys.executable
    args = [
        python_exe, ANALYZE_PY,
        req.file1,
        "--merge",
        "--file2", req.file2,
        "--key1", req.key1,
        "--key2", req.key2,
        "--join-type", req.join_type,
        "--output-merge-path", req.output_merge_path
    ]
    return await run_subprocess_simple(args)

@app.post("/query_db")
async def query_db_endpoint(req: QueryDbRequest):
    python_exe = sys.executable
    args = [
        python_exe, QUERY_DB_PY,
        "--db-type", req.db_type,
        "--query", req.query,
        "--conn-params", json.dumps(req.conn_params),
        "--output-csv", req.output_csv
    ]
    return await run_subprocess_simple(args)

# MARK: - REPL Endpoints (Phase 16: Agentic AI Analyst)

@app.post("/repl/reset")
async def repl_reset_endpoint(req: REPLResetRequest):
    """Load a new dataset into the persistent Python REPL sandbox."""
    if _repl is None:
        raise HTTPException(status_code=503, detail="REPL session module not loaded.")
    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: _repl.reset(req.file_path, req.ollama_base_url, req.ollama_model, req.cleaning_actions)
    )
    if result.get("status") == "error":
        import sys
        sys.stderr.write(f"REPL reset failed for file_path {req.file_path}: {result.get('error')}\n")
        sys.stderr.flush()
        raise HTTPException(status_code=500, detail=result.get("error", "REPL reset failed."))
    return result

@app.post("/repl/exec")
async def repl_exec_endpoint(req: REPLExecRequest):
    """Execute Python code in the persistent REPL sandbox."""
    if _repl is None:
        raise HTTPException(status_code=503, detail="REPL session module not loaded.")
    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: _repl.execute(req.code)
    )
    return result

@app.get("/repl/lineage")
async def repl_lineage_endpoint():
    """Retrieve the time-travel lineage state tree nodes."""
    if _repl is None:
        raise HTTPException(status_code=503, detail="REPL session module not loaded.")
    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: _repl.get_lineage()
    )
    return result

@app.post("/repl/rollback")
async def repl_rollback_endpoint(req: REPLRollbackRequest):
    """Roll back the REPL state to a previous state ID."""
    if _repl is None:
        raise HTTPException(status_code=503, detail="REPL session module not loaded.")
    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: _repl.rollback(req.state_id)
    )
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("error"))
    return result
@app.get("/plugins")
async def plugins_endpoint():
    """Scan and list dynamic Python plugins schema descriptions from ~/Documents/Aura/Plugins."""
    try:
        import os
        import json
        import re
        import importlib.util
        
        home = os.path.expanduser("~")
        plugins_dir = os.path.join(home, "Documents", "Aura", "Plugins")
        os.makedirs(plugins_dir, exist_ok=True)
        
        # Write a sample default plugin if the folder is empty so the user has an example!
        if not os.listdir(plugins_dir):
            sample_path = os.path.join(plugins_dir, "sample_robust_scaler.py")
            with open(sample_path, "w", encoding="utf-8") as f:
                f.write('''"""
{
  "name": "Robust Outlier Scaler",
  "description": "Scales numeric columns by trimming custom quantiles dynamically",
  "parameters": [
    {"name": "lower_quantile", "type": "slider", "min": 0.0, "max": 0.3, "default": 0.1},
    {"name": "upper_quantile", "type": "slider", "min": 0.7, "max": 1.0, "default": 0.9}
  ]
}
"""
import numpy as np

def transform(df, lower_quantile=0.1, upper_quantile=0.9):
    # Select numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        q_low = df[col].quantile(lower_quantile)
        q_high = df[col].quantile(upper_quantile)
        if q_high > q_low:
            df[col] = (df[col] - q_low) / (q_high - q_low)
    return df
''')
                
        plugins = []
        for fname in os.listdir(plugins_dir):
            if not fname.endswith(".py"):
                continue
            plugin_path = os.path.join(plugins_dir, fname)
            try:
                spec = importlib.util.spec_from_file_location(fname[:-3], plugin_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                doc = module.__doc__
                if doc:
                    doc_str = doc.strip()
                    try:
                        schema = json.loads(doc_str)
                        schema["id"] = fname[:-3]
                        plugins.append(schema)
                    except json.JSONDecodeError:
                        match = re.search(r"(\{.*\})", doc_str, re.DOTALL)
                        if match:
                            schema = json.loads(match.group(1))
                            schema["id"] = fname[:-3]
                            plugins.append(schema)
            except Exception as e:
                import sys
                sys.stderr.write(f"Warning: Failed to load plugin {fname}: {e}\n")
        return plugins
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cache/info")
async def cache_info_endpoint():
    import tempfile
    cache_dir = os.path.join(tempfile.gettempdir(), "aura_cache")
    if not os.path.exists(cache_dir):
        return {"path": cache_dir, "size_bytes": 0, "file_count": 0}
    
    total_size = 0
    file_count = 0
    for root, dirs, files in os.walk(cache_dir):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.exists(fp):
                try:
                    total_size += os.path.getsize(fp)
                    file_count += 1
                except Exception:
                    pass
    return {"path": cache_dir, "size_bytes": total_size, "file_count": file_count}

@app.post("/cache/clean")
async def cache_clean_endpoint():
    import tempfile
    import shutil
    cache_dir = os.path.join(tempfile.gettempdir(), "aura_cache")
    if os.path.exists(cache_dir):
        try:
            shutil.rmtree(cache_dir)
            os.makedirs(cache_dir, exist_ok=True)
            return {"status": "ok", "message": "Cache successfully cleared."}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")
    return {"status": "ok", "message": "Cache directory did not exist."}



if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=11435)
    args = parser.parse_args()
    
    uvicorn.run("server:app", host="127.0.0.1", port=args.port, log_level="info")
