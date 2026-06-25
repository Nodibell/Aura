import os
import sys
import json
import asyncio
import subprocess
import argparse
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="Aura Local API Server", version="0.4.2")

# Resolve paths relative to this script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYZE_PY = os.path.join(SCRIPT_DIR, "analyze.py")
QUERY_DB_PY = os.path.join(SCRIPT_DIR, "query_db.py")

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
    smart_sample: bool = False
    cleaning_actions: Optional[str] = None
    feature_selection: bool = False
    column_type_overrides: Optional[str] = None

class PreviewRequest(BaseModel):
    file_path: str
    dataset_type: str = "tabular"

class PredictRequest(BaseModel):
    model_path: str
    input_data: Dict[str, Any]

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

@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.4.2"}

async def run_subprocess_stream(args):
    """
    Runs a subprocess asynchronously and yields progress events and final stdout output.
    """
    # Disable Metal validation and enable fallback for PyTorch
    env = os.environ.copy()
    env["MTL_DEBUG_LAYER"] = "0"
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    env["OMP_NUM_THREADS"] = "1"

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
    if req.smart_sample:
        args.append("--smart-sample")
    if req.cleaning_actions:
        args += ["--cleaning-actions", req.cleaning_actions]
    if req.feature_selection:
        args.append("--feature-selection")
    if req.column_type_overrides:
        args += ["--column-type-overrides", req.column_type_overrides]

    return StreamingResponse(run_subprocess_stream(args), media_type="text/event-stream")

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
    return StreamingResponse(run_subprocess_stream(args), media_type="text/event-stream")

@app.post("/predict")
async def predict_endpoint(req: PredictRequest):
    python_exe = sys.executable
    args = [
        python_exe, ANALYZE_PY,
        "--predict",
        "--model-path", req.model_path,
        "--input-data", json.dumps(req.input_data)
    ]
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

if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=11435)
    args = parser.parse_args()
    
    uvicorn.run("server:app", host="127.0.0.1", port=args.port, log_level="info")
