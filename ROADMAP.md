# AutoEDA — Ultimate Data Analysis App: Roadmap & Vision

> **Based on:** PROJECT_ANALYSIS.md review (2026-06-15) + user feature ideas  
> **Current score:** 9.0/10 with Modular Upgrades & Advanced ML. **Target:** 9.5/10 as a real production tool.

---

## Table of Contents

1. [Current State: What Works and What Doesn't](#1-current-state)
2. [User Ideas (Immediate Priorities)](#2-user-ideas)
3. [Data Type Strategy: The Missing Foundation](#3-data-type-strategy)
4. [Feature Roadmap by Phase](#4-feature-roadmap-by-phase)
5. [Architecture Decisions](#5-architecture-decisions)
6. [Python Pipeline Overhaul Plan](#6-python-pipeline-overhaul)
7. [Swift UI Component Plan](#7-swift-ui-components)
8. [AI Integration Upgrade](#8-ai-integration)

---

## 1. Current State

### What Works Well
- ✅ End-to-end flow: drag/drop → preview → analysis → charts → AI chat → history
- ✅ CSV, TSV, Parquet, NPZ file support
- ✅ Kaggle + Hugging Face dataset download
- ✅ Real-time progress bar from Python subprocess
- ✅ Baseline ML (Logistic Regression, Random Forest)
- ✅ Ollama local AI chat with dataset context
- ✅ Analysis history with LLM-generated names
- ✅ AppLogger with in-app log viewer
- ✅ Cancel analysis button (latest addition)

### What's Broken or Missing (All Resolved)
| Resolved Gap | Status | Resolution Details |
|---|---|---|
| No Time Series support | ✅ Resolved | Added full TS pipeline with seasonal decomposition, ACF/PACF, and lags |
| No image dataset pipeline | ✅ Resolved | Added image pipeline supporting NPZ files, zip files, resizing, and average grids |
| No NLP/text-focused pipeline | ✅ Resolved | Added NLP pipeline with TF-IDF, Naive Bayes, word importances, and length histograms |
| Target column auto-detect is fragile | ✅ Resolved | User can manually select target/task configuration in preview |
| Task type heuristic is wrong | ✅ Resolved | User has dropdown overrides for ML task type |
| No user confirmation of task setup | ✅ Resolved | Sidebar configuration flow allows custom task confirmation before analysis |
| No ignored/excluded columns | ✅ Resolved | Excluded columns can be toggled in Table Preview and passed to Python via `--exclude-cols` |
| No full table view in analysis | ✅ Resolved | FullTableView allows viewing complete dataset rows up to 500 records |
| AI chart context is not structured | ✅ Resolved | Added Ask AI button on charts passing data points structure to Ollama |
| No export (MD/PDF/HTML) | ✅ Resolved | Export service generated for Markdown reports with charts and narrative |
| One train/test split, no CV | ✅ Resolved | Implemented 5-Fold Cross-Validation |
| No dummy baseline | ✅ Resolved | Fit baseline model and show baseline metrics comparison |
| Credentials in UserDefaults | ✅ Resolved | Migrated Kaggle/HF credentials securely to KeychainService |
| Hardcoded developer paths | ✅ Resolved | Implemented dynamic python interpreter path resolution |
| No requirements.txt / venv | ✅ Resolved | requirements.txt + setup_env.sh bootstrap script in root folder |
| No separate train/test datasets support | ✅ Resolved | Added optional separate validation/test dataset loading with combined feature pre-processing and target evaluation |
| No pipeline/model export | ✅ Resolved | Added model leaderboard export sheet for scikit-learn models (.joblib) and reproduction scripts (.py) |
| Performance issues on large datasets | ✅ Resolved | Implemented smart stratified/random sampling down to 100,000 rows with user warning badges |
| No interactive cleaning refits | ✅ Resolved | Integrated interactive recommendations wizard allowing users to apply mean/median/mode imputation, outlier clipping, or drop columns directly in SummaryView and re-run python analysis |
| Monolithic python backend script | ✅ Resolved | Modularized `analyze.py` into specialized submodules under `pipelines/` and `utils/` folder references with full validation coverage |

---

## 2. User Ideas (Immediate Priorities)

### Idea 1: Data Type Selector in Table Preview + Row Exclusion

**What the user wants:**
- In the **Table Preview** screen (after loading, before analysis), let the user choose the **dataset type**: Tabular, Time Series, Image, Text/NLP, Recommendations.
- Allow the user to **check/uncheck rows** to exclude specific rows from analysis (e.g., header duplicates, obviously wrong rows, test outliers).

**How to implement:**

#### Swift Side (`PreviewTableView.swift` + `ContentView.swift`)

1. Add `DatasetType` enum to `AnalysisResult.swift` or a new `AnalysisConfig.swift`:
```swift
enum DatasetType: String, CaseIterable, Codable {
    case tabular = "Tabular"
    case timeSeries = "Time Series"
    case image = "Image"
    case nlp = "Text / NLP"
    case recommendation = "Recommendations"
    
    var icon: String {
        switch self {
        case .tabular: return "tablecells"
        case .timeSeries: return "chart.line.uptrend.xyaxis"
        case .image: return "photo.stack"
        case .nlp: return "text.bubble"
        case .recommendation: return "star.fill"
        }
    }
    
    var description: String {
        switch self {
        case .tabular: return "Standard rows/columns for classification or regression"
        case .timeSeries: return "Ordered sequence with a datetime/index column"
        case .image: return "Pixel arrays (NPZ/folder), image classification"
        case .nlp: return "Text-heavy columns, sentiment, topic modeling"
        case .recommendation: return "User-item interaction matrix"
        }
    }
}
```

2. Add a `@State var selectedDatasetType: DatasetType = .tabular` in `ContentView.swift`.

3. In `PreviewTableView.swift`, add a **toolbar / top bar** with a `Picker` for dataset type:
```
┌─────────────────────────────────────────────────────────┐
│ Dataset Type: [Tabular ▼]   [Time Series ▼]   ...       │
│ ─────────────────────────────────────────────────────── │
│  ☑  | col1 | col2 | col3 | col4 ...                     │
│  ☑  | ...                                               │
│  ☐  | (row excluded from analysis)                      │
└─────────────────────────────────────────────────────────┘
```

4. Add a **row checkbox** column on the left side of `PreviewTableView`. Store excluded row indices in a `@State var excludedRows: Set<Int>`.

5. Pass `selectedDatasetType` and `excludedRows` as arguments to `runEDA()`. Encode them as CLI args or environment variables to `analyze.py`.

#### Python Side (`analyze.py`)

- Accept `--dataset-type` argument: `tabular`, `timeseries`, `image`, `nlp`, `recommendation`.
- Accept `--exclude-rows` as a comma-separated list of row indices to skip.
- Route to the correct analysis sub-pipeline based on dataset type.

**Impact:** This one change unblocks ALL other data types (Time Series, Images, etc.) because the user explicitly tells the app what type of data it is dealing with.

---

### Idea 2: Full Table View in Analysis Results

**What the user wants:**
- After running full analysis, be able to browse the **complete dataset** (all rows), not just the initial preview.

**How to implement:**

1. In `analyze.py`, include the preview rows (first N rows from the actual processed `df`) in the JSON output:
```python
# In the result dict, add:
"full_preview": {
    "columns": list(df.columns),
    "rows": df.head(500).fillna("").astype(str).values.tolist(),
    "total_rows": int(row_count)
}
```

2. Add `fullPreview` to `AnalysisResult.swift`:
```swift
struct FullTablePreview: Codable {
    let columns: [String]
    let rows: [[String]]
    let totalRows: Int
}
```

3. Add a **"Data Table"** tab alongside Summary / Charts / Correlations in `ContentView.swift`:
```swift
Picker("View", selection: $selectedTab) {
    Text("Summary").tag("Summary")
    Text("Charts").tag("Charts")
    Text("Correlations").tag("Correlations")
    Text("Data Table").tag("DataTable")  // NEW
}
```

4. Create `FullTableView.swift` that reuses the same `PreviewTableView` component but with:
   - Virtualized row rendering (only render visible rows for performance)
   - A search/filter bar
   - Column sorting
   - "Load more" pagination if > 500 rows

---

### Idea 3: Structured Chart Data Always Passed to AI

**What the user wants:**
- When the user clicks "Ask AI" or sends a message while viewing a chart, the **chart's actual data** should be included in the AI context — not just a description, but the real numbers.

**Current problem:**
The `ChatViewModel.injectContext()` sends a text summary, but chart data (feature importances, correlations, distributions) is not structured in the prompt. The AI is guessing about specifics.

**How to implement:**

1. Update `AIChatPanel.swift` to accept the currently visible `ChartConfig`:
```swift
struct AIChatPanel: View {
    @Binding var activeChart: ChartConfig?  // NEW: currently visible chart
    ...
}
```

2. When user sends a message, prepend the chart data to the prompt automatically:
```swift
private func buildContextualPrompt(_ userMessage: String) -> String {
    var context = ""
    if let chart = activeChart {
        context += "### Current Chart: \(chart.title)\n"
        context += "X-axis: \(chart.xLabel), Y-axis: \(chart.yLabel)\n"
        context += "Data points:\n"
        for point in chart.data.prefix(30) {
            if let xv = point.xVal {
                context += "- \(xv): \(String(format: "%.4f", point.y))\n"
            } else if let xn = point.xNum {
                context += "- x=\(String(format: "%.3f", xn)), y=\(String(format: "%.4f", point.y))\n"
            }
        }
        context += "\n"
    }
    return context + userMessage
}
```

3. In `ChartsListView.swift`, add an "Ask AI about this chart" button on each chart card that:
   - Sets `activeChart` to that specific chart
   - Calls `onAskAI` with the structured prompt

4. Update `ChatViewModel.injectContext()` to also serialize the top charts into JSON and include them in the system prompt:
```swift
func injectContext(_ result: AnalysisResult) {
    let chartSummary = result.charts.prefix(5).map { chart in
        let dataStr = chart.data.prefix(20).map { pt in
            pt.xVal.map { "\($0): \(pt.y)" } ?? "(\(pt.xNum ?? 0)): \(pt.y)"
        }.joined(separator: ", ")
        return "Chart '\(chart.title)': [\(dataStr)]"
    }.joined(separator: "\n")
    
    systemPrompt = """
    You are an expert data analyst. Here is the dataset context:
    ...
    Charts data:
    \(chartSummary)
    Always reference specific numbers from the data when answering.
    """
}
```

---

### Idea 4: Export MD/PDF with Formulas and Charts (using AI)

**What the user wants:**
- Export a polished **Markdown report** containing:
  - Dataset overview, summary stats
  - All charts (as rendered images or ASCII)
  - Model performance with formulas
  - AI-generated interpretation section

**How to implement:**

#### Step 1: Render Charts to Images
Use SwiftUI's `ImageRenderer` (macOS 13+) to capture chart views as PNG:
```swift
@MainActor
func captureChartImage(_ chartView: some View) async -> NSImage? {
    let renderer = ImageRenderer(content: chartView.frame(width: 600, height: 350))
    renderer.scale = 2.0  // retina
    return renderer.nsImage
}
```

#### Step 2: Generate MD Report with AI
Send the full analysis context to Ollama and request a structured report:
```swift
func generateMarkdownReport(result: AnalysisResult, chartImages: [String: URL]) async -> String {
    let prompt = """
    Generate a professional data analysis report in Markdown format for:
    Dataset: \(result.rowCount) rows × \(result.colCount) columns
    Task: \(result.taskType) on '\(result.targetColumn)'
    Best Model: \(result.metrics.model) (\(result.metrics.scoreType): \(result.metrics.score))
    Summary: \(result.summary)
    
    Include:
    1. Executive Summary
    2. Dataset Overview with statistics table
    3. Key Findings with specific numbers
    4. Model Performance Analysis with formulas (e.g. R² = ..., Accuracy = ...)
    5. Recommendations
    
    Use proper Markdown headers (##), bold key numbers, and code blocks for formulas.
    """
    // Call Ollama API...
}
```

#### Step 3: Assemble Final Document
```swift
func buildFullReport(result: AnalysisResult, aiNarrative: String, chartPaths: [String: URL]) -> String {
    var md = "# Analysis Report: \(result.targetColumn)\n\n"
    md += "_Generated by AutoEDA on \(Date().formatted())_\n\n"
    md += "---\n\n"
    md += aiNarrative + "\n\n"
    
    md += "## Charts\n\n"
    for chart in result.charts {
        if let imagePath = chartPaths[chart.title] {
            md += "### \(chart.title)\n"
            md += "![\(chart.title)](\(imagePath.path))\n\n"
        }
    }
    return md
}
```

#### Step 4: Export UI
- Add an **Export** button in the toolbar when analysis is complete.
- Show a sheet with options: Markdown / PDF / HTML.
- Use `NSSavePanel` to let user choose location.

---

## 3. Data Type Strategy

This is the most important architectural decision. The app currently only works for **tabular classification/regression**. Supporting other data types requires a **routing system** in the Python pipeline.

### 3.1 Time Series

**Unique requirements:**
- Must detect datetime/timestamp column automatically
- Chronological ordering matters — random train/test split is **wrong**
- Need temporal validation (walk-forward or last N% as test)
- Key charts: time plot, decomposition (trend/seasonality/residual), ACF/PACF
- Key models: ARIMA, Prophet, SARIMA, exponential smoothing, gradient boosting on lag features

**Python libraries needed:**
```
statsmodels     # ARIMA, decomposition, ACF/PACF
prophet         # Facebook Prophet (optional, heavy)
scikit-learn    # Gradient Boosting with lag features
```

**What `analyze.py` needs to do:**
1. Auto-detect datetime columns (try `pd.to_datetime()` on all columns)
2. Sort by datetime
3. Extract features: hour, day, weekday, month, lag_1, lag_7, rolling_mean_7
4. Use temporal split: last 20% as test
5. Return decomposition data, ACF values, and forecast chart

**New chart types to add to Swift:**
- `"line"` chart with a time axis (already partially exists)
- `"decomposition"` type with 4 subplots

### 3.2 Image Data

**Unique requirements:**
- NPZ pixel arrays (CIFAR-10 style): 50000×32×32×3
- Folder of JPG/PNG files
- Statistics: mean pixel intensity, class distribution, sample grid
- Models: Simple CNN metrics via sklearn neural_network, or just class balance analysis
- Charts: sample image grid, class distribution, mean image per class

**Python libraries needed:**
```
Pillow          # Image loading
numpy           # Array operations (already present)
```

**What `analyze.py` needs to do:**
1. Detect if input is pixel array (shape[1:] has ≥2 dims)
2. Flatten for sklearn or use it raw for image stats
3. Compute per-class mean images
4. Return class distribution, mean intensity histogram

**New chart types to add to Swift:**
- `"image_grid"` type — render a grid of sample images
- Need base64-encoded sample images in JSON response

### 3.3 Text / NLP

**Unique requirements:**
- Target is sentiment, category, topic
- Features are raw text columns
- Preprocessing: tokenization, stop words, TF-IDF (already partial), embeddings
- Key metrics: per-class precision/recall, confusion matrix
- Key models: Logistic Regression on TF-IDF (already exists but limited), Naive Bayes

**Python libraries needed:**
```
scikit-learn    # TF-IDF, Naive Bayes (already present)
nltk            # Optional: stop words, tokenization
```

**Improvements to existing code:**
- Increase `max_features` from 15 to 500 for NLP mode
- Add Multinomial Naive Bayes as a baseline
- Add word cloud data (top 50 terms by TF-IDF weight)
- Add per-class F1 breakdown in the JSON output

### 3.4 Recommendations / Collaborative Filtering

**Unique requirements:**
- Input: user_id × item_id × rating matrix (sparse)
- Models: SVD matrix factorization, KNN-based, popularity baseline
- Metrics: RMSE on held-out ratings, coverage, diversity
- Charts: rating distribution, user activity histogram, item popularity

**Python libraries needed:**
```
scipy           # Sparse matrices
scikit-learn    # SVD via TruncatedSVD
surprise        # (optional) Specialized CF library
```

---

## 4. Feature Roadmap by Phase

### Phase A — Immediate (1-2 weeks)
These are the 4 user ideas + critical stability fixes.

| # | Feature | File(s) | Status |
|---|---------|---------|--------|
| A1 | Data type selector in Preview | `PreviewTableView.swift`, `ContentView.swift` | ✅ Done |
| A2 | Row exclusion checkboxes in Preview | `PreviewTableView.swift`, `analyze.py` | ❌ Removed (User feedback) |
| A3 | Full dataset table tab in Analysis | `ContentView.swift`, new `FullTableView.swift` | ✅ Done |
| A4 | Chart data in AI prompts | `AIChatPanel.swift`, `ChatViewModel.swift` | ✅ Done |
| A5 | MD export with AI narrative + charts | new `ExportService.swift`, toolbar button | ✅ Done |
| A6 | User-confirmed target/task config | `ContentView.swift`, sidebar flow | ✅ Done |
| A7 | Fix summary rendering (MarkdownMessageView) | `SummaryView.swift` | ✅ Done |
| A8 | Cancel analysis button | `PythonRunner.swift`, `ContentView.swift` | ✅ Done |

### Phase B — Core Quality (2-4 weeks)
Make the existing tabular pipeline trustworthy.

| # | Feature | File(s) | Status |
|---|---------|---------|--------|
| B1 | Stratified split for classification | `analyze.py` | ✅ Done |
| B2 | K-fold cross-validation (k=5) | `analyze.py` | ✅ Done |
| B3 | Dummy baseline model | `analyze.py` | ✅ Done |
| B4 | Confusion matrix data in JSON | `analyze.py`, `AnalysisResult.swift` | ✅ Done |
| B5 | Full data profiling (unique, cardinality, quantiles) | `analyze.py`, `SummaryView.swift` | ✅ Done |
| B6 | Outlier detection with IQR boxes in charts | `analyze.py`, `ChartsListView.swift` | ✅ Done |
| B7 | requirements.txt + venv bootstrap script | root dir | ✅ Done |
| B8 | Keychain for Kaggle/HF credentials | `SettingsView.swift`, new `KeychainService.swift` | ✅ Done |

### Phase C — Data Type Support (4-8 weeks)
The big expansion.

| # | Feature | Pipeline | Status |
|---|---------|----------|--------|
| C1 | Time Series detection + temporal split | `analyze.py` → `analyze_timeseries.py` | ✅ Done |
| C2 | Time decomposition chart (trend/seasonal/residual) | Python + `ChartsListView.swift` | ✅ Done |
| C3 | ACF/PACF chart | Python + Swift | ✅ Done |
| C4 | Lag feature engineering for TS | `analyze.py` | ✅ Done |
| C5 | Image dataset pipeline (NPZ + folder) | `analyze.py` → `analyze_images.py` | ✅ Done |
| C6 | Image grid chart type in Swift | `ChartsListView.swift` | ✅ Done |
| C7 | NLP pipeline upgrade (Naive Bayes, bigger TF-IDF) | `analyze.py` | ✅ Done |
| C8 | Word cloud data + chart | Python + Swift | ✅ Done |
| C9 | Per-class F1 breakdown chart | Python + Swift | ✅ Done |

### Phase D — Advanced ML (8-12 weeks)

| # | Feature | Status | Details / Resolution |
|---|---------|--------|----------------------|
| D1 | SHAP values for feature importance | ✅ Done | Calculated using TreeExplainer on Random Forest model |
| D2 | Permutation importance | ✅ Done | Model-agnostic inspection calculated on test split |
| D3 | ROC/PR AUC curves | ✅ Done | Binary classification curves downsampled to 100 points |
| D4 | Residuals plots for regression | ✅ Done | Predicted vs. actual residuals scatter with zero line |
| D5 | Calibration plots for classification | ✅ Done | Probability calibration curve comparing models |
| D6 | ARIMA / Prophet for time series forecasting | ✅ Done | Statsmodels ARIMA(1,d,1) model integrated and compared |
| D7 | Data leakage detection warnings | ✅ Done | Pearson correlation >= 0.98 or target name substring match |
| D8 | Automatic cleaning recommendations | ✅ Done | Triggers warning cards for missing, outliers, cardinality, or constants |

### Phase E — Production Readiness

| # | Feature |
|---|---------|
| E1 | Hardened runtime + sandbox entitlements |
| E2 | Notarization pipeline |
| E3 | Python unit tests (pytest) |
| E4 | Swift unit tests |
| E5 | GitHub Actions CI |
| E6 | README + install guide |
| E7 | .gitignore + remove __pycache__ |
| E8 | Localization (Ukrainian / English) |

---

## 5. Architecture Decisions

### 5.1 Multi-Pipeline Python Architecture

Instead of one monolithic `analyze.py`, split into pipeline modules:

```
AutoEDA/
  analyze.py              ← Entry point: routes to correct pipeline
  pipelines/
    tabular.py            ← Current logic, improved
    timeseries.py         ← New: temporal analysis
    images.py             ← New: pixel array analysis
    nlp.py                ← New: text-heavy analysis
    recommendations.py    ← New: CF/matrix factorization
  utils/
    loader.py             ← load_dataset(), download_dataset()
    preprocessor.py       ← Feature engineering shared utils
    charts.py             ← Chart generation helpers
    reporter.py           ← Summary/report generation
```

`analyze.py` becomes a simple router:
```python
def main():
    dataset_type = args.dataset_type  # passed from Swift
    if dataset_type == "timeseries":
        from pipelines.timeseries import run
    elif dataset_type == "image":
        from pipelines.images import run
    elif dataset_type == "nlp":
        from pipelines.nlp import run
    else:
        from pipelines.tabular import run
    result = run(args)
    print(json.dumps(clean_nan(result)))
```

### 5.2 Swift CLI Arguments Schema

Current: `analyze.py <file_path> [target_column] [--preview]`

Proposed (backward-compatible):
```
analyze.py <file_path> \
  [--target <column>] \
  [--task-type auto|classification|regression|forecast] \
  [--dataset-type tabular|timeseries|image|nlp|recommendation] \
  [--exclude-rows 0,5,12] \
  [--preview]
```

### 5.3 AnalysisConfig Model (New)

Add a new `AnalysisConfig.swift` struct that holds all user-confirmed settings before analysis:

```swift
struct AnalysisConfig {
    var datasetType: DatasetType = .tabular
    var targetColumn: String = ""
    var taskType: TaskType = .auto  // auto, classification, regression, forecast
    var excludedRowIndices: Set<Int> = []
    var excludedColumns: Set<String> = []
    var timeColumn: String? = nil    // for time series
    var idColumns: [String] = []     // columns to always ignore
}
```

Pass this config into `runEDA()` and serialize all fields to Python CLI args.

### 5.4 JSON Output Schema Evolution

Add new fields to the Python JSON output without breaking existing Swift decoding (use `decodeIfPresent`):

```json
{
  "summary": "...",
  "dataset_type": "timeseries",
  "full_preview": {"columns": [...], "rows": [[...]], "total_rows": 1000},
  "profiling": {
    "unique_counts": {"col1": 42},
    "cardinality": {"col1": "low"},
    "quantiles": {"col1": {"25": 1.2, "50": 3.4, "75": 7.8}}
  },
  "confusion_matrix": {"labels": [...], "values": [[...]]},
  "cv_scores": [0.82, 0.79, 0.85, 0.80, 0.83],
  "dummy_baseline_score": 0.51,
  "time_series_data": {
    "datetime_column": "date",
    "trend": [...],
    "seasonal": [...],
    "residual": [...],
    "acf": [...],
    "pacf": [...]
  }
}
```

---

## 6. Python Pipeline Overhaul

### 6.1 Target Detection Fix (Critical)

**Current broken code** (analyze.py:363-369):
```python
target_keywords = ["target", "label", "class", "price", "outcome", "y", "survived"]
for col in columns:
    if any(kw in col.lower() for kw in target_keywords):  # BUG: substring match
```

**Fixed version:**
```python
TARGET_KEYWORDS_EXACT = {"target", "label", "class", "y", "survived", "outcome"}
TARGET_KEYWORDS_PREFIX = {"price", "score", "rate", "count", "sales", "revenue"}

def detect_target(columns):
    col_lower_map = {c: c.lower() for c in columns}
    # Exact match first
    for col, lower in col_lower_map.items():
        if lower in TARGET_KEYWORDS_EXACT:
            return col
    # Prefix/suffix match (more conservative)
    for col, lower in col_lower_map.items():
        if any(lower.startswith(kw) or lower.endswith(kw) for kw in TARGET_KEYWORDS_PREFIX):
            return col
    # Default: last column
    return columns[-1]
```

### 6.2 Task Type Detection Fix

**Current broken code:**
```python
if target_col in categorical_cols or unique_targets <= 10:
    is_classification = True
```

**Fixed version with user override:**
```python
def detect_task_type(target_series, user_override=None):
    if user_override and user_override != "auto":
        return user_override == "classification"
    
    dtype = target_series.dtype
    unique = target_series.nunique()
    
    # String/object/bool → always classification
    if dtype == object or dtype == bool:
        return True
    
    # Numeric with few unique values — only classification if looks like a label
    if pd.api.types.is_integer_dtype(dtype) and unique <= 20:
        # Check if values are 0-based contiguous integers (label-like)
        values = sorted(target_series.dropna().unique())
        is_contiguous = values == list(range(int(values[0]), int(values[-1]) + 1))
        if is_contiguous and values[0] in [0, 1]:
            return True
    
    return False  # Default: regression
```

### 6.3 Better Model Evaluation

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.dummy import DummyClassifier, DummyRegressor

# Add dummy baseline
if is_classification:
    dummy = DummyClassifier(strategy="most_frequent")
    dummy_score = cross_val_score(dummy, X_processed, y, cv=5, scoring="accuracy").mean()
    
    # Stratified CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf_cv_scores = cross_val_score(rf, X_processed, y, cv=cv, scoring="accuracy")
    
metrics["dummy_baseline_score"] = float(dummy_score)
metrics["cv_scores"] = rf_cv_scores.tolist()
metrics["cv_mean"] = float(rf_cv_scores.mean())
metrics["cv_std"] = float(rf_cv_scores.std())
```

---

## 7. Swift UI Components

### 7.1 Dataset Type Selector (Idea 1)

**New component: `DatasetTypeSelector.swift`**

A horizontal pill-selector rendered in the PreviewTableView header:

```
[ 📊 Tabular ] [ 📈 Time Series ] [ 🖼 Image ] [ 💬 Text/NLP ] [ ⭐ Recommend ]
```

- Selecting a type shows a short description below
- If "Time Series" is selected, a secondary picker appears: "Which column is the timestamp?"
- If "Image" is selected, a note explains that pixel column layout will be used

### 7.2 Row Exclusion (Idea 1, part 2)

Add a checkbox column as the first column in `PreviewTableView`:
- Header checkbox = select/deselect all
- Individual row checkboxes stored in `@Binding var excludedRows: Set<Int>`
- Excluded rows shown with dimmed text + strikethrough
- A badge in the action bar shows "N rows excluded"

### 7.3 Full Data Table Tab (Idea 2)

New `FullTableView.swift` with:
- Virtualized `LazyVStack` for performance on large datasets
- Column header tap → sort ascending/descending
- Search bar filtering rows by value
- "Showing 1–100 of 50,000 rows" pagination footer
- Export button (copy column, export selection as CSV)

### 7.4 Chart Context AI Button (Idea 3)

Each chart card in `ChartsListView.swift` gets an updated "Ask AI" button that passes the full chart data object:

```swift
Button {
    let dataDescription = chart.data.prefix(25).map { pt in
        "\(pt.xVal ?? String(pt.xNum ?? 0)): \(String(format: "%.4f", pt.y))"
    }.joined(separator: ", ")
    
    let prompt = """
    Looking at the chart "\(chart.title)" (\(chart.xLabel) vs \(chart.yLabel)):
    Data: \(dataDescription)
    
    What are the key insights from this chart?
    """
    onAskAI(prompt)
} label: {
    Label("Ask AI", systemImage: "sparkles")
}
```

### 7.5 Export Report Sheet (Idea 4)

New `ExportReportSheet.swift`:

```
┌──────────────────────────────────┐
│  Export Analysis Report          │
│  ─────────────────────────────── │
│  Format:  [○ Markdown] [○ PDF]   │
│  Include: ☑ Charts (as images)   │
│           ☑ AI Narrative         │
│           ☑ Raw Statistics       │
│           ☐ Full Data Table      │
│                                  │
│  AI Model: [llama3.2 ▼]          │
│                                  │
│  [Generate & Export...]          │
└──────────────────────────────────┘
```

---

## 8. AI Integration Upgrade

### 8.1 Structured System Prompt

Replace the current free-form context injection with a structured system prompt that includes chart data:

```swift
func buildSystemPrompt(_ result: AnalysisResult) -> String {
    """
    You are an expert data analyst assistant. Analyze the following dataset:
    
    ## Dataset
    - Rows: \(result.rowCount), Columns: \(result.colCount)
    - Task: \(result.taskType.capitalized) on target '\(result.targetColumn)'
    - Missing cells: \(result.missingValues.values.reduce(0, +))
    
    ## Model Performance
    - Best: \(result.metrics.model) — \(result.metrics.scoreType): \(String(format: "%.4f", result.metrics.score))
    - All models: \(result.modelsCompared.map { "\($0.name): \($0.score)" }.joined(separator: ", "))
    
    ## Top Correlations
    \(result.correlations.prefix(10).map { "- \($0.x) ↔ \($0.y): \(String(format: "%.3f", $0.value))" }.joined(separator: "\n"))
    
    ## Charts Data
    \(result.charts.prefix(4).map { chart in
        let data = chart.data.prefix(15).map { pt in
            "\(pt.xVal ?? String(format: "%.2f", pt.xNum ?? 0)): \(String(format: "%.4f", pt.y))"
        }.joined(separator: ", ")
        return "### \(chart.title)\n\(data)"
    }.joined(separator: "\n\n"))
    
    Always ground your answers in the specific numbers above. 
    Do not hallucinate metrics or feature names not listed here.
    """
}
```

### 8.2 Chart-Specific Quick Actions

Add quick action buttons directly on each chart card:
- "What drives this pattern?" → includes chart data + correlation context
- "Is this statistically significant?" → asks AI to evaluate the distribution
- "Suggest transformations" → asks AI to recommend feature engineering

### 8.3 Export Report AI Generation

The AI generates the narrative portion of the export report. Prompt engineering:

```
You are a senior data scientist writing an analysis report.
Generate a professional report in Markdown. Be concise but precise.
Always reference exact numbers from the data. Use this structure:

## Executive Summary
(2-3 sentences with the most important findings)

## Dataset Quality
(Comment on size, missing values, data types)

## Model Performance  
(Compare models, explain the winner, use the exact metrics)

## Key Insights
(3-5 bullet points referencing specific features and correlations)

## Recommendations
(Actionable next steps for the data owner)
```

---

## Priority Implementation Order

All initial priority items have been successfully completed:
1. **Idea 1: Data Type Selector** — ✅ Completed
2. **Idea 2: Full Table View** — ✅ Completed
3. **Idea 3: Chart AI Context** — ✅ Completed
4. **Idea 4: Export Report** — ✅ Completed
5. **Time Series pipeline** — ✅ Completed
6. **Advanced ML Upgrades (Smart Sampling, Export Model, interactive Recommendations, Holt-Winters Forecasting)** — ✅ Completed
7. **Import & UI Enhancements (Dropdown Menus & URL alert inputs)** — ✅ Completed
8. **Time Series Visualizations (Continuous Scales, Year Filters, Monthly/Daily Seasonality Aggregations)** — ✅ Completed

---

*This document was updated to reflect all Phase 1-6 modular upgrades. Version: 2026-06-17*
