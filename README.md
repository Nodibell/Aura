# Aura

A native, premium macOS application for **AI-Powered Automated Exploratory Data Analysis (Aura) & Machine Learning Modeling**. Built with a modern SwiftUI 6.0 frontend and a high-performance Python 3 subprocess pipeline backend, Aura helps you instantly profile, visualize, and model complex datasets directly on your machine.

![Aura App Icon](Aura/Assets.xcassets/AppIcon.appiconset/icon_256x256.png)

---

## 🌟 Key Features

### 1. Multi-Format Automated Data Profiling
* **Inteligent Type Detection**: Auto-detects column types: numeric, categorical, text/NLP, identifiers, and datetime.
* **Smart Exclusions**: Automatically flags and filters out datetime and identifier columns from training features to avoid leakage.
* **Correlations & Descriptives**: Provides summary statistics, histograms, and top correlation matrices.

### 2. Tabular, NLP, & Image Modeling
* **Tabular ML**: Train and compare regression and classification leaderboards (e.g., Random Forest, XGBoost, Linear/Logistic Regression).
* **Text / NLP Feature Extraction**: Auto-vectorizes text features with TF-IDF for direct document classification/regression.
* **Image Classification**: Full support for `.npz` multi-dimensional array image datasets.
* **SHAP Feature Importance**: Calculate and plot global feature impact using Tree-SHAP.

### 3. Multi-Target Time Series Forecasting
* **ARIMA Comparison**: Fits classical ARIMA alongside machine learning predictors.
* **Multi-Target Switcher**: Select multiple targets simultaneously in the sidebar and toggle between their forecasted results dynamically.
* **Seasonal & Trend Analysis**: Filter forecasts by specific years and explore calendar-seasonal cycles ("Month of Year", "Day of Month").

### 4. Automated Semantic Image Segmentation
* **Paired Image/Mask Loader**: Automatically detects paired image and mask directories.
* **Pixel-Level Random Forest**: Train classifiers on pixel grids.
* **Dice & IoU Metrics**: Calculates precision metrics and displays prediction overlay comparison grids.

### 5. Local AI Analyst (Ollama)
* **Secure Private LLM**: Connects to local models (e.g., `llama3.2`, `qwen2.5`) via Ollama. No data leaves your machine.
* **Dynamic Title Generation**: Generates clean descriptive titles for your analysis history in the background.
* **Markdown Report Export**: Compile full analyses, charts, and AI narrative reviews into clean Markdown files.

### 7. Interactive Preprocessing & AutoML (New in v0.2.0)
* **Custom Imputation & Outlier Treatments**: Configure custom imputation (Mean, Median, Mode, KNN, MICE) and outlier filters (Cap IQR, Drop IQR, Isolation Forest) per column.
* **Target Encoding**: Stateful target encoding for high-cardinality categorical variables without target leakage.
* **Optuna Hyperparameter Tuning**: Auto-tuned RandomForest and XGBoost regressors and classifiers with strict run-time timeout budgeting.

### 8. Enterprise Features & Live Inference (New in v0.3.0)
* **Interactive Live Inference**: Test predictions on-the-fly inside the app using the best-performing trained model.
* **Direct Database Connectivity**: Connect directly to SQLite, MySQL, and PostgreSQL to pull datasets into Aura.
* **Periodical Analysis Scheduler**: Automate hourly, daily, or weekly data scans and auto-export HTML/PDF reports to a designated folder.
* **Visual Run Comparison (Analysis Diff)**: Compare multiple runs of the same dataset side-by-side to track changes in metrics, feature importance, and performance.
* **Dataset Merging Engine**: Combine and merge different tables directly in-app using inner, outer, left, and right joins.
* **PDF Report Compiler**: Standardized PDF report generation, compiling data profiles, visualizations, and local AI summaries.
* **Clustering & Deep Learning Pipelines**: Added K-Means/DBSCAN clustering pipelines with PCA visualization, plus PyTorch tabular Neural Networks.

### 9. Hardening, Optimization & UI Polish (New in v0.3.1)
* **HDBSCAN Clustering**: Replaced standard DBSCAN with `HDBSCAN` in [clustering.py](file:///Users/oleksiichumak/Developer/Xcode.projects/Aura/Aura/pipelines/clustering.py) to automatically determine cluster densities and avoid manual threshold settings, integrated with column mappings in [ChartsListView.swift](file:///Users/oleksiichumak/Developer/Xcode.projects/Aura/Aura/Views/ChartsListView.swift) for interactive drill-downs.
* **PCA Image Squashing**: Compresses wide image arrays into 100 components using scikit-learn `Pipeline` and `PCA` inside the CV loop in [image.py](file:///Users/oleksiichumak/Developer/Xcode.projects/Aura/Aura/pipelines/image.py) to optimize memory usage.
* **Stepwise `pmdarima` Selection**: Automatically finds optimal $p, d, q$ ARIMA configurations using stepwise search in [timeseries.py](file:///Users/oleksiichumak/Developer/Xcode.projects/Aura/Aura/pipelines/timeseries.py).
* **Live Holt-Winters Forecasting**: Fully integrated Exponential Smoothing forecasting inside the interactive Live Inference prediction panel.
* **Light Theme Report PDF**: Refactored HTML/PDF compiling engines in [ReportCompiler.swift](file:///Users/oleksiichumak/Developer/Xcode.projects/Aura/Aura/Services/ReportCompiler.swift) to dynamically render contrast-optimized light mode layouts for printed PDFs and premium dark mode interfaces for HTML exports.
* **Interactive Chart Legend Selection**: Allows toggling and filtering data series dynamically in the SwiftUI SVD projection charts in real time.

### 10. Local Microservice & Preview Enhancements (New in v0.4.0)
* **Local FastAPI Backend**: Migrated the Swift-to-Python IPC from stdout parsing to a robust local FastAPI microservice process (`server.py`) running asynchronously on localhost.
* **Real-Time Preview Streaming**: Refactored `/preview` and the Swift client code to use Server-Sent Events (SSE), enabling live progress reporting (e.g. download percentages for large remote Kaggle/HuggingFace datasets) to prevent the UI from freezing.
* **YOLO Dataset Preview Optimization**: Implemented random file sampling for splits exceeding 1,000 images, keeping YOLO / object detection dataset previews near-instant.
* **Dynamic Prediction Picker Constraints**: Enforces integer-only steps/inputs on prediction panels when column profiling detects only integer values.
* **Manual Column Type Overrides**: Added interactive dropdown menus in the preview table headers to let users verify and override system-inferred column types before training.
* **Automated ID/Identifier Exclusions**: Automatically deselects unique identifiers/IDs from training features, with an optional toggle to re-enable them.

### 11. Local Server Control & Clean Termination (New in v0.4.1)
* **Local Server Control Panel**: Added a dedicated **Local Server** settings tab to monitor status (Running/Stopped/Starting/Stopping), address, port, and process ID (PID) of the backend.
* **Manual Lifecycle Controls**: Users can now manually **Start**, **Stop**, and **Restart** the API server directly from the settings panel.
* **Synchronous Process Termination**: Integrates a thread-safe process manager connected synchronously to application exit. Quitting the application immediately and cleanly terminates all background Python server and child processes.

### 12. Advanced Models, Data Prep & Fixes (New in v0.4.3)
* **LightGBM & CatBoost Integration**: Automatically imports and trains LightGBM and CatBoost classifiers/regressors in AutoML, exporting the winning model cleanly.
* **Multi-Metric Leaderboard Sorting**: Users can select custom sorting metrics (Weighted F1, Accuracy, Precision, Recall, MSE, RMSE, MAE) with smart-scaled charts.
* **Column Renaming in Data Cleaning**: Added live variable renaming inside the interactive Data Cleaning metadata panel.
* **Time Series Date Range Picker**: Integrated start/end date range controls for filtering datasets dynamically prior to ARIMA / ML model runs.
* **Multi-Label NLP Classification**: Inferred multi-label string formats (comma-separated tags) auto-binarize using `MultiLabelBinarizer` and train via `OneVsRestClassifier` pipelines.
* **Offline PDF Chart Polish**: Bundled local `echarts.min.js` and `marked.min.js` assets in-app to enable high-quality PDF exports without external CDN requirements. Solved chart overlaps via stacked printing grids.
* **Model Saver Memory Polish**: Solved `SIGKILL` (-9) OOM crashes during model export by adding zlib `compress=3` to `joblib.dump()`.

### 13. Agentic AI Analyst, REPL Sandbox & Jupyter Export (New in v0.5.0)
* **Agentic REPL Loop**: Aura automatically intercepts Python execution blocks (`<execute_python>`) generated by the Ollama AI Analyst, executes them safely inside a Python REPL sandbox, and feeds the results back to the LLM to complete analysis tasks.
* **Rich Dataset Context (RAG)**: Automatically generates token-efficient column profiles, statistical distributions, and row previews to feed into the local LLM's system prompt.
* **Jupyter Notebook Export**: Export entire automated modeling runs directly into standard `.ipynb` notebooks for downstream execution.
* **Custom Python Plugins**: Extend the Data Cleaning pipeline by dropping custom Python transformation scripts into the local Plugins directory.

---

## 🧩 Writing Custom Plugins

Aura allows developers to extend the Data Preprocessing and Cleaning pipeline with custom Python scripts. 

### 1. File Location
Place your custom Python script in the user's Documents folder:
`~/Documents/Aura/Plugins/<plugin_id>.py`

### 2. Plugin Structure
Each plugin must implement a `transform` function that receives a pandas DataFrame and custom parameters, and returns the modified DataFrame:

```python
import pandas as pd

def transform(df: pd.DataFrame, **params) -> pd.DataFrame:
    """
    Custom column transformation or feature engineering.
    Parameters can be passed dynamically from the cleaning configuration.
    """
    # Example: Scale a numeric column by a custom multiplier
    target_col = params.get("column", "")
    multiplier = params.get("multiplier", 1.0)
    
    if target_col in df.columns:
        df[target_col] = df[target_col] * multiplier
        
    return df
```

### 3. Execution Mappings
Plugins are dynamically executed during the cleaning cycle when a cleaning action with type `plugin:<plugin_id>:<params>` (e.g. `plugin:scale_column:column=Price,multiplier=1.5`) is added to the config.

---

## 🏗️ Architecture

Aura is built on a clean macOS-first architecture:
* **Frontend**: Native **SwiftUI 6.0 (Swift 6)**, utilizing native `NavigationSplitView`, async-await tasks, Keychain services, and Swift Charts.
* **Backend**: **Python 3.10+** script (`Aura/analyze.py`) and FastAPI microservice server (`Aura/server.py`) run as a background service.
* **Communication**: SwiftUI communicates with the FastAPI service using asynchronous HTTP requests. Training and dataset profiling progress are streamed in real time using Server-Sent Events (SSE).

---

## ⚙️ Setup & Installation

### 1. Python Environment Setup
Run the bootstrapper script to create a local virtual environment and install all packages:
```bash
./setup_env.sh
```
This installs the required packages: `pandas`, `numpy`, `scikit-learn`, `shap`, `statsmodels`, `joblib`, `pyarrow`, `kaggle`, `huggingface_hub`, `pillow`, `optuna`, `xgboost`, `torch`, and `SQLAlchemy`.

### 2. Local AI Setup (Optional)
To enable the AI Analyst panel:
1. Download and install [Ollama](https://ollama.com/).
2. Start the Ollama server:
   ```bash
   ollama serve
   ```
3. Pull a model (e.g., Llama 3.2):
   ```bash
   ollama pull llama3.2
   ```

### 3. Build & Run the App
Open `Aura.xcodeproj` in Xcode 15+ (Swift 6 compatible) and run the `Aura` target, or build it using the command line:
```bash
xcodebuild -project Aura.xcodeproj -scheme Aura -configuration Debug -destination 'platform=macOS' CODE_SIGNING_ALLOWED=NO build
```

---

## 🧪 Verification & CLI Testing

You can run the Python pipeline directly in the terminal to inspect the JSON outputs:
```bash
# Tabular analysis
.venv/bin/python Aura/analyze.py sample_data/house_prices.csv --target Price --dataset-type tabular

# Time Series analysis with multiple targets
.venv/bin/python Aura/analyze.py sample_data/airline_passengers.csv --target Passengers,Month --time-col Date --dataset-type timeseries
```

---

## 📝 License
This project is licensed under the MIT License.
