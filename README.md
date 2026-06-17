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

### 6. Pipeline Replication & Serialization
* **Model Export**: Serialize the best-performing model pipeline using `joblib`.
* **Replication Code**: Auto-generate a standalone Python script containing exact preprocessing steps (imputation, encoding, scaling, training) to run anywhere.

---

## 🏗️ Architecture

Aura is built on a clean macOS-first architecture:
* **Frontend**: Native **SwiftUI 6.0 (Swift 6)**, utilizing native `NavigationSplitView`, async-await tasks, Keychain services, and Swift Charts.
* **Backend**: **Python 3.10+** script (`Aura/analyze.py`) run as a sandboxed subprocess.
* **Communication**: Structs are serialized and communicated between Swift and Python using a structured JSON protocol over stdout/stderr. Progress bars and log status are streamed in real time.

---

## ⚙️ Setup & Installation

### 1. Python Environment Setup
Run the bootstrapper script to create a local virtual environment and install all packages:
```bash
./setup_env.sh
```
This installs the required packages: `pandas`, `numpy`, `scikit-learn`, `shap`, `statsmodels`, `joblib`, `pyarrow`, `kaggle`, `huggingface_hub`, and `pillow`.

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
.venv/bin/python Aura/analyze.py sample_data/house_prices.csv --target SalePrice --dataset-type tabular

# Time Series analysis with multiple targets
.venv/bin/python Aura/analyze.py sample_data/airline_passengers.csv --target Passengers,Month --time-col Date --dataset-type timeseries
```

---

## 📝 License
This project is licensed under the MIT License.
