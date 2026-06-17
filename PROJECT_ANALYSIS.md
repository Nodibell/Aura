# AutoEDA: аналіз проєкту та оцінка

Дата рев'ю: 2026-06-15

## Короткий висновок

AutoEDA - це сильний MVP macOS-застосунку для швидкого exploratory data analysis: користувач може завантажити датасет, побачити preview, запустити Python-пайплайн, отримати summary, базові графіки, кореляції, baseline ML-модель, історію аналізів і локальний AI-чат через Ollama.

Проєкт добре демонструє наскрізний сценарій "датасет -> аналіз -> візуалізація -> AI-пояснення". На малих табличних датасетах він справді виконує основну задачу. Але як production-ready EDA/AutoML tool він ще має суттєві прогалини: евристичне визначення target/task type, слабку ML-методологію, відсутність тестів і dependency management, hardcoded шляхи, небезпечне зберігання credentials та вимкнені macOS security settings.

Моя загальна оцінка: **7.0/10 як MVP / B-**.

Для навчального або демонстраційного продукту це виглядає переконливо. Для інструмента, якому користувач має довіряти реальні датасети й рішення, оцінка ближча до **5.5-6.0/10** через надійність, відтворюваність і якість аналітичних висновків.

## Що саме перевірено

- Структура проєкту: macOS SwiftUI app + Python backend `AutoEDA/analyze.py`.
- Збірка: `xcodebuild -project AutoEDA.xcodeproj -scheme AutoEDA -configuration Debug -destination 'platform=macOS' -derivedDataPath /private/tmp/AutoEDA_DerivedData CODE_SIGNING_ALLOWED=NO build`.
- Результат збірки: **BUILD SUCCEEDED** після запуску поза sandbox. У sandbox build падав через Swift macro/plugin server, а не через код.
- Python-середовище: `python3` 3.11.9, імпорт `pandas`, `numpy`, `sklearn` успішний.
- Sample datasets:
  - `sample_data/iris.csv`: 30 рядків, 5 колонок, classification, target `species`, Random Forest Accuracy `1.0`.
  - `sample_data/house_prices.csv`: 30 рядків, 5 колонок, regression, target `Price`, Random Forest R2 `0.9957`, RMSE `8468.95`.
- Автоматичних test target / test files / README / requirements / pyproject у проєкті не знайдено.

Важливо: sample datasets дуже малі, тому високі метрики не доводять якість пайплайна на реальних даних.

## Наскільки добре проєкт виконує задачу

### Що працює добре

1. **Наскрізний користувацький сценарій вже є.** Є drag/drop або file picker, URL input, preview, target selector, progress, tabs для Summary/Charts/Correlations, AI panel і history. Це не просто набір скриптів, а цілісний продукт.

2. **Python-пайплайн має реальну функціональність.** Він читає CSV/TSV/Parquet/NPZ, підтримує Kaggle, Hugging Face і direct URL, рахує missing values, numeric/categorical columns, кореляції, PCA, outlier/skewness warnings, тренує Linear/Logistic Regression і Random Forest.

3. **UI приємний для MVP.** Темна тема, sidebar, segmented tabs, chart cards, preview table, settings, logs і AI quick actions створюють відчуття завершеного desktop tool.

4. **Інтеграція Ollama корисна і доречна.** AI-чат отримує structured context про датасет, моделі, missing values, correlations і feature importances. Це добре відповідає ідеї "AI analyst".

5. **Є базова діагностика середовища.** Settings дозволяють перевірити Python path, побачити logs, налаштувати Ollama і credentials.

## Оцінка за категоріями

| Категорія | Оцінка | Коментар |
|---|---:|---|
| Product idea / MVP completeness | 8.0/10 | Сильний вертикальний зріз, зрозуміла цінність. |
| Core EDA usefulness | 7.0/10 | Є missing/correlation/charts/PCA, але бракує повного profiling. |
| ML quality | 5.5/10 | Baseline-моделі є, але оцінювання надто просте. |
| UI/UX | 7.5/10 | Візуально добре, але є inconsistency і fixed layouts. |
| Architecture | 6.5/10 | Поділ Models/Views/Services є, але великі файли й hardcoded paths. |
| Reliability | 6.0/10 | Build проходить, samples проходять, але немає тестів і reproducible env. |
| Security/distribution readiness | 4.5/10 | Sandbox/hardened runtime off, secrets у UserDefaults. |
| Maintainability | 6.0/10 | Код читабельний, але немає docs/tests/dependency spec, є великі файли. |

## Основні недоліки

### 1. Target column detection може легко помилитися

У `AutoEDA/analyze.py` target шукається за keywords, а якщо не знайдено - береться остання колонка. Особливо ризиковано, що keyword `"y"` перевіряється через substring match, тобто може спрацювати на назвах типу `city`, `year`, `salary`.

Джерело: `AutoEDA/analyze.py:363`, `AutoEDA/analyze.py:365`, `AutoEDA/analyze.py:369`.

Чому це важливо: якщо target визначено неправильно, усі подальші метрики, feature importances, графіки й AI-висновки стають переконливими, але хибними.

### 2. Task type detection занадто грубий

Classification визначається, якщо target categorical або має `<= 10` unique values.

Джерело: `AutoEDA/analyze.py:377`, `AutoEDA/analyze.py:380`.

Чому це важливо: числовий ordinal/regression target із малою кількістю значень буде класифікацією; ID-like або coded numeric columns можуть дати неправильну постановку задачі.

### 3. ML-оцінювання недостатньо надійне

Є один `train_test_split(test_size=0.2, random_state=42)`, без stratify, cross-validation, baseline dummy model, confidence intervals або контролю leakage.

Джерело: `AutoEDA/analyze.py:497`.

Для classification використовуються Accuracy і weighted F1; для regression - R2 і RMSE.

Джерело: `AutoEDA/analyze.py:509`, `AutoEDA/analyze.py:516`, `AutoEDA/analyze.py:539`, `AutoEDA/analyze.py:591`, `AutoEDA/analyze.py:598`, `AutoEDA/analyze.py:612`.

Чому це важливо: на imbalance datasets Accuracy може бути оманливою; на малих датасетах один split дуже нестабільний; без dummy baseline користувач не бачить, чи модель реально корисна.

### 4. Preprocessing може погано масштабуватися

One-hot encoding створюється dense array через `sparse_output=False`, а це небезпечно для великих/high-cardinality categorical columns. Text features обрізаються до `max_features=15`, що швидко, але може втрачати більшість сигналу.

Джерело: `AutoEDA/analyze.py:468`, `AutoEDA/analyze.py:480`.

Чому це важливо: застосунок може зависати або споживати багато пам'яті на реальних датасетах; водночас текстові результати можуть бути поверхневими.

### 5. Немає керованого Python dependency management

Settings перевіряє лише `pandas`, `sklearn`, `numpy`, але функціонал також може потребувати `huggingface_hub`, `kaggle`, `pyarrow` або `fastparquet`.

Джерело: `AutoEDA/Services/PythonRunner.swift:118`, `AutoEDA/Services/PythonRunner.swift:121`, `AutoEDA/analyze.py:76`, `AutoEDA/analyze.py:129`, `AutoEDA/analyze.py:229`.

Чому це важливо: користувач може отримати "Environment OK", але Kaggle/Hugging Face/Parquet впаде вже під час аналізу.

### 6. Є hardcoded абсолютні шляхи розробника

Fallback для `analyze.py` і sample datasets прив'язаний до `/Users/oleksiichumak/Developer/Xcode.projects/AutoEDA/...`.

Джерело: `AutoEDA/Services/PythonRunner.swift:151`, `AutoEDA/Services/PythonRunner.swift:348`, `AutoEDA/Views/ContentView.swift:658`.

Чому це важливо: після перенесення проєкту, зміни імені користувача, CI або distribution build fallback-логіка буде ламатися.

### 7. Немає тестів, документації та dependency spec

У проєкті не знайдено README, `requirements.txt`, `pyproject.toml`, test target або test files.

Чому це важливо: складно відтворити середовище, складно перевіряти regression bugs, складно onboard-ити нового розробника або користувача.

### 8. Security/distribution readiness слабка

Sandbox вимкнений, hardened runtime вимкнений, credentials зберігаються через UserDefaults, хоча UI використовує `SecureField`.

Джерело: `AutoEDA/AutoEDA.entitlements:5`, `project.yml:17`, `AutoEDA/Views/SettingsView.swift:170`, `AutoEDA/Views/SettingsView.swift:195`, `AutoEDA/Views/SettingsView.swift:232`, `AutoEDA/Views/SettingsView.swift:245`, `AutoEDA/Views/SettingsView.swift:248`.

Чому це важливо: для локального prototype це прийнятно, але для macOS-дистрибуції треба Keychain, sandbox/hardened runtime, signing/notarization і чіткі entitlements.

### 9. Є Swift concurrency warning, який у Swift 6 стане помилкою

Build показав warning у `OllamaStatusChecker`: captured var `models` використовується в concurrently-executing code.

Джерело: `AutoEDA/Services/OllamaStatusChecker.swift:37`.

Чому це важливо: зараз build проходить у Swift 5 mode, але міграція до Swift 6 може зламати збірку.

### 10. UX має неточності та обмеження

Drop zone пише "Supports local CSV files up to 100MB", але picker дозволяє `csv`, `tsv`, `parquet`, `npz`, і явного 100MB ліміту в коді не видно.

Джерело: `AutoEDA/Views/DragDropView.swift:130`, `AutoEDA/Views/ContentView.swift:650`, `AutoEDA/Views/ContentView.swift:651`.

Preview table має fixed width `140`, AI panel fixed width `325`, heatmap показує лише обмежений subset кореляцій.

Джерело: `AutoEDA/Views/PreviewTableView.swift:47`, `AutoEDA/Views/PreviewTableView.swift:85`, `AutoEDA/Views/ContentView.swift:57`, `AutoEDA/Views/CorrelationMatrixView.swift:188`, `AutoEDA/Views/CorrelationMatrixView.swift:192`.

Чому це важливо: на довгих назвах колонок, малих вікнах або широких датасетах UI може бути менш зручним.

### 11. Репозиторій містить generated/cache artifact

Є `AutoEDA/__pycache__/analyze.cpython-311.pyc`.

Чому це важливо: `.pyc` не має бути у source tree; варто мати `.gitignore` і чисту структуру.

## Пропозиції для наступних фіч

### Найвищий пріоритет

1. **User-confirmed target/task setup.** Після preview показувати recommended target/task, але просити користувача підтвердити або змінити: target, classification/regression, ID columns, ignored columns.

2. **Reproducible Python environment.** Додати `requirements.txt` або `pyproject.toml`, bootstrap venv, повну diagnostics-перевірку optional dependencies: `kaggle`, `huggingface_hub`, `pyarrow`.

3. **Тести.** Додати Python unit tests для `load_dataset`, target detection, preprocessing, metrics, error cases; Swift tests для `PythonRunner`, history, decoding; smoke test для sample datasets.

4. **Keychain для credentials.** Kaggle API key і HF token перенести з UserDefaults у Keychain.

5. **Export report.** Експорт Markdown/HTML/PDF з summary, charts, metrics, warnings, metadata і timestamp.

### EDA/ML фічі

6. **Повний data profiling.** Unique counts, cardinality, constant columns, duplicate rows, inferred semantic types, top categories, min/max/mean/std/quantiles, missingness matrix.

7. **Better model evaluation.** Stratified split для classification, k-fold CV, dummy baseline, confusion matrix, ROC/PR AUC, residual plots, calibration, train/test score comparison.

8. **Feature importance alternatives.** Permutation importance як model-agnostic варіант; SHAP/partial dependence як optional advanced mode.

9. **Data leakage detection.** Попереджати про ID-like target leakage, datetime leakage, duplicate target proxies, columns highly correlated with target.

10. **Cleaning recommendations.** Автоматично пропонувати: drop high-missing columns, imputation strategy, log transform skewed features, cap outliers, encode categoricals.

11. **Large dataset mode.** Sampling strategy, Polars/DuckDB або chunked reads, sparse encoders, row/column limits, progress with file size and estimated time.

### Product/UI фічі

12. **Column explorer.** Пошук колонок, per-column detail panel, distribution chart, missingness, examples, correlations with target.

13. **History improvements.** Tags, notes, compare runs, pin favorite analyses, delete all/history management UI.

14. **Dataset connector UX.** Для Kaggle/Hugging Face показувати список файлів у repo/dataset і дозволяти вибрати конкретний файл.

15. **AI grounding.** Дати AI доступ до structured JSON with chart data and metric definitions; у відповідях вимагати посилання на конкретні метрики/графіки, щоб зменшити hallucinations.

16. **Localization.** Додати українську/англійську локалізацію, бо зараз UI повністю англійський.

17. **Theme/responsive polish.** Не форсити dark-only mode, адаптувати fixed widths, покращити таблицю для довгих назв колонок.

### Distribution/engineering

18. **Signing/notarization readiness.** Увімкнути hardened runtime, продумати sandbox entitlements, network/files/process execution permissions.

19. **CI.** Build + Python tests + sample smoke tests у GitHub Actions або локальному script.

20. **Repository hygiene.** README, install guide, troubleshooting, screenshots, `.gitignore`, прибрати `__pycache__`.

## Рекомендований порядок роботи

1. Прибрати hardcoded шляхи, додати `.gitignore`, README і dependency spec.
2. Додати Python tests на sample datasets та edge cases.
3. Переробити target/task selection на user-confirmed flow.
4. Додати stratified/CV evaluation і dummy baseline.
5. Перенести secrets у Keychain.
6. Додати export report.
7. Поступово розширити EDA profiling і large dataset mode.

## Фінальна оцінка

AutoEDA вже має переконливу форму продукту і реально виконує базову задачу для невеликих табличних датасетів. Найсильніша сторона - повний end-to-end UX: імпорт, preview, аналіз, графіки, історія, локальний AI.

Головне обмеження - не UI, а довіра до результатів. Щоб стати сильним EDA/AutoML-інструментом, проєкту потрібно менше евристик "за замовчуванням", більше явного контролю користувача, краща ML-валидація, тестове покриття і reproducible environment.

**Поточна оцінка: 7.0/10 як MVP. Потенціал після доопрацювань: 8.5/10+.**
