import Foundation

// MARK: - Dataset Type

enum DatasetType: String, CaseIterable, Codable, Identifiable {
    case tabular        = "tabular"
    case timeSeries     = "timeseries"
    case image          = "image"
    case nlp            = "nlp"

    var id: String { rawValue }

    var label: String {
        switch self {
        case .tabular:    return "Tabular"
        case .timeSeries: return "Time Series"
        case .image:      return "Image"
        case .nlp:        return "Text / NLP"
        }
    }

    var icon: String {
        switch self {
        case .tabular:    return "tablecells"
        case .timeSeries: return "chart.line.uptrend.xyaxis"
        case .image:      return "photo.stack"
        case .nlp:        return "text.bubble"
        }
    }

    var description: String {
        switch self {
        case .tabular:    return "Standard rows/columns — classification or regression."
        case .timeSeries: return "Ordered sequence with a datetime or index column."
        case .image:      return "Pixel arrays (NPZ) or image folders — image classification."
        case .nlp:        return "Text-heavy columns — sentiment, topic modelling, or classification."
        }
    }

    /// Accent color for the selector pill
    var color: String {
        switch self {
        case .tabular:    return "purple"
        case .timeSeries: return "blue"
        case .image:      return "orange"
        case .nlp:        return "green"
        }
    }
}

// MARK: - Task Type Override

enum TaskTypeOverride: String, CaseIterable, Identifiable {
    case auto           = "auto"
    case classification = "classification"
    case regression     = "regression"
    case forecast       = "forecast"

    var id: String { rawValue }

    var label: String {
        switch self {
        case .auto:           return "Auto-detect"
        case .classification: return "Classification"
        case .regression:     return "Regression"
        case .forecast:       return "Forecast (TS)"
        }
    }
}

// MARK: - Cleaning Action

struct CleaningAction: Codable, Hashable, Identifiable {
    var id: String { "\(column)-\(actionType)" }
    let column: String
    let actionType: String // "drop", "impute_mean", "impute_median", "impute_mode", "clip_outliers"
}

// MARK: - Analysis Configuration

/// Holds all user-confirmed settings that are passed to analyze.py before the full run.
struct AnalysisConfig {
    var datasetType: DatasetType          = .tabular
    var taskTypeOverride: TaskTypeOverride = .auto
    var targetColumns: [String]           = []
    var targetColumn: String {
        get { targetColumns.first ?? "" }
        set {
            if newValue.isEmpty {
                targetColumns = []
            } else {
                targetColumns = [newValue]
            }
        }
    }
    var excludedColumns: Set<String>      = []
    var timeColumn: String?               = nil   // Used for Time Series
    var trainFilePath: String?            = nil   // Path to training set
    var testFilePath: String?             = nil   // Path to separate validation/test set
    var validationFilePath: String?       = nil   // Path to validation set
    var smartSample: Bool                 = false // Phase 2
    var cleaningActions: Set<CleaningAction> = [] // Phase 3
    var modelExportPath: String?          = nil   // Path to save best model (.joblib)
    var codeExportPath: String?           = nil   // Path to save reproduction code (.py)
}
