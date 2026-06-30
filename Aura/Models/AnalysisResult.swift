import Foundation

struct AnalysisResult: Codable {
    let summary: String
    let columns: [String]
    let rowCount: Int
    let colCount: Int
    let taskType: String
    let numericColCount: Int
    let categoricalColCount: Int
    let textColCount: Int
    let missingValues: [String: Int]
    let correlations: [CorrelationPair]
    let charts: [ChartConfig]
    let metrics: ModelMetrics
    let modelsCompared: [ModelComparison]
    let targetColumn: String
    let error: String?
    /// Full table preview (first 500 rows) returned by Python for the Data Table tab.
    let fullPreview: FullTablePreview?
    let testFullPreview: FullTablePreview?
    let valFullPreview: FullTablePreview?
    
    // Phase 2 smart sampling properties
    let originalRowCount: Int?
    let sampledRowCount: Int?
    
    // Phase B core quality validation fields
    let dummyBaselineScore: Double?
    let cvScores: [Double]?
    let cvMean: Double?
    let cvStd: Double?
    let confusionMatrix: ConfusionMatrixData?
    let profiling: DataProfiling?
    
    // Phase D fields
    let dataLeakageWarnings: [String]?
    let cleaningRecommendations: [CleaningRecommendation]?
    
    // Validation dataset fields
    let valMetrics: ModelMetrics?
    let valConfusionMatrix: ConfusionMatrixData?
    let targets: [String: TargetResult]?
    
    /// Optional warning message from the Python pipeline (e.g. image truncation notice).
    let warning: String?
    
    enum CodingKeys: String, CodingKey {
        case summary
        case columns
        case rowCount = "row_count"
        case colCount = "col_count"
        case taskType = "task_type"
        case numericColCount = "numeric_col_count"
        case categoricalColCount = "categorical_col_count"
        case textColCount = "text_col_count"
        case missingValues = "missing_values"
        case correlations
        case charts
        case metrics
        case modelsCompared = "models_compared"
        case targetColumn = "target_column"
        case error
        case fullPreview = "full_preview"
        case testFullPreview = "test_full_preview"
        case valFullPreview = "val_full_preview"
        
        case originalRowCount = "original_row_count"
        case sampledRowCount = "sampled_row_count"
        
        case dummyBaselineScore = "dummy_baseline_score"
        case cvScores = "cv_scores"
        case cvMean = "cv_mean"
        case cvStd = "cv_std"
        case confusionMatrix = "confusion_matrix"
        case profiling
        
        case dataLeakageWarnings = "data_leakage_warnings"
        case cleaningRecommendations = "cleaning_recommendations"
        
        case valMetrics = "val_metrics"
        case valConfusionMatrix = "val_confusion_matrix"
        case targets
        case warning
    }
}

struct TargetResult: Codable {
    let metrics: ModelMetrics
    let modelsCompared: [ModelComparison]
    let charts: [ChartConfig]
    let dummyBaselineScore: Double?
    let cvScores: [Double]?
    let cvMean: Double?
    let cvStd: Double?
    let confusionMatrix: ConfusionMatrixData?
    let valMetrics: ModelMetrics?
    let valConfusionMatrix: ConfusionMatrixData?
    
    enum CodingKeys: String, CodingKey {
        case metrics
        case modelsCompared = "models_compared"
        case charts
        case dummyBaselineScore = "dummy_baseline_score"
        case cvScores = "cv_scores"
        case cvMean = "cv_mean"
        case cvStd = "cv_std"
        case confusionMatrix = "confusion_matrix"
        case valMetrics = "val_metrics"
        case valConfusionMatrix = "val_confusion_matrix"
    }
}

extension AnalysisResult {
    func resultForTarget(_ name: String) -> AnalysisResult {
        guard let targets = targets, let targetRes = targets[name] else {
            return self
        }
        return AnalysisResult(
            summary: self.summary,
            columns: self.columns,
            rowCount: self.rowCount,
            colCount: self.colCount,
            taskType: self.taskType,
            numericColCount: self.numericColCount,
            categoricalColCount: self.categoricalColCount,
            textColCount: self.textColCount,
            missingValues: self.missingValues,
            correlations: self.correlations,
            charts: targetRes.charts,
            metrics: targetRes.metrics,
            modelsCompared: targetRes.modelsCompared,
            targetColumn: name,
            error: self.error,
            fullPreview: self.fullPreview,
            testFullPreview: self.testFullPreview,
            valFullPreview: self.valFullPreview,
            originalRowCount: self.originalRowCount,
            sampledRowCount: self.sampledRowCount,
            dummyBaselineScore: targetRes.dummyBaselineScore,
            cvScores: targetRes.cvScores,
            cvMean: targetRes.cvMean,
            cvStd: targetRes.cvStd,
            confusionMatrix: targetRes.confusionMatrix,
            profiling: self.profiling,
            dataLeakageWarnings: self.dataLeakageWarnings,
            cleaningRecommendations: self.cleaningRecommendations,
            valMetrics: targetRes.valMetrics,
            valConfusionMatrix: targetRes.valConfusionMatrix,
            targets: self.targets,
            warning: self.warning
        )
    }
}

struct CleaningRecommendation: Codable, Identifiable {
    var id: String { "\(column)-\(issue)" }
    let column: String
    let issue: String
    let recommendation: String
    let impact: String
}

/// First 500 rows of the dataset for the in-app Data Table tab.
struct FullTablePreview: Codable {
    let columns: [String]
    let rows: [[String]]
    let totalRows: Int

    enum CodingKeys: String, CodingKey {
        case columns
        case rows
        case totalRows = "total_rows"
    }
}

struct ConfusionMatrixData: Codable {
    let labels: [String]
    let values: [[Int]]
}

struct DataProfiling: Codable {
    let duplicateRows: Int
    let columns: [String: ColumnProfile]
    
    enum CodingKeys: String, CodingKey {
        case duplicateRows = "duplicate_rows"
        case columns
    }
}

struct ColumnProfile: Codable {
    let nunique: Int
    let missing: Int
    let type: String // "numeric" or "categorical"
    let stats: NumericStats?
    let topCategories: [TopCategory]?
    let isInteger: Bool?
    
    enum CodingKeys: String, CodingKey {
        case nunique
        case missing
        case type
        case stats
        case topCategories = "top_categories"
        case isInteger = "is_integer"
    }
}

struct NumericStats: Codable {
    let min: Double
    let max: Double
    let mean: Double
    let std: Double
    let p25: Double
    let p50: Double
    let p75: Double
}

struct TopCategory: Codable, Identifiable {
    var id: String { value }
    let value: String
    let count: Int
}


struct CorrelationPair: Codable, Identifiable {
    var id: String { "\(x)-\(y)" }
    let x: String
    let y: String
    let value: Double
}

struct ImageItem: Codable, Identifiable {
    var id: String { label + "-" + base64.prefix(10) }
    let label: String
    let base64: String
}

struct BoxStats: Codable {
    let min: Double
    let q1: Double
    let median: Double
    let q3: Double
    let max: Double
    let outliers: [Double]
}

struct ChartConfig: Codable, Identifiable {
    var id: String { title }
    let type: String // "line" | "bar" | "scatter" | "image_grid" | "boxplot"
    let title: String
    let xLabel: String
    let yLabel: String
    let data: [ChartPoint]
    let images: [ImageItem]?
    let boxStats: BoxStats?
    
    enum CodingKeys: String, CodingKey {
        case type
        case title
        case xLabel = "x_label"
        case yLabel = "y_label"
        case data
        case images
        case boxStats = "box_stats"
    }
}

struct ChartPoint: Codable, Identifiable {
    let id: UUID
    let xVal: String?
    let xNum: Double?
    let y: Double
    let series: String?
    
    enum CodingKeys: String, CodingKey {
        case xVal = "x_val"
        case xNum = "x_num"
        case y
        case series
    }
    
    // Provide custom decoder to generate a UUID since it is not in the JSON
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.xVal = try container.decodeIfPresent(String.self, forKey: .xVal)
        self.xNum = try container.decodeIfPresent(Double.self, forKey: .xNum)
        self.y = try container.decode(Double.self, forKey: .y)
        self.series = try container.decodeIfPresent(String.self, forKey: .series)
        self.id = UUID()
    }
    
    // Memberwise initializer for preview/test cases
    init(xVal: String? = nil, xNum: Double? = nil, y: Double, series: String? = nil) {
        self.xVal = xVal
        self.xNum = xNum
        self.y = y
        self.series = series
        self.id = UUID()
    }
}

struct ModelMetrics: Codable {
    let model: String
    let scoreType: String
    let score: Double
    let additionalMetrics: [String: Double]?
    
    enum CodingKeys: String, CodingKey {
        case model
        case scoreType = "score_type"
        case score
        case additionalMetrics = "additional_metrics"
    }
}

struct ModelComparison: Codable, Identifiable {
    var id: String { name }
    let name: String
    let score: Double
    let metric: String
    
    // Additional metrics computed for v0.4.3
    let mse: Double?
    let rmse: Double?
    let mae: Double?
    let f1: Double?
    let precision: Double?
    let recall: Double?
}


struct DatasetPreview: Codable, Sendable {
    let columns: [String]
    let previewRows: [[PreviewValue]]
    let localPath: String
    let error: String?
    /// Python-inferred hint for which dataset type to pre-select (may be nil for older responses).
    let inferredDatasetType: String?
    let availableFiles: [String]?
    let totalRows: Int?
    let columnTypes: [String: String]?

    init(
        columns: [String],
        previewRows: [[PreviewValue]],
        localPath: String,
        error: String? = nil,
        inferredDatasetType: String? = nil,
        availableFiles: [String]? = nil,
        totalRows: Int? = nil,
        columnTypes: [String: String]? = nil
    ) {
        self.columns = columns
        self.previewRows = previewRows
        self.localPath = localPath
        self.error = error
        self.inferredDatasetType = inferredDatasetType
        self.availableFiles = availableFiles
        self.totalRows = totalRows
        self.columnTypes = columnTypes
    }

    enum CodingKeys: String, CodingKey {
        case columns
        case previewRows = "preview_rows"
        case localPath = "local_path"
        case error
        case inferredDatasetType = "inferred_dataset_type"
        case availableFiles = "available_files"
        case totalRows = "total_rows"
        case columnTypes = "column_types"
    }
}

enum PreviewValue: Codable, Hashable, Identifiable, Sendable {
    var id: String {
        switch self {
        case .string(let s): return "s-\(s)-\(UUID().uuidString)"
        case .number(let n): return "n-\(n)-\(UUID().uuidString)"
        case .boolean(let b): return "b-\(b)-\(UUID().uuidString)"
        case .null: return "null-\(UUID().uuidString)"
        }
    }
    
    case string(String)
    case number(Double)
    case boolean(Bool)
    case null
    
    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let s = try? container.decode(String.self) {
            self = .string(s)
        } else if let d = try? container.decode(Double.self) {
            self = .number(d)
        } else if let b = try? container.decode(Bool.self) {
            self = .boolean(b)
        } else if container.decodeNil() {
            self = .null
        } else {
            self = .null
        }
    }
    
    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let s): try container.encode(s)
        case .number(let d): try container.encode(d)
        case .boolean(let b): try container.encode(b)
        case .null: try container.encodeNil()
        }
    }
    
    var displayString: String {
        switch self {
        case .string(let s): return s
        case .number(let d):
            if d.truncatingRemainder(dividingBy: 1) == 0 {
                return String(format: "%.0f", d)
            } else {
                return String(format: "%.4f", d)
            }
        case .boolean(let b): return b ? "True" : "False"
        case .null: return "—"
        }
    }
}

extension AnalysisResult: Equatable {
    static func == (lhs: AnalysisResult, rhs: AnalysisResult) -> Bool {
        lhs.targetColumn == rhs.targetColumn &&
        lhs.rowCount == rhs.rowCount &&
        lhs.colCount == rhs.colCount &&
        lhs.metrics.score == rhs.metrics.score &&
        lhs.summary == rhs.summary
    }
}

