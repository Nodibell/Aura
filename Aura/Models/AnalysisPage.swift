import Foundation
import Observation

@Observable
class AnalysisPage: Identifiable, Equatable {
    let id: UUID
    var title: String
    
    var selectedFileURL: URL? = nil
    var datasetURLInput: String = ""
    var isAnalyzing = false
    var isPreloading = false
    var previewResult: DatasetPreview? = nil
    var result: AnalysisResult? = nil
    var errorMessage: String? = nil
    var selectedTab: String = "Summary"
    var fileDetails: String = ""
    var selectedTargetName = ""
    var activeModelName: String? = nil {
        didSet {
            analysisConfig.activeModelName = activeModelName
        }
    }
    var analysisConfig: AnalysisConfig = AnalysisConfig()
    var trainColumns: [String] = []
    var selectedDataTab: String = "train"
    var progressFraction: Double = 0.0
    var progressMessage: String = ""
    
    var completedStages: [DashboardViewModel.ProgressStage] = []
    var currentStageMessage: String = ""
    var currentStageStartTime: Date = Date()
    
    let chatViewModel: ChatViewModel
    var currentHistoryItemId: UUID? = nil
    
    init(
        id: UUID = UUID(),
        title: String,
        fileURL: URL? = nil,
        datasetURLInput: String = "",
        previewResult: DatasetPreview? = nil,
        result: AnalysisResult? = nil,
        config: AnalysisConfig = AnalysisConfig(),
        trainColumns: [String] = [],
        historyItemId: UUID? = nil
    ) {
        self.id = id
        self.title = title
        self.selectedFileURL = fileURL
        self.datasetURLInput = datasetURLInput
        self.previewResult = previewResult
        self.result = result
        self.analysisConfig = config
        self.trainColumns = trainColumns
        self.currentHistoryItemId = historyItemId
        self.chatViewModel = ChatViewModel()
        
        if let res = result {
            self.activeModelName = res.metrics.model
            self.selectedTargetName = res.targetColumn
        }
    }
    
    static func == (lhs: AnalysisPage, rhs: AnalysisPage) -> Bool {
        lhs.id == rhs.id
    }
}
