import SwiftUI
import Observation

@MainActor
@Observable
class DashboardViewModel {
    // MARK: - Dependencies
    private let pythonRunner: any PythonRunning
    let historyService: any AnalysisHistoryServiceProtocol
    let ollamaStatus: OllamaStatusChecker
    
    // MARK: - State Properties
    var selectedFileURL: URL? = nil
    var datasetURLInput: String = ""
    var isAnalyzing = false
    var isPreloading = false
    var previewResult: DatasetPreview? = nil
    var result: AnalysisResult? = nil
    var errorMessage: String? = nil
    var selectedTab: String = "Summary"
    var fileDetails: String = ""
    var showAIPanel: Bool = false
    var showRenameAlert = false
    var renameText = ""
    var itemToRename: HistoryItem? = nil
    var showExportSheet: Bool = false
    var showModelExportSheet: Bool = false
    var showURLInputAlert = false
    var urlInputText = ""
    var selectedTargetName = ""
    var showOnboarding = false
    var showDatabaseSheet = false
    var showSchedulerSheet = false
    var currentHistoryItemId: UUID? = nil
    var showMergeSheet = false
    var mergeFile1Path = ""
    var mergeFile2Path = ""
    
    var analysisConfig: AnalysisConfig = AnalysisConfig()
    var trainColumns: [String] = []
    var selectedDataTab: String = "train"
    
    var progressFraction: Double = 0.0
    var progressMessage: String = ""
    
    struct ProgressStage: Identifiable, Equatable {
        let id = UUID()
        let message: String
        let elapsed: Double
    }
    var completedStages: [ProgressStage] = []
    var currentStageMessage: String = ""
    var currentStageStartTime: Date = Date()
    
    let chatViewModel = ChatViewModel()
    
    // MARK: - Initialization
    @MainActor
    init(
        pythonRunner: any PythonRunning = PythonRunner.shared,
        historyService: any AnalysisHistoryServiceProtocol = AnalysisHistoryService.shared,
        ollamaStatus: OllamaStatusChecker = OllamaStatusChecker.shared
    ) {
        self.pythonRunner = pythonRunner
        self.historyService = historyService
        self.ollamaStatus = ollamaStatus
    }
    
    // MARK: - Navigation Helpers
    var navigationTitleText: String {
        if let name = selectedFileURL?.lastPathComponent { return name }
        if !datasetURLInput.isEmpty { return "Web Dataset" }
        return "Aura Dashboard"
    }
    
    // MARK: - Business Logic Methods
    
    func handleDroppedFiles(_ urls: [URL]) {
        if urls.count == 2 {
            self.mergeFile1Path = urls[0].path
            self.mergeFile2Path = urls[1].path
            self.showMergeSheet = true
        } else if let first = urls.first {
            loadDroppedFile(first)
        }
    }
    
    func loadDroppedFile(_ url: URL) {
        selectedFileURL = url
        datasetURLInput = ""
        errorMessage = nil
        result = nil
        previewResult = nil
        
        // Reset configuration and columns for the new dataset
        analysisConfig = AnalysisConfig()
        analysisConfig.trainFilePath = url.path
        trainColumns = []
        
        if let size = try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize {
            let fmt = ByteCountFormatter()
            fmt.countStyle = .file
            fileDetails = fmt.string(fromByteCount: Int64(size))
            
            // Automatically enable smart sample for files > 20MB
            if size > 20 * 1024 * 1024 {
                analysisConfig.smartSample = true
            }
        } else {
            fileDetails = "Unknown size"
        }
        fetchPreview(for: url.path)
    }
    
    func selectFileManually() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = true
        panel.canChooseFiles = true
        if panel.runModal() == .OK, let url = panel.url {
            loadDroppedFile(url)
        }
    }
    
    func selectTestFileManually() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = true
        panel.canChooseFiles = true
        if panel.runModal() == .OK, let url = panel.url {
            analysisConfig.testFilePath = url.path
        }
    }
    
    func loadSampleDataset(named name: String) {
        let workspacePath = "/Users/oleksiichumak/Developer/Xcode.projects/Aura/sample_data/\(name)"
        let fileURL = URL(fileURLWithPath: workspacePath)
        if FileManager.default.fileExists(atPath: fileURL.path) {
            loadDroppedFile(fileURL)
        } else if let bundleURL = Bundle.main.url(forResource: name, withExtension: nil) {
            loadDroppedFile(bundleURL)
        } else {
            errorMessage = "Sample dataset '\(name)' could not be found."
        }
    }
    
    func clearSelection() {
        selectedFileURL = nil
        datasetURLInput = ""
        previewResult = nil
        result = nil
        errorMessage = nil
        trainColumns = []
        selectedDataTab = "train"
        analysisConfig = AnalysisConfig()
        chatViewModel.clearConversation()
    }
    
    func getURLProviderName(_ urlString: String) -> String {
        let lower = urlString.lowercased()
        if lower.contains("kaggle.com") {
            if lower.contains("/code/") || lower.contains("/kernels/") {
                return "Kaggle Notebook Output"
            }
            return "Kaggle Dataset"
        }
        if lower.contains("huggingface.co") { return "Hugging Face Dataset" }
        return "Generic Web Dataset"
    }
    
    func fetchPreview(for pathOrURL: String) {
        withAnimation {
            isPreloading = true
            errorMessage = nil
            previewResult = nil
            result = nil
            progressFraction = 0.0
            progressMessage = "Preparing preview..."
        }
        Task {
            await pythonRunner.runPreview(csvPathOrURL: pathOrURL, progress: { frac, msg in
                Task { @MainActor in
                    self.progressFraction = frac
                    self.progressMessage = msg
                }
            }) { response in
                Task { @MainActor in
                    withAnimation {
                        self.isPreloading = false
                        switch response {
                        case .success(let previewData):
                            self.previewResult = previewData
                            if self.analysisConfig.trainFilePath == nil {
                                self.analysisConfig.trainFilePath = previewData.localPath
                            }
                            if previewData.localPath == self.analysisConfig.trainFilePath {
                                self.trainColumns = previewData.columns
                            }
                            if let inferred = previewData.inferredDatasetType,
                               let type = DatasetType(rawValue: inferred) {
                                self.analysisConfig.datasetType = type
                            }
                            
                            // Auto deselect id columns
                            for (idx, col) in previewData.columns.enumerated() {
                                if self.isLikelyIdentifierColumn(name: col, columnIndex: idx, previewRows: previewData.previewRows) {
                                    self.analysisConfig.excludedColumns.insert(col)
                                }
                            }
                            
                            // Automate test and validation pre-selection
                            if let available = previewData.availableFiles {
                                if self.analysisConfig.testFilePath == nil {
                                    if let testFile = available.first(where: { $0.lowercased().contains("test") }) {
                                        self.analysisConfig.testFilePath = testFile
                                    }
                                }
                                if self.analysisConfig.validationFilePath == nil {
                                    if let valFile = available.first(where: { $0.lowercased().contains("val") || $0.lowercased().contains("valid") }) {
                                        self.analysisConfig.validationFilePath = valFile
                                    }
                                }
                            }
                        case .failure(let error):
                            if (error as NSError).code != -999 {
                                self.errorMessage = error.localizedDescription
                            }
                        }
                    }
                }
            }
        }
    }
    
    func isLikelyIdentifierColumn(name: String, columnIndex: Int, previewRows: [[PreviewValue]]) -> Bool {
        let lowerName = name.lowercased()
        
        let matchesIdName = lowerName == "id" || 
                            lowerName == "index" || 
                            lowerName == "no" || 
                            lowerName == "number" || 
                            lowerName == "num" || 
                            lowerName == "row" || 
                            lowerName == "rowid" || 
                            lowerName == "row_id" || 
                            lowerName.hasSuffix("_id") || 
                            lowerName.hasSuffix("id") || 
                            lowerName.hasPrefix("id_")
        
        guard matchesIdName else { return false }
        
        var nonNullValues: [String] = []
        for row in previewRows {
            if columnIndex < row.count {
                let val = row[columnIndex]
                switch val {
                case .string(let s):
                    if !s.isEmpty { nonNullValues.append(s) }
                case .number(let n):
                    nonNullValues.append("\(n)")
                case .boolean(let b):
                    nonNullValues.append("\(b)")
                case .null:
                    break
                }
            }
        }
        
        if nonNullValues.isEmpty { return false }
        let uniqueCount = Set(nonNullValues).count
        let ratio = Double(uniqueCount) / Double(nonNullValues.count)
        return ratio >= 0.95
    }
    
    func runEDA() {
        let csvPath: String
        if let trainPath = analysisConfig.trainFilePath {
            csvPath = trainPath
        } else if let preview = previewResult {
            csvPath = preview.localPath
        } else if let fileURL = selectedFileURL {
            csvPath = fileURL.path
        } else {
            let urlStr = datasetURLInput.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !urlStr.isEmpty else { return }
            csvPath = urlStr
        }

        withAnimation {
            isAnalyzing = true
            progressFraction = 0.0
            progressMessage = "Starting analysis..."
            errorMessage = nil
            result = nil
        }

        let targetParam: String?
        if analysisConfig.datasetType == .timeSeries && !analysisConfig.targetColumns.isEmpty {
            targetParam = analysisConfig.targetColumns.joined(separator: ",")
        } else {
            let target = analysisConfig.targetColumn.trimmingCharacters(in: .whitespacesAndNewlines)
            targetParam = target.isEmpty ? nil : target
        }

        var finalConfig = analysisConfig
        if let targetParam = targetParam {
            let targetsList = targetParam.split(separator: ",").map(String.init)
            for t in targetsList {
                finalConfig.excludedColumns.remove(t)
            }
        }

        // Reset timing stages
        self.completedStages = []
        self.currentStageMessage = ""
        self.currentStageStartTime = Date()
        
        let originalSource = selectedFileURL?.path ?? datasetURLInput.trimmingCharacters(in: .whitespacesAndNewlines)
        
        Task {
            await pythonRunner.runAnalysis(
                csvPath: csvPath,
                targetColumn: targetParam,
                config: finalConfig,
                progress: { frac, msg in
                    Task { @MainActor in
                        withAnimation(.easeInOut(duration: 0.15)) {
                            self.progressFraction = frac
                            self.progressMessage = msg
                            
                            if self.currentStageMessage != msg {
                                if !self.currentStageMessage.isEmpty {
                                    let elapsed = Date().timeIntervalSince(self.currentStageStartTime)
                                    let stage = ProgressStage(message: self.currentStageMessage, elapsed: elapsed)
                                    if !self.completedStages.contains(where: { $0.message == stage.message }) {
                                        self.completedStages.append(stage)
                                    }
                                }
                                self.currentStageMessage = msg
                                self.currentStageStartTime = Date()
                            }
                        }
                    }
                }
            ) { response in
                Task { @MainActor in
                    withAnimation {
                        if !self.currentStageMessage.isEmpty {
                            let elapsed = Date().timeIntervalSince(self.currentStageStartTime)
                            let stage = ProgressStage(message: self.currentStageMessage, elapsed: elapsed)
                            if !self.completedStages.contains(where: { $0.message == stage.message }) {
                                self.completedStages.append(stage)
                            }
                        }
                        self.isAnalyzing = false
                        switch response {
                        case .success(let data):
                            self.result = data
                            if let targetsMap = data.targets, !targetsMap.isEmpty {
                                self.selectedTargetName = targetsMap.keys.sorted().first ?? data.targetColumn
                                self.analysisConfig.targetColumns = Array(targetsMap.keys).sorted()
                            } else {
                                self.selectedTargetName = data.targetColumn
                                self.analysisConfig.targetColumn = data.targetColumn
                            }
                            let savedItem = self.historyService.saveAnalysis(result: data, datasetPath: csvPath, targetColumn: targetParam, originalSource: originalSource)
                            self.currentHistoryItemId = savedItem?.id
                            self.chatViewModel.injectContext(data, datasetURL: savedItem?.datasetURL)
                            if self.ollamaStatus.isAvailable { self.showAIPanel = true }
                        case .failure(let error):
                            if (error as NSError).code != -999 {
                                self.errorMessage = error.localizedDescription
                            }
                        }
                    }
                }
            }
        }
    }
    
    func loadHistoryItem(_ item: HistoryItem) {
        withAnimation {
            self.errorMessage = nil
            self.isAnalyzing = false
            self.isPreloading = true   // show a spinner while we read from disk
            self.progressFraction = 0.0
            self.progressMessage = ""
        }

        // Reconstruct the base configuration for this history item
        var newConfig = AnalysisConfig()
        newConfig.trainFilePath = item.datasetPath

        if let datasetURL = item.datasetURL, !datasetURL.isEmpty {
            self.datasetURLInput = datasetURL
            self.selectedFileURL = nil
            self.fileDetails = "Remote URL Dataset"
        } else if item.datasetPath.hasPrefix("http://") || item.datasetPath.hasPrefix("https://") {
            self.datasetURLInput = item.datasetPath
            self.selectedFileURL = nil
            self.fileDetails = "Remote URL Dataset"
        } else {
            self.selectedFileURL = URL(fileURLWithPath: item.datasetPath)
            self.datasetURLInput = ""
            if let size = try? selectedFileURL?.resourceValues(forKeys: [.fileSizeKey]).fileSize {
                let fmt = ByteCountFormatter()
                fmt.countStyle = .file
                self.fileDetails = fmt.string(fromByteCount: Int64(size))
            } else {
                self.fileDetails = "Local File"
            }
        }

        // Offload the potentially large file read to a background task
        Task { @MainActor in
            let loadedResult = await historyService.loadAnalysisResult(item: item)
            withAnimation {
                self.isPreloading = false
            }
            if let loadedResult {
                self.result = loadedResult
                self.currentHistoryItemId = item.id
                self.trainColumns = loadedResult.columns
                self.chatViewModel.injectContext(loadedResult, datasetURL: item.datasetURL)

                if let targetsMap = loadedResult.targets, !targetsMap.isEmpty {
                    self.selectedTargetName = targetsMap.keys.sorted().first ?? loadedResult.targetColumn
                    newConfig.targetColumns = Array(targetsMap.keys).sorted()
                } else {
                    self.selectedTargetName = loadedResult.targetColumn
                    newConfig.targetColumn = loadedResult.targetColumn
                }

                // Reconstruct a DatasetPreview from the final AnalysisResult
                let previewRows = (loadedResult.fullPreview?.rows.prefix(15) ?? []).map { row in
                    row.map { PreviewValue.string($0) }
                }

                let datasetTypeStr: String
                switch loadedResult.taskType.lowercased() {
                case "classification", "regression":
                    datasetTypeStr = "tabular"
                case "object_detection":
                    datasetTypeStr = "object_detection"
                default:
                    datasetTypeStr = loadedResult.taskType.lowercased()
                }

                self.previewResult = DatasetPreview(
                    columns: loadedResult.columns,
                    previewRows: previewRows,
                    localPath: item.datasetPath,
                    error: nil,
                    inferredDatasetType: datasetTypeStr,
                    availableFiles: [item.datasetPath],
                    totalRows: loadedResult.rowCount
                )

                if let type = DatasetType(rawValue: datasetTypeStr) {
                    newConfig.datasetType = type
                }

                self.analysisConfig = newConfig
                if self.ollamaStatus.isAvailable { self.showAIPanel = true }
                self.selectedTab = "Summary"
            } else {
                newConfig.targetColumn = item.targetColumn ?? ""
                self.analysisConfig = newConfig
                self.errorMessage = "Could not load saved analysis result from disk."
            }
        }
    }
}
