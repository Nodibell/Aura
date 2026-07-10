import SwiftUI
import Observation

@MainActor
@Observable
class DashboardViewModel {
    // MARK: - Dependencies
    private let pythonRunner: any PythonRunning
    let historyService: any AnalysisHistoryServiceProtocol
    let ollamaStatus: OllamaStatusChecker
    
    // MARK: - Page Tab Management
    var openPages: [AnalysisPage] = []
    var activePageId: UUID? = nil
    
    private var fallbackPage: AnalysisPage? = nil
    
    var activePage: AnalysisPage? {
        if openPages.isEmpty {
            if fallbackPage == nil {
                fallbackPage = AnalysisPage(title: "Aura Dashboard")
            }
            return fallbackPage
        }
        return openPages.first(where: { $0.id == activePageId })
    }
    
    // MARK: - State Properties (Shared UI state)
    var showAIPanel: Bool = false
    var showRenameAlert = false
    var renameText = ""
    var itemToRename: HistoryItem? = nil
    var showExportSheet: Bool = false
    var showModelExportSheet: Bool = false
    var showURLInputAlert = false
    var urlInputText = ""
    var showOnboarding = false
    var showDatabaseSheet = false
    var showSchedulerSheet = false
    var showMergeSheet = false
    var mergeFile1Path = ""
    var mergeFile2Path = ""
    
    // MARK: - Computed Properties Forwarding to activePage
    var selectedFileURL: URL? {
        get { activePage?.selectedFileURL }
        set { activePage?.selectedFileURL = newValue }
    }
    
    var datasetURLInput: String {
        get { activePage?.datasetURLInput ?? "" }
        set { activePage?.datasetURLInput = newValue }
    }
    
    var isAnalyzing: Bool {
        get { activePage?.isAnalyzing ?? false }
        set { activePage?.isAnalyzing = newValue }
    }
    
    var isPreloading: Bool {
        get { activePage?.isPreloading ?? false }
        set { activePage?.isPreloading = newValue }
    }
    
    var previewResult: DatasetPreview? {
        get { activePage?.previewResult }
        set { activePage?.previewResult = newValue }
    }
    
    var result: AnalysisResult? {
        get { activePage?.result }
        set { activePage?.result = newValue }
    }
    
    var errorMessage: String? {
        get { activePage?.errorMessage }
        set { activePage?.errorMessage = newValue }
    }
    
    var selectedTab: String {
        get { activePage?.selectedTab ?? "Summary" }
        set { activePage?.selectedTab = newValue }
    }
    
    var fileDetails: String {
        get { activePage?.fileDetails ?? "" }
        set { activePage?.fileDetails = newValue }
    }
    
    var selectedTargetName: String {
        get { activePage?.selectedTargetName ?? "" }
        set { activePage?.selectedTargetName = newValue }
    }
    
    var activeModelName: String? {
        get { activePage?.activeModelName }
        set { activePage?.activeModelName = newValue }
    }
    
    var analysisConfig: AnalysisConfig {
        get { activePage?.analysisConfig ?? AnalysisConfig() }
        set { activePage?.analysisConfig = newValue }
    }
    
    var trainColumns: [String] {
        get { activePage?.trainColumns ?? [] }
        set { activePage?.trainColumns = newValue }
    }
    
    var selectedDataTab: String {
        get { activePage?.selectedDataTab ?? "train" }
        set { activePage?.selectedDataTab = newValue }
    }
    
    var progressFraction: Double {
        get { activePage?.progressFraction ?? 0.0 }
        set { activePage?.progressFraction = newValue }
    }
    
    var progressMessage: String {
        get { activePage?.progressMessage ?? "" }
        set { activePage?.progressMessage = newValue }
    }
    
    struct ProgressStage: Identifiable, Equatable {
        let id = UUID()
        let message: String
        let elapsed: Double
    }
    
    var completedStages: [ProgressStage] {
        get { activePage?.completedStages ?? [] }
        set { activePage?.completedStages = newValue }
    }
    
    var currentStageMessage: String {
        get { activePage?.currentStageMessage ?? "" }
        set { activePage?.currentStageMessage = newValue }
    }
    
    var currentStageStartTime: Date {
        get { activePage?.currentStageStartTime ?? Date() }
        set { activePage?.currentStageStartTime = newValue }
    }
    
    var chatViewModel: ChatViewModel {
        activePage?.chatViewModel ?? ChatViewModel()
    }
    
    var currentHistoryItemId: UUID? {
        get { activePage?.currentHistoryItemId }
        set { activePage?.currentHistoryItemId = newValue }
    }
    
    // MARK: - Initialization
    @MainActor
    init(
        pythonRunner: (any PythonRunning)? = nil,
        historyService: (any AnalysisHistoryServiceProtocol)? = nil,
        ollamaStatus: OllamaStatusChecker? = nil
    ) {
        self.pythonRunner = pythonRunner ?? PythonRunner.shared
        self.historyService = historyService ?? AnalysisHistoryService.shared
        self.ollamaStatus = ollamaStatus ?? OllamaStatusChecker.shared
    }
    
    // MARK: - Navigation Helpers
    var navigationTitleText: String {
        if let name = selectedFileURL?.lastPathComponent { return name }
        if !datasetURLInput.isEmpty { return "Web Dataset" }
        return "Aura Dashboard"
    }
    
    // MARK: - Page Lifecycle Management
    func openNewPage(
        title: String,
        fileURL: URL? = nil,
        datasetURLInput: String = "",
        previewResult: DatasetPreview? = nil,
        result: AnalysisResult? = nil,
        config: AnalysisConfig = AnalysisConfig(),
        trainColumns: [String] = [],
        historyItemId: UUID? = nil,
        isPreview: Bool = false,
        isDataOnly: Bool = false
    ) -> AnalysisPage {
        let newPage = AnalysisPage(
            title: title,
            fileURL: fileURL,
            datasetURLInput: datasetURLInput,
            previewResult: previewResult,
            result: result,
            config: config,
            trainColumns: trainColumns,
            historyItemId: historyItemId,
            isPreview: isPreview,
            isDataOnly: isDataOnly
        )
        if isPreview, let prevIndex = openPages.firstIndex(where: { $0.isPreview }) {
            openPages[prevIndex] = newPage
        } else {
            openPages.append(newPage)
        }
        activePageId = newPage.id
        return newPage
    }
    
    func closePage(id: UUID) {
        if let index = openPages.firstIndex(where: { $0.id == id }) {
            openPages.remove(at: index)
            if activePageId == id {
                activePageId = openPages.last?.id
            }
        }
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
        let title = url.lastPathComponent
        let page = openNewPage(title: title, fileURL: url, isPreview: true)
        page.analysisConfig.trainFilePath = url.path
        
        if let size = try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize {
            let fmt = ByteCountFormatter()
            fmt.countStyle = .file
            page.fileDetails = fmt.string(fromByteCount: Int64(size))
            
            if size > 20 * 1024 * 1024 {
                page.analysisConfig.smartSample = true
            }
        } else {
            page.fileDetails = "Unknown size"
        }
        fetchPreview(for: url.path, page: page)
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
    
    func loadRemoteDataset(_ urlString: String) {
        let trimmed = urlString.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        
        let provider = getURLProviderName(trimmed)
        let title: String
        if let url = URL(string: trimmed), !url.lastPathComponent.isEmpty && url.lastPathComponent != "/" {
            title = url.lastPathComponent
        } else {
            title = provider
        }
        
        let page = openNewPage(title: title, fileURL: nil, isPreview: true)
        page.datasetURLInput = trimmed
        
        fetchPreview(for: trimmed, page: page)
    }
    
    func clearSelection() {
        if let activeId = activePageId {
            closePage(id: activeId)
        } else {
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
        if let page = activePage {
            fetchPreview(for: pathOrURL, page: page)
        }
    }
    
    func fetchPreview(for pathOrURL: String, page: AnalysisPage) {
        withAnimation {
            page.isPreloading = true
            page.errorMessage = nil
            page.previewResult = nil
            page.result = nil
            page.progressFraction = 0.0
            page.progressMessage = "Preparing preview..."
        }
        
        var cleaningActionsJson: String? = nil
        if !page.analysisConfig.cleaningActions.isEmpty {
            let actionsArray = Array(page.analysisConfig.cleaningActions)
            if let encodedData = try? JSONEncoder().encode(actionsArray),
               let jsonString = String(data: encodedData, encoding: .utf8) {
                cleaningActionsJson = jsonString
            }
        }
        
        Task {
            await pythonRunner.runPreview(
                csvPathOrURL: pathOrURL,
                datasetType: page.analysisConfig.datasetType.rawValue,
                cleaningActions: cleaningActionsJson,
                progress: { frac, msg in
                Task { @MainActor in
                    page.progressFraction = frac
                    page.progressMessage = msg
                }
            }) { response in
                Task { @MainActor in
                    withAnimation {
                        page.isPreloading = false
                        switch response {
                        case .success(let previewData):
                           page.previewResult = previewData
                           if page.analysisConfig.trainFilePath == nil {
                               if let available = previewData.availableFiles,
                                  let trainFile = available.first(where: { $0.lowercased().contains("train") }) {
                                   page.analysisConfig.trainFilePath = trainFile
                               } else {
                                   page.analysisConfig.trainFilePath = previewData.localPath
                               }
                           }
                           if previewData.localPath == page.analysisConfig.trainFilePath {
                               page.trainColumns = previewData.columns
                           }
                           if let inferred = previewData.inferredDatasetType,
                              let type = DatasetType(rawValue: inferred) {
                               page.analysisConfig.datasetType = type
                           }
                           
                           // Auto deselect id columns
                           for (idx, col) in previewData.columns.enumerated() {
                               if self.isLikelyIdentifierColumn(name: col, columnIndex: idx, previewRows: previewData.previewRows) {
                                   page.analysisConfig.excludedColumns.insert(col)
                               }
                           }
                           
                           // Automate test and validation pre-selection
                           if let available = previewData.availableFiles {
                               if page.analysisConfig.testFilePath == nil {
                                   if let testFile = available.first(where: { $0.lowercased().contains("test") }) {
                                       page.analysisConfig.testFilePath = testFile
                                   }
                               }
                               if page.analysisConfig.validationFilePath == nil {
                                   if let valFile = available.first(where: { $0.lowercased().contains("val") || $0.lowercased().contains("valid") }) {
                                       page.analysisConfig.validationFilePath = valFile
                                   }
                               }
                           }
                        case .failure(let error):
                           if (error as NSError).code != -999 {
                               page.errorMessage = error.localizedDescription
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
        if let page = activePage {
            runEDA(with: nil, page: page)
        }
    }

    func runEDA(with customConfig: AnalysisConfig?) {
        if let page = activePage {
            runEDA(with: customConfig, page: page)
        }
    }

    func runEDA(with customConfig: AnalysisConfig?, page: AnalysisPage) {
        page.isPreview = false
        let configToUse = customConfig ?? page.analysisConfig
        
        let csvPath: String
        if let trainPath = configToUse.trainFilePath {
            csvPath = trainPath
        } else if let preview = page.previewResult {
            csvPath = preview.localPath
        } else if let fileURL = page.selectedFileURL {
            csvPath = fileURL.path
        } else {
            let urlStr = page.datasetURLInput.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !urlStr.isEmpty else { return }
            csvPath = urlStr
        }

        withAnimation {
            page.isAnalyzing = true
            page.progressFraction = 0.0
            page.progressMessage = "Starting analysis..."
            page.errorMessage = nil
            page.result = nil
        }

        let targetParam: String?
        if configToUse.datasetType == .timeSeries && !configToUse.targetColumns.isEmpty {
            targetParam = configToUse.targetColumns.joined(separator: ",")
        } else {
            let target = configToUse.targetColumn.trimmingCharacters(in: .whitespacesAndNewlines)
            targetParam = target.isEmpty ? nil : target
        }

        var finalConfig = configToUse
        if let targetParam = targetParam {
            let targetsList = targetParam.split(separator: ",").map(String.init)
            for t in targetsList {
                finalConfig.excludedColumns.remove(t)
            }
        }
        
        // Reset export paths on the persistent config so they do not linger for subsequent normal runs
        page.analysisConfig.modelExportPath = nil
        page.analysisConfig.codeExportPath = nil
        page.analysisConfig.notebookExportPath = nil

        // Reset timing stages
        page.completedStages = []
        page.currentStageMessage = ""
        page.currentStageStartTime = Date()
        
        let originalSource = page.selectedFileURL?.path ?? page.datasetURLInput.trimmingCharacters(in: .whitespacesAndNewlines)
        
        Task {
            await pythonRunner.runAnalysis(
                csvPath: csvPath,
                targetColumn: targetParam,
                config: finalConfig,
                progress: { frac, msg in
                    Task { @MainActor in
                        withAnimation(.easeInOut(duration: 0.15)) {
                            page.progressFraction = frac
                            page.progressMessage = msg
                            
                            if page.currentStageMessage != msg {
                                if !page.currentStageMessage.isEmpty {
                                    let elapsed = Date().timeIntervalSince(page.currentStageStartTime)
                                    let stage = ProgressStage(message: page.currentStageMessage, elapsed: elapsed)
                                    if !page.completedStages.contains(where: { $0.message == stage.message }) {
                                        page.completedStages.append(stage)
                                    }
                                }
                                page.currentStageMessage = msg
                                page.currentStageStartTime = Date()
                            }
                        }
                    }
                }
            ) { response in
                Task { @MainActor in
                    withAnimation {
                        if !page.currentStageMessage.isEmpty {
                            let elapsed = Date().timeIntervalSince(page.currentStageStartTime)
                            let stage = ProgressStage(message: page.currentStageMessage, elapsed: elapsed)
                            if !page.completedStages.contains(where: { $0.message == stage.message }) {
                                page.completedStages.append(stage)
                            }
                        }
                        page.isAnalyzing = false
                        switch response {
                        case .success(let data):
                            page.result = data
                            page.activeModelName = data.metrics.model
                            if let targetsMap = data.targets, !targetsMap.isEmpty {
                                page.selectedTargetName = targetsMap.keys.sorted().first ?? data.targetColumn
                                page.analysisConfig.targetColumns = Array(targetsMap.keys).sorted()
                            } else {
                                page.selectedTargetName = data.targetColumn
                                page.analysisConfig.targetColumn = data.targetColumn
                            }
                            var cleaningActionsJson: String? = nil
                            if !finalConfig.cleaningActions.isEmpty {
                                let actionsArray = Array(finalConfig.cleaningActions)
                                if let encodedData = try? JSONEncoder().encode(actionsArray),
                                   let jsonString = String(data: encodedData, encoding: .utf8) {
                                    cleaningActionsJson = jsonString
                                }
                            }
                            let savedItem = self.historyService.saveAnalysis(result: data, datasetPath: csvPath, targetColumn: targetParam, originalSource: originalSource, cleaningActionsJson: cleaningActionsJson)
                            page.currentHistoryItemId = savedItem?.id
                            page.title = savedItem?.datasetName ?? page.title
                            page.chatViewModel.injectContext(data, datasetURL: savedItem?.datasetURL, cleaningActions: cleaningActionsJson, otherRunsSummary: self.buildOtherRunsSummary(for: savedItem?.datasetName ?? page.title, excludingItemWithId: savedItem?.id))
                            if self.ollamaStatus.isAvailable { self.showAIPanel = true }
                        case .failure(let error):
                            if (error as NSError).code != -999 {
                                page.errorMessage = error.localizedDescription
                            }
                        }
                    }
                }
            }
        }
    }
    
    func loadHistoryItem(_ item: HistoryItem, isPreview: Bool = false, isDataOnly: Bool = false) {
        // If already open, make active
        if let existing = openPages.first(where: { $0.currentHistoryItemId == item.id && $0.isDataOnly == isDataOnly }) {
            activePageId = existing.id
            return
        }
        
        let groupItems = historyService.items.filter { $0.datasetName == item.datasetName }.sorted(by: { $0.timestamp < $1.timestamp })
        let versionNum = (groupItems.firstIndex(where: { $0.id == item.id }) ?? 0) + 1
        
        let title = isDataOnly ? "\(item.datasetName) • Data (v\(versionNum))" : "\(item.datasetName) (v\(versionNum))"
        let page = openNewPage(title: title, historyItemId: item.id, isPreview: isPreview, isDataOnly: isDataOnly)
        
        withAnimation {
            page.errorMessage = nil
            page.isAnalyzing = false
            page.isPreloading = true   // show a spinner while we read from disk
            page.progressFraction = 0.0
            page.progressMessage = ""
        }

        // Reconstruct the base configuration for this history item
        var newConfig = AnalysisConfig()
        newConfig.trainFilePath = item.datasetPath
        
        if let jsonStr = item.cleaningActionsJson,
           let jsonData = jsonStr.data(using: .utf8),
           let actions = try? JSONDecoder().decode([CleaningAction].self, from: jsonData) {
            newConfig.cleaningActions = Set(actions)
        }

        if let datasetURL = item.datasetURL, !datasetURL.isEmpty {
            page.datasetURLInput = datasetURL
            page.selectedFileURL = nil
            page.fileDetails = "Remote URL Dataset"
        } else if item.datasetPath.hasPrefix("http://") || item.datasetPath.hasPrefix("https://") {
            page.datasetURLInput = item.datasetPath
            page.selectedFileURL = nil
            page.fileDetails = "Remote URL Dataset"
        } else {
            page.selectedFileURL = URL(fileURLWithPath: item.datasetPath)
            page.datasetURLInput = ""
            if let size = try? page.selectedFileURL?.resourceValues(forKeys: [.fileSizeKey]).fileSize {
                let fmt = ByteCountFormatter()
                fmt.countStyle = .file
                page.fileDetails = fmt.string(fromByteCount: Int64(size))
            } else {
                page.fileDetails = "Local File"
            }
        }

        // Offload the potentially large file read to a background task
        Task { @MainActor in
            let loadedResult = await historyService.loadAnalysisResult(item: item)
            withAnimation {
                page.isPreloading = false
            }
            if let loadedResult {
                page.result = loadedResult
                page.activeModelName = item.bestModel ?? loadedResult.metrics.model
                page.trainColumns = loadedResult.columns
                page.chatViewModel.injectContext(loadedResult, datasetURL: item.datasetURL, cleaningActions: item.cleaningActionsJson, otherRunsSummary: self.buildOtherRunsSummary(for: item.datasetName, excludingItemWithId: item.id))

                if let targetsMap = loadedResult.targets, !targetsMap.isEmpty {
                    page.selectedTargetName = targetsMap.keys.sorted().first ?? loadedResult.targetColumn
                    newConfig.targetColumns = Array(targetsMap.keys).sorted()
                } else {
                    page.selectedTargetName = loadedResult.targetColumn
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

                page.previewResult = DatasetPreview(
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

                page.analysisConfig = newConfig
                if isDataOnly {
                    self.fetchPreview(for: item.datasetPath, page: page)
                }
                if self.ollamaStatus.isAvailable { self.showAIPanel = true }
                page.selectedTab = "Summary"
            } else {
                newConfig.targetColumn = item.targetColumn ?? ""
                page.analysisConfig = newConfig
            }
        }
    }
    
    var groupedDatasets: [GroupedDataset] {
        let items = historyService.items
        let grouped = Dictionary(grouping: items, by: { $0.datasetName })
        return grouped.map { name, runs in
            let sortedRuns = runs.sorted(by: { $0.timestamp > $1.timestamp })
            return GroupedDataset(name: name, runs: sortedRuns)
        }.sorted(by: { a, b in
            let aTime = a.runs.first?.timestamp ?? Date.distantPast
            let bTime = b.runs.first?.timestamp ?? Date.distantPast
            return aTime > bTime
        })
    }
    
    func buildOtherRunsSummary(for datasetName: String, excludingItemWithId: UUID? = nil) -> String {
        let otherRuns = historyService.items.filter {
            $0.datasetName == datasetName && $0.id != excludingItemWithId
        }
        guard !otherRuns.isEmpty else { return "" }
        
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        
        return otherRuns.map { run in
            let dateStr = formatter.string(from: run.timestamp)
            let task = run.taskType ?? "EDA"
            let target = run.targetColumn ?? "N/A"
            let best = run.bestModel ?? "N/A"
            let scoreVal = run.bestScore != nil ? String(format: "%.4f", run.bestScore!) : "N/A"
            let scoreT = run.scoreType ?? "score"
            return "- Task: \(task), Target: \(target), Best Model: \(best) (\(scoreT): \(scoreVal)), Run Date: \(dateStr)"
        }.joined(separator: "\n")
    }
}

struct GroupedDataset: Identifiable {
    var id: String { name }
    let name: String
    let runs: [HistoryItem]
}
