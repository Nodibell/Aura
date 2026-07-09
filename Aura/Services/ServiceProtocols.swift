import Foundation

// MARK: - PythonRunning Protocol (PythonRunner)
protocol PythonRunning: Sendable {
    func verifyPythonEnvironment(at pythonPath: String) -> Bool
    func resolvePythonPath() -> String
    func setCustomPythonPath(_ path: String)
    func resetPythonPath()
    
    func startServerManual() async throws
    func stopServerManual() async
    func restartServerManual() async throws
    func cancelActiveAnalysis() async
    func isServerRunning() async -> Bool
    func getServerPID() -> Int32?
    func stopServer() async
    
    func runAnalysis(
        csvPath: String,
        targetColumn: String?,
        config: AnalysisConfig,
        progress: @escaping @Sendable (Double, String) -> Void,
        completion: @escaping @Sendable (Result<AnalysisResult, Error>) -> Void
    ) async
    
    func runPreview(
        csvPathOrURL: String,
        progress: @escaping @Sendable (Double, String) -> Void,
        completion: @escaping @Sendable (Result<DatasetPreview, Error>) -> Void
    ) async
    
    func runDatabaseQuery(
        dbType: String,
        query: String,
        connParams: [String: String],
        outputCSVPath: String
    ) async throws -> (rowCount: Int, columns: [String])
    
    func runMerge(
        file1: String,
        file2: String,
        key1: String,
        key2: String,
        joinType: String,
        outputMergePath: String
    ) async throws -> (rowCount: Int, columns: [String])
    
    func runInference(modelPath: String, inputData: [String: Any]) async throws -> PredictionResult
    func runBatchInference(modelPath: String, inputFilePath: String, outputFilePath: String) async throws -> BatchPredictionResult
    func getCacheInfo() async throws -> PythonRunner.CacheInfo
    func cleanCache() async throws
}

struct BatchPredictionResult: Codable, Sendable {
    let success: Bool
    let outputPath: String
    let rowCount: Int
    
    enum CodingKeys: String, CodingKey {
        case success
        case outputPath = "output_path"
        case rowCount = "row_count"
    }
}

// MARK: - AIServiceProtocol
protocol AIServiceProtocol: AnyObject, Sendable {
    func streamChat(
        messages: [OllamaChatMessage],
        systemPrompt: String,
        provider: LLMProvider,
        model: String,
        temperature: Double,
        maxTokens: Int
    ) -> AsyncThrowingStream<String, Error>
}

// MARK: - AnalysisHistoryServiceProtocol
@MainActor
protocol AnalysisHistoryServiceProtocol: AnyObject {
    var items: [HistoryItem] { get }
    func loadMetadata()
    func renameItem(_ item: HistoryItem, to newName: String)
    func togglePinItem(_ item: HistoryItem)
    func saveAnalysis(result: AnalysisResult, datasetPath: String, targetColumn: String?, originalSource: String?) -> HistoryItem?
    func loadAnalysisResult(item: HistoryItem) async -> AnalysisResult?
    func deleteItem(_ item: HistoryItem)
    func clearHistory()
}

// MARK: - OllamaServiceProtocol
protocol OllamaServiceProtocol: AnyObject, Sendable {
    func checkAvailability() async -> Bool
    func listModels() async -> [OllamaModelInfo]
    func streamChat(
        messages: [OllamaChatMessage],
        systemPrompt: String,
        model: String,
        temperature: Double,
        maxTokens: Int
    ) -> AsyncThrowingStream<String, Error>
}

// MARK: - REPLServiceProtocol
protocol REPLServiceProtocol {
    func reset(filePath: String, ollamaBaseURL: String, ollamaModel: String, cleaningActions: String?) async throws
    func execute(_ code: String) async throws -> REPLResult
    func getLineage() async throws -> [REPLService.LineageNode]
    func rollback(stateId: Int) async throws -> REPLService.RollbackResult
    func getPlugins() async throws -> [REPLService.PluginInfo]
}

// MARK: - KeychainServiceProtocol
protocol KeychainServiceProtocol: AnyObject, Sendable {
    func save(_ string: String, forKey key: String) -> Bool
    func load(forKey key: String) -> String?
    func delete(forKey key: String)
    func getSecureString(forKey key: String) -> String?
}
