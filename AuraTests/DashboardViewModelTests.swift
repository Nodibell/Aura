import Testing
import Foundation
@testable import Aura

@MainActor
struct DashboardViewModelTests {
    
    // MARK: - Mocks
    
    class MockPythonRunner: PythonRunning, @unchecked Sendable {
        var verifyPythonEnvironmentResult = true
        var resolvedPythonPath = "/usr/bin/python3"
        var customPythonPath: String?
        var resetPythonPathCalled = false
        var startServerManualCalled = false
        var stopServerManualCalled = false
        var restartServerManualCalled = false
        var cancelActiveAnalysisCalled = false
        var serverRunningResult = false
        var serverPIDResult: Int32? = 1234
        var stopServerCalled = false
        
        func verifyPythonEnvironment(at pythonPath: String) -> Bool {
            return verifyPythonEnvironmentResult
        }
        
        func resolvePythonPath() -> String {
            return resolvedPythonPath
        }
        
        func setCustomPythonPath(_ path: String) {
            customPythonPath = path
        }
        
        func resetPythonPath() {
            resetPythonPathCalled = true
        }
        
        func startServerManual() async throws {
            startServerManualCalled = true
        }
        
        func stopServerManual() async {
            stopServerManualCalled = true
        }
        
        func restartServerManual() async throws {
            restartServerManualCalled = true
        }
        
        func cancelActiveAnalysis() async {
            cancelActiveAnalysisCalled = true
        }
        
        func isServerRunning() async -> Bool {
            return serverRunningResult
        }
        
        func getServerPID() -> Int32? {
            return serverPIDResult
        }
        
        func stopServer() async {
            stopServerCalled = true
        }
        
        func runAnalysis(
            csvPath: String,
            targetColumn: String?,
            config: AnalysisConfig,
            progress: @escaping @Sendable (Double, String) -> Void,
            completion: @escaping @Sendable (Result<AnalysisResult, Error>) -> Void
        ) async {
            // No-op for mocks in unit tests
        }
        
        func runPreview(
            csvPathOrURL: String,
            progress: @escaping @Sendable (Double, String) -> Void,
            completion: @escaping @Sendable (Result<DatasetPreview, Error>) -> Void
        ) async {
            // No-op for mocks in unit tests
        }
        
        func runDatabaseQuery(
            dbType: String,
            query: String,
            connParams: [String: String],
            outputCSVPath: String
        ) async throws -> (rowCount: Int, columns: [String]) {
            return (0, [])
        }
        
        func runMerge(
            file1: String,
            file2: String,
            key1: String,
            key2: String,
            joinType: String,
            outputMergePath: String
        ) async throws -> (rowCount: Int, columns: [String]) {
            return (0, [])
        }
        
        func runInference(modelPath: String, inputData: [String: Any]) async throws -> PredictionResult {
            throw NSError(domain: "Mock", code: -1)
        }
        
        func getCacheInfo() async throws -> PythonRunner.CacheInfo {
            return PythonRunner.CacheInfo(path: "/tmp", sizeBytes: 0, fileCount: 0)
        }
        
        func cleanCache() async throws {}
    }
    
    class MockHistoryService: AnalysisHistoryServiceProtocol {
        var items: [HistoryItem] = []
        var loadMetadataCalled = false
        var renameItemCalled = false
        var saveAnalysisCalled = false
        var loadAnalysisResultCalled = false
        var deleteItemCalled = false
        var clearHistoryCalled = false
        
        func loadMetadata() {
            loadMetadataCalled = true
        }
        
        func renameItem(_ item: HistoryItem, to newName: String) {
            renameItemCalled = true
        }
        
        func saveAnalysis(
            result: AnalysisResult,
            datasetPath: String,
            targetColumn: String?,
            originalSource: String?
        ) -> HistoryItem? {
            saveAnalysisCalled = true
            return nil
        }
        
        func loadAnalysisResult(item: HistoryItem) async -> AnalysisResult? {
            loadAnalysisResultCalled = true
            return nil
        }
        
        func deleteItem(_ item: HistoryItem) {
            deleteItemCalled = true
        }
        
        func clearHistory() {
            clearHistoryCalled = true
        }
    }
    
    // MARK: - Tests
    
    @Test func testNavigationTitle() {
        let mockRunner = MockPythonRunner()
        let mockHistory = MockHistoryService()
        let vm = DashboardViewModel(pythonRunner: mockRunner, historyService: mockHistory)
        
        #expect(vm.navigationTitleText == "Aura Dashboard")
        
        vm.selectedFileURL = URL(fileURLWithPath: "/path/to/my_data.csv")
        #expect(vm.navigationTitleText == "my_data.csv")
        
        vm.selectedFileURL = nil
        vm.datasetURLInput = "https://example.com/data.csv"
        #expect(vm.navigationTitleText == "Web Dataset")
    }
    
    @Test func testURLProviderName() {
        let mockRunner = MockPythonRunner()
        let mockHistory = MockHistoryService()
        let vm = DashboardViewModel(pythonRunner: mockRunner, historyService: mockHistory)
        
        #expect(vm.getURLProviderName("https://kaggle.com/code/some-user/notebook-name") == "Kaggle Notebook Output")
        #expect(vm.getURLProviderName("https://kaggle.com/datasets/some-user/dataset-name") == "Kaggle Dataset")
        #expect(vm.getURLProviderName("https://huggingface.co/datasets/some-user/dataset") == "Hugging Face Dataset")
        #expect(vm.getURLProviderName("https://example.com/raw.csv") == "Generic Web Dataset")
    }
    
    @Test func testClearSelection() {
        let mockRunner = MockPythonRunner()
        let mockHistory = MockHistoryService()
        let vm = DashboardViewModel(pythonRunner: mockRunner, historyService: mockHistory)
        
        vm.selectedFileURL = URL(fileURLWithPath: "/tmp/data.csv")
        vm.datasetURLInput = "https://example.com/raw.csv"
        vm.previewResult = DatasetPreview(columns: ["col"], previewRows: [], localPath: "/tmp/data.csv", error: nil, inferredDatasetType: "tabular", availableFiles: [], totalRows: 0)
        vm.result = AnalysisResult(columns: [], rowCount: 0, taskType: "classification", targetColumn: "label")
        vm.errorMessage = "Some error"
        
        vm.clearSelection()
        
        #expect(vm.selectedFileURL == nil)
        #expect(vm.datasetURLInput.isEmpty)
        #expect(vm.previewResult == nil)
        #expect(vm.result == nil)
        #expect(vm.errorMessage == nil)
    }
    
    @Test func testIsLikelyIdentifierColumn() {
        let mockRunner = MockPythonRunner()
        let mockHistory = MockHistoryService()
        let vm = DashboardViewModel(pythonRunner: mockRunner, historyService: mockHistory)
        
        // Matches ID suffix with unique ratio >= 0.95
        let previewRows: [[PreviewValue]] = [
            [.string("id_1")], [.string("id_2")], [.string("id_3")], [.string("id_4")], [.string("id_5")]
        ]
        #expect(vm.isLikelyIdentifierColumn(name: "user_id", columnIndex: 0, previewRows: previewRows) == true)
        
        // Matches ID suffix but low unique ratio (duplicate IDs)
        let duplicateRows: [[PreviewValue]] = [
            [.string("id_1")], [.string("id_1")], [.string("id_1")], [.string("id_2")], [.string("id_2")]
        ]
        #expect(vm.isLikelyIdentifierColumn(name: "user_id", columnIndex: 0, previewRows: duplicateRows) == false)
        
        // Unique values but name does not match ID pattern
        #expect(vm.isLikelyIdentifierColumn(name: "measurement_value", columnIndex: 0, previewRows: previewRows) == false)
    }
    
    @Test func testHandleDroppedFiles() {
        let mockRunner = MockPythonRunner()
        let mockHistory = MockHistoryService()
        let vm = DashboardViewModel(pythonRunner: mockRunner, historyService: mockHistory)
        
        // Single file dropped
        let singleURL = URL(fileURLWithPath: "/tmp/test.csv")
        vm.handleDroppedFiles([singleURL])
        #expect(vm.selectedFileURL == singleURL)
        #expect(vm.showMergeSheet == false)
        
        // Double files dropped (triggers merge)
        let file1 = URL(fileURLWithPath: "/tmp/file1.csv")
        let file2 = URL(fileURLWithPath: "/tmp/file2.csv")
        vm.handleDroppedFiles([file1, file2])
        #expect(vm.mergeFile1Path == file1.path)
        #expect(vm.mergeFile2Path == file2.path)
        #expect(vm.showMergeSheet == true)
    }
}
