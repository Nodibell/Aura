import Foundation
import AppKit

/// Thread-safe manager to track the active server process and allow synchronous termination.
final class ServerProcessManager {
    static let shared = ServerProcessManager()
    private let lock = NSLock()
    private var activeProcess: Process?
    
    private init() {}
    
    func setProcess(_ process: Process?) {
        lock.lock()
        defer { lock.unlock() }
        self.activeProcess = process
    }
    
    func getProcess() -> Process? {
        lock.lock()
        defer { lock.unlock() }
        return activeProcess
    }
    
    func terminateProcess() {
        lock.lock()
        defer { lock.unlock() }
        if let process = activeProcess {
            if process.isRunning {
                process.terminate()
            }
            activeProcess = nil
        }
    }
    
    func cleanOrphanedProcesses(port: Int) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/sbin/lsof")
        process.arguments = ["-t", "-i", ":\(port)"]
        
        let outputPipe = Pipe()
        process.standardOutput = outputPipe
        process.standardError = Pipe()
        
        do {
            try process.run()
            process.waitUntilExit()
            
            let data = outputPipe.fileHandleForReading.readDataToEndOfFile()
            if let output = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines),
               !output.isEmpty {
                let pids = output.components(separatedBy: .newlines).compactMap { Int32($0) }
                for pid in pids {
                    if pid != ProcessInfo.processInfo.processIdentifier {
                        // Terminate gracefully first, then kill -9 if still running
                        kill(pid, SIGTERM)
                        try? Thread.sleep(forTimeInterval: 0.05)
                        kill(pid, SIGKILL)
                    }
                }
            }
        } catch {
            // Ignore if lsof is not available or fails
        }
    }
}

actor PythonRunner: PythonRunning {
    static let shared = PythonRunner()
    
    private let defaultPythonPathKey = "Aura_PythonPath"
    private let serverPort = 11435
    
    private var serverProcess: Process?
    private var isServerStarting = false
    private var isCancelled = false
    private var activeTask: Task<Void, Never>?
    
    private init() {
        // Stop the server when the application terminates (synchronously!)
        NotificationCenter.default.addObserver(
            forName: NSApplication.willTerminateNotification,
            object: nil,
            queue: nil
        ) { _ in
            ServerProcessManager.shared.terminateProcess()
        }
        
        // Clean up any orphaned uvicorn server processes listening on port 11435
        ServerProcessManager.shared.cleanOrphanedProcesses(port: serverPort)
    }
    
    // MARK: - Logging Helpers (Safely bounces to MainActor)
    private nonisolated func logInfo(_ message: String, category: String = "PythonRunner") {
        Task { @MainActor in
            AppLogger.shared.info(message, category: category)
        }
    }
    
    private nonisolated func logError(_ message: String, category: String = "PythonRunner") {
        Task { @MainActor in
            AppLogger.shared.error(message, category: category)
        }
    }
    
    private nonisolated func logWarning(_ message: String, category: String = "PythonRunner") {
        Task { @MainActor in
            AppLogger.shared.warning(message, category: category)
        }
    }
    
    // MARK: - Python Path Resolution (Non-isolated to be synchronously callable)
    nonisolated func resolvePythonPath() -> String {
        if let savedPath = UserDefaults.standard.string(forKey: defaultPythonPathKey),
           FileManager.default.fileExists(atPath: savedPath),
           verifyPythonEnvironment(at: savedPath) {
            return savedPath
        }
        
        let homeDir = FileManager.default.homeDirectoryForCurrentUser.path
        var candidates: [String] = []
        
        let currentDir = FileManager.default.currentDirectoryPath
        candidates.append((currentDir as NSString).appendingPathComponent(".venv/bin/python3"))
        candidates.append((currentDir as NSString).appendingPathComponent(".venv/bin/python"))
        
        let devWorkspaceVenv3 = (homeDir as NSString).appendingPathComponent("Developer/Xcode.projects/Aura/.venv/bin/python3")
        let devWorkspaceVenv = (homeDir as NSString).appendingPathComponent("Developer/Xcode.projects/Aura/.venv/bin/python")
        candidates.append(devWorkspaceVenv3)
        candidates.append(devWorkspaceVenv)
        
        candidates.append("\(homeDir)/.pyenv/shims/python3")
        
        let pyenvVersionsDir = "\(homeDir)/.pyenv/versions"
        if let versions = try? FileManager.default.contentsOfDirectory(atPath: pyenvVersionsDir) {
            for version in versions {
                candidates.append("\(pyenvVersionsDir)/\(version)/bin/python3")
            }
        }
        
        if let shellPath = runShellCommand("which python3") {
            candidates.append(shellPath)
        }
        
        let commonPaths = [
            "/opt/homebrew/bin/python3",
            "/usr/local/bin/python3",
            "/usr/bin/python3"
        ]
        candidates.append(contentsOf: commonPaths)
        
        var uniqueExistingCandidates: [String] = []
        for path in candidates {
            let cleaned = path.trimmingCharacters(in: .whitespacesAndNewlines)
            if FileManager.default.fileExists(atPath: cleaned) && !uniqueExistingCandidates.contains(cleaned) {
                uniqueExistingCandidates.append(cleaned)
            }
        }
        
        for candidate in uniqueExistingCandidates {
            if verifyPythonEnvironment(at: candidate) {
                UserDefaults.standard.set(candidate, forKey: defaultPythonPathKey)
                return candidate
            }
        }
        
        if let firstExisting = uniqueExistingCandidates.first {
            UserDefaults.standard.set(firstExisting, forKey: defaultPythonPathKey)
            return firstExisting
        }
        
        return "/usr/bin/python3"
    }
    
    nonisolated func setCustomPythonPath(_ path: String) {
        UserDefaults.standard.set(path, forKey: defaultPythonPathKey)
    }
    
    nonisolated func resetPythonPath() {
        UserDefaults.standard.removeObject(forKey: defaultPythonPathKey)
    }
    
    private nonisolated func runShellCommand(_ command: String) -> String? {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/zsh")
        process.arguments = ["-l", "-c", command]
        
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = Pipe()
        
        do {
            try process.run()
            process.waitUntilExit()
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            if let path = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines),
               !path.isEmpty {
                return path
            }
        } catch {
            return nil
        }
        return nil
    }
    
    nonisolated func verifyPythonEnvironment(at pythonPath: String) -> Bool {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: pythonPath)
        process.arguments = ["-c", "import pandas, sklearn, numpy, torch, fastapi, uvicorn; import sys; sys.exit(0)"]
        
        do {
            try process.run()
            process.waitUntilExit()
            return process.terminationStatus == 0
        } catch {
            return false
        }
    }
    
    // MARK: - Server management
    private func checkServerHealth(url: String) async -> Bool {
        guard let healthURL = URL(string: "\(url)/health") else { return false }
        var request = URLRequest(url: healthURL)
        request.httpMethod = "GET"
        request.timeoutInterval = 0.5
        do {
            let (_, response) = try await URLSession.shared.data(for: request)
            return (response as? HTTPURLResponse)?.statusCode == 200
        } catch {
            return false
        }
    }
    
    private func ensureServerRunning() async throws -> String {
        let isHybrid = UserDefaults.standard.bool(forKey: "Aura_HybridMode")
        if isHybrid {
            let remoteURL = UserDefaults.standard.string(forKey: "Aura_RemoteServerURL") ?? "http://127.0.0.1:11435"
            return remoteURL.trimmingCharacters(in: .whitespacesAndNewlines)
        }

        let baseURL = "http://127.0.0.1:\(serverPort)"
        
        if await checkServerHealth(url: baseURL) {
            return baseURL
        }

        
        if isServerStarting {
            for _ in 0..<25 {
                try? await Task.sleep(nanoseconds: 200_000_000)
                if await checkServerHealth(url: baseURL) {
                    return baseURL
                }
            }
            throw NSError(domain: "PythonRunner", code: 503, userInfo: [NSLocalizedDescriptionKey: "Server is starting up but took too long."])
        }
        
        isServerStarting = true
        defer { isServerStarting = false }
        
        logInfo("Aura Local API Server is not running. Launching...")
        let pythonExecutable = self.resolvePythonPath()
        
        let serverScriptPath: String
        if let scriptURL = Bundle.main.url(forResource: "server", withExtension: "py") {
            serverScriptPath = scriptURL.path
        } else {
            let currentDir = FileManager.default.currentDirectoryPath
            let possibleWorkspacePath = (currentDir as NSString).appendingPathComponent("Aura/server.py")
            let homeDir = FileManager.default.homeDirectoryForCurrentUser.path
            let possibleHomeWorkspacePath = (homeDir as NSString).appendingPathComponent("Developer/Xcode.projects/Aura/Aura/server.py")
            
            if FileManager.default.fileExists(atPath: possibleWorkspacePath) {
                serverScriptPath = possibleWorkspacePath
            } else {
                serverScriptPath = possibleHomeWorkspacePath
            }
        }
        
        guard FileManager.default.fileExists(atPath: serverScriptPath) else {
            let errMsg = "server.py script not found."
            logError(errMsg)
            throw NSError(domain: "PythonRunner", code: 404, userInfo: [NSLocalizedDescriptionKey: errMsg])
        }
        
        let process = Process()
        process.executableURL = URL(fileURLWithPath: pythonExecutable)
        process.arguments = [serverScriptPath, "--port", "\(serverPort)"]
        
        let logURL: URL? = {
            let fileManager = FileManager.default
            if let appSupport = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first {
                let auraDir = appSupport.appendingPathComponent("Aura", isDirectory: true)
                try? fileManager.createDirectory(at: auraDir, withIntermediateDirectories: true, attributes: nil)
                let url = auraDir.appendingPathComponent("server.log")
                if !fileManager.fileExists(atPath: url.path) {
                    fileManager.createFile(atPath: url.path, contents: nil)
                }
                return url
            }
            return nil
        }()

        if let logURL = logURL, let fileHandle = try? FileHandle(forWritingTo: logURL) {
            fileHandle.seekToEndOfFile()
            process.standardOutput = fileHandle
            process.standardError = fileHandle
        } else {
            process.standardOutput = FileHandle.nullDevice
            process.standardError = FileHandle.nullDevice
        }

        
        do {
            try process.run()
            self.serverProcess = process
            ServerProcessManager.shared.setProcess(process)
            
            for _ in 0..<75 {
                try? await Task.sleep(nanoseconds: 200_000_000)
                if await checkServerHealth(url: baseURL) {
                    logInfo("Aura Local API Server successfully started on port \(serverPort).")
                    return baseURL
                }
            }
            
            process.terminate()

            self.serverProcess = nil
            ServerProcessManager.shared.setProcess(nil)
            throw NSError(domain: "PythonRunner", code: 500, userInfo: [NSLocalizedDescriptionKey: "Local API server failed to respond in time."])
        } catch {
            self.serverProcess = nil
            ServerProcessManager.shared.setProcess(nil)
            logError("Failed to launch local API server process: \(error.localizedDescription)")
            throw error
        }
    }
    
    func stopServer() {
        logInfo("Terminating Aura Local API Server process...")
        ServerProcessManager.shared.terminateProcess()
        self.serverProcess = nil
    }
    
    func isServerRunning() async -> Bool {
        let baseURL = "http://127.0.0.1:\(serverPort)"
        return await checkServerHealth(url: baseURL)
    }
    
    nonisolated func getServerPID() -> Int32? {
        if let process = ServerProcessManager.shared.getProcess(), process.isRunning {
            return process.processIdentifier
        }
        return nil
    }
    
    func startServerManual() async throws {
        _ = try await ensureServerRunning()
    }
    
    func stopServerManual() async {
        stopServer()
    }
    
    func restartServerManual() async throws {
        stopServer()
        try? await Task.sleep(nanoseconds: 500_000_000)
        _ = try await ensureServerRunning()
    }
    
    func cancelActiveAnalysis() {
        logInfo("Request to cancel active analysis received.")
        self.isCancelled = true
        if let task = self.activeTask {
            logInfo("Cancelling active Swift network task...")
            task.cancel()
            self.activeTask = nil
        }
    }
    
    // MARK: - Core Execution Methods
    func runAnalysis(
        csvPath: String,
        targetColumn: String?,
        config: AnalysisConfig = AnalysisConfig(),
        progress: @escaping @Sendable (Double, String) -> Void,
        completion: @escaping @Sendable (Result<AnalysisResult, Error>) -> Void
    ) {
        logInfo("Starting runAnalysis via FastAPI. Path/URL: \(csvPath). Target: \(targetColumn ?? "Auto-detect").")
        
        self.isCancelled = false
        
        var modelExportPath = config.modelExportPath
        if (modelExportPath == nil || modelExportPath!.isEmpty) {
            let csvURL = URL(fileURLWithPath: csvPath)
            let folder = csvURL.deletingLastPathComponent()
            let baseName = csvURL.deletingPathExtension().lastPathComponent
            modelExportPath = folder.appendingPathComponent("\(baseName)_model.joblib").path
        }
        
        var bodyDict: [String: Any] = [
            "file_path": csvPath,
            "dataset_type": config.datasetType.rawValue,
            "task_type_override": config.taskTypeOverride.rawValue,
            "smart_sample": config.smartSample,
            "feature_selection": config.featureSelection
        ]
        if let target = targetColumn {
            bodyDict["target_col"] = target
        }
                if let timeCol = config.timeColumn, !timeCol.isEmpty {
            bodyDict["time_col"] = timeCol
        }
        if !config.excludedColumns.isEmpty {
            bodyDict["exclude_cols"] = config.excludedColumns.joined(separator: ",")
        }
        if let start = config.timeRangeStart, !start.isEmpty {
            bodyDict["time_range_start"] = start
        }
        if let end = config.timeRangeEnd, !end.isEmpty {
            bodyDict["time_range_end"] = end
        }
        if let testPath = config.testFilePath, !testPath.isEmpty {
            bodyDict["test_file_path"] = testPath
        }
        if let valPath = config.validationFilePath, !valPath.isEmpty {
            bodyDict["val_file_path"] = valPath
        }
        if let modelPath = modelExportPath, !modelPath.isEmpty {
            bodyDict["model_export_path"] = modelPath
        }
        if let codePath = config.codeExportPath, !codePath.isEmpty {
            bodyDict["code_export_path"] = codePath
        }
        if let notebookPath = config.notebookExportPath, !notebookPath.isEmpty {
            bodyDict["notebook_export_path"] = notebookPath
        }

        if !config.cleaningActions.isEmpty {
            let actionsArray = Array(config.cleaningActions)
            if let encodedData = try? JSONEncoder().encode(actionsArray),
               let jsonString = String(data: encodedData, encoding: .utf8) {
                bodyDict["cleaning_actions"] = jsonString
            }
        }
        if !config.columnTypeOverrides.isEmpty {
            if let encodedData = try? JSONEncoder().encode(config.columnTypeOverrides),
               let jsonString = String(data: encodedData, encoding: .utf8) {
                bodyDict["column_type_overrides"] = jsonString
            }
        }
        
        let progressHandler = progress
        let completionHandler = completion
        
        self.activeTask = Task {
            var tempArrowPath: String? = nil
            defer {
                if let path = tempArrowPath, FileManager.default.fileExists(atPath: path) {
                    try? FileManager.default.removeItem(atPath: path)
                }
            }
            do {

                let baseURL = try await self.ensureServerRunning()
                
                let isRemoteURL = csvPath.lowercased().hasPrefix("http://") || csvPath.lowercased().hasPrefix("https://")
                
                var isDir: ObjCBool = false
                let fileExists = FileManager.default.fileExists(atPath: csvPath, isDirectory: &isDir)
                let isDirectory = fileExists && isDir.boolValue
                
                let ext = (csvPath as NSString).pathExtension.lowercased()
                let isTabularFile = ext == "csv" || ext == "tsv" || ext == "parquet"
                
                let useArrow = !isRemoteURL && !isDirectory && isTabularFile
                let requestURL: URL
                var requestBodyData: Data? = nil
                var isArrowPayload = false
                
                if useArrow {
                    // Local file: use Arrow IPC for 100x faster binary transfer
                    progressHandler(0.08, "Serializing dataset with Apache Arrow...")
                    let arrowFile = NSTemporaryDirectory() + UUID().uuidString + ".arrow"
                    tempArrowPath = arrowFile
                    
                    try await self.runArrowConversion(csvPath: csvPath, tempArrowPath: arrowFile)
                    
                    // Compile binary payload: [4 bytes JSON len] + [JSON bytes] + [Arrow IPC bytes]
                    guard let jsonData = try? JSONSerialization.data(withJSONObject: bodyDict) else {
                        throw NSError(domain: "PythonRunner", code: 400, userInfo: [NSLocalizedDescriptionKey: "Failed to serialize JSON config."])
                    }
                    var jsonLen = UInt32(jsonData.count).bigEndian
                    let lenData = withUnsafeBytes(of: &jsonLen) { Data($0) }
                    
                    var bodyData = Data()
                    bodyData.append(lenData)
                    bodyData.append(jsonData)
                    
                    let arrowData = try Data(contentsOf: URL(fileURLWithPath: arrowFile))
                    bodyData.append(arrowData)
                    
                    requestBodyData = bodyData
                    requestURL = URL(string: "\(baseURL)/analyze/arrow")!
                    isArrowPayload = true
                } else {
                    // Remote URL or Directory/Non-tabular local file: fallback to normal JSON
                    requestURL = URL(string: "\(baseURL)/analyze")!
                    requestBodyData = try? JSONSerialization.data(withJSONObject: bodyDict)
                }
                
                var request = URLRequest(url: requestURL)
                request.httpMethod = "POST"
                if isArrowPayload {
                    request.setValue("application/octet-stream", forHTTPHeaderField: "Content-Type")
                } else {
                    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                }
                request.setValue("text/event-stream", forHTTPHeaderField: "Accept")
                request.httpBody = requestBodyData
                request.timeoutInterval = 600
                
                let (bytes, response) = try await URLSession.shared.bytes(for: request)
                guard (response as? HTTPURLResponse)?.statusCode == 200 else {
                    let (data, _) = try await URLSession.shared.data(for: request)
                    let responseString = String(data: data, encoding: .utf8) ?? "Unknown server error"
                    completionHandler(.failure(NSError(domain: "PythonRunner", code: 500, userInfo: [NSLocalizedDescriptionKey: "Server returned error: \(responseString)"])))
                    if let path = tempArrowPath, FileManager.default.fileExists(atPath: path) {
                        try? FileManager.default.removeItem(atPath: path)
                    }
                    return
                }

                
                var finalResultData: Data? = nil
                var serverError: String? = nil
                
                for try await line in bytes.lines {
                    if Task.isCancelled {
                        completionHandler(.failure(NSError(domain: "PythonRunner", code: -999, userInfo: [NSLocalizedDescriptionKey: "Analysis was cancelled by user."])))
                        return
                    }
                    
                    if line.hasPrefix("data: ") {
                        let dataStr = line.dropFirst("data: ".count).trimmingCharacters(in: .whitespacesAndNewlines)
                        self.logInfo("SSE line received. Length: \(dataStr.count)")
                        
                        guard let eventData = dataStr.data(using: .utf8) else {
                            self.logError("Failed to convert SSE line to UTF-8 data.")
                            continue
                        }
                        
                        do {
                            if let eventJson = try JSONSerialization.jsonObject(with: eventData) as? [String: Any],
                               let type = eventJson["type"] as? String {
                                
                                if type == "progress" {
                                    if let frac = eventJson["progress"] as? Double,
                                       let msg = eventJson["message"] as? String {
                                        progressHandler(frac, msg)
                                    }
                                } else if type == "result" {
                                    if let resultDict = eventJson["data"] as? [String: Any] {
                                        if let errStr = resultDict["error"] as? String, !errStr.isEmpty {
                                            serverError = errStr
                                            self.logError("Result payload contained an error: \(errStr)")
                                        } else if let serializedData = try? JSONSerialization.data(withJSONObject: resultDict) {
                                            finalResultData = serializedData
                                            self.logInfo("Successfully parsed final result data from SSE event.")
                                        }
                                    } else {
                                        self.logError("SSE result event did not contain a valid 'data' dictionary.")
                                    }
                                } else if type == "error" {
                                    serverError = eventJson["error"] as? String
                                    self.logError("Received SSE error event: \(serverError ?? "nil")")
                                } else if type == "log" {
                                    if let msg = eventJson["message"] as? String {
                                        self.logInfo("[Subprocess Log] \(msg)")
                                    }
                                }
                            } else {
                                self.logError("SSE event JSON was parsed but type or structure was missing.")
                            }
                        } catch {
                            self.logError("Failed to parse SSE event JSON: \(error.localizedDescription)")
                        }
                    } else if !line.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        self.logInfo("Non-data SSE line received: \(line.prefix(100))...")
                    }
                }
                
                if let serverError = serverError {
                    completionHandler(.failure(NSError(domain: "PythonRunner", code: 500, userInfo: [NSLocalizedDescriptionKey: serverError])))
                } else if let resultData = finalResultData {
                    do {
                        let decoder = JSONDecoder()
                        let result = try decoder.decode(AnalysisResult.self, from: resultData)
                        completionHandler(.success(result))
                    } catch {
                        completionHandler(.failure(error))
                    }
                } else {
                    completionHandler(.failure(NSError(domain: "PythonRunner", code: 500, userInfo: [NSLocalizedDescriptionKey: "No result received from server."])))
                }
            } catch {
                if Task.isCancelled {
                    completionHandler(.failure(NSError(domain: "PythonRunner", code: -999, userInfo: [NSLocalizedDescriptionKey: "Analysis was cancelled by user."])))
                } else {
                    completionHandler(.failure(error))
                }
            }
        }
    }
    
    func runPreview(
        csvPathOrURL: String,
        progress: @escaping @Sendable (Double, String) -> Void,
        completion: @escaping @Sendable (Result<DatasetPreview, Error>) -> Void
    ) {
        logInfo("Starting runPreview via FastAPI. Path/URL: \(csvPathOrURL)")
        
        self.isCancelled = false
        
        let bodyDict: [String: Any] = [
            "file_path": csvPathOrURL
        ]
        
        let progressHandler = progress
        let completionHandler = completion
        
        self.activeTask = Task {
            do {
                let baseURL = try await self.ensureServerRunning()
                
                guard let url = URL(string: "\(baseURL)/preview") else {
                    completionHandler(.failure(NSError(domain: "PythonRunner", code: 400, userInfo: [NSLocalizedDescriptionKey: "Invalid API URL."])))
                    return
                }
                
                var request = URLRequest(url: url)
                request.httpMethod = "POST"
                request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                request.setValue("text/event-stream", forHTTPHeaderField: "Accept")
                request.httpBody = try? JSONSerialization.data(withJSONObject: bodyDict)
                request.timeoutInterval = 600
                
                let (bytes, response) = try await URLSession.shared.bytes(for: request)
                guard (response as? HTTPURLResponse)?.statusCode == 200 else {
                    let (data, _) = try await URLSession.shared.data(for: request)
                    let responseString = String(data: data, encoding: .utf8) ?? "Unknown server error"
                    completionHandler(.failure(NSError(domain: "PythonRunner", code: 500, userInfo: [NSLocalizedDescriptionKey: "Server returned error: \(responseString)"])))
                    return
                }
                
                var finalResultData: Data? = nil
                var serverError: String? = nil
                
                for try await line in bytes.lines {
                    if Task.isCancelled {
                        completionHandler(.failure(NSError(domain: "PythonRunner", code: -999, userInfo: [NSLocalizedDescriptionKey: "Preview was cancelled by user."])))
                        return
                    }
                    
                    if line.hasPrefix("data: ") {
                        let dataStr = line.dropFirst("data: ".count).trimmingCharacters(in: .whitespacesAndNewlines)
                        self.logInfo("SSE line received. Length: \(dataStr.count)")
                        
                        guard let eventData = dataStr.data(using: .utf8) else {
                            self.logError("Failed to convert SSE line to UTF-8 data.")
                            continue
                        }
                        
                        do {
                            if let eventJson = try JSONSerialization.jsonObject(with: eventData) as? [String: Any],
                               let type = eventJson["type"] as? String {
                                
                                if type == "progress" {
                                    if let frac = eventJson["progress"] as? Double,
                                       let msg = eventJson["message"] as? String {
                                        progressHandler(frac, msg)
                                    }
                                } else if type == "result" {
                                    if let resultDict = eventJson["data"] as? [String: Any] {
                                        if let errStr = resultDict["error"] as? String, !errStr.isEmpty {
                                            serverError = errStr
                                            self.logError("Result payload contained a preview error: \(errStr)")
                                        } else if let serializedData = try? JSONSerialization.data(withJSONObject: resultDict) {
                                            finalResultData = serializedData
                                            self.logInfo("Successfully parsed final preview result from SSE event.")
                                        }
                                    } else {
                                        self.logError("SSE result event did not contain a valid 'data' dictionary.")
                                    }
                                } else if type == "error" {
                                    serverError = eventJson["error"] as? String
                                    self.logError("Received SSE error event: \(serverError ?? "nil")")
                                } else if type == "log" {
                                    if let msg = eventJson["message"] as? String {
                                        self.logInfo("[Subprocess Log] \(msg)")
                                    }
                                }
                            } else {
                                self.logError("SSE event JSON was parsed but type or structure was missing.")
                            }
                        } catch {
                            self.logError("Failed to parse SSE event JSON: \(error.localizedDescription)")
                        }
                    }
                }
                
                if let serverError = serverError {
                    completionHandler(.failure(NSError(domain: "PythonRunner", code: 500, userInfo: [NSLocalizedDescriptionKey: serverError])))
                } else if let resultData = finalResultData {
                    do {
                        let decoder = JSONDecoder()
                        let result = try decoder.decode(DatasetPreview.self, from: resultData)
                        completionHandler(.success(result))
                    } catch {
                        completionHandler(.failure(error))
                    }
                } else {
                    completionHandler(.failure(NSError(domain: "PythonRunner", code: 500, userInfo: [NSLocalizedDescriptionKey: "No result received from server."])))
                }
            } catch {
                if Task.isCancelled {
                    completionHandler(.failure(NSError(domain: "PythonRunner", code: -999, userInfo: [NSLocalizedDescriptionKey: "Preview was cancelled by user."])))
                } else {
                    completionHandler(.failure(error))
                }
            }
        }
    }
    
    func runDatabaseQuery(
        dbType: String,
        query: String,
        connParams: [String: String],
        outputCSVPath: String
    ) async throws -> (rowCount: Int, columns: [String]) {
        logInfo("Starting runDatabaseQuery via FastAPI. Type: \(dbType), Output: \(outputCSVPath)")
        
        self.isCancelled = false
        
        let bodyDict: [String: Any] = [
            "db_type": dbType,
            "query": query,
            "conn_params": connParams,
            "output_csv": outputCSVPath
        ]
        
        let baseURL = try await self.ensureServerRunning()
        
        guard let url = URL(string: "\(baseURL)/query_db") else {
            throw NSError(domain: "PythonRunner", code: 400, userInfo: [NSLocalizedDescriptionKey: "Invalid API URL."])
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try? JSONSerialization.data(withJSONObject: bodyDict)
        request.timeoutInterval = 300
        
        let (data, response) = try await URLSession.shared.data(for: request)
        
        guard (response as? HTTPURLResponse)?.statusCode == 200 else {
            let errStr = String(data: data, encoding: .utf8) ?? "Unknown server error"
            throw NSError(domain: "PythonRunner", code: 500, userInfo: [NSLocalizedDescriptionKey: "Server returned error: \(errStr)"])
        }
        
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let success = json["success"] as? Bool else {
            throw NSError(domain: "PythonRunner", code: 499, userInfo: [NSLocalizedDescriptionKey: "Failed to parse database query result JSON."])
        }
        
        if success {
            let rowCount = json["row_count"] as? Int ?? 0
            let columns = json["columns"] as? [String] ?? []
            return (rowCount, columns)
        } else {
            let errorMsg = json["error"] as? String ?? "Unknown database logic error."
            throw NSError(domain: "PythonRunner", code: 500, userInfo: [NSLocalizedDescriptionKey: errorMsg])
        }
    }
    
    func runMerge(
        file1: String,
        file2: String,
        key1: String,
        key2: String,
        joinType: String,
        outputMergePath: String
    ) async throws -> (rowCount: Int, columns: [String]) {
        logInfo("Starting runMerge via FastAPI. File1: \(file1), File2: \(file2)")
        
        self.isCancelled = false
        
        let bodyDict: [String: Any] = [
            "file1": file1,
            "file2": file2,
            "key1": key1,
            "key2": key2,
            "join_type": joinType,
            "output_merge_path": outputMergePath
        ]
        
        let baseURL = try await self.ensureServerRunning()
        
        guard let url = URL(string: "\(baseURL)/merge") else {
            throw NSError(domain: "PythonRunner", code: 400, userInfo: [NSLocalizedDescriptionKey: "Invalid API URL."])
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try? JSONSerialization.data(withJSONObject: bodyDict)
        request.timeoutInterval = 120
        
        let (data, response) = try await URLSession.shared.data(for: request)
        
        guard (response as? HTTPURLResponse)?.statusCode == 200 else {
            let errStr = String(data: data, encoding: .utf8) ?? "Unknown server error"
            throw NSError(domain: "PythonRunner", code: 500, userInfo: [NSLocalizedDescriptionKey: "Server returned error: \(errStr)"])
        }
        
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let success = json["success"] as? Bool else {
            throw NSError(domain: "PythonRunner", code: 499, userInfo: [NSLocalizedDescriptionKey: "Failed to parse merge result JSON."])
        }
        
        if success {
            let rowCount = json["row_count"] as? Int ?? 0
            let columns = json["columns"] as? [String] ?? []
            return (rowCount, columns)
        } else {
            let errorMsg = json["error"] as? String ?? "Unknown merge logic error."
            throw NSError(domain: "PythonRunner", code: 500, userInfo: [NSLocalizedDescriptionKey: errorMsg])
        }
    }
    
    func runInference(modelPath: String, inputData: [String: Any]) async throws -> PredictionResult {
        logInfo("Starting runInference via FastAPI. Model: \(modelPath)")
        
        self.isCancelled = false
        
        let bodyDict: [String: Any] = [
            "model_path": modelPath,
            "input_data": inputData
        ]
        
        let baseURL = try await self.ensureServerRunning()
        
        guard let url = URL(string: "\(baseURL)/predict") else {
            throw NSError(domain: "PythonRunner", code: 400, userInfo: [NSLocalizedDescriptionKey: "Invalid API URL."])
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try? JSONSerialization.data(withJSONObject: bodyDict)
        request.timeoutInterval = 60
        
        let (data, response) = try await URLSession.shared.data(for: request)
        
        guard (response as? HTTPURLResponse)?.statusCode == 200 else {
            let errStr = String(data: data, encoding: .utf8) ?? "Unknown server error"
            throw NSError(domain: "PythonRunner", code: 500, userInfo: [NSLocalizedDescriptionKey: "Server returned error: \(errStr)"])
        }
        
        let decoder = JSONDecoder()
        let result = try decoder.decode(PredictionResult.self, from: data)
        return result
    }

    private func runArrowConversion(csvPath: String, tempArrowPath: String) async throws {
        let pythonExecutable = self.resolvePythonPath()
        
        let arrowHelperPath: String
        if let scriptURL = Bundle.main.url(forResource: "arrow_helper", withExtension: "py") {
            arrowHelperPath = scriptURL.path
        } else {
            let currentDir = FileManager.default.currentDirectoryPath
            let possibleWorkspacePath = (currentDir as NSString).appendingPathComponent("Aura/utils/arrow_helper.py")
            let homeDir = FileManager.default.homeDirectoryForCurrentUser.path
            let possibleHomeWorkspacePath = (homeDir as NSString).appendingPathComponent("Developer/Xcode.projects/Aura/Aura/utils/arrow_helper.py")
            
            if FileManager.default.fileExists(atPath: possibleWorkspacePath) {
                arrowHelperPath = possibleWorkspacePath
            } else {
                arrowHelperPath = possibleHomeWorkspacePath
            }
        }
        
        guard FileManager.default.fileExists(atPath: arrowHelperPath) else {
            throw NSError(domain: "PythonRunner", code: 404, userInfo: [NSLocalizedDescriptionKey: "arrow_helper.py not found."])
        }
        
        let process = Process()
        process.executableURL = URL(fileURLWithPath: pythonExecutable)
        process.arguments = [arrowHelperPath, "--to-arrow", csvPath, tempArrowPath]
        
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            process.terminationHandler = { proc in
                if proc.terminationStatus == 0 {
                    continuation.resume()
                } else {
                    continuation.resume(throwing: NSError(domain: "PythonRunner", code: Int(proc.terminationStatus), userInfo: [NSLocalizedDescriptionKey: "Arrow conversion exited with code \(proc.terminationStatus)."]))
                }
            }
            do {
                try process.run()
            } catch {
                continuation.resume(throwing: error)
            }
        }
    }

    // MARK: - Cache Management

    struct CacheInfo: Decodable {
        let path: String
        let sizeBytes: Int64
        let fileCount: Int
        
        enum CodingKeys: String, CodingKey {
            case path
            case sizeBytes = "size_bytes"
            case fileCount = "file_count"
        }
    }

    func getCacheInfo() async throws -> CacheInfo {
        let baseURL = try await self.ensureServerRunning()
        let url = URL(string: "\(baseURL)/cache/info")!
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.timeoutInterval = 10
        
        let (data, response) = try await URLSession.shared.data(for: request)
        guard (response as? HTTPURLResponse)?.statusCode == 200 else {
            let errStr = String(data: data, encoding: .utf8) ?? "Unknown server error"
            throw NSError(domain: "PythonRunner", code: 500, userInfo: [NSLocalizedDescriptionKey: "Server returned error: \(errStr)"])
        }
        
        let decoder = JSONDecoder()
        return try decoder.decode(CacheInfo.self, from: data)
    }

    func cleanCache() async throws {
        let baseURL = try await self.ensureServerRunning()
        let url = URL(string: "\(baseURL)/cache/clean")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.timeoutInterval = 15
        
        let (data, response) = try await URLSession.shared.data(for: request)
        guard (response as? HTTPURLResponse)?.statusCode == 200 else {
            let errStr = String(data: data, encoding: .utf8) ?? "Unknown server error"
            throw NSError(domain: "PythonRunner", code: 500, userInfo: [NSLocalizedDescriptionKey: "Server returned error: \(errStr)"])
        }
    }
}

