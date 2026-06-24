import Foundation

// A thread-safe class for accumulating Data from FileHandles concurrently.
final class ProtectedData: @unchecked Sendable {
    private var data = Data()
    private let lock = NSLock()
    
    func append(_ newData: Data) {
        lock.lock()
        defer { lock.unlock() }
        data.append(newData)
    }
    
    func get() -> Data {
        lock.lock()
        defer { lock.unlock() }
        return data
    }
}

actor PythonRunner {
    static let shared = PythonRunner()
    
    private let defaultPythonPathKey = "Aura_PythonPath"
    
    private var activeProcess: Process?
    private var isCancelled = false
    
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
           FileManager.default.fileExists(atPath: savedPath) {
            return savedPath
        }
        
        let homeDir = FileManager.default.homeDirectoryForCurrentUser.path
        var candidates: [String] = []
        
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
        process.arguments = ["-c", "import pandas, sklearn, numpy, torch; import sys; sys.exit(0 if torch.backends.mps.is_available() else 1)"]
        
        do {
            try process.run()
            process.waitUntilExit()
            return process.terminationStatus == 0
        } catch {
            return false
        }
    }
    
    // MARK: - Process Execution (Non-blocking async runner)
    private func runProcessAsync(_ process: Process) async throws -> Int32 {
        try await withCheckedThrowingContinuation { continuation in
            process.terminationHandler = { proc in
                continuation.resume(returning: proc.terminationStatus)
            }
            do {
                try process.run()
            } catch {
                continuation.resume(throwing: error)
            }
        }
    }
    
    func cancelActiveAnalysis() {
        logInfo("Request to cancel active analysis received.")
        self.isCancelled = true
        if let process = self.activeProcess {
            logInfo("Terminating active process...")
            process.terminate()
        } else {
            logInfo("No active process to terminate.")
        }
    }
    
    // MARK: - Core Execution
    func runAnalysis(
        csvPath: String,
        targetColumn: String?,
        config: AnalysisConfig = AnalysisConfig(),
        progress: @escaping @Sendable (Double, String) -> Void,
        completion: @escaping @Sendable (Result<AnalysisResult, Error>) -> Void
    ) {
        logInfo("Starting runAnalysis for path/URL: \(csvPath). Target: \(targetColumn ?? "Auto-detect"). Type: \(config.datasetType.rawValue)")
        
        self.isCancelled = false
        let pythonExecutable = self.resolvePythonPath()
        
        let scriptPath: String
        if let scriptURL = Bundle.main.url(forResource: "analyze", withExtension: "py") {
            scriptPath = scriptURL.path
        } else {
            scriptPath = "/Users/oleksiichumak/Developer/Xcode.projects/Aura/Aura/analyze.py"
        }
        
        guard FileManager.default.fileExists(atPath: scriptPath) else {
            let errMsg = "analyze.py script not found."
            self.logError(errMsg)
            completion(.failure(NSError(domain: "PythonRunner", code: 404, userInfo: [NSLocalizedDescriptionKey: errMsg])))
            return
        }
        
        Task {
            await self.executePythonScript(
                pythonPath: pythonExecutable,
                scriptPath: scriptPath,
                csvPath: csvPath,
                targetColumn: targetColumn,
                config: config,
                progress: progress,
                completion: completion
            )
        }
    }
    
    nonisolated func buildArguments(
        scriptPath: String,
        csvPath: String,
        targetColumn: String?,
        config: AnalysisConfig
    ) -> [String] {
        var arguments = [scriptPath, csvPath]
        if let target = targetColumn, !target.isEmpty {
            arguments += ["--target", target]
        }
        arguments += ["--dataset-type", config.datasetType.rawValue]
        if config.taskTypeOverride != .auto {
            arguments += ["--task-type", config.taskTypeOverride.rawValue]
        }
        if let timeCol = config.timeColumn, !timeCol.isEmpty {
            arguments += ["--time-col", timeCol]
        }
        if !config.excludedColumns.isEmpty {
            let cols = config.excludedColumns.sorted().joined(separator: ",")
            arguments += ["--exclude-cols", cols]
        }
        if let testPath = config.testFilePath, !testPath.isEmpty {
            arguments += ["--test-file", testPath]
        }
        if let valPath = config.validationFilePath, !valPath.isEmpty {
            arguments += ["--val-file", valPath]
        }
        if config.smartSample {
            arguments += ["--smart-sample"]
        }
        if !config.cleaningActions.isEmpty {
            let encoder = JSONEncoder()
            if let data = try? encoder.encode(config.cleaningActions),
               let jsonStr = String(data: data, encoding: .utf8) {
                arguments += ["--cleaning-actions", jsonStr]
            }
        }
        var modelPath = config.modelExportPath
        if modelPath == nil || modelPath!.isEmpty {
            let csvURL = URL(fileURLWithPath: csvPath)
            if csvPath.starts(with: "http://") || csvPath.starts(with: "https://") {
                let tempDir = FileManager.default.temporaryDirectory
                let baseName = csvURL.deletingPathExtension().lastPathComponent
                modelPath = tempDir.appendingPathComponent("\(baseName)_model.joblib").path
            } else {
                let folder = csvURL.deletingLastPathComponent()
                let baseName = csvURL.deletingPathExtension().lastPathComponent
                modelPath = folder.appendingPathComponent("\(baseName)_model.joblib").path
            }
        }
        if let modelPath = modelPath, !modelPath.isEmpty {
            arguments += ["--model-export-path", modelPath]
        }
        if let codePath = config.codeExportPath, !codePath.isEmpty {
            arguments += ["--code-export-path", codePath]
        }
        return arguments
    }
    
    private func executePythonScript(
        pythonPath: String,
        scriptPath: String,
        csvPath: String,
        targetColumn: String?,
        config: AnalysisConfig,
        progress: @escaping @Sendable (Double, String) -> Void,
        completion: @escaping @Sendable (Result<AnalysisResult, Error>) -> Void
    ) async {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: pythonPath)
        
        let arguments = buildArguments(scriptPath: scriptPath, csvPath: csvPath, targetColumn: targetColumn, config: config)
        process.arguments = arguments
        
        logInfo("Launching Python subprocess with arguments: \(arguments)")
        
        var env = ProcessInfo.processInfo.environment
        let pathLower = csvPath.lowercased()
        let isKaggle = pathLower.contains("kaggle.com")
        let isHF = pathLower.contains("huggingface.co")
        
        if isKaggle {
            if let kaggleUser = KeychainService.shared.getSecureString(forKey: "Aura_KaggleUsername"), !kaggleUser.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                env["KAGGLE_USERNAME"] = kaggleUser.trimmingCharacters(in: .whitespacesAndNewlines)
            }
            if let kaggleKey = KeychainService.shared.getSecureString(forKey: "Aura_KaggleKey"), !kaggleKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                env["KAGGLE_KEY"] = kaggleKey.trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }
        
        if isHF {
            if let hfToken = KeychainService.shared.getSecureString(forKey: "Aura_HFToken"), !hfToken.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                env["HF_TOKEN"] = hfToken.trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }
        process.environment = env
        
        let outPipe = Pipe()
        let errPipe = Pipe()
        
        process.standardOutput = outPipe
        process.standardError = errPipe
        
        let accumulatedOutData = ProtectedData()
        let accumulatedErrData = ProtectedData()
        
        outPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            accumulatedOutData.append(data)
        }
        
        let progressHandler = progress
        errPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            accumulatedErrData.append(data)
            
            if let text = String(data: data, encoding: .utf8) {
                let lines = text.components(separatedBy: .newlines)
                for line in lines {
                    let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
                    guard !trimmed.isEmpty else { continue }
                    
                    if trimmed.hasPrefix("PROGRESS: ") {
                        let parts = trimmed.dropFirst("PROGRESS: ".count).split(separator: ":", maxSplits: 1)
                        if parts.count == 2,
                           let frac = Double(parts[0].trimmingCharacters(in: .whitespaces)) {
                            let msg = String(parts[1]).trimmingCharacters(in: .whitespaces)
                            
                            Task { @MainActor in
                                progressHandler(frac, msg)
                            }
                        }
                    }
                }
            }
        }
        
        if self.isCancelled {
            completion(.failure(NSError(domain: "PythonRunner", code: -999, userInfo: [NSLocalizedDescriptionKey: "Analysis was cancelled by user."])))
            return
        }
        
        self.activeProcess = process
        
        do {
            let status = try await runProcessAsync(process)
            
            outPipe.fileHandleForReading.readabilityHandler = nil
            errPipe.fileHandleForReading.readabilityHandler = nil
            
            let leftoverOut = outPipe.fileHandleForReading.readDataToEndOfFile()
            if !leftoverOut.isEmpty {
                accumulatedOutData.append(leftoverOut)
            }
            let leftoverErr = errPipe.fileHandleForReading.readDataToEndOfFile()
            if !leftoverErr.isEmpty {
                accumulatedErrData.append(leftoverErr)
            }
            
            self.activeProcess = nil
            
            if self.isCancelled {
                logInfo("Analysis subprocess was terminated due to cancellation.")
                completion(.failure(NSError(domain: "PythonRunner", code: -999, userInfo: [NSLocalizedDescriptionKey: "Analysis was cancelled by user."])))
                return
            }
            
            let outData = accumulatedOutData.get()
            let errData = accumulatedErrData.get()
            
            logInfo("Python subprocess finished with termination status: \(status)")
            
            if status != 0 {
                let errString = String(data: errData, encoding: .utf8) ?? "Unknown Python execution error"
                logError("Subprocess execution failed: \(errString)")
                completion(.failure(NSError(domain: "PythonRunner", code: Int(status), userInfo: [NSLocalizedDescriptionKey: errString])))
                return
            }
            
            if let errorDict = try? JSONSerialization.jsonObject(with: outData) as? [String: Any],
               let errorMsg = errorDict["error"] as? String, !errorMsg.isEmpty {
                logError("Subprocess returned logic error: \(errorMsg)")
                completion(.failure(NSError(domain: "PythonRunner", code: 500, userInfo: [NSLocalizedDescriptionKey: errorMsg])))
                return
            }
            
            do {
                let decoder = JSONDecoder()
                let result = try decoder.decode(AnalysisResult.self, from: outData)
                if let errorMsg = result.error {
                    logError("Decoded AnalysisResult indicates error: \(errorMsg)")
                    completion(.failure(NSError(domain: "PythonRunner", code: 500, userInfo: [NSLocalizedDescriptionKey: errorMsg])))
                } else {
                    logInfo("Successfully completed analysis of \(result.rowCount) rows, \(result.colCount) columns.")
                    completion(.success(result))
                }
            } catch {
                let rawOutput = String(data: outData, encoding: .utf8) ?? "Unreadable output"
                let errString = String(data: errData, encoding: .utf8) ?? ""
                let detail = "JSON Decoding Failed: \(error.localizedDescription)\n\nPython Output:\n\(rawOutput)\n\nErrors:\n\(errString)"
                logError(detail)
                completion(.failure(NSError(domain: "PythonRunner", code: 499, userInfo: [NSLocalizedDescriptionKey: detail])))
            }
            
        } catch {
            outPipe.fileHandleForReading.readabilityHandler = nil
            errPipe.fileHandleForReading.readabilityHandler = nil
            self.activeProcess = nil
            logError("Failed to run subprocess: \(error.localizedDescription)")
            completion(.failure(error))
        }
    }
    
    func runPreview(
        csvPathOrURL: String,
        progress: @escaping @Sendable (Double, String) -> Void,
        completion: @escaping @Sendable (Result<DatasetPreview, Error>) -> Void
    ) {
        logInfo("Starting runPreview for path/URL: \(csvPathOrURL)")
        
        self.isCancelled = false
        let pythonExecutable = self.resolvePythonPath()
        
        let scriptPath: String
        if let scriptURL = Bundle.main.url(forResource: "analyze", withExtension: "py") {
            scriptPath = scriptURL.path
        } else {
            scriptPath = "/Users/oleksiichumak/Developer/Xcode.projects/Aura/Aura/analyze.py"
        }
        
        guard FileManager.default.fileExists(atPath: scriptPath) else {
            let errMsg = "analyze.py script not found."
            self.logError(errMsg)
            completion(.failure(NSError(domain: "PythonRunner", code: 404, userInfo: [NSLocalizedDescriptionKey: errMsg])))
            return
        }
        
        Task {
            await self.executePythonPreview(
                pythonPath: pythonExecutable,
                scriptPath: scriptPath,
                csvPath: csvPathOrURL,
                progress: progress,
                completion: completion
            )
        }
    }
    
    private func executePythonPreview(
        pythonPath: String,
        scriptPath: String,
        csvPath: String,
        progress: @escaping @Sendable (Double, String) -> Void,
        completion: @escaping @Sendable (Result<DatasetPreview, Error>) -> Void
    ) async {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: pythonPath)
        process.arguments = [scriptPath, csvPath, "--preview"]
        
        logInfo("Launching Python preview subprocess with arguments: \(process.arguments ?? [])")
        
        var env = ProcessInfo.processInfo.environment
        let pathLower = csvPath.lowercased()
        let isKaggle = pathLower.contains("kaggle.com")
        let isHF = pathLower.contains("huggingface.co")
        
        if isKaggle {
            if let kaggleUser = KeychainService.shared.getSecureString(forKey: "Aura_KaggleUsername"), !kaggleUser.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                env["KAGGLE_USERNAME"] = kaggleUser.trimmingCharacters(in: .whitespacesAndNewlines)
            }
            if let kaggleKey = KeychainService.shared.getSecureString(forKey: "Aura_KaggleKey"), !kaggleKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                env["KAGGLE_KEY"] = kaggleKey.trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }
        
        if isHF {
            if let hfToken = KeychainService.shared.getSecureString(forKey: "Aura_HFToken"), !hfToken.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                env["HF_TOKEN"] = hfToken.trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }
        process.environment = env
        
        let outPipe = Pipe()
        let errPipe = Pipe()
        
        process.standardOutput = outPipe
        process.standardError = errPipe
        
        let accumulatedOutData = ProtectedData()
        let accumulatedErrData = ProtectedData()
        
        outPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            accumulatedOutData.append(data)
        }
        
        let progressHandler = progress
        errPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            accumulatedErrData.append(data)
            
            if let text = String(data: data, encoding: .utf8) {
                let lines = text.components(separatedBy: .newlines)
                for line in lines {
                    let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
                    guard !trimmed.isEmpty else { continue }
                    
                    if trimmed.hasPrefix("PROGRESS: ") {
                        let parts = trimmed.dropFirst("PROGRESS: ".count).split(separator: ":", maxSplits: 1)
                        if parts.count == 2,
                           let frac = Double(parts[0].trimmingCharacters(in: .whitespaces)) {
                            let msg = String(parts[1]).trimmingCharacters(in: .whitespaces)
                            
                            Task { @MainActor in
                                progressHandler(frac, msg)
                            }
                        }
                    }
                }
            }
        }
        
        if self.isCancelled {
            completion(.failure(NSError(domain: "PythonRunner", code: -999, userInfo: [NSLocalizedDescriptionKey: "Preview was cancelled by user."])))
            return
        }
        
        self.activeProcess = process
        
        do {
            let status = try await runProcessAsync(process)
            
            outPipe.fileHandleForReading.readabilityHandler = nil
            errPipe.fileHandleForReading.readabilityHandler = nil
            
            let leftoverOut = outPipe.fileHandleForReading.readDataToEndOfFile()
            if !leftoverOut.isEmpty {
                accumulatedOutData.append(leftoverOut)
            }
            let leftoverErr = errPipe.fileHandleForReading.readDataToEndOfFile()
            if !leftoverErr.isEmpty {
                accumulatedErrData.append(leftoverErr)
            }
            
            self.activeProcess = nil
            
            if self.isCancelled {
                logInfo("Preview subprocess was terminated due to cancellation.")
                completion(.failure(NSError(domain: "PythonRunner", code: -999, userInfo: [NSLocalizedDescriptionKey: "Preview was cancelled by user."])))
                return
            }
            
            let outData = accumulatedOutData.get()
            let errData = accumulatedErrData.get()
            
            logInfo("Python preview subprocess finished with termination status: \(status)")
            
            if status != 0 {
                let errString = String(data: errData, encoding: .utf8) ?? "Preview execution failed."
                completion(.failure(NSError(domain: "PythonRunner", code: Int(status), userInfo: [NSLocalizedDescriptionKey: errString])))
                return
            }
            
            if let errorDict = try? JSONSerialization.jsonObject(with: outData) as? [String: Any],
               let errorMsg = errorDict["error"] as? String, !errorMsg.isEmpty {
                completion(.failure(NSError(domain: "PythonRunner", code: 500, userInfo: [NSLocalizedDescriptionKey: errorMsg])))
                return
            }
            
            do {
                let decoder = JSONDecoder()
                let result = try decoder.decode(DatasetPreview.self, from: outData)
                if let errorMsg = result.error {
                    completion(.failure(NSError(domain: "PythonRunner", code: 500, userInfo: [NSLocalizedDescriptionKey: errorMsg])))
                } else {
                    logInfo("Successfully completed preview of \(result.columns.count) columns.")
                    completion(.success(result))
                }
            } catch {
                let rawOutput = String(data: outData, encoding: .utf8) ?? "Unreadable output"
                let errString = String(data: errData, encoding: .utf8) ?? ""
                let detail = "JSON Decoding Failed (Preview): \(error.localizedDescription)\n\nPython Output:\n\(rawOutput)\n\nErrors:\n\(errString)"
                completion(.failure(NSError(domain: "PythonRunner", code: 499, userInfo: [NSLocalizedDescriptionKey: detail])))
            }
        } catch {
            outPipe.fileHandleForReading.readabilityHandler = nil
            errPipe.fileHandleForReading.readabilityHandler = nil
            self.activeProcess = nil
            logError("Failed to run preview subprocess: \(error.localizedDescription)")
            completion(.failure(error))
        }
    }
    
    func runDatabaseQuery(
        dbType: String,
        query: String,
        connParams: [String: String],
        outputCSVPath: String
    ) async throws -> (rowCount: Int, columns: [String]) {
        logInfo("Starting runDatabaseQuery. Type: \(dbType), Output: \(outputCSVPath)")
        
        self.isCancelled = false
        let pythonExecutable = self.resolvePythonPath()
        
        let scriptPath: String
        if let scriptURL = Bundle.main.url(forResource: "query_db", withExtension: "py") {
            scriptPath = scriptURL.path
        } else {
            scriptPath = "/Users/oleksiichumak/Developer/Xcode.projects/Aura/Aura/query_db.py"
        }
        
        guard FileManager.default.fileExists(atPath: scriptPath) else {
            let errMsg = "query_db.py script not found."
            self.logError(errMsg)
            throw NSError(domain: "PythonRunner", code: 404, userInfo: [NSLocalizedDescriptionKey: errMsg])
        }
        
        guard let connParamsData = try? JSONSerialization.data(withJSONObject: connParams),
              let connParamsJSON = String(data: connParamsData, encoding: .utf8) else {
            throw NSError(domain: "PythonRunner", code: 400, userInfo: [NSLocalizedDescriptionKey: "Invalid connection parameters."])
        }
        
        let process = Process()
        process.executableURL = URL(fileURLWithPath: pythonExecutable)
        process.arguments = [
            scriptPath,
            "--db-type", dbType,
            "--query", query,
            "--conn-params", connParamsJSON,
            "--output-csv", outputCSVPath
        ]
        
        let outPipe = Pipe()
        let errPipe = Pipe()
        process.standardOutput = outPipe
        process.standardError = errPipe
        
        let accumulatedOutData = ProtectedData()
        let accumulatedErrData = ProtectedData()
        
        outPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            accumulatedOutData.append(data)
        }
        
        errPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            accumulatedErrData.append(data)
        }
        
        if self.isCancelled {
            throw NSError(domain: "PythonRunner", code: -999, userInfo: [NSLocalizedDescriptionKey: "Database query was cancelled."])
        }
        
        self.activeProcess = process
        
        do {
            let status = try await runProcessAsync(process)
            
            outPipe.fileHandleForReading.readabilityHandler = nil
            errPipe.fileHandleForReading.readabilityHandler = nil
            
            let leftoverOut = outPipe.fileHandleForReading.readDataToEndOfFile()
            if !leftoverOut.isEmpty {
                accumulatedOutData.append(leftoverOut)
            }
            let leftoverErr = errPipe.fileHandleForReading.readDataToEndOfFile()
            if !leftoverErr.isEmpty {
                accumulatedErrData.append(leftoverErr)
            }
            
            self.activeProcess = nil
            
            if self.isCancelled {
                throw NSError(domain: "PythonRunner", code: -999, userInfo: [NSLocalizedDescriptionKey: "Database query was cancelled."])
            }
            
            let outData = accumulatedOutData.get()
            let errData = accumulatedErrData.get()
            
            if status != 0 {
                let errString = String(data: errData, encoding: .utf8) ?? "Database execution failed."
                throw NSError(domain: "PythonRunner", code: Int(status), userInfo: [NSLocalizedDescriptionKey: errString])
            }
            
            guard let json = try? JSONSerialization.jsonObject(with: outData) as? [String: Any] else {
                let rawOutput = String(data: outData, encoding: .utf8) ?? "Unreadable output"
                let errString = String(data: errData, encoding: .utf8) ?? ""
                throw NSError(domain: "PythonRunner", code: 499, userInfo: [NSLocalizedDescriptionKey: "Failed to parse database runner output: \(rawOutput)\nErrors: \(errString)"])
            }
            
            if let success = json["success"] as? Bool, success {
                let rowCount = json["row_count"] as? Int ?? 0
                let columns = json["columns"] as? [String] ?? []
                return (rowCount, columns)
            } else {
                let errorMsg = json["error"] as? String ?? "Unknown database logic error."
                throw NSError(domain: "PythonRunner", code: 500, userInfo: [NSLocalizedDescriptionKey: errorMsg])
            }
            
        } catch {
            outPipe.fileHandleForReading.readabilityHandler = nil
            errPipe.fileHandleForReading.readabilityHandler = nil
            self.activeProcess = nil
            throw error
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
        logInfo("Starting runMerge. File1: \(file1), File2: \(file2), Output: \(outputMergePath)")
        
        self.isCancelled = false
        let pythonExecutable = self.resolvePythonPath()
        
        let scriptPath: String
        if let scriptURL = Bundle.main.url(forResource: "analyze", withExtension: "py") {
            scriptPath = scriptURL.path
        } else {
            scriptPath = "/Users/oleksiichumak/Developer/Xcode.projects/Aura/Aura/analyze.py"
        }
        
        guard FileManager.default.fileExists(atPath: scriptPath) else {
            let errMsg = "analyze.py script not found."
            self.logError(errMsg)
            throw NSError(domain: "PythonRunner", code: 404, userInfo: [NSLocalizedDescriptionKey: errMsg])
        }
        
        let process = Process()
        process.executableURL = URL(fileURLWithPath: pythonExecutable)
        process.arguments = [
            scriptPath,
            file1,
            "--merge",
            "--file2", file2,
            "--key1", key1,
            "--key2", key2,
            "--join-type", joinType,
            "--output-merge-path", outputMergePath
        ]
        
        let outPipe = Pipe()
        let errPipe = Pipe()
        process.standardOutput = outPipe
        process.standardError = errPipe
        
        let accumulatedOutData = ProtectedData()
        let accumulatedErrData = ProtectedData()
        
        outPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            accumulatedOutData.append(data)
        }
        
        errPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            accumulatedErrData.append(data)
        }
        
        if self.isCancelled {
            throw NSError(domain: "PythonRunner", code: -999, userInfo: [NSLocalizedDescriptionKey: "Merge was cancelled."])
        }
        
        self.activeProcess = process
        
        do {
            let status = try await runProcessAsync(process)
            
            outPipe.fileHandleForReading.readabilityHandler = nil
            errPipe.fileHandleForReading.readabilityHandler = nil
            
            let leftoverOut = outPipe.fileHandleForReading.readDataToEndOfFile()
            if !leftoverOut.isEmpty {
                accumulatedOutData.append(leftoverOut)
            }
            let leftoverErr = errPipe.fileHandleForReading.readDataToEndOfFile()
            if !leftoverErr.isEmpty {
                accumulatedErrData.append(leftoverErr)
            }
            
            self.activeProcess = nil
            
            if self.isCancelled {
                throw NSError(domain: "PythonRunner", code: -999, userInfo: [NSLocalizedDescriptionKey: "Merge was cancelled."])
            }
            
            let outData = accumulatedOutData.get()
            let errData = accumulatedErrData.get()
            
            if status != 0 {
                let errString = String(data: errData, encoding: .utf8) ?? "Merge execution failed."
                throw NSError(domain: "PythonRunner", code: Int(status), userInfo: [NSLocalizedDescriptionKey: errString])
            }
            
            guard let json = try? JSONSerialization.jsonObject(with: outData) as? [String: Any] else {
                let rawOutput = String(data: outData, encoding: .utf8) ?? "Unreadable output"
                let errString = String(data: errData, encoding: .utf8) ?? ""
                throw NSError(domain: "PythonRunner", code: 499, userInfo: [NSLocalizedDescriptionKey: "Failed to parse merge runner output: \(rawOutput)\nErrors: \(errString)"])
            }
            
            if let success = json["success"] as? Bool, success {
                let rowCount = json["row_count"] as? Int ?? 0
                let columns = json["columns"] as? [String] ?? []
                return (rowCount, columns)
            } else {
                let errorMsg = json["error"] as? String ?? "Unknown merge logic error."
                throw NSError(domain: "PythonRunner", code: 500, userInfo: [NSLocalizedDescriptionKey: errorMsg])
            }
            
        } catch {
            outPipe.fileHandleForReading.readabilityHandler = nil
            errPipe.fileHandleForReading.readabilityHandler = nil
            self.activeProcess = nil
            throw error
        }
    }
    
    func runInference(modelPath: String, inputData: [String: Any]) async throws -> PredictionResult {
        logInfo("Starting runInference for model: \(modelPath)")
        logInfo("[Predict] Input features (\(inputData.count)): \(inputData.keys.sorted().joined(separator: ", "))")

        // --- Model file check ---
        let modelExists = FileManager.default.fileExists(atPath: modelPath)
        logInfo("[Predict] Model file exists at path: \(modelExists)")
        if !modelExists {
            let errMsg = "[Predict] Model file not found: \(modelPath)"
            logError(errMsg)
            throw NSError(domain: "PythonRunner", code: 404, userInfo: [NSLocalizedDescriptionKey: errMsg])
        }

        let pythonExecutable = self.resolvePythonPath()
        logInfo("[Predict] Python executable: \(pythonExecutable)")

        let scriptPath: String
        if let scriptURL = Bundle.main.url(forResource: "analyze", withExtension: "py") {
            scriptPath = scriptURL.path
        } else {
            scriptPath = "/Users/oleksiichumak/Developer/Xcode.projects/Aura/Aura/analyze.py"
        }
        logInfo("[Predict] Script path: \(scriptPath)")

        guard FileManager.default.fileExists(atPath: scriptPath) else {
            let errMsg = "analyze.py script not found at: \(scriptPath)"
            self.logError(errMsg)
            throw NSError(domain: "PythonRunner", code: 404, userInfo: [NSLocalizedDescriptionKey: errMsg])
        }

        // --- Serialize input data ---
        guard let inputDataJSONData = try? JSONSerialization.data(withJSONObject: inputData, options: .sortedKeys),
              let inputDataJSON = String(data: inputDataJSONData, encoding: .utf8) else {
            throw NSError(domain: "PythonRunner", code: 400, userInfo: [NSLocalizedDescriptionKey: "Invalid input data format."])
        }
        logInfo("[Predict] Input JSON: \(inputDataJSON.prefix(300))")

        let process = Process()
        process.executableURL = URL(fileURLWithPath: pythonExecutable)
        process.arguments = [
            scriptPath,
            "--predict",
            "--model-path", modelPath,
            "--input-data", inputDataJSON
        ]
        logInfo("[Predict] Launching subprocess with args: --predict --model-path <path> --input-data <json>")

        let outPipe = Pipe()
        let errPipe = Pipe()
        process.standardOutput = outPipe
        process.standardError = errPipe

        let accumulatedOutData = ProtectedData()
        let accumulatedErrData = ProtectedData()

        outPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            accumulatedOutData.append(data)
        }

        errPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            accumulatedErrData.append(data)
        }

        do {
            let status = try await runProcessAsync(process)

            outPipe.fileHandleForReading.readabilityHandler = nil
            errPipe.fileHandleForReading.readabilityHandler = nil

            let leftoverOut = outPipe.fileHandleForReading.readDataToEndOfFile()
            if !leftoverOut.isEmpty {
                accumulatedOutData.append(leftoverOut)
            }
            let leftoverErr = errPipe.fileHandleForReading.readDataToEndOfFile()
            if !leftoverErr.isEmpty {
                accumulatedErrData.append(leftoverErr)
            }

            let outData = accumulatedOutData.get()
            let errData = accumulatedErrData.get()

            // Always log stderr (Python warnings, tracebacks, debug prints)
            if !errData.isEmpty, let errString = String(data: errData, encoding: .utf8) {
                let trimmed = errString.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty {
                    for line in trimmed.components(separatedBy: "\n") {
                        if line.lowercased().contains("error") || line.lowercased().contains("traceback") || line.lowercased().contains("exception") {
                            logError("[Predict stderr] \(line)")
                        } else {
                            logInfo("[Predict stderr] \(line)")
                        }
                    }
                }
            }

            logInfo("[Predict] Subprocess exited with status: \(status)")
            logInfo("[Predict] stdout size: \(outData.count) bytes")

            if status != 0 {
                let errString = String(data: errData, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? "Prediction execution failed (exit \(status))."
                let msg = errString.isEmpty ? "Python subprocess exited with code \(status)." : errString
                logError("[Predict] Non-zero exit: \(msg)")
                throw NSError(domain: "PythonRunner", code: Int(status), userInfo: [NSLocalizedDescriptionKey: msg])
            }

            // Log raw stdout for debugging
            if let rawOut = String(data: outData, encoding: .utf8) {
                logInfo("[Predict] Raw stdout: \(rawOut.prefix(500))")
            }

            guard let json = try? JSONSerialization.jsonObject(with: outData) as? [String: Any] else {
                let rawOut = String(data: outData, encoding: .utf8) ?? "<unreadable>"
                let msg = "Failed to parse prediction JSON. Raw output: \(rawOut.prefix(300))"
                logError("[Predict] \(msg)")
                throw NSError(domain: "PythonRunner", code: 499, userInfo: [NSLocalizedDescriptionKey: msg])
            }

            if let errorMsg = json["error"] as? String, !errorMsg.isEmpty {
                logError("[Predict] Python returned error field: \(errorMsg)")
                throw NSError(domain: "PythonRunner", code: 500, userInfo: [NSLocalizedDescriptionKey: errorMsg])
            }

            logInfo("[Predict] JSON keys in response: \(json.keys.sorted().joined(separator: ", "))")

            let decoder = JSONDecoder()
            let result = try decoder.decode(PredictionResult.self, from: outData)
            logInfo("[Predict] Prediction decoded successfully: \(result.prediction)")
            return result

        } catch {
            outPipe.fileHandleForReading.readabilityHandler = nil
            errPipe.fileHandleForReading.readabilityHandler = nil
            logError("[Predict] runInference failed: \(error.localizedDescription)")
            throw error
        }
    }
}
