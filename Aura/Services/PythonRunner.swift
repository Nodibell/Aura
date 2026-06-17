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

final class PythonRunner: @unchecked Sendable {
    static let shared = PythonRunner()
    
    private let defaultPythonPathKey = "Aura_PythonPath"
    
    // Using NSLock for thread-safe state management
    private let processLock = NSLock()
    private var activeProcess: Process?
    private var isCancelled = false
    
    // MARK: - Logging Helpers (Safely bounces to MainActor)
    private func logInfo(_ message: String, category: String = "PythonRunner") {
        Task { @MainActor in
            AppLogger.shared.info(message, category: category)
        }
    }
    
    private func logError(_ message: String, category: String = "PythonRunner") {
        Task { @MainActor in
            AppLogger.shared.error(message, category: category)
        }
    }
    
    private func logWarning(_ message: String, category: String = "PythonRunner") {
        Task { @MainActor in
            AppLogger.shared.warning(message, category: category)
        }
    }
    
    // MARK: - Python Path Resolution
    func resolvePythonPath() -> String {
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
    
    func setCustomPythonPath(_ path: String) {
        UserDefaults.standard.set(path, forKey: defaultPythonPathKey)
    }
    
    func resetPythonPath() {
        UserDefaults.standard.removeObject(forKey: defaultPythonPathKey)
    }
    
    func cancelActiveAnalysis() {
        logInfo("Request to cancel active analysis received.")
        processLock.lock()
        defer { processLock.unlock() }
        
        self.isCancelled = true
        if let process = self.activeProcess {
            logInfo("Terminating active process...")
            process.terminate()
        } else {
            logInfo("No active process to terminate.")
        }
    }
    
    private func runShellCommand(_ command: String) -> String? {
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
    
    func verifyPythonEnvironment(at pythonPath: String) -> Bool {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: pythonPath)
        process.arguments = ["-c", "import pandas, sklearn, numpy"]
        
        do {
            try process.run()
            process.waitUntilExit()
            return process.terminationStatus == 0
        } catch {
            return false
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
        
        processLock.lock()
        self.isCancelled = false
        processLock.unlock()
        
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            
            let pythonExecutable = self.resolvePythonPath()
            self.logInfo("Resolved Python Executable: \(pythonExecutable)")
            
            if let scriptURL = Bundle.main.url(forResource: "analyze", withExtension: "py") {
                self.logInfo("Using bundled analyze.py script at: \(scriptURL.path)")
                self.executePythonScript(pythonPath: pythonExecutable, scriptPath: scriptURL.path, csvPath: csvPath, targetColumn: targetColumn, config: config, progress: progress, completion: completion)
            } else {
                let workspacePath = "/Users/oleksiichumak/Developer/Xcode.projects/Aura/Aura/analyze.py"
                if FileManager.default.fileExists(atPath: workspacePath) {
                    self.logInfo("Using workspace analyze.py script at: \(workspacePath)")
                    self.executePythonScript(pythonPath: pythonExecutable, scriptPath: workspacePath, csvPath: csvPath, targetColumn: targetColumn, config: config, progress: progress, completion: completion)
                } else {
                    let errMsg = "analyze.py script not found. Make sure it is bundled or in the project directory."
                    self.logError(errMsg)
                    completion(.failure(NSError(domain: "PythonRunner", code: 404, userInfo: [NSLocalizedDescriptionKey: errMsg])))
                }
            }
        }
    }
    
    private func executePythonScript(
        pythonPath: String,
        scriptPath: String,
        csvPath: String,
        targetColumn: String?,
        config: AnalysisConfig,
        progress: @escaping @Sendable (Double, String) -> Void,
        completion: @escaping @Sendable (Result<AnalysisResult, Error>) -> Void
    ) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: pythonPath)
        
        // Build argument list with new config flags
        var arguments = [scriptPath, csvPath]
        if let target = targetColumn, !target.isEmpty {
            arguments += ["--target", target]
        }
        // Dataset type (always explicit so Python doesn't have to guess)
        arguments += ["--dataset-type", config.datasetType.rawValue]
        // Task type override
        if config.taskTypeOverride != .auto {
            arguments += ["--task-type", config.taskTypeOverride.rawValue]
        }
        // Time column for Time Series
        if let timeCol = config.timeColumn, !timeCol.isEmpty {
            arguments += ["--time-col", timeCol]
        }
        // Excluded columns (comma-separated names)
        if !config.excludedColumns.isEmpty {
            let cols = config.excludedColumns.sorted().joined(separator: ",")
            arguments += ["--exclude-cols", cols]
        }
        // Test dataset file path
        if let testPath = config.testFilePath, !testPath.isEmpty {
            arguments += ["--test-file", testPath]
        }
        // Validation dataset file path
        if let valPath = config.validationFilePath, !valPath.isEmpty {
            arguments += ["--val-file", valPath]
        }
        // Smart sampling (Phase 2)
        if config.smartSample {
            arguments += ["--smart-sample"]
        }
        // Cleaning actions (Phase 3)
        if !config.cleaningActions.isEmpty {
            let encoder = JSONEncoder()
            if let data = try? encoder.encode(config.cleaningActions),
               let jsonStr = String(data: data, encoding: .utf8) {
                arguments += ["--cleaning-actions", jsonStr]
            }
        }
        // Model & Code Export (Phase 1)
        if let modelPath = config.modelExportPath, !modelPath.isEmpty {
            arguments += ["--model-export-path", modelPath]
        }
        if let codePath = config.codeExportPath, !codePath.isEmpty {
            arguments += ["--code-export-path", codePath]
        }
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
        errPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
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
                            
                            self?.logInfo("Progress \(Int(frac * 100))%: \(msg)", category: "PythonSubprocess")
                            Task { @MainActor in
                                progressHandler(frac, msg)
                            }
                        }
                    } else {
                        self?.logWarning(trimmed, category: "PythonSubprocess")
                    }
                }
            }
        }
        
        var wasCancelledBeforeRun = false
        processLock.lock()
        if self.isCancelled {
            wasCancelledBeforeRun = true
        } else {
            self.activeProcess = process
        }
        processLock.unlock()
        
        if wasCancelledBeforeRun {
            completion(.failure(NSError(domain: "PythonRunner", code: -999, userInfo: [NSLocalizedDescriptionKey: "Analysis was cancelled by user."])))
            return
        }
        
        defer {
            processLock.lock()
            if self.activeProcess === process {
                self.activeProcess = nil
            }
            processLock.unlock()
        }
        
        do {
            try process.run()
            process.waitUntilExit()
            
            outPipe.fileHandleForReading.readabilityHandler = nil
            errPipe.fileHandleForReading.readabilityHandler = nil
            
            var wasCancelled = false
            processLock.lock()
            wasCancelled = self.isCancelled
            processLock.unlock()
            
            if wasCancelled {
                logInfo("Analysis subprocess was terminated due to cancellation.")
                completion(.failure(NSError(domain: "PythonRunner", code: -999, userInfo: [NSLocalizedDescriptionKey: "Analysis was cancelled by user."])))
                return
            }
            
            let outData = accumulatedOutData.get()
            let errData = accumulatedErrData.get()
            
            logInfo("Python subprocess finished with termination status: \(process.terminationStatus)")
            
            if process.terminationStatus != 0 {
                let errString = String(data: errData, encoding: .utf8) ?? "Unknown Python execution error"
                logError("Subprocess execution failed: \(errString)")
                completion(.failure(NSError(domain: "PythonRunner", code: Int(process.terminationStatus), userInfo: [NSLocalizedDescriptionKey: errString])))
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
            errPipe.fileHandleForReading.readabilityHandler = nil
            logError("Failed to run subprocess: \(error.localizedDescription)")
            completion(.failure(error))
        }
    }
    
    func runPreview(csvPathOrURL: String, completion: @escaping @Sendable (Result<DatasetPreview, Error>) -> Void) {
        logInfo("Starting runPreview for path/URL: \(csvPathOrURL)")
        
        processLock.lock()
        self.isCancelled = false
        processLock.unlock()
        
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            
            let pythonExecutable = self.resolvePythonPath()
            self.logInfo("Resolved Python Executable: \(pythonExecutable)")
            
            let scriptPath: String
            if let scriptURL = Bundle.main.url(forResource: "analyze", withExtension: "py") {
                scriptPath = scriptURL.path
            } else {
                scriptPath = "/Users/oleksiichumak/Developer/Xcode.projects/Aura/Aura/analyze.py"
            }
            self.logInfo("Using analyze.py script at: \(scriptPath)")
            
            guard FileManager.default.fileExists(atPath: scriptPath) else {
                let errMsg = "analyze.py script not found."
                self.logError(errMsg)
                completion(.failure(NSError(domain: "PythonRunner", code: 404, userInfo: [NSLocalizedDescriptionKey: errMsg])))
                return
            }
            
            self.executePythonPreview(pythonPath: pythonExecutable, scriptPath: scriptPath, csvPath: csvPathOrURL, completion: completion)
        }
    }
    
    private func executePythonPreview(pythonPath: String, scriptPath: String, csvPath: String, completion: @escaping @Sendable (Result<DatasetPreview, Error>) -> Void) {
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
        
        errPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            accumulatedErrData.append(data)
        }
        
        var wasCancelledBeforeRun = false
        processLock.lock()
        if self.isCancelled {
            wasCancelledBeforeRun = true
        } else {
            self.activeProcess = process
        }
        processLock.unlock()
        
        if wasCancelledBeforeRun {
            completion(.failure(NSError(domain: "PythonRunner", code: -999, userInfo: [NSLocalizedDescriptionKey: "Preview was cancelled by user."])))
            return
        }
        
        defer {
            processLock.lock()
            if self.activeProcess === process {
                self.activeProcess = nil
            }
            processLock.unlock()
        }
        
        do {
            try process.run()
            process.waitUntilExit()
            
            outPipe.fileHandleForReading.readabilityHandler = nil
            errPipe.fileHandleForReading.readabilityHandler = nil
            
            var wasCancelled = false
            processLock.lock()
            wasCancelled = self.isCancelled
            processLock.unlock()
            
            if wasCancelled {
                logInfo("Preview subprocess was terminated due to cancellation.")
                completion(.failure(NSError(domain: "PythonRunner", code: -999, userInfo: [NSLocalizedDescriptionKey: "Preview was cancelled by user."])))
                return
            }
            
            let outData = accumulatedOutData.get()
            let errData = accumulatedErrData.get()
            
            let errString = String(data: errData, encoding: .utf8) ?? ""
            if !errString.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                logWarning("Preview Subprocess stderr: \(errString.trimmingCharacters(in: .whitespacesAndNewlines))")
            }
            
            logInfo("Python preview subprocess finished with termination status: \(process.terminationStatus)")
            
            if process.terminationStatus != 0 {
                logError("Preview subprocess failed: \(errString)")
                completion(.failure(NSError(domain: "PythonRunner", code: Int(process.terminationStatus), userInfo: [NSLocalizedDescriptionKey: errString])))
                return
            }
            
            if let errorDict = try? JSONSerialization.jsonObject(with: outData) as? [String: Any],
               let errorMsg = errorDict["error"] as? String, !errorMsg.isEmpty {
                logError("Preview subprocess returned logic error: \(errorMsg)")
                completion(.failure(NSError(domain: "PythonRunner", code: 500, userInfo: [NSLocalizedDescriptionKey: errorMsg])))
                return
            }
            
            do {
                let decoder = JSONDecoder()
                let result = try decoder.decode(DatasetPreview.self, from: outData)
                if let errorMsg = result.error {
                    logError("Decoded DatasetPreview indicates error: \(errorMsg)")
                    completion(.failure(NSError(domain: "PythonRunner", code: 500, userInfo: [NSLocalizedDescriptionKey: errorMsg])))
                } else {
                    logInfo("Successfully completed preview of \(result.columns.count) columns.")
                    completion(.success(result))
                }
            } catch {
                let rawOutput = String(data: outData, encoding: .utf8) ?? "Unreadable output"
                let detail = "JSON Decoding Failed (Preview): \(error.localizedDescription)\n\nPython Output:\n\(rawOutput)\n\nErrors:\n\(errString)"
                logError(detail)
                completion(.failure(NSError(domain: "PythonRunner", code: 499, userInfo: [NSLocalizedDescriptionKey: detail])))
            }
        } catch {
            logError("Failed to run preview subprocess: \(error.localizedDescription)")
            completion(.failure(error))
        }
    }
}
