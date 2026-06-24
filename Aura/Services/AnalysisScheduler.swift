import Foundation
import UserNotifications

enum ExportFormat: String, Codable, CaseIterable, Identifiable {
    case pdf = "PDF"
    case html = "HTML"
    case markdown = "Markdown"
    
    var id: String { rawValue }
}

struct ScheduledTask: Identifiable, Codable, Equatable {
    let id: UUID
    var name: String
    var datasetPath: String
    var targetColumn: String?
    var taskType: DatasetType
    var recurrence: Recurrence
    var exportFormat: ExportFormat
    var exportFolderPath: String
    var isActive: Bool
    var lastRun: Date?
    var nextRun: Date
    var config: AnalysisConfig
}

enum Recurrence: Codable, Equatable, Hashable {
    case hourly(Int)
    case daily
    case weekly
    
    var label: String {
        switch self {
        case .hourly(let hrs): return hrs == 1 ? "Every hour" : "Every \(hrs) hours"
        case .daily: return "Daily"
        case .weekly: return "Weekly"
        }
    }
}

final class AnalysisScheduler: @unchecked Sendable {
    static let shared = AnalysisScheduler()
    
    private let lock = NSLock()
    private var tasks: [ScheduledTask] = []
    private var timer: Timer?
    private let fileManager = FileManager.default
    
    // MARK: - Logging Helpers (Safely bounces to MainActor)
    private func logInfo(_ message: String, category: String = "AnalysisScheduler") {
        Task { @MainActor in
            AppLogger.shared.info(message, category: category)
        }
    }
    
    private func logError(_ message: String, category: String = "AnalysisScheduler") {
        Task { @MainActor in
            AppLogger.shared.error(message, category: category)
        }
    }
    
    private func logWarning(_ message: String, category: String = "AnalysisScheduler") {
        Task { @MainActor in
            AppLogger.shared.warning(message, category: category)
        }
    }
    
    private init() {
        loadTasks()
        startTimer()
        requestNotificationPermission()
    }
    
    private var schedulesFileURL: URL {
        let appSupport = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let auraDir = appSupport.appendingPathComponent("Aura", isDirectory: true)
        try? fileManager.createDirectory(at: auraDir, withIntermediateDirectories: true)
        return auraDir.appendingPathComponent("schedules.json")
    }
    
    func getTasks() -> [ScheduledTask] {
        lock.lock()
        defer { lock.unlock() }
        return tasks
    }
    
    private func loadTasks() {
        lock.lock()
        defer { lock.unlock() }
        
        let path = schedulesFileURL.path
        guard fileManager.fileExists(atPath: path) else { return }
        
        do {
            let data = try Data(contentsOf: schedulesFileURL)
            let decoder = JSONDecoder()
            tasks = try decoder.decode([ScheduledTask].self, from: data)
        } catch {
            logError("Failed to load scheduled tasks: \(error.localizedDescription)")
        }
    }
    
    private func saveTasks() {
        do {
            let data = try JSONEncoder().encode(tasks)
            try data.write(to: schedulesFileURL, options: .atomic)
        } catch {
            logError("Failed to save scheduled tasks: \(error.localizedDescription)")
        }
    }
    
    func addTask(_ task: ScheduledTask) {
        lock.lock()
        tasks.append(task)
        saveTasks()
        lock.unlock()
        logInfo("Added scheduled task: \(task.name)")
    }
    
    func updateTask(_ task: ScheduledTask) {
        lock.lock()
        if let idx = tasks.firstIndex(where: { $0.id == task.id }) {
            tasks[idx] = task
            saveTasks()
        }
        lock.unlock()
    }
    
    func removeTask(withId id: UUID) {
        lock.lock()
        tasks.removeAll(where: { $0.id == id })
        saveTasks()
        lock.unlock()
        logInfo("Removed scheduled task ID: \(id)")
    }
    
    func toggleTaskActive(withId id: UUID) {
        lock.lock()
        if let idx = tasks.firstIndex(where: { $0.id == id }) {
            tasks[idx].isActive.toggle()
            if tasks[idx].isActive {
                // Reset nextRun to now so it catches up
                tasks[idx].nextRun = Date()
            }
            saveTasks()
        }
        lock.unlock()
    }
    
    func startTimer() {
        lock.lock()
        defer { lock.unlock() }
        
        timer?.invalidate()
        timer = Timer.scheduledTimer(withTimeInterval: 60, repeats: true) { [weak self] _ in
            self?.checkSchedules()
        }
    }
    
    func stopTimer() {
        lock.lock()
        defer { lock.unlock() }
        timer?.invalidate()
        timer = nil
    }
    
    func checkSchedules() {
        let now = Date()
        var tasksToRun: [ScheduledTask] = []
        
        lock.lock()
        for idx in 0..<tasks.count {
            let task = tasks[idx]
            if task.isActive && now >= task.nextRun {
                tasks[idx].lastRun = now
                tasks[idx].nextRun = calculateNextRun(from: now, recurrence: task.recurrence)
                tasksToRun.append(tasks[idx])
            }
        }
        if !tasksToRun.isEmpty {
            saveTasks()
        }
        lock.unlock()
        
        for task in tasksToRun {
            Task {
                await executeTask(task)
            }
        }
    }
    
    func triggerImmediately(_ task: ScheduledTask) {
        Task {
            await executeTask(task)
        }
    }
    
    private func calculateNextRun(from date: Date, recurrence: Recurrence) -> Date {
        let calendar = Calendar.current
        switch recurrence {
        case .hourly(let hrs):
            return calendar.date(byAdding: .hour, value: hrs, to: date) ?? date.addingTimeInterval(Double(hrs) * 3600)
        case .daily:
            return calendar.date(byAdding: .day, value: 1, to: date) ?? date.addingTimeInterval(86400)
        case .weekly:
            return calendar.date(byAdding: .day, value: 7, to: date) ?? date.addingTimeInterval(7 * 86400)
        }
    }
    
    private func executeTask(_ task: ScheduledTask) async {
        logInfo("Executing scheduled task: \(task.name)")
        
        await PythonRunner.shared.runAnalysis(
            csvPath: task.datasetPath,
            targetColumn: task.targetColumn,
            config: task.config,
            progress: { _, _ in }
        ) { result in
            Task {
                switch result {
                case .success(let analysisResult):
                    await self.handleSuccess(task: task, result: analysisResult)
                case .failure(let error):
                    self.handleFailure(task: task, error: error)
                }
            }
        }
    }
    
    private func handleSuccess(task: ScheduledTask, result: AnalysisResult) async {
        logInfo("Scheduled task \(task.name) completed analysis successfully.")
        
        // Save history item so it displays in main dashboard
        await MainActor.run {
            _ = AnalysisHistoryService.shared.saveAnalysis(
                result: result,
                datasetPath: task.datasetPath,
                targetColumn: task.targetColumn
            )
        }
        
        // Generate AI Narrative
        let model = UserDefaults.standard.string(forKey: "Aura_OllamaModel") ?? "llama3.2"
        let prompt = buildPromptForTask(task, result: result)
        let narrative = await fetchOllamaNarrative(prompt: prompt, model: model)
        
        // Export file
        let timestamp = ISO8601DateFormatter().string(from: Date())
            .replacingOccurrences(of: ":", with: "-")
            .replacingOccurrences(of: "T", with: "_")
            .prefix(16)
        
        let safeName = task.name.replacingOccurrences(of: " ", with: "_")
        let filename = "\(safeName)_Report_\(timestamp)"
        
        let folderURL = URL(fileURLWithPath: task.exportFolderPath)
        try? fileManager.createDirectory(at: folderURL, withIntermediateDirectories: true)
        
        // Capture narrative as immutable constant so Swift 6 concurrency is satisfied
        let capturedNarrative = narrative
        
        do {
            switch task.exportFormat {
            case .markdown:
                let md = await MainActor.run {
                    ReportCompiler.buildMarkdownReport(result: result, narrative: capturedNarrative)
                }
                let destURL = folderURL.appendingPathComponent("\(filename).md")
                try md.write(to: destURL, atomically: true, encoding: .utf8)
                sendNotification(title: "Schedule Succeeded", body: "Exported report to \(destURL.lastPathComponent)")
                
            case .html:
                let html = await MainActor.run {
                    ReportCompiler.buildHTMLReport(result: result, narrative: capturedNarrative)
                }
                let destURL = folderURL.appendingPathComponent("\(filename).html")
                try html.write(to: destURL, atomically: true, encoding: .utf8)
                sendNotification(title: "Schedule Succeeded", body: "Exported report to \(destURL.lastPathComponent)")
                
            case .pdf:
                let html = await MainActor.run {
                    ReportCompiler.buildHTMLReport(result: result, narrative: capturedNarrative)
                }
                let pdfData = try await HTMLToPDFConverter.shared.convert(html: html)
                let destURL = folderURL.appendingPathComponent("\(filename).pdf")
                try pdfData.write(to: destURL)
                sendNotification(title: "Schedule Succeeded", body: "Exported report to \(destURL.lastPathComponent)")
            }
        } catch {
            logError("Failed to write scheduled export: \(error.localizedDescription)")
            sendNotification(title: "Schedule Succeeded with Export Warning", body: "Analysis complete, but export failed: \(error.localizedDescription)")
        }
    }
    
    private func handleFailure(task: ScheduledTask, error: Error) {
        logError("Scheduled task \(task.name) failed: \(error.localizedDescription)")
        sendNotification(title: "Schedule Failed", body: "Task: \(task.name). Error: \(error.localizedDescription)")
    }
    
    private func requestNotificationPermission() {
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound]) { granted, error in
            if let error = error {
                self.logWarning("Notification permission not granted: \(error.localizedDescription)")
            } else {
                self.logInfo("Notification permission status: \(granted)")
            }
        }
    }
    
    private func sendNotification(title: String, body: String) {
        let content = UNMutableNotificationContent()
        content.title = title
        content.body = body
        content.sound = .default
        
        let request = UNNotificationRequest(identifier: UUID().uuidString, content: content, trigger: nil)
        UNUserNotificationCenter.current().add(request)
    }
    
    private func buildPromptForTask(_ task: ScheduledTask, result: AnalysisResult) -> String {
        var extraContext = ""
        if let warnings = result.dataLeakageWarnings, !warnings.isEmpty {
            extraContext += "- Data Leakage Warnings: \(warnings.joined(separator: "; "))\n"
        }
        if let recs = result.cleaningRecommendations, !recs.isEmpty {
            let recsStr = recs.map { "\($0.column) (\($0.issue) -> \($0.recommendation) [Impact: \($0.impact)])" }.joined(separator: "; ")
            extraContext += "- Data Quality Issues & Recommendations: \(recsStr)\n"
        }
        
        return """
        You are a Senior Data Scientist. Write a professional, highly polished data analysis review in Markdown based on the following metrics:
        - Dataset size: \(result.rowCount) rows × \(result.colCount) columns
        - Target variable: '\(result.targetColumn)' (\(result.taskType.capitalized) task)
        - Best Model: \(result.metrics.model) (\(result.metrics.scoreType): \(String(format: "%.4f", result.metrics.score)))
        \(extraContext)
        - Top correlations: \(result.correlations.prefix(5).map { "\($0.x) ↔ \($0.y) = \(String(format: "%.3f", $0.value))" }.joined(separator: ", "))
        - Missing values: \(result.missingValues.filter { $0.value > 0 }.count) columns with missing values.

        Ensure the response strictly follows this Markdown structure:
        
        ## 🧠 AI Analysis Summary
        Write 2-3 detailed paragraphs. Discuss the dataset size and task. Assess the model performance (R² / Accuracy / F1) critically. Discuss whether any data quality or data leakage warnings were flagged.
        
        ## 💡 Key Findings
        Provide 3-5 specific, bulleted insights. Each bullet MUST reference exact numbers, features, correlations, or performance metrics from the data provided above.
        
        ## 📋 Recommendations
        Provide 2-4 concrete, actionable next steps. Address data quality issues, recommended column actions, feature engineering suggestions, or model improvements.
        """
    }
    
    private func fetchOllamaNarrative(prompt: String, model: String) async -> String? {
        guard let url = URL(string: "http://localhost:11434/api/generate") else { return nil }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 120

        let body: [String: Any] = [
            "model": model,
            "prompt": prompt,
            "stream": false,
            "options": ["temperature": 0.4, "num_predict": 1200]
        ]
        request.httpBody = try? JSONSerialization.data(withJSONObject: body)

        return await withCheckedContinuation { continuation in
            URLSession.shared.dataTask(with: request) { data, _, _ in
                if let data = data,
                   let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let text = json["response"] as? String {
                    continuation.resume(returning: text)
                } else {
                    continuation.resume(returning: nil)
                }
            }.resume()
        }
    }
}
