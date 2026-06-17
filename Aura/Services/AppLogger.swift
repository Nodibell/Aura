import Foundation
import Observation

@MainActor @Observable
class AppLogger {
    static let shared = AppLogger()
    
    private(set) var logs: [LogEntry] = []
    
    struct LogEntry: Identifiable, Codable {
        let id: UUID
        let timestamp: Date
        let category: String
        let level: LogLevel
        let message: String
        
        var formattedString: String {
            let formatter = DateFormatter()
            formatter.dateFormat = "yyyy-MM-dd HH:mm:ss.SSS"
            return "[\(formatter.string(from: timestamp))] [\(level.rawValue.uppercased())] [\(category)] \(message)"
        }
    }
    
    enum LogLevel: String, Codable {
        case debug
        case info
        case warning
        case error
    }
    
    private let queue = DispatchQueue(label: "com.aura.logger", attributes: .concurrent)
    private let maxLogEntries = 2000
    private let logFileWriter: LogFileWriter?
    
    private init() {
        self.logFileWriter = LogFileWriter()
        log(category: "AppLogger", level: .info, message: "Logger initialized.")
    }
    
    func log(category: String, level: LogLevel = .info, message: String) {
        let entry = LogEntry(id: UUID(), timestamp: Date(), category: category, level: level, message: message)
        
        // Print to standard console
        print(entry.formattedString)
        
        // Write to file
        logFileWriter?.write(entry.formattedString)
        
        // Update logs array on main actor to avoid data races
        Task { @MainActor in
            self.logs.append(entry)
            if self.logs.count > self.maxLogEntries {
                self.logs.removeFirst()
            }
        }
    }
    
    func debug(_ message: String, category: String = "App") {
        log(category: category, level: .debug, message: message)
    }
    
    func info(_ message: String, category: String = "App") {
        log(category: category, level: .info, message: message)
    }
    
    func warning(_ message: String, category: String = "App") {
        log(category: category, level: .warning, message: message)
    }
    
    func error(_ message: String, category: String = "App") {
        log(category: category, level: .error, message: message)
    }
    
    func clearLogs() {
        Task { @MainActor in
            self.logs.removeAll()
        }
        logFileWriter?.clear()
    }
    
    func getRawLogs() -> String {
        var result = ""
        queue.sync {
            result = logs.map { $0.formattedString }.joined(separator: "\n")
        }
        return result
    }
    
    func getLogFileURL() -> URL? {
        return logFileWriter?.fileURL
    }
}

private class LogFileWriter {
    let fileURL: URL?
    private let queue = DispatchQueue(label: "com.aura.logger.file", qos: .utility)
    
    init() {
        let fileManager = FileManager.default
        if let appSupport = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first {
            let auraDir = appSupport.appendingPathComponent("Aura", isDirectory: true)
            try? fileManager.createDirectory(at: auraDir, withIntermediateDirectories: true, attributes: nil)
            self.fileURL = auraDir.appendingPathComponent("app_debug.log")
        } else {
            self.fileURL = nil
        }
    }
    
    func write(_ line: String) {
        guard let url = fileURL else { return }
        queue.async {
            let logLine = line + "\n"
            if let data = logLine.data(using: .utf8) {
                if FileManager.default.fileExists(atPath: url.path) {
                    if let fileHandle = try? FileHandle(forWritingTo: url) {
                        fileHandle.seekToEndOfFile()
                        fileHandle.write(data)
                        fileHandle.closeFile()
                    }
                } else {
                    try? data.write(to: url)
                }
            }
        }
    }
    
    func clear() {
        guard let url = fileURL else { return }
        queue.async {
            try? "".write(to: url, atomically: true, encoding: .utf8)
        }
    }
}

