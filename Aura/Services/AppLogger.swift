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
    private var persistentHandle: FileHandle?

    init() {
        let fileManager = FileManager.default
        if let appSupport = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first {
            let auraDir = appSupport.appendingPathComponent("Aura", isDirectory: true)
            try? fileManager.createDirectory(at: auraDir, withIntermediateDirectories: true, attributes: nil)
            let url = auraDir.appendingPathComponent("app_debug.log")
            self.fileURL = url

            // Create file if it doesn't exist
            if !fileManager.fileExists(atPath: url.path) {
                fileManager.createFile(atPath: url.path, contents: nil)
            }
            // Open once and keep open for the lifetime of the logger
            self.persistentHandle = try? FileHandle(forWritingTo: url)
            self.persistentHandle?.seekToEndOfFile()
        } else {
            self.fileURL = nil
        }
    }

    deinit {
        queue.sync { persistentHandle?.closeFile() }
    }

    func write(_ line: String) {
        guard let handle = persistentHandle else { return }
        queue.async {
            guard let data = (line + "\n").data(using: .utf8) else { return }
            handle.write(data)
        }
    }

    func clear() {
        guard let handle = persistentHandle else { return }
        queue.async {
            handle.truncateFile(atOffset: 0)
            handle.seek(toFileOffset: 0)
        }
    }
}

