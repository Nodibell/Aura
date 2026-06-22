import Foundation
import Observation
import AppKit

@MainActor
@Observable
class OllamaStatusChecker {
    static let shared = OllamaStatusChecker()

    var isAvailable: Bool = false
    var availableModels: [OllamaModelInfo] = []
    var isChecking: Bool = false

    private var pollTask: Task<Void, Never>?
    private var observers: [NSObjectProtocol] = []

    init() {
        // Pause polling when app resigns active (minimized / Cmd-Tabbed away)
        let resign = NotificationCenter.default.addObserver(
            forName: NSApplication.didResignActiveNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor in self?.stopPolling() }
        }

        // Resume polling immediately when app becomes active again
        let become = NotificationCenter.default.addObserver(
            forName: NSApplication.didBecomeActiveNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor in self?.startPolling() }
        }

        observers = [resign, become]
        startPolling()
    }

    func startPolling() {
        pollTask?.cancel()
        pollTask = Task {
            while !Task.isCancelled {
                await refresh()
                try? await Task.sleep(nanoseconds: 30_000_000_000) // 30 seconds
            }
        }
    }

    func stopPolling() {
        pollTask?.cancel()
        pollTask = nil
    }

    func refresh() async {
        isChecking = true
        let available = await OllamaService.shared.checkAvailability()
        var models: [OllamaModelInfo] = []
        if available {
            models = await OllamaService.shared.listModels()
        }
        self.isAvailable = available
        self.availableModels = models
        self.isChecking = false
    }

    @MainActor
    deinit {
        pollTask?.cancel()
        observers.forEach { NotificationCenter.default.removeObserver($0) }
    }
}
