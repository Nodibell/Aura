import Foundation
import Observation

@MainActor
@Observable
class OllamaStatusChecker {
    static let shared = OllamaStatusChecker()

    var isAvailable: Bool = false
    var availableModels: [OllamaModelInfo] = []
    var isChecking: Bool = false

    private var pollTask: Task<Void, Never>?

    init() {
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
    }
}
