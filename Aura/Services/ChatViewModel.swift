import Foundation
import Observation

@Observable
class ChatViewModel {
    var messages: [ChatMessage] = []
    var isStreaming: Bool = false
    var inputText: String = ""

    private var streamTask: Task<Void, Never>?
    private var systemPrompt: String = ""

    // MARK: - Context Injection

    func injectContext(_ result: AnalysisResult) {
        // Cap correlation pairs to avoid flooding a local LLM's context window
        let maxCorr = 10
        let corrList = result.correlations.prefix(maxCorr).map {
            "\($0.x) ↔ \($0.y) (\(String(format: "%.3f", $0.value)))"
        }.joined(separator: ", ")
        let topCorr = corrList.isEmpty ? "none" : corrList
            + (result.correlations.count > maxCorr ? " … (\(result.correlations.count - maxCorr) more)" : "")

        // Cap missing values at 10 columns
        let maxMissing = 10
        let missingFiltered = result.missingValues.filter { $0.value > 0 }
            .sorted { $0.value > $1.value }
        let topMissing = missingFiltered.prefix(maxMissing)
            .map { "\($0.key): \($0.value)" }
            .joined(separator: ", ")
            + (missingFiltered.count > maxMissing ? " … and \(missingFiltered.count - maxMissing) more" : "")

        let modelLeaderboard = result.modelsCompared
            .map { "\($0.name) → \($0.metric) \(String(format: "%.4f", $0.score))" }
            .joined(separator: " | ")

        // Cap feature importances at 15 to avoid token overflow
        let maxFeatures = 15
        var featureData = result.charts.first(where: { $0.title.contains("SHAP") })?.data
        if featureData == nil {
            featureData = result.charts.first(where: { $0.title.lowercased().contains("importance") })?.data
        }
        let featurePoints = featureData ?? []
        let featureImportances = featurePoints.prefix(maxFeatures).compactMap { point -> String? in
            guard let name = point.xVal else { return nil }
            return "\(name): \(String(format: "%.4f", point.y))"
        }.joined(separator: ", ")
        + (featurePoints.count > maxFeatures ? " … (\(featurePoints.count - maxFeatures) more)" : "")

        var validationDetails = ""
        if let cvMean = result.cvMean {
            validationDetails += "\n- 5-Fold Cross-Validation Score: \(String(format: "%.4f", cvMean))"
            if let cvStd = result.cvStd {
                validationDetails += " (±\(String(format: "%.4f", cvStd)))"
            }
        }
        if let dummy = result.dummyBaselineScore {
            validationDetails += "\n- Dummy Baseline Score: \(String(format: "%.4f", dummy))"
        }
        if let duplicateRows = result.profiling?.duplicateRows, duplicateRows > 0 {
            validationDetails += "\n- Duplicate Rows Count: \(duplicateRows)"
        }

        systemPrompt = """
        You are an expert data scientist providing concise, insightful analysis. The user loaded this dataset:
        
        - Size: \(result.rowCount) rows × \(result.colCount) columns
        - Task: \(result.taskType.uppercased())
        - Target column: "\(result.targetColumn)"
        - Numeric columns: \(result.numericColCount), Categorical: \(result.categoricalColCount), Text (TF-IDF): \(result.textColCount)\(validationDetails)
        - Models trained: \(modelLeaderboard)
        - Best model: \(result.metrics.model) (\(result.metrics.scoreType): \(String(format: "%.4f", result.metrics.score)))
        - Top feature importances: \(featureImportances.isEmpty ? "none" : featureImportances)
        - Top correlations (showing \(min(result.correlations.count, maxCorr)) of \(result.correlations.count)): \(topCorr)
        - Missing values: \(topMissing.isEmpty ? "none" : topMissing)
        
        Answer questions about this specific dataset concisely using markdown formatting. Use bullet points and headers where helpful. Be direct and data-specific.
        You are Aura, an advanced AI Data Analyst. 
        A pandas DataFrame named `df` is already loaded in memory.
        If you need to analyze the data, you MUST write Python code to inspect it. 
        To execute code, output EXACTLY like this:
        <execute_python>
        print(df.describe())
        </execute_python>

        Wait for the system to return the <execution_result> before answering the user.
        """

        // Clear and re-inject
        messages = []
    }

    func clearConversation() {
        cancelGeneration()
        messages = []
    }

    // MARK: - Message Sending

    @MainActor func sendMessage(_ text: String, model: String, temperature: Double, maxTokens: Int) {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }

        let userMessage = ChatMessage(role: .user, content: text)
        messages.append(userMessage)

        let assistantMessage = ChatMessage(role: .assistant, content: "", state: .streaming)
        messages.append(assistantMessage)
        let assistantId = assistantMessage.id

        isStreaming = true
        inputText = ""

        let historyMessages = buildHistoryMessages()
        let providerStr = UserDefaults.standard.string(forKey: "Aura_LLMProvider") ?? "Ollama"
        let provider = LLMProvider(rawValue: providerStr) ?? .ollama
        let service = AIService.shared

        streamTask = Task {
            do {
                let stream = service.streamChat(
                    messages: historyMessages,
                    systemPrompt: systemPrompt,
                    provider: provider,
                    model: model,
                    temperature: temperature,
                    maxTokens: maxTokens
                )

                for try await token in stream {
                    if Task.isCancelled { break }
                    await MainActor.run {
                        if let idx = self.messages.firstIndex(where: { $0.id == assistantId }) {
                            self.messages[idx].content += token
                        }
                    }
                }

                await MainActor.run {
                    if let idx = self.messages.firstIndex(where: { $0.id == assistantId }) {
                        self.messages[idx].state = .complete
                    }
                    self.isStreaming = false
                }
            } catch {
                await MainActor.run {
                    if let idx = self.messages.firstIndex(where: { $0.id == assistantId }) {
                        let rawMsg = error.localizedDescription
                        let errMsg: String
                        if rawMsg.contains("refused") || rawMsg.contains("connect") {
                            errMsg = "Ollama is not reachable. Make sure it is running (`ollama serve`) or check your custom endpoint."
                        } else {
                            errMsg = rawMsg
                        }
                        self.messages[idx].content = errMsg
                        self.messages[idx].state = .error
                    }
                    self.isStreaming = false
                }
            }
        }
    }

    func cancelGeneration() {
        streamTask?.cancel()
        streamTask = nil
        if let idx = messages.indices.last, messages[idx].state == .streaming {
            messages[idx].state = .complete
        }
        isStreaming = false
    }

    // MARK: - Helpers

    private func buildHistoryMessages() -> [OllamaChatMessage] {
        var result: [OllamaChatMessage] = []
        for msg in messages.dropLast() { // drop last (empty assistant placeholder)
            let role = msg.role == .user ? "user" : "assistant"
            if !msg.content.isEmpty {
                result.append(OllamaChatMessage(role: role, content: msg.content))
            }
        }
        return result
    }
}
