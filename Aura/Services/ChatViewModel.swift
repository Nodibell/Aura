import Foundation
import Observation

@Observable
class ChatViewModel {
    var messages: [ChatMessage] = []
    var isStreaming: Bool = false
    var inputText: String = ""

    private var streamTask: Task<Void, Never>?
    private var systemPrompt: String = ""
    private var replLoopDepth: Int = 0
    private static let maxREPLDepth = 5


    // MARK: - Context Injection

    func injectContext(_ result: AnalysisResult, datasetURL: String? = nil, cleaningActions: String? = nil) {
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
        You are an agentic AI Data Analyst running inside Aura, a macOS machine learning app.
        A pandas DataFrame `df` is already loaded in the Python environment.

        To inspect or compute anything about the data, output Python code wrapped EXACTLY like this:
        <execute_python>
        print(df['column'].describe())
        </execute_python>
        The system will execute the code and return <execution_result>…</execution_result>.
        You MUST wait for the result before giving your final answer.

        For processing large text columns chunk-by-chunk, use the built-in recursive helper:
          result = llm_query(chunk=df['review'][0], question="Summarise sentiment")
        This performs a lightweight sub-LLM call on just that chunk of text.

        Rules:
        - Never hallucinate execution results — always wait for <execution_result>.
        - Keep code blocks focused (one task per block).
        - Maximum \(ChatViewModel.maxREPLDepth) code execution rounds per question.
        - Answer in markdown. Use bullet points and headers.
        - Be direct and data-specific.

        Dataset context (auto-generated):
        - Size: \(result.rowCount) rows × \(result.colCount) columns
        - Task: \(result.taskType.uppercased())
        - Target column: "\(result.targetColumn)"
        - Numeric columns: \(result.numericColCount), Categorical: \(result.categoricalColCount)\(validationDetails)
        - Models trained: \(modelLeaderboard)
        - Best model: \(result.metrics.model) (\(result.metrics.scoreType): \(String(format: "%.4f", result.metrics.score)))
        - Top feature importances: \(featureImportances.isEmpty ? "none" : featureImportances)
        - Top correlations: \(topCorr)
        - Missing values: \(topMissing.isEmpty ? "none" : topMissing)
        """

        // Append rich dataset context snapshot (column stats + sample rows)
        if let ctx = result.datasetContext, !ctx.isEmpty {
            systemPrompt += "\n\n" + ctx
        }

        // Clear and re-inject
        messages = []

        // Reset the REPL sandbox so df reflects the new dataset (Phase 16)
        var pathForREPL = result.filePath
        if let path = pathForREPL, !FileManager.default.fileExists(atPath: path), let url = datasetURL, !url.isEmpty {
            pathForREPL = url
        }
        
        if let filePath = pathForREPL {
            Task {
                try? await REPLService.shared.reset(filePath: filePath, cleaningActions: cleaningActions)
            }
        }
    }


    func clearConversation() {
        cancelGeneration()
        messages = []
    }

    // MARK: - Message Sending

    @MainActor func sendMessage(
        _ text: String,
        model: String,
        temperature: Double,
        maxTokens: Int,
        isREPLInjection: Bool = false
    ) {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }

        // Only add a visible user message for real user inputs (not REPL result injections)
        if !isREPLInjection {
            let userMessage = ChatMessage(role: .user, content: text)
            messages.append(userMessage)
        }

        let assistantMessage = ChatMessage(role: .assistant, content: "", state: .streaming)
        messages.append(assistantMessage)

        let assistantId = assistantMessage.id

        isStreaming = true
        inputText = ""
        if !isREPLInjection {
            replLoopDepth = 0
        }

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

                // ── Phase 16: RLM agentic REPL loop ──
                let finalContent = await MainActor.run {
                    self.messages.first(where: { $0.id == assistantId })?.content ?? ""
                }
                await self.runREPLLoopIfNeeded(
                    assistantContent: finalContent,
                    model: model,
                    temperature: temperature,
                    maxTokens: maxTokens
                )

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
        if let idx = messages.indices.last, messages[idx].state == .streaming || messages[idx].state == .executingCode {
            messages[idx].state = .complete
        }
        isStreaming = false
    }

    // MARK: - RLM REPL Loop

    @MainActor
    private func runREPLLoopIfNeeded(
        assistantContent: String,
        model: String,
        temperature: Double,
        maxTokens: Int
    ) async {
        guard replLoopDepth < ChatViewModel.maxREPLDepth else { return }

        // Extract first <execute_python>…</execute_python> block
        guard let code = extractFirstCodeBlock(from: assistantContent) else { return }

        replLoopDepth += 1
        isStreaming = true

        // Show a "⚙ Executing code…" tool message
        let toolMsg = ChatMessage(role: .tool, content: "⚙ Executing Python code…", state: .executingCode)
        messages.append(toolMsg)
        let toolMsgId = toolMsg.id

        // Execute in background
        do {
            let result = try await REPLService.shared.execute(code)

            // Build the execution result string
            var resultText = ""
            if let err = result.error, !err.isEmpty {
                resultText = "<execution_result>\nError:\n\(err)\n</execution_result>"
            } else {
                let out = result.stdout.isEmpty ? "(no output)" : result.stdout
                resultText = "<execution_result>\n\(out)\n</execution_result>"
            }

            // Attach figure count note if any
            if !result.figures.isEmpty {
                resultText += "\n\n_(\(result.figures.count) figure(s) captured — displayed above)_"
            }

            // Update tool message with result + store figures
            if let idx = messages.firstIndex(where: { $0.id == toolMsgId }) {
                messages[idx].content = resultText
                messages[idx].state = .complete
                messages[idx].figures = result.figures
            }

            // Inject result as "user" turn and re-trigger AI
            sendMessage(
                resultText,
                model: model,
                temperature: temperature,
                maxTokens: maxTokens,
                isREPLInjection: true
            )

        } catch {
            if let idx = messages.firstIndex(where: { $0.id == toolMsgId }) {
                messages[idx].content = "⚙ Code execution failed: \(error.localizedDescription)"
                messages[idx].state = .error
            }
            isStreaming = false
        }
    }

    private func extractFirstCodeBlock(from text: String) -> String? {
        guard let start = text.range(of: "<execute_python>"),
              let end   = text.range(of: "</execute_python>") else { return nil }
        let code = String(text[start.upperBound..<end.lowerBound])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return code.isEmpty ? nil : code
    }


    // MARK: - Helpers

    private func buildHistoryMessages() -> [OllamaChatMessage] {
        var result: [OllamaChatMessage] = []
        for msg in messages.dropLast() { // drop last (empty assistant placeholder)
            switch msg.role {
            case .user, .tool:
                // Tool messages (REPL results) are injected as user turns so the
                // model sees the execution output in context.
                if !msg.content.isEmpty {
                    result.append(OllamaChatMessage(role: "user", content: msg.content))
                }
            case .assistant:
                if !msg.content.isEmpty {
                    result.append(OllamaChatMessage(role: "assistant", content: msg.content))
                }
            }
        }
        return result
    }
}