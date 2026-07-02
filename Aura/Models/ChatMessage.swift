import Foundation

// MARK: - Chat Role & State

enum ChatRole {
    case user
    case assistant
    case tool          // Phase 16: REPL execution result
}

enum MessageState {
    case complete
    case streaming
    case error
    case executingCode // Phase 16: waiting for REPL result
}

// MARK: - Chat Message

struct ChatMessage: Identifiable {
    let id: UUID
    let role: ChatRole
    var content: String
    var state: MessageState
    var figures: [String] = []   // base64 PNG strings from REPL execution (Phase 16)

    var formattedContent: String {
        content
            .replacingOccurrences(of: "<execute_python>", with: "```python\n")
            .replacingOccurrences(of: "</execute_python>", with: "\n```")
            .replacingOccurrences(of: "<execution_result>", with: "```\n")
            .replacingOccurrences(of: "</execution_result>", with: "\n```")
    }

    init(id: UUID = UUID(), role: ChatRole, content: String, state: MessageState = .complete) {
        self.id = id
        self.role = role
        self.content = content
        self.state = state
    }
}

// MARK: - Quick Action Chips

struct QuickAction: Identifiable {
    let id = UUID()
    let emoji: String
    let label: String
    let prompt: String
}

extension QuickAction {
    static func actionsFor(result: AnalysisResult) -> [QuickAction] {
        var actions: [QuickAction] = [
            QuickAction(
                emoji: "📋",
                label: "Summarize findings",
                prompt: "Summarize the key findings from this EDA in 3-5 bullet points. Include dataset size, task type, best model performance, and the most important features or correlations."
            ),
            QuickAction(
                emoji: "🎯",
                label: "Model performance",
                prompt: "Explain the model performance results in plain English. Compare \(result.modelsCompared.map { "\($0.name): \(String(format: "%.3f", $0.score))" }.joined(separator: " vs ")). Is \(String(format: "%.3f", result.metrics.score)) a good score for a \(result.taskType) task on this dataset?"
            ),
            QuickAction(
                emoji: "💡",
                label: "Next steps",
                prompt: "Based on these EDA results, what are the top 5 concrete next steps a data scientist should take? Consider feature engineering, data quality issues, model improvements, and business insights."
            )
        ]
        if result.missingValues.values.reduce(0, +) > 0 {
            actions.insert(QuickAction(
                emoji: "⚠️",
                label: "Missing data advice",
                prompt: "The dataset has missing values in these columns: \(result.missingValues.filter { $0.value > 0 }.sorted { $0.value > $1.value }.map { "\($0.key) (\($0.value) missing)" }.joined(separator: ", ")). What are the best strategies to handle this missing data?"
            ), at: 1)
        }
        if !result.correlations.isEmpty {
            actions.append(QuickAction(
                emoji: "🔗",
                label: "Top correlations",
                prompt: "Explain the significance of the top correlations found: \(result.correlations.prefix(5).map { "\($0.x) ↔ \($0.y): \(String(format: "%.3f", $0.value))" }.joined(separator: ", ")). What do these relationships mean for the \(result.taskType) task?"
            ))
        }
        return actions
    }
}
