import Foundation
import Observation
import SwiftData

@MainActor
@Observable
class AnalysisHistoryService {
    static let shared = AnalysisHistoryService()
    
    private(set) var items: [HistoryItem] = []
    
    private let fileManager = FileManager.default
    private let container: ModelContainer
    private var context: ModelContext {
        container.mainContext
    }
    
    private var baseDirectory: URL {
        let appSupport = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let auraDir = appSupport.appendingPathComponent("Aura", isDirectory: true)
        let historyDir = auraDir.appendingPathComponent("history", isDirectory: true)
        
        // Create directory structures if missing
        try? fileManager.createDirectory(at: historyDir, withIntermediateDirectories: true, attributes: nil)
        return historyDir
    }
    
    private init() {
        do {
            container = try ModelContainer(for: HistoryItem.self)
            loadMetadata()
        } catch {
            fatalError("Failed to initialize ModelContainer for HistoryItem: \(error)")
        }
    }
    
    func loadMetadata() {
        let descriptor = FetchDescriptor<HistoryItem>(sortBy: [SortDescriptor(\.timestamp, order: .reverse)])
        do {
            let fetchedItems = try context.fetch(descriptor)
            self.items = fetchedItems
            AppLogger.shared.info("Loaded \(self.items.count) history items from SwiftData", category: "History")
        } catch {
            self.items = []
            AppLogger.shared.error("Failed to fetch history items from SwiftData: \(error)", category: "History")
        }
    }
    
    private func saveMetadata() {
        do {
            try context.save()
            AppLogger.shared.info("Saved SwiftData changes successfully.", category: "History")
        } catch {
            AppLogger.shared.error("Failed to save SwiftData changes: \(error)", category: "History")
        }
    }
    
    private func generateDisplayName(source: String, result: AnalysisResult) -> String {
        let trimmed = source.trimmingCharacters(in: .whitespacesAndNewlines)
        
        var baseName = ""
        if trimmed.lowercased().hasPrefix("http://") || trimmed.lowercased().hasPrefix("https://") {
            if let url = URL(string: trimmed) {
                let pathComponents = url.pathComponents.filter { $0 != "/" }
                
                // Kaggle format: /datasets/owner/dataset-name
                if trimmed.contains("kaggle.com"), pathComponents.contains("datasets"),
                   let index = pathComponents.firstIndex(of: "datasets"), index + 2 < pathComponents.count {
                    baseName = pathComponents[index + 2].replacingOccurrences(of: "-", with: " ").replacingOccurrences(of: "_", with: " ").capitalized
                }
                // Hugging Face format: /datasets/owner/dataset-name
                else if trimmed.contains("huggingface.co"), pathComponents.contains("datasets"),
                   let index = pathComponents.firstIndex(of: "datasets"), index + 2 < pathComponents.count {
                    baseName = pathComponents[index + 2].replacingOccurrences(of: "-", with: " ").replacingOccurrences(of: "_", with: " ").capitalized
                }
                // Fallback: last path component
                else if let last = pathComponents.last {
                    baseName = last.replacingOccurrences(of: ".csv", with: "")
                               .replacingOccurrences(of: ".parquet", with: "")
                               .replacingOccurrences(of: ".tsv", with: "")
                               .replacingOccurrences(of: "-", with: " ")
                               .replacingOccurrences(of: "_", with: " ")
                               .capitalized
                }
            }
            if baseName.isEmpty {
                baseName = "Web Dataset"
            }
        } else {
            // Local File Path
            let filename = URL(fileURLWithPath: trimmed).lastPathComponent
            let nameWithoutExt = (filename as NSString).deletingPathExtension
            
            // Check if filename is a SHA256 hex string (64 characters)
            if nameWithoutExt.count == 64 && nameWithoutExt.range(of: "^[0-9a-fA-F]+$", options: .regularExpression) != nil {
                baseName = "Cached Dataset"
            } else {
                baseName = nameWithoutExt
                    .replacingOccurrences(of: "-", with: " ")
                    .replacingOccurrences(of: "_", with: " ")
                    .capitalized
            }
        }
        
        let targetName = result.targetColumn.replacingOccurrences(of: "_", with: " ").capitalized
        let taskName = result.taskType.capitalized
        return "\(baseName) (\(targetName) \(taskName))"
    }
    
    func renameItem(_ item: HistoryItem, to newName: String) {
        if let index = items.firstIndex(where: { $0.id == item.id }) {
            items[index].datasetName = newName
            AppLogger.shared.info("Renamed history item '\(item.datasetName)' to '\(newName)' in SwiftData.", category: "History")
            saveMetadata()
        }
    }
    
    private func triggerBackgroundTitleGeneration(for id: UUID, result: AnalysisResult) {
        Task {
            let ollamaChecker = OllamaStatusChecker.shared
            // Make sure Ollama is available
            guard ollamaChecker.isAvailable else { return }
            
            let savedModel = UserDefaults.standard.string(forKey: "Aura_OllamaModel") ?? ""
            let modelToUse: String
            if !savedModel.isEmpty && ollamaChecker.availableModels.contains(where: { $0.name == savedModel }) {
                modelToUse = savedModel
            } else if let first = ollamaChecker.availableModels.first {
                modelToUse = first.name
            } else {
                return
            }
            
            let prompt = """
            Based on the following summary of a dataset analysis, generate a very short, clean title for the analysis (maximum 4 words, no quotes, no markdown, e.g., 'Credit Risk Scoring' or 'House Prices Regression'). Do not include any introductory or explanatory text.
            
            Summary:
            \(result.summary.prefix(1000))
            """
            
            let messages = [OllamaChatMessage(role: "user", content: prompt)]
            var generated = ""
            do {
                let stream = OllamaService.shared.streamChat(
                    messages: messages,
                    systemPrompt: "",
                    model: modelToUse,
                    temperature: 0.1,
                    maxTokens: 30
                )
                for try await chunk in stream {
                    generated += chunk
                }
                
                let cleanedTitle = generated.trimmingCharacters(in: .whitespacesAndNewlines)
                    .replacingOccurrences(of: "\"", with: "")
                    .replacingOccurrences(of: "'", with: "")
                    .replacingOccurrences(of: "*", with: "")
                    .replacingOccurrences(of: "`", with: "")
                
                guard !cleanedTitle.isEmpty && cleanedTitle.count < 100 else { return }
                
                // Update the title in items and save
                await MainActor.run {
                    if let index = self.items.firstIndex(where: { $0.id == id }) {
                        self.items[index].datasetName = cleanedTitle
                        AppLogger.shared.info("Background title generated via LLM: '\(cleanedTitle)' (replaced in SwiftData)", category: "History")
                        self.saveMetadata()
                    }
                }
            } catch {
                AppLogger.shared.error("Background name generation failed: \(error)", category: "History")
            }
        }
    }
    
    @discardableResult
    func saveAnalysis(result: AnalysisResult, datasetPath: String, targetColumn: String?, originalSource: String? = nil) -> HistoryItem? {
        let sourceForName = originalSource ?? datasetPath
        let datasetName = generateDisplayName(source: sourceForName, result: result)
        let resultId = UUID()
        let resultFileName = "\(resultId.uuidString).json"
        let resultURL = baseDirectory.appendingPathComponent(resultFileName)
        
        do {
            let data = try JSONEncoder().encode(result)
            try data.write(to: resultURL)
            
            let isRemote = originalSource?.lowercased().hasPrefix("http://") == true || originalSource?.lowercased().hasPrefix("https://") == true
            let finalURL = isRemote ? originalSource : nil

            let newItem = HistoryItem(
                id: resultId,
                datasetName: datasetName,
                datasetPath: datasetPath,
                targetColumn: targetColumn,
                timestamp: Date(),
                resultFileName: resultFileName,
                taskType: result.taskType,
                bestModel: result.metrics.model,
                bestScore: result.metrics.score,
                scoreType: result.metrics.scoreType,
                rowCount: result.rowCount,
                colCount: result.colCount,
                datasetURL: finalURL
            )
            
            context.insert(newItem)
            items.insert(newItem, at: 0)
            AppLogger.shared.info("Saved analysis for \(datasetName) (path: \(datasetPath)) in SwiftData", category: "History")
            saveMetadata()
            
            // Asynchronously generate a title via LLM if available
            triggerBackgroundTitleGeneration(for: resultId, result: result)
            return newItem
        } catch {
            AppLogger.shared.error("Failed to save analysis result in SwiftData: \(error)", category: "History")
            return nil
        }
    }
    
    func loadAnalysisResult(item: HistoryItem) async -> AnalysisResult? {
        let resultURL = baseDirectory.appendingPathComponent(item.resultFileName)
        let fileName = item.resultFileName
        let displayName = item.datasetName
        return await Task.detached(priority: .userInitiated) {
            guard let data = try? Data(contentsOf: resultURL),
                  let result = try? JSONDecoder().decode(AnalysisResult.self, from: data) else {
                await MainActor.run {
                    AppLogger.shared.error(
                        "Failed to load/decode analysis result for '\(displayName)' from file '\(fileName)'",
                        category: "History"
                    )
                }
                return nil
            }
            await MainActor.run {
                AppLogger.shared.info(
                    "Loaded analysis result for '\(displayName)' from file '\(fileName)'",
                    category: "History"
                )
            }
            return result
        }.value
    }
    
    func deleteItem(_ item: HistoryItem) {
        let resultURL = baseDirectory.appendingPathComponent(item.resultFileName)
        try? fileManager.removeItem(at: resultURL)
        
        context.delete(item)
        items.removeAll(where: { $0.id == item.id })
        AppLogger.shared.info("Deleted history item '\(item.datasetName)' from SwiftData", category: "History")
        saveMetadata()
    }
    
    func clearAll() {
        for item in items {
            let resultURL = baseDirectory.appendingPathComponent(item.resultFileName)
            try? fileManager.removeItem(at: resultURL)
            context.delete(item)
        }
        items.removeAll()
        AppLogger.shared.info("Cleared all history items from SwiftData", category: "History")
        saveMetadata()
    }
}
