import Foundation
import SwiftData
import SwiftUI

@Model
class HistoryItem: Identifiable, Equatable {
    @Attribute(.unique) var id: UUID
    var datasetName: String
    var datasetPath: String
    var targetColumn: String?
    var timestamp: Date
    var resultFileName: String
    
    // Expanded metadata
    var taskType: String?
    var bestModel: String?
    var bestScore: Double?
    var scoreType: String?
    var rowCount: Int?
    var colCount: Int?
    var datasetURL: String?
    var isPinned: Bool? = false
    var cleaningActionsJson: String?

    init(
        id: UUID = UUID(),
        datasetName: String,
        datasetPath: String,
        targetColumn: String?,
        timestamp: Date = Date(),
        resultFileName: String,
        taskType: String? = nil,
        bestModel: String? = nil,
        bestScore: Double? = nil,
        scoreType: String? = nil,
        rowCount: Int? = nil,
        colCount: Int? = nil,
        datasetURL: String? = nil,
        isPinned: Bool? = false,
        cleaningActionsJson: String? = nil
    ) {
        self.id = id
        self.datasetName = datasetName
        self.datasetPath = datasetPath
        self.targetColumn = targetColumn
        self.timestamp = timestamp
        self.resultFileName = resultFileName
        self.taskType = taskType
        self.bestModel = bestModel
        self.bestScore = bestScore
        self.scoreType = scoreType
        self.rowCount = rowCount
        self.colCount = colCount
        self.datasetURL = datasetURL
        self.isPinned = isPinned
        self.cleaningActionsJson = cleaningActionsJson
    }
    
    static func == (lhs: HistoryItem, rhs: HistoryItem) -> Bool {
        lhs.id == rhs.id
    }
}

extension HistoryItem {
    var uiColor: Color {
        guard let task = taskType?.lowercased() else { return .secondary }
        if task.contains("regress") {
            return .purple
        } else if task.contains("class") {
            return .indigo
        } else if task.contains("time") || task.contains("forecast") {
            return .blue
        } else if task.contains("nlp") {
            return .green
        } else if task.contains("image") {
            return .orange
        } else if task.contains("object") || task.contains("vision") {
            return .red
        } else if task.contains("cluster") {
            return .yellow
        }
        return .secondary
    }
    
    var shortLabel: String {
        guard let task = taskType?.lowercased() else { return "EDA" }
        if task.contains("regress") {
            return "REG"
        } else if task.contains("class") {
            return "CLS"
        } else if task.contains("time") || task.contains("forecast") {
            return "TS"
        } else if task.contains("nlp") {
            return "NLP"
        } else if task.contains("image") {
            return "IMG"
        } else if task.contains("object") || task.contains("vision") {
            return "CV"
        } else if task.contains("cluster") {
            return "CLST"
        }
        return "EDA"
    }
}
