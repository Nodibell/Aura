import Foundation
import SwiftData

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
        colCount: Int? = nil
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
    }
    
    static func == (lhs: HistoryItem, rhs: HistoryItem) -> Bool {
        lhs.id == rhs.id
    }
}
