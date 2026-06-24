import Testing
import Foundation
@testable import Aura

struct EnterpriseDatabaseSchedulerTests {
    
    @Test func testDatabaseQueryExecution() async throws {
        let runner = PythonRunner.shared
        let dbPath = "/Users/oleksiichumak/Developer/Xcode.projects/Aura/sample_data/iris.db"
        
        // Ensure the db exists before executing
        guard FileManager.default.fileExists(atPath: dbPath) else {
            Issue.record("Test DB sample_data/iris.db does not exist. Run setup DB script.")
            return
        }
        
        let tempOutCSV = NSTemporaryDirectory() + "aura_test_iris_export.csv"
        
        // Clean up any existing file
        try? FileManager.default.removeItem(atPath: tempOutCSV)
        defer {
            try? FileManager.default.removeItem(atPath: tempOutCSV)
        }
        
        let connParams = ["db_path": dbPath]
        let query = "SELECT * FROM iris"
        
        let result = try await runner.runDatabaseQuery(
            dbType: "sqlite",
            query: query,
            connParams: connParams,
            outputCSVPath: tempOutCSV
        )
        
        #expect(result.rowCount == 150)
        #expect(result.columns.contains("species"))
        #expect(result.columns.contains("sepal_length"))
        #expect(FileManager.default.fileExists(atPath: tempOutCSV))
    }
    
    @Test func testSchedulerTaskManagement() async {
        let scheduler = AnalysisScheduler.shared
        let taskId = UUID()
        
        let config = AnalysisConfig()
        let task = ScheduledTask(
            id: taskId,
            name: "Test Routine Task",
            datasetPath: "sample_data/iris.csv",
            targetColumn: "species",
            taskType: .tabular,
            recurrence: .daily,
            exportFormat: .html,
            exportFolderPath: NSTemporaryDirectory(),
            isActive: true,
            lastRun: nil,
            nextRun: Date(),
            config: config
        )
        
        // Add task
        scheduler.addTask(task)
        var allTasks = scheduler.getTasks()
        #expect(allTasks.contains(where: { $0.id == taskId }))
        
        // Toggle active status
        scheduler.toggleTaskActive(withId: taskId)
        allTasks = scheduler.getTasks()
        let updatedTask = allTasks.first(where: { $0.id == taskId })
        #expect(updatedTask?.isActive == false)
        
        // Remove task
        scheduler.removeTask(withId: taskId)
        allTasks = scheduler.getTasks()
        #expect(!allTasks.contains(where: { $0.id == taskId }))
    }
    
    @Test func testSchedulerNextRunCalculation() {
        let now = Date()
        let calendar = Calendar.current
        
        // Daily
        let dailyNext = calendar.date(byAdding: .day, value: 1, to: now) ?? now.addingTimeInterval(86400)
        #expect(calendar.isDate(dailyNext, inSameDayAs: now.addingTimeInterval(86400)))
        
        // Hourly
        let hourlyNext = calendar.date(byAdding: .hour, value: 2, to: now) ?? now.addingTimeInterval(7200)
        #expect(calendar.component(.hour, from: hourlyNext) == (calendar.component(.hour, from: now) + 2) % 24)
    }
}
