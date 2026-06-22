import Testing
import Foundation
@testable import Aura

struct AutoMLCleaningTests {
    
    @Test func testCleaningActionSerialization() throws {
        let action = CleaningAction(column: "age", actionType: "impute_mean")
        
        let encoder = JSONEncoder()
        let data = try encoder.encode(action)
        
        let decoder = JSONDecoder()
        let decoded = try decoder.decode(CleaningAction.self, from: data)
        
        #expect(decoded.column == "age")
        #expect(decoded.actionType == "impute_mean")
        #expect(decoded.id == "age-impute_mean")
    }
    
    @Test func testAnalysisConfigCleaningActions() {
        var config = AnalysisConfig()
        #expect(config.cleaningActions.isEmpty)
        
        let action1 = CleaningAction(column: "age", actionType: "impute_mean")
        let action2 = CleaningAction(column: "salary", actionType: "clip_outliers")
        
        config.cleaningActions.insert(action1)
        config.cleaningActions.insert(action2)
        
        #expect(config.cleaningActions.count == 2)
        #expect(config.cleaningActions.contains(action1))
        #expect(config.cleaningActions.contains(action2))
    }
    
    @Test func testPythonRunnerArgumentBuildingWithCleaningActions() throws {
        let runner = PythonRunner.shared
        var config = AnalysisConfig()
        
        // Add cleaning actions
        let action1 = CleaningAction(column: "age", actionType: "impute_mean")
        let action2 = CleaningAction(column: "income", actionType: "isolation_forest")
        config.cleaningActions.insert(action1)
        config.cleaningActions.insert(action2)
        
        let args = runner.buildArguments(
            scriptPath: "/path/to/analyze.py",
            csvPath: "/path/to/data.csv",
            targetColumn: "target",
            config: config
        )
        
        #expect(args.contains("/path/to/analyze.py"))
        #expect(args.contains("/path/to/data.csv"))
        #expect(args.contains("--target"))
        #expect(args.contains("target"))
        #expect(args.contains("--cleaning-actions"))
        
        // Find the index of "--cleaning-actions"
        if let idx = args.firstIndex(of: "--cleaning-actions") {
            let jsonValueIdx = args.index(after: idx)
            #expect(jsonValueIdx < args.endIndex)
            let jsonString = args[jsonValueIdx]
            
            let data = jsonString.data(using: .utf8)!
            let decoder = JSONDecoder()
            let decodedActions = try decoder.decode([CleaningAction].self, from: data)
            
            #expect(decodedActions.count == 2)
            let ageAction = decodedActions.first(where: { $0.column == "age" })
            #expect(ageAction != nil)
            #expect(ageAction?.actionType == "impute_mean")
            
            let incomeAction = decodedActions.first(where: { $0.column == "income" })
            #expect(incomeAction != nil)
            #expect(incomeAction?.actionType == "isolation_forest")
        } else {
            Issue.record("Arguments list should contain '--cleaning-actions'")
        }
    }
}
