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
}
