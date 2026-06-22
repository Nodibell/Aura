import Testing
import Foundation
@testable import Aura

struct ProtectedDataTests {
    @Test func testProtectedDataThreadSafety() async {
        let protectedData = ProtectedData()
        
        // Append concurrently from multiple tasks
        await withTaskGroup(of: Void.self) { group in
            for i in 0..<100 {
                group.addTask {
                    let segment = Data([UInt8(i)])
                    protectedData.append(segment)
                }
            }
        }
        
        let resultData = protectedData.get()
        #expect(resultData.count == 100)
    }
}
