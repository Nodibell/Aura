import Testing
import Foundation
@testable import Aura

struct PythonRunnerTests {
    @Test func testPythonPathResolution() {
        let runner = PythonRunner.shared
        let path = runner.resolvePythonPath()
        #expect(!path.isEmpty)
        #expect(FileManager.default.fileExists(atPath: path))
    }
    
    @Test func testVerifyPythonEnvironment() {
        let runner = PythonRunner.shared
        let path = runner.resolvePythonPath()
        let isEnvValid = runner.verifyPythonEnvironment(at: path)
        #expect(isEnvValid == true)
    }
}
