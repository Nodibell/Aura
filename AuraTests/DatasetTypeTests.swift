import Testing
@testable import Aura

struct DatasetTypeTests {
    @Test func testDatasetTypeEnum() {
        let odType = DatasetType.objectDetection
        #expect(odType.rawValue == "object_detection")
        #expect(odType.label == "Object Detection")
        #expect(odType.icon == "viewfinder.rectangular")
    }
    
    @Test func testAnalysisConfigInitialization() {
        let config = AnalysisConfig()
        #expect(config.datasetType == .tabular)
        #expect(config.testFilePath == nil)
        #expect(config.validationFilePath == nil)
    }
}
