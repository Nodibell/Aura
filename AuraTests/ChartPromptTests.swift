import Testing
import Foundation
@testable import Aura

struct ChartPromptTests {
    
    @Test func testWordcloudPrompt() {
        let points = [
            ChartPoint(xVal: "hello", y: 0.95),
            ChartPoint(xVal: "world", y: 0.85)
        ]
        let config = ChartConfig(type: "wordcloud", title: "Word Importance", xLabel: "Word", yLabel: "Weight", data: points, images: nil, boxStats: nil)
        
        let prompt = buildChartPrompt(config)
        #expect(prompt.contains("Analyze this word frequency cloud"))
        #expect(prompt.contains("hello: 0.9500"))
        #expect(prompt.contains("world: 0.8500"))
    }
    
    @Test func testBoxplotPrompt() {
        let stats = BoxStats(min: 1.0, q1: 2.0, median: 3.0, q3: 4.0, max: 5.0, outliers: [0.2, 9.8])
        let config = ChartConfig(type: "boxplot", title: "Feature Range", xLabel: "", yLabel: "Value", data: [], images: nil, boxStats: stats)
        
        let prompt = buildChartPrompt(config)
        #expect(prompt.contains("Analyze this statistical outlier box plot"))
        #expect(prompt.contains("Median: 3.0000"))
        #expect(prompt.contains("Outliers Count: 2"))
    }
    
    @Test func testNumericXAndYPromptWithoutSampling() {
        let points = [
            ChartPoint(xNum: 0.1, y: 10.0),
            ChartPoint(xNum: 0.5, y: 20.0),
            ChartPoint(xNum: 0.9, y: 30.0)
        ]
        let config = ChartConfig(type: "scatter", title: "Simple Scatter", xLabel: "X", yLabel: "Y", data: points, images: nil, boxStats: nil)
        
        let prompt = buildChartPrompt(config)
        
        // Assert global Y stats (10, 20, 30)
        #expect(prompt.contains("Min: 10.000000"))
        #expect(prompt.contains("Max: 30.000000"))
        #expect(prompt.contains("Mean: 20.000000"))
        #expect(prompt.contains("Median: 20.000000"))
        
        // Assert global X stats (0.1, 0.5, 0.9)
        #expect(prompt.contains("Min: 0.100000"))
        #expect(prompt.contains("Max: 0.900000"))
        #expect(prompt.contains("Mean: 0.500000"))
        #expect(prompt.contains("Median: 0.500000"))
        
        // Assert no sampling used
        #expect(prompt.contains("Showing all 3 data points"))
        #expect(prompt.contains("0.1000: 10.000000"))
        #expect(prompt.contains("0.5000: 20.000000"))
        #expect(prompt.contains("0.9000: 30.000000"))
    }
    
    @Test func testCategoricalXPrompt() {
        let points = [
            ChartPoint(xVal: "Apple", y: 1.0),
            ChartPoint(xVal: "Banana", y: 2.0),
            ChartPoint(xVal: "Apple", y: 3.0),
            ChartPoint(xVal: "Orange", y: 4.0)
        ]
        let config = ChartConfig(type: "bar", title: "Fruit Counts", xLabel: "Fruit", yLabel: "Count", data: points, images: nil, boxStats: nil)
        
        let prompt = buildChartPrompt(config)
        
        #expect(prompt.contains("Unique Categories: 3"))
        #expect(prompt.contains("Apple: 2 points"))
        #expect(prompt.contains("Banana: 1 points"))
        #expect(prompt.contains("Orange: 1 points"))
    }
    
    @Test func testSeriesPrompt() {
        let points = [
            ChartPoint(xNum: 1.0, y: 10.0, series: "A"),
            ChartPoint(xNum: 2.0, y: 20.0, series: "B"),
            ChartPoint(xNum: 3.0, y: 30.0, series: "A")
        ]
        let config = ChartConfig(type: "line", title: "Series Chart", xLabel: "X", yLabel: "Y", data: points, images: nil, boxStats: nil)
        
        let prompt = buildChartPrompt(config)
        
        #expect(prompt.contains("Series/Groups (by column):"))
        // Mean for Series A is (10 + 30)/2 = 20.0
        #expect(prompt.contains("A: 2 points (Mean Y = 20.0000)"))
        // Mean for Series B is 20.0
        #expect(prompt.contains("B: 1 points (Mean Y = 20.0000)"))
    }
    
    @Test func testEvenSamplingForLargeData() {
        // Generate 200 data points: X from 0.0 to 199.0, Y from 0.0 to 199.0
        var points: [ChartPoint] = []
        for i in 0..<200 {
            points.append(ChartPoint(xNum: Double(i), y: Double(i)))
        }
        let config = ChartConfig(type: "scatter", title: "Large Scatter", xLabel: "X", yLabel: "Y", data: points, images: nil, boxStats: nil)
        
        let prompt = buildChartPrompt(config)
        
        // Global stats should reflect the entire 200 points
        #expect(prompt.contains("Dataset Summary (200 points total):"))
        #expect(prompt.contains("Min: 0.000000"))
        #expect(prompt.contains("Max: 199.000000"))
        #expect(prompt.contains("Mean: 99.500000"))
        
        // Sampling note
        #expect(prompt.contains("Showing a representative sample of 100 data points spaced evenly across the range of X"))
        
        // Check that exactly 100 points are present in the data lines
        let parts = prompt.components(separatedBy: "**Data Points")
        #expect(parts.count == 2)
        let dataPointsPart = parts[1]
        let dataLines = dataPointsPart.components(separatedBy: "\n").filter { $0.hasPrefix("  - ") }
        #expect(dataLines.count == 100)
        
        // First point should be X=0.0
        #expect(dataLines.first?.contains("0.0000: 0.000000") == true)
        // Last point should be X=199.0
        #expect(dataLines.last?.contains("199.0000: 199.000000") == true)
    }
}
