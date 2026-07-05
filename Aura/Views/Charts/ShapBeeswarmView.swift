import SwiftUI
import Charts

struct ShapBeeswarmView: View {
    let config: ChartConfig
    
    // Stacking point structure
    struct StackingPoint: Identifiable {
        let id: UUID
        let x: Double
        let y: Double
        let yOffset: Double
        let featureIndex: Int
    }
    
    private var uniqueFeatures: [String] {
        var list: [String] = []
        for point in config.data {
            if let feat = point.xVal, !list.contains(feat) {
                list.append(feat)
            }
        }
        return list
    }
    
    private var stackedPoints: [StackingPoint] {
        let features = uniqueFeatures
        var allStacked: [StackingPoint] = []
        for (idx, feature) in features.enumerated() {
            let featurePoints = config.data.filter { $0.xVal == feature }
            let stackedForFeature = computeStacking(points: featurePoints, featureIndex: idx, featuresCount: features.count)
            allStacked.append(contentsOf: stackedForFeature)
        }
        return allStacked
    }
    
    var body: some View {
        let features = uniqueFeatures
        
        Chart {
            ForEach(stackedPoints) { point in
                let reversedY = Double(features.count - 1 - point.featureIndex) + point.yOffset
                PointMark(
                    x: .value(config.xLabel, point.x),
                    y: .value(config.yLabel, reversedY)
                )
                .foregroundStyle(featureValueColor(point.y))
                .symbolSize(22) // slightly smaller for a denser, cleaner look
            }
        }
        .chartXAxis {
            AxisMarks()
        }
        .chartYAxis {
            AxisMarks(position: .leading, values: Array(0..<features.count)) { value in
                if let index = value.as(Int.self), index >= 0, index < features.count {
                    AxisValueLabel {
                        Text(features[features.count - 1 - index])
                            .font(.system(size: 10, weight: .medium))
                    }
                }
                AxisGridLine()
            }
        }
        .chartXScale(domain: .automatic(includesZero: false))
        .chartYScale(domain: -0.5...Double(max(1, features.count)) - 0.5)
        .overlay(alignment: .bottomTrailing) {
            colorLegend
        }
    }
    
    private func computeStacking(points: [ChartPoint], featureIndex: Int, featuresCount: Int) -> [StackingPoint] {
        let sortedPoints = points.compactMap { pt -> (id: UUID, x: Double, y: Double)? in
            guard let x = pt.xNum else { return nil }
            return (id: pt.id, x: x, y: pt.y)
        }.sorted { $0.x < $1.x }
        
        var stacked: [StackingPoint] = []
        stacked.reserveCapacity(sortedPoints.count)
        
        let xMin = sortedPoints.first?.x ?? 0.0
        let xMax = sortedPoints.last?.x ?? 1.0
        let xRange = xMax - xMin
        let cellWidth = xRange > 0 ? (xRange * 0.02) : 0.015
        let diameter = 0.07 // vertical stacking step
        
        for pt in sortedPoints {
            var neighbors: [StackingPoint] = []
        
            for i in stride(from: stacked.count - 1, through: 0, by: -1) {
                let diff = pt.x - stacked[i].x
                if diff <= cellWidth {
                    neighbors.append(stacked[i])
                } else {
                    break
                }
            }
            
            var offset = 0.0
            var found = false
            var step = 0
            
            while !found {
                let candidate: Double
                if step == 0 {
                    candidate = 0.0
                } else if step % 2 == 1 {
                    candidate = Double((step + 1) / 2) * diameter
                } else {
                    candidate = -Double(step / 2) * diameter
                }
                
                // Limit overflow to avoid overlapping next feature row
                let collides = neighbors.contains { abs($0.yOffset - candidate) < (diameter * 0.85) }
                if !collides || abs(candidate) >= 0.4 {
                    offset = min(max(candidate, -0.4), 0.4)
                    found = true
                } else {
                    step += 1
                }
            }
            
            stacked.append(
                StackingPoint(
                    id: pt.id,
                    x: pt.x,
                    y: pt.y,
                    yOffset: offset,
                    featureIndex: featureIndex
                )
            )
        }
        return stacked
    }
    
    private func featureValueColor(_ value: Double) -> Color {
        let r = 0.2 + value * 0.8
        let g = 0.4 - value * 0.3
        let b = 1.0 - value * 0.6
        return Color(red: r, green: g, blue: b)
    }
    
    private var colorLegend: some View {
        HStack(spacing: 6) {
            Text("Low")
                .font(.system(size: 8))
                .foregroundColor(.secondary)
            
            LinearGradient(
                colors: [Color(red: 0.2, green: 0.4, blue: 1.0), Color(red: 1.0, green: 0.1, blue: 0.4)],
                startPoint: .leading,
                endPoint: .trailing
            )
            .frame(width: 60, height: 6)
            .cornerRadius(3)
            
            Text("High")
                .font(.system(size: 8))
                .foregroundColor(.secondary)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(.regularMaterial)
        .cornerRadius(6)
        .padding(.trailing, 8)
        .padding(.bottom, 8)
    }
}
