import SwiftUI
import Charts

struct RidgelineChartView: View {
    let config: ChartConfig
    
    @State private var selectedXNum: Double? = nil
    
    var body: some View {
        Chart {
            ForEach(config.data) { point in
                if let xNum = point.xNum {
                    AreaMark(
                        x: .value(config.xLabel, xNum),
                        y: .value(config.yLabel, point.y)
                    )
                    .foregroundStyle(by: .value("Cluster", point.series ?? ""))
                    .opacity(0.35)
                    .accessibilityLabel("Bin: \(xNum), Cluster: \(point.series ?? "")")
                    .accessibilityValue("Density: \(point.y)")
                    
                    LineMark(
                        x: .value(config.xLabel, xNum),
                        y: .value(config.yLabel, point.y)
                    )
                    .foregroundStyle(by: .value("Cluster", point.series ?? ""))
                    .lineStyle(StrokeStyle(lineWidth: 2.0))
                }
            }
            
            if let selectedXNum = selectedXNum {
                RuleMark(x: .value("Selected", selectedXNum))
                    .foregroundStyle(Color.purple.opacity(0.4))
                    .lineStyle(StrokeStyle(lineWidth: 1.5, dash: [4, 4]))
                    .annotation(position: .top, spacing: 0, overflowResolution: .init(x: .fit, y: .fit)) {
                        tooltipView(for: selectedXNum)
                    }
            }
        }
        .chartXSelection(value: $selectedXNum)
        .chartXAxis {
            AxisMarks()
        }
        .chartYAxis {
            AxisMarks(position: .leading)
        }
        .chartXScale(domain: .automatic(includesZero: false))
        .chartYScale(domain: .automatic(includesZero: true))
        .chartLegend(.visible)
        .padding(.all, 8)
    }

    @ViewBuilder
    private func tooltipView(for xNum: Double) -> some View {
        let closestX = config.data.compactMap { $0.xNum }.min(by: { abs($0 - xNum) < abs($1 - xNum) })
        if let targetX = closestX {
            let matchedPoints = config.data.filter { $0.xNum == targetX }
            VStack(alignment: .leading, spacing: 6) {
                Text(String(format: "Bin Center: %.2f", targetX))
                    .font(.system(size: 11, weight: .bold))
                    .foregroundColor(.primary)
                Divider().background(Color.primary.opacity(0.1))
                ForEach(matchedPoints) { pt in
                    HStack(spacing: 12) {
                        Text(pt.series ?? "Value")
                            .font(.system(size: 10))
                            .foregroundColor(.secondary)
                        Spacer()
                        Text(String(format: "%.4f", pt.y))
                            .font(.system(size: 10, weight: .semibold, design: .rounded))
                            .foregroundColor(.purple)
                    }
                }
            }
            .padding(8)
            .background(.ultraThinMaterial)
            .cornerRadius(8)
            .shadow(color: .black.opacity(0.15), radius: 4)
            .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.primary.opacity(0.1), lineWidth: 0.5))
        }
    }
}
