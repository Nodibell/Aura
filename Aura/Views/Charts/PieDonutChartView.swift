import SwiftUI
import Charts

struct PieDonutChartView: View {
    let config: ChartConfig
    
    @State private var selectedValue: Double? = nil
    @State private var hoveredID: UUID? = nil

    private var processedData: [ChartPoint] {
        config.data
            .filter { $0.y > 0 }
            .sorted { ($0.xVal ?? "") < ($1.xVal ?? "") }
    }

    private var totalY: Double {
        processedData.reduce(0) { $0 + $1.y }
    }

    private func matchedPoint(for targetValue: Double) -> ChartPoint? {
        var sum = 0.0
        for pt in processedData {
            sum += pt.y
            if targetValue <= sum {
                return pt
            }
        }
        return processedData.last
    }

    var body: some View {
        let total = totalY
        
        Chart {
            ForEach(processedData) { point in
                let category = point.xVal ?? "Unknown"
                let value = point.y
                let percentage = total > 0 ? (value / total) * 100 : 0
                let percentageString = String(format: "%.1f%%", percentage)
                let isSelected = hoveredID == nil || hoveredID == point.id
                
                SectorMark(
                    angle: .value("Value", value),
                    innerRadius: .ratio(config.type == "donut" ? 0.6 : 0.0),
                    angularInset: 1.5
                )
                .foregroundStyle(by: .value("Category", category))
                .cornerRadius(4)
                .opacity(isSelected ? 1.0 : 0.55)
                .accessibilityLabel("Category: \(category), Value: \(value)")
                .accessibilityValue("Percentage: \(percentageString)")
            }
        }
        .chartLegend(.visible)
        .chartAngleSelection(value: $selectedValue)
        .onChange(of: selectedValue) { oldValue, newValue in
            if let val = newValue {
                hoveredID = matchedPoint(for: val)?.id
            } else {
                hoveredID = nil
            }
        }
        .chartOverlay { proxy in
            if let val = selectedValue, let pt = matchedPoint(for: val) {
                ZStack {
                    VStack(alignment: .center, spacing: 4) {
                        Text(pt.xVal ?? "")
                            .font(.system(size: 11, weight: .bold))
                            .lineLimit(1)
                        Text(formatValue(pt.y))
                            .font(Theme.Font.sectionTitle)
                            .foregroundColor(.purple)
                        Text(String(format: "%.1f%%", total > 0 ? (pt.y / total) * 100 : 0))
                            .font(.system(size: 10))
                            .foregroundColor(.secondary)
                    }
                    .padding(8)
                    .background(.ultraThinMaterial)
                    .cornerRadius(8)
                    .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.primary.opacity(0.1), lineWidth: 0.5))
                    .shadow(color: .black.opacity(0.1), radius: 3)
                    .frame(width: 110, height: 75)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .padding(.all, 8)
    }

    private func formatValue(_ value: Double) -> String {
        if value == 0 { return "0" }
        let absVal = abs(value)
        if absVal >= 1000 {
            return String(format: "%.0f", value)
        } else if absVal >= 1 {
            if value.truncatingRemainder(dividingBy: 1) == 0 {
                return String(format: "%.0f", value)
            } else {
                return String(format: "%.1f", value)
            }
        } else if absVal >= 0.01 {
            return String(format: "%.2f", value)
        } else {
            return String(format: "%.3f", value)
        }
    }
}
