import SwiftUI
import Charts

struct BarChartView: View {
    let config: ChartConfig
    var onTapPoint: ((ChartPoint) -> Void)? = nil

    @State private var selectedXVal: String? = nil
    @State private var selectedXNum: Double? = nil

    @State private var persistentSelectedXVal: String? = nil
    @State private var persistentSelectedXNum: Double? = nil

    private var selectedPoint: ChartPoint? {
        if let xVal = persistentSelectedXVal {
            return config.data.first { $0.xVal == xVal }
        }
        if let xNum = persistentSelectedXNum {
            return config.data.min {
                let diff1 = abs(($0.xNum ?? 0.0) - xNum)
                let diff2 = abs(($1.xNum ?? 0.0) - xNum)
                return diff1 < diff2
            }
        }
        return nil
    }

    var body: some View {
        let hasMultipleSeries = config.data.contains(where: { $0.series != nil })
        let isCategorical = config.data.first?.xVal != nil
        
        let visibleCount = hasMultipleSeries ? 6 : 12
        let needsScrolling = config.data.count > visibleCount

        VStack(spacing: 8) {
            if isCategorical {
                Chart {
                    ForEach(Array(config.data.enumerated()), id: \.element.id) { index, point in
                        if let xVal = point.xVal {
                            if hasMultipleSeries, let series = point.series {
                                BarMark(x: .value(config.xLabel, xVal), y: .value(config.yLabel, point.y))
                                    .foregroundStyle(by: .value("Series", series))
                                    .cornerRadius(4)
                                    .accessibilityLabel("Category: \(xVal), Series: \(series)")
                                    .accessibilityValue("Value: \(formatValue(point.y))")
                            } else {
                                BarMark(x: .value(config.xLabel, xVal), y: .value(config.yLabel, point.y))
                                    .foregroundStyle(barGradient(index: index, total: config.data.count))
                                    .cornerRadius(4)
                                    .accessibilityLabel("Category: \(xVal)")
                                    .accessibilityValue("Value: \(formatValue(point.y))")
                            }
                        }
                    }
                    
                    if let selectedXVal = selectedXVal {
                        RuleMark(x: .value("Selected", selectedXVal))
                            .foregroundStyle(Color.purple.opacity(0.4))
                            .lineStyle(StrokeStyle(lineWidth: 1.5, dash: [4, 4]))
                            .annotation(position: .top, spacing: 0, overflowResolution: .init(x: .fit, y: .fit)) {
                                tooltipView(for: selectedXVal)
                            }
                    }
                }
                .chartXSelection(value: $selectedXVal)
                .chartXAxis {
                    AxisMarks(values: .automatic) { value in
                        AxisGridLine()
                        AxisTick()
                        AxisValueLabel(anchor: .top) {
                            if let val = value.as(String.self) {
                                Text(val)
                                    .font(.system(size: 9))
                            }
                        }
                    }
                }
                .chartYAxis { AxisMarks(position: .leading) }
                .chartLegend(hasMultipleSeries ? .visible : .hidden)
                .chartXScale(domain: .automatic(includesZero: false))
                .chartScrollableAxes(needsScrolling ? .horizontal : [])
                .chartXVisibleDomain(length: needsScrolling ? visibleCount : config.data.count)
                .frame(height: 200)
                .padding(.horizontal, 8)
                .padding(.top, 8)
                .padding(.bottom, 24)
                
            } else {
                Chart {
                    ForEach(Array(config.data.enumerated()), id: \.element.id) { index, point in
                        if let xNum = point.xNum {
                            if hasMultipleSeries, let series = point.series {
                                BarMark(x: .value(config.xLabel, xNum), y: .value(config.yLabel, point.y))
                                    .foregroundStyle(by: .value("Series", series))
                                    .cornerRadius(4)
                                    .accessibilityLabel("Value: \(xNum), Series: \(series)")
                                    .accessibilityValue("Value: \(formatValue(point.y))")
                            } else {
                                BarMark(x: .value(config.xLabel, xNum), y: .value(config.yLabel, point.y))
                                    .foregroundStyle(LinearGradient(colors: [.blue, .purple], startPoint: .bottom, endPoint: .top))
                                    .cornerRadius(4)
                                    .accessibilityLabel("Value: \(xNum)")
                                    .accessibilityValue("Value: \(formatValue(point.y))")
                            }
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
                .chartXAxis { AxisMarks() }
                .chartYAxis { AxisMarks(position: .leading) }
                .chartLegend(hasMultipleSeries ? .visible : .hidden)
                .chartXScale(domain: .automatic(includesZero: false))
                .frame(height: 180)
                .padding(.all, 8)
            }
            
            // Drill down button
            if let selectedPoint = selectedPoint, onTapPoint != nil {
                Button {
                    onTapPoint?(selectedPoint)
                } label: {
                    let desc = selectedPoint.xVal ?? (selectedPoint.xNum != nil ? String(format: "%.2f", selectedPoint.xNum!) : "")
                    Label("Drill Down Details: \(desc)", systemImage: "arrow.up.left.and.down.right.magnifyingglass")
                        .font(.system(size: 11, weight: .bold))
                }
                .buttonStyle(.borderedProminent)
                .tint(Theme.Color.primaryAction)
                .padding(.bottom, 4)
            }
        }
        .onChange(of: selectedXVal) { oldValue, newValue in
            if let val = newValue {
                persistentSelectedXVal = val
                persistentSelectedXNum = nil
            }
        }
        .onChange(of: selectedXNum) { oldValue, newValue in
            if let val = newValue {
                persistentSelectedXNum = val
                persistentSelectedXVal = nil
            }
        }
    }

    @ViewBuilder
    private func tooltipView(for xVal: String) -> some View {
        let matchedPoints = config.data.filter { $0.xVal == xVal }
        if !matchedPoints.isEmpty {
            VStack(alignment: .leading, spacing: 6) {
                Text(xVal)
                    .font(.system(size: 11, weight: .bold))
                    .foregroundColor(.primary)
                Divider().background(Color.primary.opacity(0.1))
                ForEach(matchedPoints) { pt in
                    HStack(spacing: 12) {
                        if let series = pt.series {
                            Text(series)
                                .font(.system(size: 10))
                                .foregroundColor(.secondary)
                        }
                        Spacer()
                        Text(formatValue(pt.y))
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

    @ViewBuilder
    private func tooltipView(for xNum: Double) -> some View {
        let closestX = config.data.compactMap { $0.xNum }.min(by: { abs($0 - xNum) < abs($1 - xNum) })
        if let targetX = closestX {
            let matchedPoints = config.data.filter { $0.xNum == targetX }
            VStack(alignment: .leading, spacing: 6) {
                Text(String(format: "%.2f", targetX))
                    .font(.system(size: 11, weight: .bold))
                    .foregroundColor(.primary)
                Divider().background(Color.primary.opacity(0.1))
                ForEach(matchedPoints) { pt in
                    HStack(spacing: 12) {
                        if let series = pt.series {
                            Text(series)
                                .font(.system(size: 10))
                                .foregroundColor(.secondary)
                        }
                        Spacer()
                        Text(formatValue(pt.y))
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

    private func barGradient(index: Int, total: Int) -> LinearGradient {
        let ratio = total > 1 ? Double(index) / Double(total - 1) : 0.5
        let startColor = Color(hue: 0.65 - ratio * 0.15, saturation: 0.7, brightness: 0.9)
        let endColor = Color(hue: 0.65 - ratio * 0.15, saturation: 0.5, brightness: 0.7)
        return LinearGradient(colors: [startColor, endColor], startPoint: .top, endPoint: .bottom)
    }
}