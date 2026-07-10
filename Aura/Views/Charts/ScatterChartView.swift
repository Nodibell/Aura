import SwiftUI
import Charts

struct ScatterChartView: View {
    let config: ChartConfig
    var onTapPoint: ((ChartPoint) -> Void)? = nil
    @State private var visibleSeries: Set<String> = []
    
    @State private var selectedXVal: String? = nil
    @State private var selectedXNum: Double? = nil

    @State private var persistentSelectedXVal: String? = nil
    @State private var persistentSelectedXNum: Double? = nil

    // Swift Charts draws one real mark per point with no built-in
    // decimation, so a large test set (e.g. a Predicted vs Actual or
    // Residual plot on tens of thousands of rows) can make this view slow
    // to render regardless of how cheap our own filtering is. Cap what
    // actually reaches the Chart; the full `config.data` is still used for
    // axis bounds so scaling doesn't shift.
    private let maxRenderPoints = 1500

    private func decimated(_ points: [ChartPoint]) -> [ChartPoint] {
        guard points.count > maxRenderPoints else { return points }
        var sampled: [ChartPoint] = []
        sampled.reserveCapacity(maxRenderPoints)
        for i in 0..<maxRenderPoints {
            let idx = i * (points.count - 1) / (maxRenderPoints - 1)
            sampled.append(points[idx])
        }
        return sampled
    }

    private func getSelectedPoint(filteredData: [ChartPoint]) -> ChartPoint? {
        if let xVal = persistentSelectedXVal {
            return filteredData.first { $0.xVal == xVal }
        }
        if let xNum = persistentSelectedXNum {
            return filteredData.min {
                let diff1 = abs(($0.xNum ?? 0.0) - xNum)
                let diff2 = abs(($1.xNum ?? 0.0) - xNum)
                return diff1 < diff2
            }
        }
        return nil
    }

    private func getSelectedPointDescription(point: ChartPoint) -> String {
        if let xVal = point.xVal {
            return xVal
        }
        if let xNum = point.xNum {
            return String(format: "%.2f", xNum)
        }
        return ""
    }

    private func resetVisibleSeries() {
        let unique = Set(config.data.compactMap { $0.series })
        visibleSeries = unique
    }

    var body: some View {
        let xValues = config.data.compactMap { $0.xNum }
        let yValues = config.data.map { $0.y }
        
        let isPredictedVsActual = config.title.lowercased().contains("predicted vs actual") ||
                                  config.title.lowercased().contains("actual vs predicted")
        let isResidualPlot = config.title.lowercased().contains("residual")
        
        let uniqueSeries = Set(config.data.compactMap { $0.series })
        let hasMultipleSeries = uniqueSeries.count > 1
        
        let filteredData = config.data.filter { point in
            if let series = point.series {
                return visibleSeries.contains(series)
            }
            return true
        }

        VStack(alignment: .leading, spacing: 10) {
            if hasMultipleSeries {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        Button(action: {
                            if visibleSeries.count == uniqueSeries.count {
                                visibleSeries.removeAll()
                            } else {
                                visibleSeries = uniqueSeries
                            }
                        }) {
                            Text(visibleSeries.count == uniqueSeries.count ? "Deselect All" : "Select All")
                                .font(.caption.bold())
                                .padding(.horizontal, 10)
                                .padding(.vertical, 6)
                                .background(Color.primary.opacity(0.08))
                                .cornerRadius(6)
                        }
                        .buttonStyle(.plain)
                        
                        ForEach(Array(uniqueSeries).sorted(), id: \.self) { series in
                            let isSelected = visibleSeries.contains(series)
                            Button(action: {
                                if isSelected {
                                    visibleSeries.remove(series)
                                } else {
                                    visibleSeries.insert(series)
                                }
                            }) {
                                HStack(spacing: 4) {
                                    Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                                        .foregroundColor(isSelected ? .purple : .secondary)
                                    Text(series)
                                        .foregroundColor(isSelected ? .primary : .secondary)
                                }
                                .font(.caption)
                                .padding(.horizontal, 10)
                                .padding(.vertical, 6)
                                .background(
                                    RoundedRectangle(cornerRadius: 6)
                                        .stroke(isSelected ? Color.purple.opacity(0.5) : Color.primary.opacity(0.1), lineWidth: 1)
                                        .background(isSelected ? Color.purple.opacity(0.1) : Color.clear)
                                )
                                .cornerRadius(6)
                            }
                            .buttonStyle(.plain)
                        }
                    }
                    .padding(.horizontal, 4)
                    .padding(.vertical, 2)
                }
            }
            
            let baseChart = Chart {
                // Perfect prediction reference line (y=x) ONLY for prediction comparisons
                if isPredictedVsActual, let minX = xValues.min(), let maxX = xValues.max(),
                                        let minY = yValues.min(), let maxY = yValues.max() {
                    let idealMin = min(minX, minY)
                    let idealMax = max(maxX, maxY)
                    
                    LineMark(x: .value("Ideal", idealMin), y: .value("Ideal", idealMin))
                        .foregroundStyle(Color.green.opacity(0.4))
                        .lineStyle(StrokeStyle(lineWidth: 1.5, dash: [5, 4]))
                    LineMark(x: .value("Ideal", idealMax), y: .value("Ideal", idealMax))
                        .foregroundStyle(Color.green.opacity(0.4))
                        .lineStyle(StrokeStyle(lineWidth: 1.5, dash: [5, 4]))
                }

                // Zero-reference line (y=0) for residual plots
                if isResidualPlot {
                    RuleMark(y: .value("Zero Reference", 0.0))
                        .foregroundStyle(Color.red.opacity(0.4))
                        .lineStyle(StrokeStyle(lineWidth: 1.5, dash: [5, 4]))
                }

                ForEach(decimated(filteredData)) { point in
                    if let xVal = point.xVal {
                        PointMark(x: .value(config.xLabel, xVal), y: .value(config.yLabel, point.y))
                            .foregroundStyle(by: .value("Series", point.series ?? "Value"))
                            .symbolSize(30)
                    } else if let xNum = point.xNum {
                        PointMark(x: .value(config.xLabel, xNum), y: .value(config.yLabel, point.y))
                            .foregroundStyle(by: .value("Series", point.series ?? "Value"))
                            .symbolSize(28)
                    }
                }
                
                // Rule mark line
                if let selectedXVal = selectedXVal {
                    RuleMark(x: .value("Selected", selectedXVal))
                        .foregroundStyle(Color.purple.opacity(0.4))
                        .lineStyle(StrokeStyle(lineWidth: 1.5, dash: [4, 4]))
                } else if let selectedXNum = selectedXNum {
                    RuleMark(x: .value("Selected", selectedXNum))
                        .foregroundStyle(Color.purple.opacity(0.4))
                        .lineStyle(StrokeStyle(lineWidth: 1.5, dash: [4, 4]))
                }
            }
            .chartXSelection(value: $selectedXVal)
            .chartXSelection(value: $selectedXNum)
            .chartXAxis {
                AxisMarks()
            }
            .chartYAxis {
                AxisMarks(position: .leading)
            }
            .chartLegend(hasMultipleSeries ? .visible : .hidden)
            .overlay(alignment: .topTrailing) {
                if let selectedXVal = selectedXVal {
                    tooltipView(for: selectedXVal, data: filteredData)
                        .padding(8)
                } else if let selectedXNum = selectedXNum {
                    tooltipView(for: selectedXNum, data: filteredData)
                        .padding(8)
                }
            }

            // Apply scale bounds dynamically
            if xValues.isEmpty {
                // Categorical X-Axis
                baseChart
                    .chartYScale(domain: .automatic(includesZero: false))
                    .frame(height: 180)
            } else {
                // Numeric X-Axis: Calculate exact bounds with a 10% visual padding
                let minX = xValues.min() ?? 0
                let maxX = xValues.max() ?? 1
                let minY = yValues.min() ?? 0
                let maxY = yValues.max() ?? 1
                
                let xPad = max((maxX - minX) * 0.1, 0.1)
                let yPad = max((maxY - minY) * 0.1, 0.1)
                
                baseChart
                    .chartXScale(domain: (minX - xPad)...(maxX + xPad))
                    .chartYScale(domain: (minY - yPad)...(maxY + yPad))
                    .frame(height: 180)
            }
            
            // Drill down button for ScatterChartView
            if let selectedPoint = getSelectedPoint(filteredData: filteredData), onTapPoint != nil {
                Button {
                    onTapPoint?(selectedPoint)
                } label: {
                    Label("Drill Down Details: \(getSelectedPointDescription(point: selectedPoint))", systemImage: "arrow.up.left.and.down.right.magnifyingglass")
                        .font(.system(size: 11, weight: .bold))
                }
                .buttonStyle(.borderedProminent)
                .tint(Theme.Color.primaryAction)
                .padding(.top, 4)
                .frame(maxWidth: .infinity, alignment: .center)
            }
        }
        .onAppear {
            resetVisibleSeries()
        }
        .onChange(of: config.id, initial: false) {
            resetVisibleSeries()
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
    
    // MARK: - Tooltip Methods
    
    @ViewBuilder
    private func tooltipView(for xVal: String, data: [ChartPoint]) -> some View {
        let matchedPoints = data.filter { $0.xVal == xVal }
        if !matchedPoints.isEmpty {
            VStack(alignment: .leading, spacing: 6) {
                Text(xVal)
                    .font(.system(size: 11, weight: .bold))
                    .foregroundColor(.primary)
                Divider().background(Color.primary.opacity(0.1))
                
                ForEach(matchedPoints.prefix(5)) { pt in
                    HStack(spacing: 12) {
                        Text(pt.series ?? "Value")
                            .font(.system(size: 10))
                            .foregroundColor(.secondary)
                        Spacer()
                        Text(formatValue(pt.y))
                            .font(.system(size: 10, weight: .semibold, design: .rounded))
                            .foregroundColor(.purple)
                    }
                }
                if matchedPoints.count > 5 {
                    Text("+ \(matchedPoints.count - 5) more")
                        .font(.system(size: 9))
                        .foregroundColor(.secondary)
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
    private func tooltipView(for xNum: Double, data: [ChartPoint]) -> some View {
        let closestX = data.compactMap { $0.xNum }.min(by: { abs($0 - xNum) < abs($1 - xNum) })
        if let targetX = closestX {
            let matchedPoints = data.filter { $0.xNum == targetX }
            VStack(alignment: .leading, spacing: 6) {
                let titleText = matchedPoints.first?.xVal ?? String(format: "%.2f", targetX)
                Text(titleText)
                    .font(.system(size: 11, weight: .bold))
                    .foregroundColor(.primary)
                Divider().background(Color.primary.opacity(0.1))
                
                ForEach(matchedPoints.prefix(5)) { pt in
                    HStack(spacing: 12) {
                        Text(pt.series ?? "Value")
                            .font(.system(size: 10))
                            .foregroundColor(.secondary)
                        Spacer()
                        Text(formatValue(pt.y))
                            .font(.system(size: 10, weight: .semibold, design: .rounded))
                            .foregroundColor(.purple)
                    }
                }
                if matchedPoints.count > 5 {
                    Text("+ \(matchedPoints.count - 5) more")
                        .font(.system(size: 9))
                        .foregroundColor(.secondary)
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
}
