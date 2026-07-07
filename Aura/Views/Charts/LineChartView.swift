import SwiftUI
import Charts

struct LineChartView: View {
    let config: ChartConfig
    var onTapPoint: ((ChartPoint) -> Void)? = nil
    
    private static let formatters: [DateFormatter] = {
        let formats = [
            "yyyy-MM-dd",
            "M/d/yyyy",
            "d/M/yyyy",
            "dd-mm-yyyy HH:mm:ss",
            "dd.mm.yyyy HH:mm:ss",
            "dd.dd.yyyy HH:mm:ss",
            "yyyy-MM-dd HH:mm:ss",
            "yyyy-MM-dd'T'HH:mm:ssZ",
            "yyyy-MM-dd'T'HH:mm:ss.SSSZ",
            "yyyy"
        ]
        return formats.map { fmt in
            let df = DateFormatter()
            df.dateFormat = fmt
            return df
        }
    }()
    private static let isoFormatter = ISO8601DateFormatter()
    
    @State private var viewMode: ViewMode = .raw
    @State private var selectedYear: String = "All"

    @State private var selectedDate: Date? = nil
    @State private var selectedXVal: String? = nil
    @State private var State_selectedXNum: Double? = nil // Avoid conflicts
    @State private var selectedXNum: Double? = nil

    @State private var persistentSelectedDate: Date? = nil
    @State private var persistentSelectedXVal: String? = nil
    @State private var persistentSelectedXNum: Double? = nil

    // Re-applied fix (was lost in a later merge): parseDate() tries up to 11
    // DateFormatters per string. It must run ONCE per chart, not on every
    // body re-evaluation. Cached here, populated by .task(id:) below.
    @State private var parsedCache: [(point: ChartPoint, date: Date?)] = []
    @State private var tsInfo: (isTS: Bool, years: [String]) = (false, [])

    // NEW: hard cap on how many points we actually hand to Swift Charts for
    // a single line/series. Swift Charts lays out a real mark per data
    // point, so tens of thousands of marks are slow to draw regardless of
    // how fast our own data prep is. 1500 keeps the visual shape intact
    // (evenly-spaced sampling, same technique already used in
    // buildChartPrompt below) while staying smooth to render and scroll.
    private let maxRenderPoints = 1500

    private var selectedPoint: ChartPoint? {
        if let selDate = persistentSelectedDate {
            let processed = getProcessedPoints()
            guard let closest = processed.min(by: {
                let diff1 = abs(($0.xDate?.timeIntervalSince1970 ?? 0.0) - selDate.timeIntervalSince1970)
                let diff2 = abs(($1.xDate?.timeIntervalSince1970 ?? 0.0) - selDate.timeIntervalSince1970)
                return diff1 < diff2
            }) else { return nil }
            return config.data.first { $0.xVal == closest.xVal }
        }
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

    private var selectedPointDescription: String {
        if let selDate = persistentSelectedDate {
            let df = DateFormatter()
            df.dateStyle = .medium
            df.timeStyle = .none
            return df.string(from: selDate)
        }
        if let xVal = persistentSelectedXVal {
            return xVal
        }
        if let xNum = persistentSelectedXNum {
            return String(format: "%.2f", xNum)
        }
        return ""
    }
    
    enum ViewMode: String, CaseIterable, Identifiable {
        case raw = "All Data"
        case monthly = "Month of Year"
        case daily = "Day of Month"
        
        var id: String { rawValue }
    }

    private func parseDate(_ string: String) -> Date? {
        if let date = Self.isoFormatter.date(from: string) {
            return date
        }
        for formatter in Self.formatters {
            if let date = formatter.date(from: string) {
                return date
            }
        }
        return nil
    }
    
    /// Runs the (up to 11-formatter) date-parsing pass over `config.data`
    /// exactly once, and caches both the per-point parse result and the
    /// derived time-series summary. Call this from `.task(id: config.id)`,
    /// never from `body` directly.
    private func recomputeCache() {
        var parsedCount = 0
        var yearsSet = Set<String>()
        var newCache: [(point: ChartPoint, date: Date?)] = []
        newCache.reserveCapacity(config.data.count)

        for pt in config.data {
            var date: Date? = nil
            if let xVal = pt.xVal {
                date = parseDate(xVal)
                if let date {
                    parsedCount += 1
                    yearsSet.insert(String(Calendar.current.component(.year, from: date)))
                }
            }
            newCache.append((pt, date))
        }

        let isTS = !config.data.isEmpty && parsedCount >= Int(Double(config.data.count) * 0.7)
        self.parsedCache = newCache
        self.tsInfo = (isTS, yearsSet.sorted())
    }
    
    struct ProcessedPoint: Identifiable {
        let id: String
        let xVal: String?
        let xNum: Double?
        let xDate: Date?
        let y: Double
        let series: String
    }

    /// Evenly-spaced downsample so Swift Charts never has to lay out more
    /// than `maxRenderPoints` marks for one series. Keeps first/last points
    /// and the overall shape; same stride technique buildChartPrompt below
    /// already uses for the AI-summary sample.
    private func decimatedForRender(_ points: [ProcessedPoint]) -> [ProcessedPoint] {
        guard points.count > maxRenderPoints else { return points }
        var sampled: [ProcessedPoint] = []
        sampled.reserveCapacity(maxRenderPoints)
        for i in 0..<maxRenderPoints {
            let idx = i * (points.count - 1) / (maxRenderPoints - 1)
            sampled.append(points[idx])
        }
        return sampled
    }
    
    private func getProcessedPoints() -> [ProcessedPoint] {
        // Reads pre-parsed dates from `parsedCache` (populated once by
        // recomputeCache()) instead of calling parseDate() again here.
        let filtered = parsedCache.compactMap { item -> (point: ChartPoint, date: Date?)? in
            if let date = item.date {
                let yearStr = String(Calendar.current.component(.year, from: date))
                if selectedYear == "All" || selectedYear == yearStr {
                    return item
                }
                return nil
            }
            if selectedYear == "All" {
                return item
            }
            return nil
        }
        
        switch viewMode {
        case .raw:
            let all = filtered.map { item in
                ProcessedPoint(
                    id: item.point.id.uuidString,
                    xVal: item.point.xVal,
                    xNum: item.point.xNum,
                    xDate: item.date,
                    y: item.point.y,
                    series: item.point.series ?? "Value"
                )
            }
            return decimatedForRender(all)
            
        case .monthly:
            var monthGroups: [Int: [Double]] = [:]
            for item in filtered {
                if let date = item.date {
                    let month = Calendar.current.component(.month, from: date)
                    monthGroups[month, default: []].append(item.point.y)
                }
            }
            let monthNames = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            var results: [ProcessedPoint] = []
            for m in 1...12 {
                if let vals = monthGroups[m], !vals.isEmpty {
                    let avg = vals.reduce(0, +) / Double(vals.count)
                    let name = monthNames[m - 1]
                    results.append(ProcessedPoint(
                        id: "monthly-\(name)",
                        xVal: name,
                        xNum: nil,
                        xDate: nil,
                        y: avg,
                        series: "Monthly Average"
                    ))
                }
            }
            return results
            
        case .daily:
            var dayGroups: [Int: [Double]] = [:]
            for item in filtered {
                if let date = item.date {
                    let day = Calendar.current.component(.day, from: date)
                    dayGroups[day, default: []].append(item.point.y)
                }
            }
            var results: [ProcessedPoint] = []
            for d in 1...31 {
                if let vals = dayGroups[d], !vals.isEmpty {
                    let avg = vals.reduce(0, +) / Double(vals.count)
                    let name = String(d)
                    results.append(ProcessedPoint(
                        id: "daily-\(name)",
                        xVal: name,
                        xNum: Double(d),
                        xDate: nil,
                        y: avg,
                        series: "Daily Average"
                    ))
                }
            }
            return results
        }
    }
    
    var body: some View {
        let processed = getProcessedPoints()
        let hasMultipleSeries = config.data.contains(where: { $0.series != nil }) && viewMode == .raw
        
        VStack(alignment: .leading, spacing: 12) {
            if tsInfo.isTS {
                HStack(spacing: 12) {
                    HStack(spacing: 4) {
                        ForEach(ViewMode.allCases) { mode in
                            Button {
                                withAnimation(.spring(response: 0.2, dampingFraction: 0.85)) {
                                    viewMode = mode
                                    selectedDate = nil
                                    selectedXVal = nil
                                    selectedXNum = nil
                                }
                            } label: {
                                Text(mode.rawValue)
                                    .font(.system(size: 9, weight: .bold))
                                    .padding(.horizontal, 8)
                                    .padding(.vertical, 4)
                                    .background(viewMode == mode ? Color.purple : Color.primary.opacity(0.04))
                                    .foregroundColor(viewMode == mode ? .white : .secondary)
                                    .cornerRadius(8)
                            }
                            .buttonStyle(.plain)
                        }
                    }
                    
                    Divider().frame(height: 14)
                    
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 4) {
                            Button {
                                withAnimation(.spring(response: 0.2, dampingFraction: 0.85)) {
                                    selectedYear = "All"
                                    selectedDate = nil
                                }
                            } label: {
                                Text("All Years")
                                    .font(.system(size: 9, weight: .bold))
                                    .padding(.horizontal, 8)
                                    .padding(.vertical, 4)
                                    .background(selectedYear == "All" ? Color.indigo : Color.primary.opacity(0.04))
                                    .foregroundColor(selectedYear == "All" ? .white : .secondary)
                                    .cornerRadius(8)
                            }
                            .buttonStyle(.plain)
                            
                            ForEach(tsInfo.years, id: \.self) { year in
                                Button {
                                    withAnimation(.spring(response: 0.2, dampingFraction: 0.85)) {
                                        selectedYear = year
                                        selectedDate = nil
                                    }
                                } label: {
                                    Text(year)
                                        .font(.system(size: 9, weight: .bold))
                                        .padding(.horizontal, 8)
                                        .padding(.vertical, 4)
                                        .background(selectedYear == year ? Color.indigo : Color.primary.opacity(0.04))
                                        .foregroundColor(selectedYear == year ? .white : .secondary)
                                        .cornerRadius(8)
                                }
                                .buttonStyle(.plain)
                            }
                        }
                    }
                }
                .padding(.horizontal, 2)
            }
            
            GeometryReader { geo in
                if tsInfo.isTS {
                    if viewMode == .raw {
                        Chart {
                            ForEach(processed) { point in
                                if let date = point.xDate {
                                    LineMark(
                                        x: .value(config.xLabel, date),
                                        y: .value(config.yLabel, point.y)
                                    )
                                    .foregroundStyle(by: .value("Series", point.series))
                                    .lineStyle(StrokeStyle(lineWidth: 1.8))
                                    .accessibilityLabel("Date: \(date), Series: \(point.series)")
                                    .accessibilityValue("Value: \(formatValue(point.y))")
                                }
                            }
                            
                            if let selectedDate = selectedDate {
                                RuleMark(x: .value("Selected", selectedDate))
                                    .foregroundStyle(Color.purple.opacity(0.4))
                                    .lineStyle(StrokeStyle(lineWidth: 1.5, dash: [4, 4]))
                                    .annotation(position: .top, spacing: 0, overflowResolution: .init(x: .fit, y: .fit)) {
                                        tooltipView(for: selectedDate, points: processed)
                                    }
                            }
                        }
                        .chartXSelection(value: $selectedDate)
                        .chartXAxis {
                            AxisMarks()
                        }
                        .chartYAxis {
                            AxisMarks(position: .leading)
                        }
                        .chartXScale(domain: .automatic(includesZero: false))
                        .chartYScale(domain: .automatic(includesZero: false))
                        .chartLegend(hasMultipleSeries ? .visible : .hidden)
                        .padding(.all, 4)
                    } else if viewMode == .monthly {
                        Chart {
                            ForEach(processed) { point in
                                if let xVal = point.xVal {
                                    LineMark(
                                        x: .value(config.xLabel, xVal),
                                        y: .value(config.yLabel, point.y)
                                    )
                                    .foregroundStyle(Color.purple)
                                    .lineStyle(StrokeStyle(lineWidth: 2.2))
                                    .accessibilityLabel("Month: \(xVal)")
                                    .accessibilityValue("Value: \(formatValue(point.y))")
                                }
                            }
                            
                            if let selectedXVal = selectedXVal {
                                RuleMark(x: .value("Selected", selectedXVal))
                                    .foregroundStyle(Color.purple.opacity(0.4))
                                    .lineStyle(StrokeStyle(lineWidth: 1.5, dash: [4, 4]))
                                    .annotation(position: .top, spacing: 0, overflowResolution: .init(x: .fit, y: .fit)) {
                                        tooltipView(for: selectedXVal, points: processed)
                                    }
                            }
                        }
                        .chartXSelection(value: $selectedXVal)
                        .chartXAxis {
                            AxisMarks()
                        }
                        .chartYAxis {
                            AxisMarks(position: .leading)
                        }
                        .chartXScale(domain: .automatic(includesZero: false))
                        .chartYScale(domain: .automatic(includesZero: false))
                        .chartLegend(.hidden)
                        .padding(.all, 4)
                    } else if viewMode == .daily {
                        Chart {
                            ForEach(processed) { point in
                                if let xNum = point.xNum {
                                    LineMark(
                                        x: .value(config.xLabel, xNum),
                                        y: .value(config.yLabel, point.y)
                                    )
                                    .foregroundStyle(Color.indigo)
                                    .lineStyle(StrokeStyle(lineWidth: 2.2))
                                    .accessibilityLabel("Day: \(xNum)")
                                    .accessibilityValue("Value: \(formatValue(point.y))")
                                }
                            }
                            
                            if let selectedXNum = selectedXNum {
                                RuleMark(x: .value("Selected", selectedXNum))
                                    .foregroundStyle(Color.purple.opacity(0.4))
                                    .lineStyle(StrokeStyle(lineWidth: 1.5, dash: [4, 4]))
                                    .annotation(position: .top, spacing: 0, overflowResolution: .init(x: .fit, y: .fit)) {
                                        tooltipView(for: selectedXNum, points: processed)
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
                        .chartYScale(domain: .automatic(includesZero: false))
                        .chartLegend(.hidden)
                        .padding(.all, 4)
                    }
                } else {
                    let uniqueXCount = Set(config.data.map { point -> String in
                        if let xVal = point.xVal { return xVal }
                        if let xNum = point.xNum { return String(xNum) }
                        return ""
                    }).count
                    
                    let isCat = config.data.first?.xVal != nil
                    let visibleLength = 12
                    let needsScroll = uniqueXCount > visibleLength

                    if isCat {
                        Chart {
                            ForEach(config.data) { point in
                                if let xVal = point.xVal {
                                    LineMark(x: .value(config.xLabel, xVal), y: .value(config.yLabel, point.y))
                                        .foregroundStyle(by: .value("Series", point.series ?? "Value"))
                                        .lineStyle(StrokeStyle(lineWidth: 2.5))
                                        .accessibilityLabel("Category: \(xVal)")
                                        .accessibilityValue("Value: \(formatValue(point.y))")
                                }
                            }
                            
                            if let selectedXVal = selectedXVal {
                                RuleMark(x: .value("Selected", selectedXVal))
                                    .foregroundStyle(Color.purple.opacity(0.4))
                                    .lineStyle(StrokeStyle(lineWidth: 1.5, dash: [4, 4]))
                                    .annotation(position: .top, spacing: 0, overflowResolution: .init(x: .fit, y: .fit)) {
                                        tooltipView(for: selectedXVal, points: processed)
                                    }
                            }
                        }
                        .chartXSelection(value: $selectedXVal)
                        .chartXAxis {
                            AxisMarks()
                        }
                        .chartYAxis {
                            AxisMarks(position: .leading)
                        }
                        .chartXScale(domain: .automatic(includesZero: false))
                        .chartYScale(domain: .automatic(includesZero: false))
                        .chartLegend(hasMultipleSeries ? .visible : .hidden)
                        .chartScrollableAxes(needsScroll ? .horizontal : [])
                        .chartXVisibleDomain(length: needsScroll ? visibleLength : uniqueXCount)
                        .padding(.all, 8)
                    } else {
                        Chart {
                            ForEach(config.data) { point in
                                if let xNum = point.xNum {
                                    LineMark(x: .value(config.xLabel, xNum), y: .value(config.yLabel, point.y))
                                        .foregroundStyle(by: .value("Series", point.series ?? "Value"))
                                        .lineStyle(StrokeStyle(lineWidth: 2.5))
                                        .accessibilityLabel("Value: \(xNum)")
                                        .accessibilityValue("Value: \(formatValue(point.y))")
                                }
                            }
                            
                            if let selectedXNum = selectedXNum {
                                RuleMark(x: .value("Selected", selectedXNum))
                                    .foregroundStyle(Color.purple.opacity(0.4))
                                    .lineStyle(StrokeStyle(lineWidth: 1.5, dash: [4, 4]))
                                    .annotation(position: .top, spacing: 0, overflowResolution: .init(x: .fit, y: .fit)) {
                                        tooltipView(for: selectedXNum, points: processed)
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
                        .chartYScale(domain: .automatic(includesZero: false))
                        .chartLegend(hasMultipleSeries ? .visible : .hidden)
                        .chartScrollableAxes(needsScroll ? .horizontal : [])
                        .chartXVisibleDomain(length: needsScroll ? visibleLength : uniqueXCount)
                        .padding(.all, 8)
                    }
                }
            }
            .frame(height: 180)
            
            // Drill down button for LineChartView
            if let selectedPoint = selectedPoint, onTapPoint != nil {
                Button {
                    onTapPoint?(selectedPoint)
                } label: {
                    Label("Drill Down Details: \(selectedPointDescription)", systemImage: "arrow.up.left.and.down.right.magnifyingglass")
                        .font(.system(size: 11, weight: .bold))
                }
                .buttonStyle(.borderedProminent)
                .tint(.purple)
                .padding(.bottom, 8)
            }
        }
        .onChange(of: selectedDate) { oldValue, newValue in
            if let val = newValue {
                persistentSelectedDate = val
                persistentSelectedXVal = nil
                persistentSelectedXNum = nil
            }
        }
        .onChange(of: selectedXVal) { oldValue, newValue in
            if let val = newValue {
                persistentSelectedXVal = val
                persistentSelectedDate = nil
                persistentSelectedXNum = nil
            }
        }
        .onChange(of: selectedXNum) { oldValue, newValue in
            if let val = newValue {
                persistentSelectedXNum = val
                persistentSelectedDate = nil
                persistentSelectedXVal = nil
            }
        }
        .task(id: config.id) {
            recomputeCache()
        }
    }

    @ViewBuilder
    private func tooltipView(for date: Date, points: [ProcessedPoint]) -> some View {
        let closestPt = points.compactMap { pt -> (ProcessedPoint, TimeInterval)? in
            guard let ptDate = pt.xDate else { return nil }
            return (pt, abs(ptDate.timeIntervalSince(date)))
        }
        .min(by: { $0.1 < $1.1 })?.0

        if let targetPt = closestPt, let targetDate = targetPt.xDate {
            let matchedPoints = points.filter { pt in
                guard let d = pt.xDate else { return false }
                return Calendar.current.isDate(d, inSameDayAs: targetDate)
            }
            
            let dateFormatter: DateFormatter = {
                let df = DateFormatter()
                df.dateStyle = .medium
                df.timeStyle = .none
                return df
            }()

            VStack(alignment: .leading, spacing: 6) {
                Text(dateFormatter.string(from: targetDate))
                    .font(.system(size: 11, weight: .bold))
                    .foregroundColor(.primary)
                Divider().background(Color.primary.opacity(0.1))
                ForEach(matchedPoints) { pt in
                    HStack(spacing: 12) {
                        Text(pt.series)
                            .font(.system(size: 10))
                            .foregroundColor(.secondary)
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
    private func tooltipView(for xVal: String, points: [ProcessedPoint]) -> some View {
        let matchedPoints = points.filter { $0.xVal == xVal }
        if !matchedPoints.isEmpty {
            VStack(alignment: .leading, spacing: 6) {
                Text(xVal)
                    .font(.system(size: 11, weight: .bold))
                    .foregroundColor(.primary)
                Divider().background(Color.primary.opacity(0.1))
                ForEach(matchedPoints) { pt in
                    HStack(spacing: 12) {
                        Text(pt.series)
                            .font(.system(size: 10))
                            .foregroundColor(.secondary)
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
    private func tooltipView(for xNum: Double, points: [ProcessedPoint]) -> some View {
        let closestX = points.compactMap { $0.xNum }.min(by: { abs($0 - xNum) < abs($1 - xNum) })
        if let targetX = closestX {
            let matchedPoints = points.filter { $0.xNum == targetX }
            VStack(alignment: .leading, spacing: 6) {
                Text(String(format: "%.2f", targetX))
                    .font(.system(size: 11, weight: .bold))
                    .foregroundColor(.primary)
                Divider().background(Color.primary.opacity(0.1))
                ForEach(matchedPoints) { pt in
                    HStack(spacing: 12) {
                        Text(pt.series)
                            .font(.system(size: 10))
                            .foregroundColor(.secondary)
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
}
