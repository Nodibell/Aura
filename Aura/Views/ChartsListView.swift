import SwiftUI
import Charts

struct DrillDownItem: Identifiable {
    var id: String { title }
    let title: String
    let preview: FullTablePreview
}

struct ChartsListView: View {
    let result: AnalysisResult
    let onAskAI: (String) -> Void

    @State private var drillDownPreview: FullTablePreview? = nil
    @State private var drillDownTitle: String = ""

    var body: some View {
        ScrollView {
            VStack(spacing: 24) {
                if result.charts.isEmpty {
                    VStack(spacing: 12) {
                        Image(systemName: "chart.bar.doc.horizontal")
                            .font(.system(size: 48))
                            .foregroundColor(.secondary)
                        Text("No charts available for this dataset.")
                            .foregroundColor(.secondary)
                    }
                    .frame(maxWidth: .infinity, minHeight: 300)
                } else {
                    ForEach(result.charts) { chartConfig in
                        ChartCard(config: chartConfig, onAskAI: onAskAI) { point in
                            handleDrillDown(chartConfig: chartConfig, point: point)
                        }
                    }
                }
            }
            .padding()
        }
        .sheet(item: Binding<DrillDownItem?>(
            get: {
                if let preview = drillDownPreview {
                    return DrillDownItem(title: drillDownTitle, preview: preview)
                }
                return nil
            },
            set: { val in
                if val == nil {
                    drillDownPreview = nil
                    drillDownTitle = ""
                }
            }
        )) { item in
            VStack(spacing: 0) {
                HStack {
                    Text(item.title)
                        .font(.headline)
                    Spacer()
                    Button {
                        drillDownPreview = nil
                        drillDownTitle = ""
                    } label: {
                        Text("Close")
                            .fontWeight(.medium)
                    }
                    .buttonStyle(.bordered)
                }
                .padding()
                
                Divider()
                
                FullTableView(preview: item.preview)
            }
            .frame(minWidth: 800, minHeight: 500)
            .background(Color(NSColor.windowBackgroundColor))
        }
    }

    private func handleDrillDown(chartConfig: ChartConfig, point: ChartPoint) {
        guard let preview = result.fullPreview else { return }
        
        let colName = findColumnName(title: chartConfig.title, columns: preview.columns, targetColumn: result.targetColumn) ?? chartConfig.xLabel
        
        let filtered = filterRows(for: colName, selectedPoint: point, allData: chartConfig.data, preview: preview)
        
        let labelVal = point.xVal ?? (point.xNum != nil ? String(format: "%.2f", point.xNum!) : "")
        drillDownTitle = "Data Points where '\(colName)' matches '\(labelVal)' (\(filtered.rows.count) rows)"
        drillDownPreview = filtered
    }

    private func findColumnName(title: String, columns: [String], targetColumn: String) -> String? {
        let lowerTitle = title.lowercased()
        
        // Explicit mappings
        if lowerTitle.contains("target class") || lowerTitle.contains("target value") || lowerTitle.contains("predictions vs actual") {
            return targetColumn
        }
        if lowerTitle.contains("k-means cluster") {
            return "K-Means Cluster"
        }
        if lowerTitle.contains("dbscan cluster") {
            return "DBSCAN Cluster"
        }
        
        // Check if any column name is in the title
        for col in columns {
            if lowerTitle.contains(col.lowercased()) {
                return col
            }
        }
        
        return nil
    }

    private func filterRows(for colName: String, selectedPoint: ChartPoint, allData: [ChartPoint], preview: FullTablePreview) -> FullTablePreview {
        guard let colIdx = preview.columns.firstIndex(of: colName) else { return preview }
        
        var filteredRows: [[String]] = []
        
        if let xVal = selectedPoint.xVal {
            let cleanXVal = xVal.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            
            var possibleMatches = [cleanXVal]
            if let firstWord = xVal.components(separatedBy: " ").first?.lowercased() {
                possibleMatches.append(firstWord)
            }
            
            let parts = xVal.split(separator: "-")
            if parts.count == 2, let low = Double(parts[0].trimmingCharacters(in: .whitespaces)), let high = Double(parts[1].trimmingCharacters(in: .whitespaces)) {
                for row in preview.rows {
                    if colIdx < row.count, let val = Double(row[colIdx]) {
                        if val >= low && val <= high {
                            filteredRows.append(row)
                        }
                    }
                }
            } else {
                for row in preview.rows {
                    if colIdx < row.count {
                        let cellVal = row[colIdx].trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
                        if possibleMatches.contains(cellVal) || cellVal.contains(cleanXVal) || cleanXVal.contains(cellVal) {
                            filteredRows.append(row)
                        }
                    }
                }
            }
        } else if let xNum = selectedPoint.xNum {
            var binWidth: Double = 0.0
            let nums = allData.compactMap { $0.xNum }.sorted()
            if nums.count > 1 {
                var minDiff = Double.greatestFiniteMagnitude
                for i in 0..<(nums.count - 1) {
                    let diff = nums[i+1] - nums[i]
                    if diff > 0 && diff < minDiff {
                        minDiff = diff
                    }
                }
                if minDiff != Double.greatestFiniteMagnitude {
                    binWidth = minDiff
                }
            }
            
            let halfWidth = binWidth > 0 ? binWidth / 2.0 : 1e-5
            let low = xNum - halfWidth
            let high = xNum + halfWidth
            
            for row in preview.rows {
                if colIdx < row.count, let val = Double(row[colIdx]) {
                    if val >= low && val <= high {
                        filteredRows.append(row)
                    }
                }
            }
        } else {
            filteredRows = preview.rows
        }
        
        return FullTablePreview(columns: preview.columns, rows: filteredRows, totalRows: filteredRows.count)
    }
}

// MARK: - Chart AI Prompt Builder

#if canImport(AppKit)
import AppKit
typealias PlatformImage = NSImage
#else
import UIKit
typealias PlatformImage = UIImage
#endif

/// Builds a rich AI prompt that includes the actual data points from the chart.
func buildChartPrompt(_ config: ChartConfig) -> String {
    if config.type == "image_grid" {
        let labels = (config.images ?? []).map { $0.label }.joined(separator: ", ")
        return """
        Analyze these representative class images and the overall image dataset.
        
        **Title:** \(config.title)
        **Detected classes / image labels:** \(labels)
        
        Please answer:
        1. What patterns, visual distinctions, or characteristics define each class?
        2. How well-separated are these classes based on the visual representations?
        3. What recommendations do you have for training a machine learning model on these images?
        """
    }
    
    if config.type == "wordcloud" {
        let maxPoints = 30
        let sample = config.data.prefix(maxPoints)
        let dataLines = sample.map { pt -> String in
            let x = pt.xVal ?? "?"
            return "  \(x): \(String(format: "%.4f", pt.y))"
        }.joined(separator: "\n")
        let moreNote = config.data.count > maxPoints
            ? "\n  … (\(config.data.count - maxPoints) more words not shown)"
            : ""
            
        return """
        Analyze this word frequency cloud and provide data-driven NLP insights:
        
        **Chart:** \(config.title)
        **Top Words & TF-IDF Weights:**
        \(dataLines)\(moreNote)
        
        Please answer:
        1. What are the most dominant themes, topics, or terms in this text corpus?
        2. Are there any unexpected or anomalous words appearing with high weight?
        3. How might these terms be useful for building a text classifier, topic model, or sentiment analyzer?
        """
    }

    if config.type == "boxplot", let stats = config.boxStats {
        return """
        Analyze this statistical outlier box plot and provide data-driven insights:
        
        **Chart:** \(config.title)
        **Variable:** \(config.yLabel)
        **Key Statistics:**
          - Lower Whisker (Min): \(String(format: "%.4f", stats.min))
          - 25th Percentile (Q1): \(String(format: "%.4f", stats.q1))
          - Median: \(String(format: "%.4f", stats.median))
          - 75th Percentile (Q3): \(String(format: "%.4f", stats.q3))
          - Upper Whisker (Max): \(String(format: "%.4f", stats.max))
          - Interquartile Range (IQR): \(String(format: "%.4f", stats.q3 - stats.q1))
          - Outliers Count: \(stats.outliers.count) (Values: \(stats.outliers.prefix(10).map { String(format: "%.4f", $0) }.joined(separator: ", "))\(stats.outliers.count > 10 ? "..." : ""))
        
        Please answer:
        1. What does the distribution look like (skewness, spread, range)?
        2. Are the outliers significant, and how might they impact machine learning models?
        3. Do you recommend any data transformations or outlier treatments (e.g. log transform, capping, removal)?
        """
    }

    // For general charts (line, bar, scatter):
    let totalPoints = config.data.count
    
    // Sort the data first to make sure samples are ordered by X value (important for trends)
    let sortedData = config.data.sorted { (pt1, pt2) -> Bool in
        if let x1 = pt1.xNum, let x2 = pt2.xNum {
            return x1 < x2
        }
        if let x1 = pt1.xVal, let x2 = pt2.xVal {
            return x1 < x2
        }
        return false
    }
    
    // Statistics for Y-axis (always numeric)
    let yValues = sortedData.map { $0.y }
    let yMin = yValues.min() ?? 0.0
    let yMax = yValues.max() ?? 0.0
    let ySum = yValues.reduce(0.0, +)
    let yMean = yValues.isEmpty ? 0.0 : ySum / Double(yValues.count)
    
    let ySorted = yValues.sorted()
    let yMedian: Double
    if ySorted.isEmpty {
        yMedian = 0.0
    } else if ySorted.count % 2 == 1 {
        yMedian = ySorted[ySorted.count / 2]
    } else {
        let mid = ySorted.count / 2
        yMedian = (ySorted[mid - 1] + ySorted[mid]) / 2.0
    }
    
    // Statistics for X-axis (if numeric)
    let xNums = sortedData.compactMap { $0.xNum }
    let hasNumericX = !xNums.isEmpty && xNums.count == totalPoints
    
    var xMin: Double?
    var xMax: Double?
    var xMean: Double?
    var xMedian: Double?
    
    if hasNumericX {
        xMin = xNums.min()
        xMax = xNums.max()
        let xSum = xNums.reduce(0.0, +)
        xMean = xSum / Double(xNums.count)
        
        let xSorted = xNums.sorted()
        if !xSorted.isEmpty {
            if xSorted.count % 2 == 1 {
                xMedian = xSorted[xSorted.count / 2]
            } else {
                let mid = xSorted.count / 2
                xMedian = (xSorted[mid - 1] + xSorted[mid]) / 2.0
            }
        }
    }
    
    // Statistics for X-axis (if categorical)
    let xVals = sortedData.compactMap { $0.xVal }
    let uniqueCategoriesCount = Set(xVals).count
    var categoricalDistributionString = ""
    if !xVals.isEmpty {
        var counts: [String: Int] = [:]
        for val in xVals {
            counts[val, default: 0] += 1
        }
        let sortedCounts = counts.sorted { $0.value > $1.value }
        let topCounts = sortedCounts.prefix(10)
        let distributionLines = topCounts.map { "    - \($0.key): \($0.value) points (\(String(format: "%.1f", Double($0.value) / Double(totalPoints) * 100.0))%)" }
        categoricalDistributionString = distributionLines.joined(separator: "\n")
        if sortedCounts.count > 10 {
            let otherCount = sortedCounts.dropFirst(10).reduce(0) { $0 + $1.value }
            categoricalDistributionString += "\n    - Others: \(otherCount) points (\(String(format: "%.1f", Double(otherCount) / Double(totalPoints) * 100.0))%)"
        }
    }
    
    // Series statistics (if present)
    let seriesList = sortedData.compactMap { $0.series }
    let hasSeries = !seriesList.isEmpty
    var seriesSummaryString = ""
    if hasSeries {
        var seriesCounts: [String: Int] = [:]
        var seriesYValues: [String: [Double]] = [:]
        for pt in sortedData {
            if let series = pt.series {
                seriesCounts[series, default: 0] += 1
                seriesYValues[series, default: []].append(pt.y)
            }
        }
        
        let sortedSeries = seriesCounts.sorted { $0.value > $1.value }
        let seriesSummaryLines = sortedSeries.map { pair -> String in
            let name = pair.key
            let count = pair.value
            let yVals = seriesYValues[name] ?? []
            let avgY = yVals.isEmpty ? 0.0 : yVals.reduce(0.0, +) / Double(yVals.count)
            return "    - \(name): \(count) points (Mean \(config.yLabel) = \(String(format: "%.4f", avgY)))"
        }
        seriesSummaryString = seriesSummaryLines.joined(separator: "\n")
    }
    
    // Sampling representative data points
    let maxPoints = 100
    var sampledPoints: [ChartPoint] = []
    if totalPoints <= maxPoints {
        sampledPoints = sortedData
    } else {
        // Sample exactly maxPoints points evenly across sortedData
        for i in 0..<maxPoints {
            let idx = i * (totalPoints - 1) / (maxPoints - 1)
            sampledPoints.append(sortedData[idx])
        }
    }
    
    // Format the sample data lines
    let dataLines = sampledPoints.map { pt -> String in
        let x = pt.xVal ?? (pt.xNum.map { String(format: "%.4f", $0) } ?? "?")
        let seriesPrefix = pt.series.map { "[\($0)] " } ?? ""
        return "  - \(seriesPrefix)\(x): \(String(format: "%.6f", pt.y))"
    }.joined(separator: "\n")
    
    let isSampled = totalPoints > maxPoints
    let samplingNote = isSampled 
        ? "Showing a representative sample of \(maxPoints) data points spaced evenly across the range of X (sorted ascending) out of \(totalPoints) total points."
        : "Showing all \(totalPoints) data points (sorted by X ascending)."

    // Build the stats block
    var statsBlock = """
    **Dataset Summary (\(totalPoints) points total):**
      - **Y-axis (\(config.yLabel)):**
        - Min: \(String(format: "%.6f", yMin))
        - Max: \(String(format: "%.6f", yMax))
        - Mean: \(String(format: "%.6f", yMean))
        - Median: \(String(format: "%.6f", yMedian))
    """
    
    if hasNumericX, let xMinVal = xMin, let xMaxVal = xMax, let xMeanVal = xMean, let xMedianVal = xMedian {
        statsBlock += """
        
          - **X-axis (\(config.xLabel), Numeric):**
            - Min: \(String(format: "%.6f", xMinVal))
            - Max: \(String(format: "%.6f", xMaxVal))
            - Mean: \(String(format: "%.6f", xMeanVal))
            - Median: \(String(format: "%.6f", xMedianVal))
        """
    } else if uniqueCategoriesCount > 0 {
        statsBlock += """
        
          - **X-axis (\(config.xLabel), Categorical):**
            - Unique Categories: \(uniqueCategoriesCount)
            - Category Distribution:
        \(categoricalDistributionString)
        """
    }
    
    if hasSeries {
        statsBlock += """
        
          - **Series/Groups (by column):**
        \(seriesSummaryString)
        """
    }

    return """
    Analyze this chart dataset and provide specific, data-driven insights. Do not base your analysis solely on the subset of data points if they are sampled, but look at the statistical summary below first, and then cross-reference with the representative points.
    
    **Chart:** \(config.title)
    
    \(statsBlock)
    
    **Data Points (\(samplingNote)):**
    \(dataLines)
    
    Please answer:
    1. What pattern or trend do you see in the overall dataset statistics and spatial distribution?
    2. Which values or ranges stand out as most important, anomalous, or surprising?
    3. What does this tell us about the dataset and its implications for machine learning/modeling?
    Reference both the global statistics and the sample data points in your answer.
    """
}

// MARK: - Chart Card

struct ChartCard: View {
    let config: ChartConfig
    let onAskAI: (String) -> Void
    var onTapPoint: ((ChartPoint) -> Void)? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(config.title)
                        .font(.headline)
                    if !config.xLabel.isEmpty || !config.yLabel.isEmpty {
                        HStack(spacing: 6) {
                            Text(config.xLabel)
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Text("→")
                                .font(.caption2)
                                .foregroundColor(.secondary)
                            Text(config.yLabel)
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                }
                Spacer()
                Button {
                    onAskAI(buildChartPrompt(config))
                } label: {
                    Label("AI Insight", systemImage: "sparkles")
                        .font(.caption)
                        .fontWeight(.medium)
                }
                .buttonStyle(.bordered)
                .tint(.purple)
            }

            chartView
                .frame(height: (config.type == "image_grid" || config.type == "boxplot" || config.type == "wordcloud") ? 340 : 260)
        }
        .padding(20)
        .background(Color.primary.opacity(0.02))
        .cornerRadius(16)
        .overlay(RoundedRectangle(cornerRadius: 16).stroke(Color.primary.opacity(0.05), lineWidth: 1))
    }

    @ViewBuilder
    private var chartView: some View {
        if config.type == "bar" {
            BarChartView(config: config, onTapPoint: onTapPoint)
        } else if config.type == "line" {
            LineChartView(config: config)
        } else if config.type == "scatter" {
            ScatterChartView(config: config)
        } else if config.type == "shap_beeswarm" {
            ShapBeeswarmView(config: config)
        } else if config.type == "pdp_ice" {
            PdpIceView(config: config)
        } else if config.type == "image_grid" {
            ImageGridView(config: config)
        } else if config.type == "boxplot" {
            BoxPlotView(config: config)
        } else if config.type == "wordcloud" {
            WordCloudView(config: config)
        } else {
            Text("Unsupported chart type: \(config.type)")
                .foregroundColor(.secondary)
        }
    }
}

// MARK: - Bar Chart

struct BarChartView: View {
    let config: ChartConfig
    var onTapPoint: ((ChartPoint) -> Void)? = nil

    var body: some View {
        let hasMultipleSeries = config.data.contains(where: { $0.series != nil })

        GeometryReader { geo in
            ScrollView(.horizontal, showsIndicators: true) {
                Chart {
                    ForEach(Array(config.data.enumerated()), id: \.element.id) { index, point in
                        if let xVal = point.xVal {
                            if hasMultipleSeries, let series = point.series {
                                BarMark(x: .value(config.xLabel, xVal), y: .value(config.yLabel, point.y))
                                    .foregroundStyle(by: .value("Series", series))
                                    .cornerRadius(4)
                                    .annotation(position: .top) {
                                        Text(formatValue(point.y))
                                            .font(.system(size: 9, weight: .bold, design: .rounded))
                                            .foregroundColor(.secondary)
                                    }
                            } else {
                                BarMark(x: .value(config.xLabel, xVal), y: .value(config.yLabel, point.y))
                                    .foregroundStyle(barGradient(index: index, total: config.data.count))
                                    .cornerRadius(4)
                                    .annotation(position: .top) {
                                        Text(formatValue(point.y))
                                            .font(.system(size: 9, weight: .bold, design: .rounded))
                                            .foregroundColor(.secondary)
                                    }
                            }
                        } else if let xNum = point.xNum {
                            if hasMultipleSeries, let series = point.series {
                                BarMark(x: .value(config.xLabel, xNum), y: .value(config.yLabel, point.y))
                                    .foregroundStyle(by: .value("Series", series))
                                    .cornerRadius(4)
                                    .annotation(position: .top) {
                                        Text(formatValue(point.y))
                                            .font(.system(size: 9, weight: .bold, design: .rounded))
                                            .foregroundColor(.secondary)
                                    }
                            } else {
                                BarMark(x: .value(config.xLabel, xNum), y: .value(config.yLabel, point.y))
                                    .foregroundStyle(LinearGradient(colors: [.blue, .purple], startPoint: .bottom, endPoint: .top))
                                    .cornerRadius(4)
                                    .annotation(position: .top) {
                                        Text(formatValue(point.y))
                                            .font(.system(size: 9, weight: .bold, design: .rounded))
                                            .foregroundColor(.secondary)
                                    }
                            }
                        }
                    }
                }
                .chartXAxis {
                    AxisMarks()
                }
                .chartYAxis {
                    AxisMarks(position: .leading)
                }
                .chartLegend(hasMultipleSeries ? .visible : .hidden)
                // Removes the empty space on the left by un-anchoring 0
                .chartXScale(domain: .automatic(includesZero: false))
                .frame(width: max(geo.size.width, CGFloat(config.data.count) * (hasMultipleSeries ? 50 : 30)))
                .padding(.all, 8)
                .chartOverlay { proxy in
                    Color.clear
                        .contentShape(Rectangle())
                        .onTapGesture { location in
                            if let xVal = proxy.value(atX: location.x, as: String.self) {
                                if let matchedPoint = config.data.first(where: { $0.xVal == xVal }) {
                                    onTapPoint?(matchedPoint)
                                }
                            } else if let xNum = proxy.value(atX: location.x, as: Double.self) {
                                let matchedPoint = config.data.min(by: { a, b in
                                    let aDiff = abs((a.xNum ?? 0.0) - xNum)
                                    let bDiff = abs((b.xNum ?? 0.0) - xNum)
                                    return aDiff < bDiff
                                })
                                if let matched = matchedPoint {
                                    onTapPoint?(matched)
                                }
                            }
                        }
                }
            }
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

// MARK: - Line Chart

struct LineChartView: View {
    let config: ChartConfig
    
    @State private var viewMode: ViewMode = .raw
    @State private var selectedYear: String = "All"
    
    enum ViewMode: String, CaseIterable, Identifiable {
        case raw = "All Data"
        case monthly = "Month of Year"
        case daily = "Day of Month"
        
        var id: String { rawValue }
    }
    
    private func parseDate(_ string: String) -> Date? {
        let formatters = [
            "yyyy-MM-dd",
            "yyyy-MM-dd HH:mm:ss",
            "yyyy-MM-dd'T'HH:mm:ssZ",
            "yyyy-MM-dd'T'HH:mm:ss.SSSZ",
            "yyyy"
        ]
        let isoFormatter = ISO8601DateFormatter()
        if let date = isoFormatter.date(from: string) {
            return date
        }
        let formatter = DateFormatter()
        for fmt in formatters {
            formatter.dateFormat = fmt
            if let date = formatter.date(from: string) {
                return date
            }
        }
        return nil
    }
    
    private var timeSeriesInfo: (isTS: Bool, years: [String]) {
        var parsedCount = 0
        var yearsSet = Set<String>()
        for pt in config.data {
            if let xVal = pt.xVal, let date = parseDate(xVal) {
                parsedCount += 1
                let year = Calendar.current.component(.year, from: date)
                yearsSet.insert(String(year))
            }
        }
        let isTS = !config.data.isEmpty && parsedCount >= Int(Double(config.data.count) * 0.7)
        return (isTS, yearsSet.sorted())
    }
    
    struct ProcessedPoint: Identifiable {
        let id = UUID()
        let xVal: String?
        let xNum: Double?
        let xDate: Date?
        let y: Double
        let series: String
    }
    
    private func getProcessedPoints() -> [ProcessedPoint] {
        let filtered = config.data.compactMap { pt -> (point: ChartPoint, date: Date?)? in
            if let xVal = pt.xVal, let date = parseDate(xVal) {
                let yearStr = String(Calendar.current.component(.year, from: date))
                if selectedYear == "All" || selectedYear == yearStr {
                    return (pt, date)
                }
                return nil
            }
            if selectedYear == "All" {
                return (pt, nil)
            }
            return nil
        }
        
        switch viewMode {
        case .raw:
            return filtered.map { item in
                ProcessedPoint(
                    xVal: item.point.xVal,
                    xNum: item.point.xNum,
                    xDate: item.date,
                    y: item.point.y,
                    series: item.point.series ?? "Value"
                )
            }
            
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
                    results.append(ProcessedPoint(
                        xVal: monthNames[m - 1],
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
                    results.append(ProcessedPoint(
                        xVal: String(d),
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
        let tsInfo = timeSeriesInfo
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
                    Chart {
                        ForEach(processed) { point in
                            if viewMode == .raw, let date = point.xDate {
                                LineMark(
                                    x: .value(config.xLabel, date),
                                    y: .value(config.yLabel, point.y)
                                )
                                .foregroundStyle(by: .value("Series", point.series))
                                .lineStyle(StrokeStyle(lineWidth: 1.8))
                            } else if viewMode == .monthly, let xVal = point.xVal {
                                LineMark(
                                    x: .value(config.xLabel, xVal),
                                    y: .value(config.yLabel, point.y)
                                )
                                .foregroundStyle(Color.purple)
                                .lineStyle(StrokeStyle(lineWidth: 2.2))
                            } else if viewMode == .daily, let xNum = point.xNum {
                                LineMark(
                                    x: .value(config.xLabel, xNum),
                                    y: .value(config.yLabel, point.y)
                                )
                                .foregroundStyle(Color.indigo)
                                .lineStyle(StrokeStyle(lineWidth: 2.2))
                            }
                        }
                    }
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
                } else {
                    let uniqueXCount = Set(config.data.map { point -> String in
                        if let xVal = point.xVal { return xVal }
                        if let xNum = point.xNum { return String(xNum) }
                        return ""
                    }).count
                    
                    ScrollView(.horizontal, showsIndicators: true) {
                        Chart {
                            ForEach(config.data) { point in
                                if let xVal = point.xVal {
                                    LineMark(x: .value(config.xLabel, xVal), y: .value(config.yLabel, point.y))
                                        .foregroundStyle(by: .value("Series", point.series ?? "Value"))
                                        .lineStyle(StrokeStyle(lineWidth: 2.5))
                                        .interpolationMethod(.catmullRom)
                                } else if let xNum = point.xNum {
                                    LineMark(x: .value(config.xLabel, xNum), y: .value(config.yLabel, point.y))
                                        .foregroundStyle(by: .value("Series", point.series ?? "Value"))
                                        .lineStyle(StrokeStyle(lineWidth: 2.5))
                                        .interpolationMethod(.catmullRom)
                                }
                            }
                        }
                        .chartXAxis {
                            AxisMarks()
                        }
                        .chartYAxis {
                            AxisMarks(position: .leading)
                        }
                        .chartXScale(domain: .automatic(includesZero: false))
                        .chartYScale(domain: .automatic(includesZero: false))
                        .chartLegend(hasMultipleSeries ? .visible : .hidden)
                        .frame(width: max(geo.size.width, CGFloat(uniqueXCount) * 30))
                        .padding(.all, 8)
                    }
                }
            }
        }
    }
}

// MARK: - Scatter Chart

struct ScatterChartView: View {
    let config: ChartConfig

    var body: some View {
        let xValues = config.data.compactMap { $0.xNum }
        let yValues = config.data.map { $0.y }
        
        let isPredictedVsActual = config.title.lowercased().contains("predicted vs actual") ||
                                  config.title.lowercased().contains("actual vs predicted")
        let isResidualPlot = config.title.lowercased().contains("residual")
        
        let hasMultipleSeries = config.data.contains(where: { $0.series != nil })

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

            ForEach(config.data) { point in
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
        }
        .chartXAxis {
            AxisMarks()
        }
        .chartYAxis {
            AxisMarks(position: .leading)
        }
        .chartLegend(hasMultipleSeries ? .visible : .hidden)

        // Apply strict scale bounds dynamically
        if xValues.isEmpty {
            // Categorical X-Axis
            baseChart
                .chartYScale(domain: .automatic(includesZero: false))
        } else {
            // Numeric X-Axis: Calculate exact bounds with a 5% visual padding
            let minX = xValues.min() ?? 0
            let maxX = xValues.max() ?? 1
            let minY = yValues.min() ?? 0
            let maxY = yValues.max() ?? 1
            
            let xPad = max((maxX - minX) * 0.1, 0.1)
            let yPad = max((maxY - minY) * 0.1, 0.1)
            
            baseChart
                .chartXScale(domain: (minX - xPad)...(maxX + xPad))
                .chartYScale(domain: (minY - yPad)...(maxY + yPad))
        }
}
}

// MARK: - SHAP Beeswarm Chart

struct ShapBeeswarmView: View {
    let config: ChartConfig
    
    // Stacking point structure
    struct StackingPoint: Identifiable {
        let id = UUID()
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
        let sortedPoints = points.compactMap { pt -> (x: Double, y: Double)? in
            guard let x = pt.xNum else { return nil }
            return (x: x, y: pt.y)
        }.sorted { $0.x < $1.x }
        
        var stacked: [StackingPoint] = []
        
        let xs = sortedPoints.map { $0.x }
        let xMin = xs.min() ?? 0.0
        let xMax = xs.max() ?? 1.0
        let xRange = xMax - xMin
        let cellWidth = xRange > 0 ? (xRange * 0.02) : 0.015
        let diameter = 0.07 // vertical stacking step
        
        for pt in sortedPoints {
            let neighbors = stacked.filter { abs($0.x - pt.x) < cellWidth }
            
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

// MARK: - PDP & ICE Chart

struct PdpIceView: View {
    let config: ChartConfig
    
    var body: some View {
        Chart {
            ForEach(config.data.filter { $0.series != "PDP" }) { point in
                if let xNum = point.xNum {
                    LineMark(
                        x: .value(config.xLabel, xNum),
                        y: .value(config.yLabel, point.y),
                        series: .value("Series", point.series ?? "")
                    )
                    .foregroundStyle(Color.primary.opacity(0.12))
                    .lineStyle(StrokeStyle(lineWidth: 0.8))
                }
            }
            
            ForEach(config.data.filter { $0.series == "PDP" }) { point in
                if let xNum = point.xNum {
                    LineMark(
                        x: .value(config.xLabel, xNum),
                        y: .value(config.yLabel, point.y),
                        series: .value("Series", "PDP")
                    )
                    .foregroundStyle(Color.purple)
                    .lineStyle(StrokeStyle(lineWidth: 3.0))
                }
            }
        }
        .chartXAxis {
            AxisMarks()
        }
        .chartYAxis {
            AxisMarks(position: .leading)
        }
        .chartXScale(domain: .automatic(includesZero: false))
        .chartYScale(domain: .automatic(includesZero: false))
        .chartLegend(.hidden)
    }
}

// MARK: - Image Grid View

struct ImageGridView: View {
    let config: ChartConfig
    @State private var selectedImage: ImageItem? = nil

    private let columns = [
        GridItem(.adaptive(minimum: 120, maximum: 160), spacing: 16)
    ]

    var body: some View {
        let items = config.images ?? []
        ScrollView {
            LazyVGrid(columns: columns, spacing: 16) {
                ForEach(items) { item in
                    VStack(alignment: .center, spacing: 8) {
                        if let platImg = platformImage(from: item.base64) {
                            #if canImport(AppKit)
                            Image(nsImage: platImg)
                                .resizable()
                                .interpolation(.none) // keeps pixel-art look (nearest neighbor)
                                .aspectRatio(contentMode: .fit)
                                .frame(width: 100, height: 100)
                                .cornerRadius(8)
                                .shadow(radius: 2)
                            #else
                            Image(uiImage: platImg)
                                .resizable()
                                .interpolation(.none)
                                .aspectRatio(contentMode: .fit)
                                .frame(width: 100, height: 100)
                                .cornerRadius(8)
                                .shadow(radius: 2)
                            #endif
                        } else {
                            RoundedRectangle(cornerRadius: 8)
                                .fill(Color.gray.opacity(0.2))
                                .frame(width: 100, height: 100)
                                .overlay(Image(systemName: "photo").foregroundColor(.secondary))
                        }
                        
                        Text(item.label)
                            .font(.caption)
                            .fontWeight(.medium)
                            .lineLimit(2)
                            .multilineTextAlignment(.center)
                            .foregroundColor(.primary)
                    }
                    .padding(8)
                    .background(Color.primary.opacity(0.03))
                    .cornerRadius(12)
                    .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color.primary.opacity(0.05), lineWidth: 1))
                    .onTapGesture {
                        selectedImage = item
                    }
                }
            }
            .padding(.top, 4)
        }
        .sheet(item: $selectedImage) { item in
            LightboxView(item: item)
        }
    }

    private func platformImage(from base64String: String) -> PlatformImage? {
        guard let data = Data(base64Encoded: base64String) else { return nil }
        return PlatformImage(data: data)
    }
}

// MARK: - Lightbox View

struct LightboxView: View {
    let item: ImageItem
    @Environment(\.dismiss) var dismiss

    var body: some View {
        VStack(spacing: 16) {
            HStack {
                Text(item.label)
                    .font(.headline)
                Spacer()
                Button("Close") {
                    dismiss()
                }
                .keyboardShortcut(.cancelAction)
            }
            .padding([.top, .horizontal])

            Divider()

            Spacer()

            if let data = Data(base64Encoded: item.base64),
               let platImg = PlatformImage(data: data) {
                #if canImport(AppKit)
                Image(nsImage: platImg)
                    .resizable()
                    .interpolation(.none)
                    .aspectRatio(contentMode: .fit)
                    .frame(maxWidth: 400, maxHeight: 400)
                    .cornerRadius(8)
                    .shadow(radius: 4)
                #else
                Image(uiImage: platImg)
                    .resizable()
                    .interpolation(.none)
                    .aspectRatio(contentMode: .fit)
                    .frame(maxWidth: 400, maxHeight: 400)
                    .cornerRadius(8)
                    .shadow(radius: 4)
                #endif
            } else {
                Image(systemName: "exclamationmark.triangle")
                    .font(.largeTitle)
                    .foregroundColor(.red)
                Text("Failed to decode image data.")
            }

            Spacer()
        }
        .frame(minWidth: 450, minHeight: 480)
        .padding(.bottom)
    }
}

// MARK: - Box Plot View

struct BoxPlotView: View {
    let config: ChartConfig

    var body: some View {
        if let stats = config.boxStats {
            GeometryReader { geo in
                Chart {
                    // Box from Q1 to Q3
                    RectangleMark(
                        xStart: .value("Left", 0.8),
                        xEnd: .value("Right", 1.2),
                        yStart: .value("Q1", stats.q1),
                        yEnd: .value("Q3", stats.q3)
                    )
                    .foregroundStyle(Color.purple.opacity(0.25))
                    
                    // Bottom border of box
                    RuleMark(
                        xStart: .value("Left", 0.8),
                        xEnd: .value("Right", 1.2),
                        y: .value("Q1", stats.q1)
                    )
                    .foregroundStyle(Color.purple)
                    .lineStyle(StrokeStyle(lineWidth: 1.5))
                    
                    // Top border of box
                    RuleMark(
                        xStart: .value("Left", 0.8),
                        xEnd: .value("Right", 1.2),
                        y: .value("Q3", stats.q3)
                    )
                    .foregroundStyle(Color.purple)
                    .lineStyle(StrokeStyle(lineWidth: 1.5))

                    // Whiskers (vertical rules at x = 1.0)
                    RuleMark(
                        x: .value("X", 1.0),
                        yStart: .value("Min", stats.min),
                        yEnd: .value("Q1", stats.q1)
                    )
                    .lineStyle(StrokeStyle(lineWidth: 1.5, dash: [4, 4]))
                    .foregroundStyle(Color.purple)
                    
                    RuleMark(
                        x: .value("X", 1.0),
                        yStart: .value("Q3", stats.q3),
                        yEnd: .value("Max", stats.max)
                    )
                    .lineStyle(StrokeStyle(lineWidth: 1.5, dash: [4, 4]))
                    .foregroundStyle(Color.purple)
                    
                    // Whisker caps (horizontal rules)
                    RuleMark(
                        xStart: .value("Left", 0.9),
                        xEnd: .value("Right", 1.1),
                        y: .value("Min", stats.min)
                    )
                    .foregroundStyle(Color.purple)
                    .lineStyle(StrokeStyle(lineWidth: 1.5))
                    
                    RuleMark(
                        xStart: .value("Left", 0.9),
                        xEnd: .value("Right", 1.1),
                        y: .value("Max", stats.max)
                    )
                    .foregroundStyle(Color.purple)
                    .lineStyle(StrokeStyle(lineWidth: 1.5))
                    
                    // Median line (horizontal rule at x = 1.0)
                    RuleMark(
                        xStart: .value("Left", 0.8),
                        xEnd: .value("Right", 1.2),
                        y: .value("Median", stats.median)
                    )
                    .foregroundStyle(Color.purple)
                    .lineStyle(StrokeStyle(lineWidth: 2.5))
                    
                    // Outliers (points at x = 1.0)
                    ForEach(stats.outliers, id: \.self) { val in
                        PointMark(
                            x: .value("X", 1.0),
                            y: .value("Outlier", val)
                        )
                        .foregroundStyle(Color.red)
                        .symbolSize(20)
                    }
                }
                .chartXAxis {
                    AxisMarks(values: [1.0]) { value in
                        AxisValueLabel {
                            Text(config.yLabel)
                        }
                    }
                }
                .chartYAxis {
                    AxisMarks(position: .leading)
                }
                .chartXScale(domain: 0.5...1.5)
                .chartYScale(domain: .automatic(includesZero: false))
                .padding(.all, 8)
            }
        } else {
            Text("No box stats available.")
                .foregroundColor(.secondary)
        }
    }
}

// MARK: - Word Cloud View

struct WordCloudView: View {
    let config: ChartConfig
    
    @State private var hoveredWord: String? = nil
    
    private let colors: [Color] = [
        .blue, .purple, .pink, .indigo, .teal, .orange, .cyan, .mint
    ]
    
    var body: some View {
        let maxWeight = config.data.map { $0.y }.max() ?? 1.0
        let minWeight = config.data.map { $0.y }.min() ?? 0.0
        let weightRange = max(0.0001, maxWeight - minWeight)
        
        ScrollView(.vertical, showsIndicators: true) {
            WordCloudLayout {
                ForEach(Array(config.data.enumerated()), id: \.element.id) { index, point in
                    if let word = point.xVal {
                        let normalizedWeight = (point.y - minWeight) / weightRange
                        let size = 12 + normalizedWeight * 20 // size from 12 to 32
                        
                        Text(word)
                            .font(.system(size: size, weight: .semibold, design: .rounded))
                            .foregroundColor(colors[index % colors.count])
                            .padding(.horizontal, 6)
                            .padding(.vertical, 3)
                            .background(
                                RoundedRectangle(cornerRadius: 6)
                                    .fill(colors[index % colors.count].opacity(hoveredWord == word ? 0.15 : 0.03))
                            )
                            .scaleEffect(hoveredWord == word ? 1.08 : 1.0)
                            .animation(.spring(response: 0.2, dampingFraction: 0.6), value: hoveredWord)
                            .onHover { isHovered in
                                hoveredWord = isHovered ? word : nil
                            }
                            .help(String(format: "TF-IDF Weight: %.4f", point.y))
                    }
                }
            }
            .padding(10)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color.primary.opacity(0.01))
        .cornerRadius(12)
    }
}

// MARK: - Word Cloud Custom Layout

struct WordCloudLayout: Layout {
    var spacing: CGFloat = 8
    
    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let width = proposal.width ?? 400
        var currentX: CGFloat = 0
        var currentY: CGFloat = 0
        var lineHeight: CGFloat = 0
        var maxWidth: CGFloat = 0
        
        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if currentX + size.width > width {
                maxWidth = max(maxWidth, currentX)
                currentX = 0
                currentY += lineHeight + spacing
                lineHeight = 0
            }
            currentX += size.width + spacing
            lineHeight = max(lineHeight, size.height)
        }
        maxWidth = max(maxWidth, currentX)
        return CGSize(width: maxWidth, height: currentY + lineHeight)
    }
    
    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        var currentX: CGFloat = bounds.minX
        var currentY: CGFloat = bounds.minY
        var lineHeight: CGFloat = 0
        
        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if currentX + size.width > bounds.maxX {
                currentX = bounds.minX
                currentY += lineHeight + spacing
                lineHeight = 0
            }
            subview.place(
                at: CGPoint(x: currentX, y: currentY),
                proposal: ProposedViewSize(size)
            )
            currentX += size.width + spacing
            lineHeight = max(lineHeight, size.height)
        }
    }
}

