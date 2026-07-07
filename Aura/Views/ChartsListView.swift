import SwiftUI
import Charts

struct DrillDownItem: Identifiable {
    var id: String { title }
    let title: String
    let preview: FullTablePreview
}

struct CategoryTabButton: View {
    let title: String
    let iconName: String?
    let isSelected: Bool
    let action: () -> Void
    
    @State private var isHovered = false
    
    var body: some View {
        Button(action: action) {
            HStack(spacing: 5) {
                if let icon = iconName {
                    Image(systemName: icon)
                        .font(.system(size: 10, weight: .bold))
                }
                Text(title)
                    .font(.system(size: 11, weight: .semibold))
            }
            .foregroundColor(isSelected || isHovered ? .white : .secondary)
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(
                Capsule()
                    .fill(isSelected || isHovered ? Color.accentColor : Color.primary.opacity(0.04))
            )
            .overlay(
                Capsule()
                    .stroke(isSelected || isHovered ? Color.accentColor : Color.primary.opacity(0.08), lineWidth: 1)
            )
        }
        .buttonStyle(.plain)
        .onHover { hovering in
            withAnimation(.easeInOut(duration: 0.12)) {
                isHovered = hovering
            }
        }
    }
}

struct ChartsListView: View {
    let result: AnalysisResult
    let onAskAI: (String) -> Void

    @State private var drillDownPreview: FullTablePreview? = nil
    @State private var drillDownTitle: String = ""
    @State private var selectedCategory: String = "All"
    @State private var searchText: String = ""

    var body: some View {
        ScrollViewReader { proxy in
            VStack(spacing: 0) {
                // Category Filter Picker
                HStack(spacing: 12) {
                    Label("Jump to:", systemImage: "arrow.right.circle.fill")
                        .font(.system(size: 11, weight: .bold))
                        .foregroundColor(.secondary)
                        .padding(.trailing, 4)
                    
                    CategoryTabButton(title: "All Plots", iconName: "circle.grid.2x2.fill", isSelected: selectedCategory == "All") {
                        selectedCategory = "All"
                        withAnimation(.easeInOut(duration: 0.25)) {
                            proxy.scrollTo("top", anchor: .top)
                        }
                    }
                    
                    CategoryTabButton(title: "Model Quality", iconName: "bolt.badge.a.fill", isSelected: selectedCategory == "Model Quality") {
                        selectedCategory = "Model Quality"
                        withAnimation(.easeInOut(duration: 0.25)) {
                            proxy.scrollTo("Model Quality", anchor: .top)
                        }
                    }
                    
                    CategoryTabButton(title: "Feature Importance", iconName: "sparkles", isSelected: selectedCategory == "Feature Importance") {
                        selectedCategory = "Feature Importance"
                        withAnimation(.easeInOut(duration: 0.25)) {
                            proxy.scrollTo("Feature Importance", anchor: .top)
                        }
                    }
                    
                    CategoryTabButton(title: "Data Distributions", iconName: "chart.bar.fill", isSelected: selectedCategory == "Data Distributions") {
                        selectedCategory = "Data Distributions"
                        withAnimation(.easeInOut(duration: 0.25)) {
                            proxy.scrollTo("Data Distributions", anchor: .top)
                        }
                    }
                    
                    Spacer()
                }
                .padding(.horizontal, 20)
                .padding(.vertical, 12)
                .background(Color(NSColor.controlBackgroundColor))
                
                Divider()
                
                ScrollView {
                    LazyVStack(spacing: 28) {
                        Color.clear
                            .frame(height: 1)
                            .id("top")
                        
                        let modelCharts = result.charts.filter { self.category(for: $0) == "Model Quality" && self.matchesSearch($0) }
                        let featureCharts = result.charts.filter { self.category(for: $0) == "Feature Importance" && self.matchesSearch($0) }
                        let dataCharts = result.charts.filter { self.category(for: $0) == "Data Distributions" && self.matchesSearch($0) }
                        
                        if modelCharts.isEmpty && featureCharts.isEmpty && dataCharts.isEmpty {
                            VStack(spacing: 12) {
                                Image(systemName: "chart.bar.doc.horizontal")
                                    .font(.system(size: 48))
                                    .foregroundColor(.secondary)
                                Text("No charts match your search.")
                                    .foregroundColor(.secondary)
                                    .font(.headline)
                            }
                            .frame(maxWidth: .infinity, minHeight: 300)
                        } else {
                            if !modelCharts.isEmpty {
                                VStack(alignment: .leading, spacing: 12) {
                                    HStack {
                                        Label("Model Quality", systemImage: "bolt.badge.a.fill")
                                            .font(.title3.bold())
                                            .foregroundColor(.purple)
                                        Spacer()
                                    }
                                    .id("Model Quality")
                                    .padding(.top, 8)
                                    
                                    LazyVStack(spacing: 12) {
                                        ForEach(modelCharts) { chartConfig in
                                            ChartCard(config: chartConfig, onAskAI: onAskAI) { point in
                                                handleDrillDown(chartConfig: chartConfig, point: point)
                                            }
                                        }
                                    }
                                }
                            }
                            
                            if !featureCharts.isEmpty {
                                VStack(alignment: .leading, spacing: 12) {
                                    HStack {
                                        Label("Feature Importance", systemImage: "sparkles")
                                            .font(.title3.bold())
                                            .foregroundColor(.purple)
                                        Spacer()
                                    }
                                    .id("Feature Importance")
                                    .padding(.top, 8)
                                    
                                    LazyVStack(spacing: 12) {
                                        ForEach(featureCharts) { chartConfig in
                                            ChartCard(config: chartConfig, onAskAI: onAskAI) { point in
                                                handleDrillDown(chartConfig: chartConfig, point: point)
                                            }
                                        }
                                    }
                                }
                            }
                            
                            if !dataCharts.isEmpty {
                                VStack(alignment: .leading, spacing: 12) {
                                    HStack {
                                        Label("Data Distributions", systemImage: "chart.bar.fill")
                                            .font(.title3.bold())
                                            .foregroundColor(.purple)
                                        Spacer()
                                    }
                                    .id("Data Distributions")
                                    .padding(.top, 8)
                                    
                                    LazyVStack(spacing: 12) {
                                        ForEach(dataCharts) { chartConfig in
                                            ChartCard(config: chartConfig, onAskAI: onAskAI) { point in
                                                handleDrillDown(chartConfig: chartConfig, point: point)
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    .padding()
                }
            }
        }
        .searchable(text: $searchText, placement: .sidebar, prompt: "Search charts...")
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
        if lowerTitle.contains("k-means") || lowerTitle.contains("kmeans") {
            return "K-Means Cluster"
        }
        if lowerTitle.contains("dbscan") {
            return "DBSCAN Cluster"
        }
        if lowerTitle.contains("hdbscan") {
            return "HDBSCAN Cluster"
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

    private func matchesSearch(_ chart: ChartConfig) -> Bool {
        searchText.isEmpty ||
            chart.title.localizedCaseInsensitiveContains(searchText) ||
            chart.xLabel.localizedCaseInsensitiveContains(searchText) ||
            chart.yLabel.localizedCaseInsensitiveContains(searchText)
    }

    private func category(for chart: ChartConfig) -> String {
        let title = chart.title.lowercased()
        
        if title.contains("learning curve") || title.contains("roc") || title.contains("precision-recall") ||
           title.contains("confusion") || title.contains("residual") || title.contains("forecast") ||
           title.contains("actual vs") || title.contains("prediction") || title.contains("pdp") ||
           title.contains("ice") || title.contains("calibration") || title.contains("quality") {
            return "Model Quality"
        }
        
        if title.contains("importance") || title.contains("shap") || title.contains("beeswarm") ||
           title.contains("coefficient") || title.contains("weight") || title.contains("tf-idf") {
            return "Feature Importance"
        }
        
        return "Data Distributions"
    }
}

// MARK: - Chart AI Prompt Builder

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
