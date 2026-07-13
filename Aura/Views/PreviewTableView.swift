import SwiftUI
import WebKit

// MARK: - Preview Table with Type Selector + Column Selection

struct FilterColumnWrapper: Identifiable {
    var id: String { name }
    let name: String
}

struct PreviewTableView: View {
    let preview: DatasetPreview

    @Binding var config: AnalysisConfig
    var onPreviewFileRequested: ((String) -> Void)? = nil
    var onRefreshPreview: (() -> Void)? = nil
    var isSidebar: Bool = false

    @State private var isShowingStartCalendar = false
    @State private var isShowingEndCalendar = false
    @State private var filteringColumn: FilterColumnWrapper? = nil

    private let colWidth: CGFloat = 148

    private func parseDateString(_ str: String) -> Date? {
        let formats = [
            "yyyy-MM-dd",
            "yyyy-MM-dd HH:mm:ss",
            "yyyy/MM/dd",
            "MM/dd/yyyy",
            "dd-MM-yyyy"
        ]
        let df = DateFormatter()
        for fmt in formats {
            df.dateFormat = fmt
            if let date = df.date(from: str) {
                return date
            }
        }
        return nil
    }

    private var timeRangeStartDateBinding: Binding<Date> {
        Binding(
            get: {
                if let startStr = config.timeRangeStart, let date = parseDateString(startStr) {
                    return date
                }
                if let timeColumn = config.timeColumn,
                   let datetimeRange = preview.datetimeRange,
                   let bounds = datetimeRange[timeColumn],
                   let date = parseDateString(bounds.min) {
                    return date
                }
                return Date()
            },
            set: { newDate in
                let df = DateFormatter()
                df.dateFormat = "yyyy-MM-dd"
                config.timeRangeStart = df.string(from: newDate)
            }
        )
    }

    private var timeRangeEndDateBinding: Binding<Date> {
        Binding(
            get: {
                if let endStr = config.timeRangeEnd, let date = parseDateString(endStr) {
                    return date
                }
                if let timeColumn = config.timeColumn,
                   let datetimeRange = preview.datetimeRange,
                   let bounds = datetimeRange[timeColumn],
                   let date = parseDateString(bounds.max) {
                    return date
                }
                return Date()
            },
            set: { newDate in
                let df = DateFormatter()
                df.dateFormat = "yyyy-MM-dd"
                config.timeRangeEnd = df.string(from: newDate)
            }
        )
    }

    private var dateRangeLimit: ClosedRange<Date> {
        let defaultMin = Calendar.current.date(byAdding: .year, value: -100, to: Date()) ?? Date.distantPast
        let defaultMax = Calendar.current.date(byAdding: .year, value: 100, to: Date()) ?? Date.distantFuture
        
        guard let timeColumn = config.timeColumn,
              let datetimeRange = preview.datetimeRange,
              let bounds = datetimeRange[timeColumn] else {
            return defaultMin...defaultMax
        }
        
        let minDate = parseDateString(bounds.min) ?? defaultMin
        let maxDate = parseDateString(bounds.max) ?? defaultMax
        
        if minDate <= maxDate {
            return minDate...maxDate
        } else {
            return defaultMin...defaultMax
        }
    }

    private func formatNumber(_ value: Int) -> String {
        let formatter = NumberFormatter()
        formatter.numberStyle = .decimal
        formatter.groupingSeparator = Locale.current.groupingSeparator
        return formatter.string(from: NSNumber(value: value)) ?? String(value)
    }

    var body: some View {
        // FIX: Replaced the rigid outer VStack with a master ScrollView.
        // This guarantees the view will never push the action bar off the screen.
        ScrollView(.vertical, showsIndicators: true) {
            VStack(alignment: .leading, spacing: 0) {
                // ── Header ─────────────────────────────────────────────────────
                headerSection
                    .padding(.horizontal, isSidebar ? 12 : 20)
                    .padding(.top, isSidebar ? 10 : 16)
                    .padding(.bottom, isSidebar ? 8 : 12)

                Divider().background(Color.primary.opacity(0.06))

                // Optional: time-column picker for Time Series
                if config.datasetType == .timeSeries {
                    Divider().background(Color.primary.opacity(0.06))
                    
                    timeColumnPicker
                        .padding(.horizontal, isSidebar ? 12 : 20)
                        .padding(.vertical, isSidebar ? 8 : 12)
                    
                    Divider().background(Color.primary.opacity(0.06))
                    
                    dateRangePickerSection
                        .padding(.horizontal, isSidebar ? 12 : 20)
                        .padding(.vertical, isSidebar ? 8 : 12)
                }

                // ── Exclusion Banner ────────────────────────────────────────────
                if !config.excludedColumns.isEmpty {
                    Divider().background(Color.primary.opacity(0.06))
                    
                    exclusionBanner
                        .padding(.horizontal, isSidebar ? 12 : 20)
                        .padding(.vertical, isSidebar ? 8 : 12)
                }

                Divider().background(Color.primary.opacity(0.06))

                // ── Data Table (Horizontal Scroll Only) ───────────────────────────
                ScrollView(.horizontal, showsIndicators: true) {
                    VStack(alignment: .leading, spacing: 0) {
                        tableHeader
                            .padding(.horizontal, isSidebar ? 12 : 20)
                        
                        // FIX: Removed the nested vertical ScrollView.
                        // The table now flows perfectly with the master page scroll.
                        LazyVStack(alignment: .leading, spacing: 0) {
                            tableRows
                        }
                        .padding(.horizontal, isSidebar ? 12 : 20)
                    }
                }
                .padding(.top, 16)
                .padding(.bottom, 32)
            }
        }
        .popover(item: $filteringColumn) { colWrapper in
            filterPopoverView(for: colWrapper.name)
        }
    }

    // MARK: - Header Section

    private var headerSection: some View {
        HStack(alignment: .center, spacing: 12) {
            VStack(alignment: .leading, spacing: 3) {
                Text(isSidebar ? "Dataset Schema & Preview" : "Dataset Preview")
                    .font(isSidebar ? .system(size: 13, weight: .bold) : .title3.bold())
                if let total = preview.totalRows {
                    Text(isSidebar ? "\(preview.columns.count) cols • \(formatNumber(total)) rows" : "First \(preview.previewRows.count) rows of \(formatNumber(total)) total • \(preview.columns.count) columns")
                        .font(.caption)
                        .foregroundColor(.secondary)
                } else {
                    Text(isSidebar ? "\(preview.columns.count) cols • \(preview.previewRows.count)+ rows" : "First \(preview.previewRows.count) rows • \(preview.columns.count) columns")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }

            Spacer()

            if !isSidebar {
                HStack(spacing: 8) {
                    Button(action: {
                        let isDeduplicated = config.cleaningActions.contains(where: { $0.column == "all" && $0.actionType == "remove_duplicates" })
                        if isDeduplicated {
                            config.cleaningActions = config.cleaningActions.filter { !($0.column == "all" && $0.actionType == "remove_duplicates") }
                        } else {
                            config.cleaningActions.insert(CleaningAction(column: "all", actionType: "remove_duplicates"))
                        }
                        onRefreshPreview?()
                    }) {
                        let isDeduplicated = config.cleaningActions.contains(where: { $0.column == "all" && $0.actionType == "remove_duplicates" })
                        Label("Remove Duplicate Rows", systemImage: isDeduplicated ? "checkmark.circle.fill" : "circle")
                            .font(.system(size: 11, weight: .medium))
                            .foregroundColor(isDeduplicated ? .purple : .secondary)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 5)
                            .background(isDeduplicated ? Color.purple.opacity(0.12) : Color.primary.opacity(0.04))
                            .cornerRadius(8)
                    }
                    .buttonStyle(.plain)
                    
                    if !preview.localPath.isEmpty {
                        Label("Cached Locally", systemImage: "checkmark.circle.fill")
                            .font(.caption2.bold())
                            .foregroundColor(.green)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 5)
                            .background(Color.green.opacity(0.1))
                            .cornerRadius(20)
                    }
                }
            }
        }
    }

    // MARK: - Dataset Type Selector

    private var datasetTypeSelector: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Dataset Type", systemImage: "square.stack.3d.up")
                .font(.system(size: 10, weight: .bold, design: .rounded))
                .foregroundColor(.secondary.opacity(0.7))
                .tracking(0.5)

            HStack(spacing: 8) {
                ForEach(DatasetType.allCases) { dtype in
                    TypePill(
                        type: dtype,
                        isSelected: config.datasetType == dtype
                    ) {
                        withAnimation(.easeInOut(duration: 0.18)) {
                            config.datasetType = dtype
                        }
                    }
                }
            }

            // Description of the selected type
            Text(config.datasetType.description)
                .font(.system(size: 11))
                .foregroundColor(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var smartSamplingSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Smart Sampling (Large Datasets)", systemImage: "sparkles.rectangle.stack")
                .font(.system(size: 10, weight: .bold, design: .rounded))
                .foregroundColor(.secondary.opacity(0.7))
                .tracking(0.5)
            
            HStack(spacing: 12) {
                Toggle(isOn: $config.smartSample) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Enable Smart Sampling")
                            .font(.system(size: 12, weight: .semibold))
                        Text("Automatically sample classification/regression datasets down to 100,000 rows to prevent memory limits and speed up fitting.")
                            .font(.system(size: 10))
                            .foregroundColor(.secondary)
                    }
                }
                .toggleStyle(.checkbox)
                
                Spacer()
                
                if config.smartSample {
                    Text("ACTIVE")
                        .font(.system(size: 8, weight: .bold))
                        .foregroundColor(.white)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2.5)
                        .background(Capsule().fill(Color.purple))
                }
            }
            .padding(10)
            .background(Color.primary.opacity(0.02))
            .cornerRadius(8)
            .overlay(RoundedRectangle(cornerRadius: 8).stroke(config.smartSample ? Color.purple.opacity(0.2) : Color.primary.opacity(0.06), lineWidth: 1))
        }
    }

    // MARK: - Time Column Picker

    private var timeColumnPicker: some View {
        VStack(alignment: .leading, spacing: 6) {
            Label("Datetime Column", systemImage: "calendar.badge.clock")
                .font(.system(size: 10, weight: .bold, design: .rounded))
                .foregroundColor(.secondary.opacity(0.7))
                .tracking(0.5)

            Menu {
                Button("Auto-detect") { config.timeColumn = nil }
                Divider()
                ForEach(preview.columns, id: \.self) { col in
                    Button(col) { config.timeColumn = col }
                }
            } label: {
                HStack {
                    Text(config.timeColumn ?? "Auto-detect")
                        .font(.system(size: 12))
                        .foregroundColor(config.timeColumn == nil ? .secondary : .primary)
                    Spacer()
                    Image(systemName: "chevron.up.chevron.down")
                        .font(.system(size: 9))
                        .foregroundColor(.secondary)
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 7)
                .background(Color.blue.opacity(0.06))
                .cornerRadius(7)
                .overlay(RoundedRectangle(cornerRadius: 7).stroke(Color.blue.opacity(0.2), lineWidth: 1))
            }
            .menuStyle(.borderlessButton)
        }
    }
    
    private var dateRangePickerSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Date Range Filter (Optional)", systemImage: "calendar")
                .font(.system(size: 10, weight: .bold, design: .rounded))
                .foregroundColor(.secondary.opacity(0.7))
                .tracking(0.5)
            
            HStack(spacing: 12) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Start Date")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundColor(.secondary)
                    
                    HStack(spacing: 4) {
                        TextField("YYYY-MM-DD", text: Binding(
                            get: { self.config.timeRangeStart ?? "" },
                            set: { self.config.timeRangeStart = $0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : $0 }
                        ))
                        .textFieldStyle(.plain)
                        .font(.system(size: 11, design: .monospaced))
                        
                        Button(action: { isShowingStartCalendar = true }) {
                            Image(systemName: "calendar")
                                .font(.system(size: 11))
                                .foregroundColor(.secondary)
                                .padding(.horizontal, 4)
                                .padding(.vertical, 2)
                                .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                        .popover(isPresented: $isShowingStartCalendar, arrowEdge: .bottom) {
                            VStack(spacing: 8) {
                                DatePicker(
                                    "",
                                    selection: timeRangeStartDateBinding,
                                    in: dateRangeLimit,
                                    displayedComponents: .date
                                )
                                .datePickerStyle(.graphical)
                                .labelsHidden()
                                .frame(width: 250, height: 250)
                                
                                Button("Clear Filter") {
                                    config.timeRangeStart = nil
                                    isShowingStartCalendar = false
                                }
                                .font(.system(size: 11, weight: .semibold))
                                .foregroundColor(.red)
                                .buttonStyle(.plain)
                                .padding(.bottom, 8)
                            }
                            .padding(8)
                        }
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 5)
                    .background(Color.primary.opacity(0.04))
                    .cornerRadius(5)
                    .overlay(RoundedRectangle(cornerRadius: 5).stroke(Color.primary.opacity(0.1), lineWidth: 1))
                }
                
                VStack(alignment: .leading, spacing: 4) {
                    Text("End Date")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundColor(.secondary)
                    
                    HStack(spacing: 4) {
                        TextField("YYYY-MM-DD", text: Binding(
                            get: { self.config.timeRangeEnd ?? "" },
                            set: { self.config.timeRangeEnd = $0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : $0 }
                        ))
                        .textFieldStyle(.plain)
                        .font(.system(size: 11, design: .monospaced))
                        
                        Button(action: { isShowingEndCalendar = true }) {
                            Image(systemName: "calendar")
                                .font(.system(size: 11))
                                .foregroundColor(.secondary)
                                .padding(.horizontal, 4)
                                .padding(.vertical, 2)
                                .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                        .popover(isPresented: $isShowingEndCalendar, arrowEdge: .bottom) {
                            VStack(spacing: 8) {
                                DatePicker(
                                    "",
                                    selection: timeRangeEndDateBinding,
                                    in: dateRangeLimit,
                                    displayedComponents: .date
                                )
                                .datePickerStyle(.graphical)
                                .labelsHidden()
                                .frame(width: 250, height: 250)
                                
                                Button("Clear Filter") {
                                    config.timeRangeEnd = nil
                                    isShowingEndCalendar = false
                                }
                                .font(.system(size: 11, weight: .semibold))
                                .foregroundColor(.red)
                                .buttonStyle(.plain)
                                .padding(.bottom, 8)
                            }
                            .padding(8)
                        }
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 5)
                    .background(Color.primary.opacity(0.04))
                    .cornerRadius(5)
                    .overlay(RoundedRectangle(cornerRadius: 5).stroke(Color.primary.opacity(0.1), lineWidth: 1))
                }
            }
            
            if let hint = dateRangeRangeHint {
                HStack(spacing: 6) {
                    Text(hint)
                        .font(.system(size: 10, weight: .medium))
                        .foregroundColor(.indigo)
                    
                    Button(action: applyEstimatedDateRange) {
                        Text("Apply Range")
                            .font(.system(size: 9, weight: .bold))
                            .foregroundColor(.white)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(Color.indigo)
                            .cornerRadius(4)
                    }
                    .buttonStyle(.plain)
                }
                .padding(.top, 2)
            }
        }
    }

    private var dateRangeRangeHint: String? {
        guard let timeColumn = config.timeColumn else { return nil }
        
        // Use full dataset range if available from Python
        if let datetimeRange = preview.datetimeRange, let bounds = datetimeRange[timeColumn] {
            return "Dataset range: \(bounds.min) to \(bounds.max)"
        }
        
        guard let colIndex = preview.columns.firstIndex(of: timeColumn) else { return nil }
        
        let formatters = [
            "yyyy-MM-dd",
            "yyyy-MM-dd HH:mm:ss",
            "yyyy/MM/dd",
            "MM/dd/yyyy",
            "dd-MM-yyyy"
        ].map { fmt -> DateFormatter in
            let df = DateFormatter()
            df.dateFormat = fmt
            return df
        }
        
        var dates: [Date] = []
        for row in preview.previewRows {
            guard colIndex < row.count else { continue }
            let valStr = row[colIndex].displayString.trimmingCharacters(in: .whitespacesAndNewlines)
            if valStr.isEmpty { continue }
            
            var parsedDate: Date? = nil
            for formatter in formatters {
                if let date = formatter.date(from: valStr) {
                    parsedDate = date
                    break
                }
            }
            if let date = parsedDate {
                dates.append(date)
            }
        }
        
        guard !dates.isEmpty else { return nil }
        let sortedDates = dates.sorted()
        if let minDate = sortedDates.first, let maxDate = sortedDates.last {
            let df = DateFormatter()
            df.dateFormat = "yyyy-MM-dd"
            return "Estimated range: \(df.string(from: minDate)) to \(df.string(from: maxDate))"
        }
        return nil
    }

    private func applyEstimatedDateRange() {
        guard let timeColumn = config.timeColumn else { return }
        
        // Use full dataset range if available from Python
        if let datetimeRange = preview.datetimeRange, let bounds = datetimeRange[timeColumn] {
            config.timeRangeStart = bounds.min
            config.timeRangeEnd = bounds.max
            return
        }
        
        guard let colIndex = preview.columns.firstIndex(of: timeColumn) else { return }
        
        let formatters = [
            "yyyy-MM-dd",
            "yyyy-MM-dd HH:mm:ss",
            "yyyy/MM/dd",
            "MM/dd/yyyy",
            "dd-MM-yyyy"
        ].map { fmt -> DateFormatter in
            let df = DateFormatter()
            df.dateFormat = fmt
            return df
        }
        
        var dates: [Date] = []
        for row in preview.previewRows {
            guard colIndex < row.count else { continue }
            let valStr = row[colIndex].displayString.trimmingCharacters(in: .whitespacesAndNewlines)
            if valStr.isEmpty { continue }
            
            var parsedDate: Date? = nil
            for formatter in formatters {
                if let date = formatter.date(from: valStr) {
                    parsedDate = date
                    break
                }
            }
            if let date = parsedDate {
                dates.append(date)
            }
        }
        
        guard !dates.isEmpty else { return }
        let sortedDates = dates.sorted()
        if let minDate = sortedDates.first, let maxDate = sortedDates.last {
            let df = DateFormatter()
            df.dateFormat = "yyyy-MM-dd"
            config.timeRangeStart = df.string(from: minDate)
            config.timeRangeEnd = df.string(from: maxDate)
        }
    }
    
    private var uniqueDatesInDataset: [String] {
        guard let timeColumn = config.timeColumn else { return [] }
        
        var valuesSet = Set<String>()
        
        // Inject min and max of entire dataset
        if let datetimeRange = preview.datetimeRange, let bounds = datetimeRange[timeColumn] {
            valuesSet.insert(bounds.min)
            valuesSet.insert(bounds.max)
        }
        
        if let colIndex = preview.columns.firstIndex(of: timeColumn) {
            for row in preview.previewRows {
                guard colIndex < row.count else { continue }
                let valStr = row[colIndex].displayString.trimmingCharacters(in: .whitespacesAndNewlines)
                if !valStr.isEmpty && valStr != "—" {
                    valuesSet.insert(valStr)
                }
            }
        }
        
        let formatters = [
            "yyyy-MM-dd",
            "yyyy-MM-dd HH:mm:ss",
            "yyyy/MM/dd",
            "MM/dd/yyyy",
            "dd-MM-yyyy"
        ].map { fmt -> DateFormatter in
            let df = DateFormatter()
            df.dateFormat = fmt
            return df
        }
        
        return valuesSet.sorted { val1, val2 in
            var d1: Date? = nil
            var d2: Date? = nil
            for formatter in formatters {
                if d1 == nil { d1 = formatter.date(from: val1) }
                if d2 == nil { d2 = formatter.date(from: val2) }
            }
            if let d1 = d1, let d2 = d2 {
                return d1 < d2
            }
            return val1 < val2
        }
    }


    // MARK: - Exclusion Banner

    private var exclusionBanner: some View {
        HStack(spacing: 8) {
            Image(systemName: "eye.slash.fill")
                .foregroundColor(.orange)
                .font(Theme.Font.caption)
            Text("\(config.excludedColumns.count) column\(config.excludedColumns.count == 1 ? "" : "s") excluded from analysis")
                .font(Theme.Font.captionBold)
                .foregroundColor(.orange)
            Spacer()
            Button("Clear All") {
                withAnimation { config.excludedColumns.removeAll() }
            }
            .font(.system(size: 10, weight: .semibold))
            .foregroundColor(.orange.opacity(0.8))
            .buttonStyle(.plain)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 7)
        .background(Color.orange.opacity(0.08))
        .cornerRadius(8)
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.orange.opacity(0.2), lineWidth: 1))
    }

    private var detectedTargetColumn: String {
        if !config.targetColumn.isEmpty {
            return config.targetColumn
        }
        let targetKeywords = ["target", "label", "class", "price", "outcome", "y", "survived"]
        for col in preview.columns {
            let lower = col.lowercased()
            if targetKeywords.contains(where: { lower.contains($0) }) {
                return col
            }
        }
        return preview.columns.last ?? ""
    }

    private func isColumnTarget(_ col: String) -> Bool {
        if !config.targetColumns.isEmpty {
            return config.targetColumns.contains(where: { $0.lowercased() == col.lowercased() })
        }
        return col.lowercased() == detectedTargetColumn.lowercased()
    }
    
    private func columnTypeDisplayString(for col: String) -> String {
        let inferredType = preview.columnTypes?[col] ?? "categorical"
        let activeType = config.columnTypeOverrides[col] ?? inferredType
        switch activeType {
        case "numeric": return "Numeric"
        case "categorical": return "Categorical"
        case "text": return "Text / NLP"
        case "datetime": return "Datetime"
        case "identifier": return "Identifier"
        default: return activeType.capitalized
        }
    }

    // MARK: - Table Header Row

    private var tableHeader: some View {
        HStack(spacing: 0) {
            ForEach(0..<preview.columns.count, id: \.self) { colIndex in
                let col = preview.columns[colIndex]
                let isExcluded = config.excludedColumns.contains(col)
                let isTarget = isColumnTarget(col)
                let columnValues: [PreviewValue] = preview.previewRows.map { rowIndex in
                    if colIndex < rowIndex.count {
                        return rowIndex[colIndex]
                    }
                    return .null
                }
                
                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 6) {
                        if isTarget {
                            Text("TARGET")
                                .font(.system(size: 8, weight: .bold, design: .rounded))
                                .foregroundColor(.purple)
                                .padding(.horizontal, 4)
                                .padding(.vertical, 1)
                                .background(Color.purple.opacity(0.15))
                                .cornerRadius(3)
                        } else {
                            Button {
                                withAnimation(.easeInOut(duration: 0.12)) {
                                    if isExcluded {
                                        config.excludedColumns.remove(col)
                                    } else {
                                        config.excludedColumns.insert(col)
                                    }
                                }
                            } label: {
                                Image(systemName: isExcluded ? "square" : "checkmark.square.fill")
                                    .font(Theme.Font.controlLabel)
                                    .foregroundColor(isExcluded ? .secondary.opacity(0.4) : .blue)
                                    .frame(width: 20, height: 20, alignment: .leading)
                                    .contentShape(Rectangle())
                            }
                            .buttonStyle(.plain)
                            .help(isExcluded ? "Include column in analysis" : "Exclude column from analysis")
                        }
                        
                        if !isTarget && !isExcluded {
                            Menu {
                                Button("Numeric") { config.columnTypeOverrides[col] = "numeric" }
                                Button("Categorical") { config.columnTypeOverrides[col] = "categorical" }
                                Button("Text / NLP") { config.columnTypeOverrides[col] = "text" }
                                Button("Datetime") { config.columnTypeOverrides[col] = "datetime" }
                                Button("Identifier") { config.columnTypeOverrides[col] = "identifier" }
                            } label: {
                                HStack(spacing: 3) {
                                    Text(columnTypeDisplayString(for: col))
                                        .font(.system(size: 8, weight: .bold, design: .rounded))
                                    Image(systemName: "chevron.down")
                                        .font(.system(size: 7))
                                }
                                .foregroundColor(.blue.opacity(0.8))
                                .padding(.horizontal, 4)
                                .padding(.vertical, 2)
                                .background(Color.blue.opacity(0.08))
                                .cornerRadius(4)
                            }
                            .menuStyle(.borderlessButton)
                        }
                    }
                    
                    HStack(spacing: 4) {
                        Text(col)
                            .font(.system(size: 11, weight: .bold, design: .rounded))
                            .foregroundColor(isExcluded ? .secondary.opacity(0.5) : .primary)
                            .lineLimit(1)
                        
                        if !isExcluded && !isTarget {
                            Spacer()
                            Button {
                                filteringColumn = FilterColumnWrapper(name: col)
                            } label: {
                                let hasFilter = config.cleaningActions.contains(where: { $0.column == col })
                                Image(systemName: hasFilter ? "line.3.horizontal.decrease.circle.fill" : "line.3.horizontal.decrease.circle")
                                    .font(.system(size: 10))
                                    .foregroundColor(hasFilter ? .purple : .secondary)
                            }
                            .buttonStyle(.plain)
                        }
                    }
                    
                    if !isExcluded {
                        Spacer(minLength: 2)
                        SparklineView(values: columnValues, isExcluded: isExcluded)
                    }
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .frame(width: colWidth, height: 80, alignment: .leading)
                .background(isExcluded ? Color.primary.opacity(0.02) : Color.primary.opacity(0.04))
                .border(Color.primary.opacity(0.07), width: 0.5)
            }
        }
    }

    // MARK: - Table Data Rows

    private var tableRows: some View {
        ForEach(0..<preview.previewRows.count, id: \.self) { rowIndex in
            let row = preview.previewRows[rowIndex]
            HStack(spacing: 0) {
                ForEach(0..<row.count, id: \.self) { colIndex in
                    let colName = colIndex < preview.columns.count ? preview.columns[colIndex] : ""
                    let isExcluded = config.excludedColumns.contains(colName)
                    PreviewCellView(
                        cell: row[colIndex],
                        rowIndex: rowIndex,
                        isExcluded: isExcluded
                    )
                }
            }
        }
    }

    private func availableFilesPicker(_ files: [String]) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Multi-file Dataset Sources", systemImage: "folder.badge.gearshape")
                .font(.system(size: 10, weight: .bold, design: .rounded))
                .foregroundColor(.secondary.opacity(0.7))
                .tracking(0.5)

            HStack(spacing: 16) {
                // Primary/Train File Selection
                VStack(alignment: .leading, spacing: 4) {
                    Text("Primary Dataset (Train)")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundColor(.secondary)
                    
                    Menu {
                        ForEach(files, id: \.self) { file in
                            let filename = URL(fileURLWithPath: file).lastPathComponent
                            Button(filename) {
                                config.trainFilePath = file
                                onPreviewFileRequested?(file)
                            }
                        }
                    } label: {
                        HStack {
                            let selectedFilename = config.trainFilePath.map { URL(fileURLWithPath: $0).lastPathComponent } ?? URL(fileURLWithPath: files.first ?? preview.localPath).lastPathComponent
                            
                            Text(selectedFilename)
                                .font(.system(size: 12))
                                .foregroundColor(.primary)
                                .lineLimit(1)
                                .truncationMode(.middle)
                                
                            Spacer()
                            Image(systemName: "chevron.up.chevron.down")
                                .font(.system(size: 9))
                                .foregroundColor(.secondary)
                        }
                        .padding(.horizontal, 10)
                        .padding(.vertical, 7)
                        .background(Color.primary.opacity(0.04))
                        .cornerRadius(7)
                        .overlay(RoundedRectangle(cornerRadius: 7).stroke(Color.primary.opacity(0.08), lineWidth: 1))
                    }
                    .menuStyle(.borderlessButton)
                }
                .frame(maxWidth: .infinity)

                // Optional Test File Selection
                VStack(alignment: .leading, spacing: 4) {
                    Text("Test Dataset (Optional)")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundColor(.secondary)
                    
                    Menu {
                        Button("None / Auto-split") {
                            let wasPreviewingTest = (preview.localPath == config.testFilePath)
                            config.testFilePath = nil
                            
                            if wasPreviewingTest {
                                let fallback = config.trainFilePath ?? files.first ?? preview.localPath
                                onPreviewFileRequested?(fallback)
                            }
                        }
                        Divider()
                        ForEach(files, id: \.self) { file in
                            if file != (config.trainFilePath ?? files.first) {
                                let filename = URL(fileURLWithPath: file).lastPathComponent
                                Button(filename) {
                                    config.testFilePath = file
                                    onPreviewFileRequested?(file)
                                }
                            }
                        }
                    } label: {
                        HStack {
                            let selectedFilename = config.testFilePath.map { URL(fileURLWithPath: $0).lastPathComponent } ?? "None / Auto-split"
                            Text(selectedFilename)
                                .font(.system(size: 12))
                                .foregroundColor(config.testFilePath == nil ? .secondary : .primary)
                                .lineLimit(1)
                                .truncationMode(.middle)
                                
                            Spacer()
                            Image(systemName: "chevron.up.chevron.down")
                                .font(.system(size: 9))
                                .foregroundColor(.secondary)
                        }
                        .padding(.horizontal, 10)
                        .padding(.vertical, 7)
                        .background(Color.primary.opacity(0.04))
                        .cornerRadius(7)
                        .overlay(RoundedRectangle(cornerRadius: 7).stroke(Color.primary.opacity(0.08), lineWidth: 1))
                    }
                    .menuStyle(.borderlessButton)
                }
                .frame(maxWidth: .infinity)

                // Optional Validation File Selection
                VStack(alignment: .leading, spacing: 4) {
                    Text("Validation Dataset (Optional)")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundColor(.secondary)
                    
                    Menu {
                        Button("None / Auto-split") {
                            let wasPreviewingVal = (preview.localPath == config.validationFilePath)
                            config.validationFilePath = nil
                            
                            if wasPreviewingVal {
                                let fallback = config.trainFilePath ?? files.first ?? preview.localPath
                                onPreviewFileRequested?(fallback)
                            }
                        }
                        Divider()
                        ForEach(files, id: \.self) { file in
                            if file != (config.trainFilePath ?? files.first) && file != config.testFilePath {
                                let filename = URL(fileURLWithPath: file).lastPathComponent
                                Button(filename) {
                                    config.validationFilePath = file
                                    onPreviewFileRequested?(file)
                                }
                            }
                        }
                    } label: {
                        HStack {
                            let selectedFilename = config.validationFilePath.map { URL(fileURLWithPath: $0).lastPathComponent } ?? "None / Auto-split"
                            Text(selectedFilename)
                                .font(.system(size: 12))
                                .foregroundColor(config.validationFilePath == nil ? .secondary : .primary)
                                .lineLimit(1)
                                .truncationMode(.middle)
                                
                            Spacer()
                            Image(systemName: "chevron.up.chevron.down")
                                .font(.system(size: 9))
                                .foregroundColor(.secondary)
                        }
                        .padding(.horizontal, 10)
                        .padding(.vertical, 7)
                        .background(Color.primary.opacity(0.04))
                        .cornerRadius(7)
                        .overlay(RoundedRectangle(cornerRadius: 7).stroke(Color.primary.opacity(0.08), lineWidth: 1))
                    }
                    .menuStyle(.borderlessButton)
                }
                .frame(maxWidth: .infinity)
            }

            // Segmented Switcher for Active Table Preview
            HStack(spacing: 12) {
                Text("Select Table to Preview:")
                    .font(.system(size: 10, weight: .semibold, design: .rounded))
                    .foregroundColor(.secondary)
                
                let previewBinding = Binding<String>(
                    get: {
                        if config.validationFilePath != nil && preview.localPath == config.validationFilePath {
                            return "val"
                        } else if config.testFilePath != nil && preview.localPath == config.testFilePath {
                            return "test"
                        }
                        return "train"
                    },
                    set: { val in
                        var targetPath: String?
                        if val == "train" {
                            targetPath = config.trainFilePath ?? files.first ?? preview.localPath
                        } else if val == "test" {
                            targetPath = config.testFilePath
                        } else if val == "val" {
                            targetPath = config.validationFilePath
                        }
                        
                        if let path = targetPath {
                            onPreviewFileRequested?(path)
                        }
                    }
                )
                let previewItems: [(String, String)] = {
                    var list = [("Train", "train")]
                    if config.testFilePath != nil {
                        list.append(("Test", "test"))
                    }
                    if config.validationFilePath != nil {
                        list.append(("Validation", "val"))
                    }
                    return list
                }()
                CustomSegmentedPicker(
                    selection: previewBinding,
                    items: previewItems
                )
                .frame(width: 280)
            }
            .padding(.top, 4)
        }
    }
}


// MARK: - Type Pill Button

private struct TypePill: View {
    let type: DatasetType
    let isSelected: Bool
    let onTap: () -> Void

    private var accentColor: Color {
        switch type {
        case .tabular:          return .purple
        case .timeSeries:       return .blue
        case .image:            return .orange
        case .nlp:              return .green
        case .objectDetection:  return .indigo
        }
    }

    var body: some View {
        Button(action: onTap) {
            HStack(spacing: 5) {
                Image(systemName: type.icon)
                    .font(.system(size: 10, weight: .semibold))
                Text(type.label)
                    .font(.system(size: 11, weight: .semibold, design: .rounded))
            }
            .foregroundColor(isSelected ? .white : .secondary)
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(
                isSelected
                    ? AnyShapeStyle(accentColor.gradient.opacity(0.9))
                    : AnyShapeStyle(Color.primary.opacity(0.04))
            )
            .cornerRadius(20)
            .overlay(
                RoundedRectangle(cornerRadius: 20)
                    .stroke(isSelected ? accentColor.opacity(0.4) : Color.primary.opacity(0.07), lineWidth: 1)
            )
            .shadow(color: isSelected ? accentColor.opacity(0.25) : .clear, radius: 4, y: 2)
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Preview Cell

struct PreviewCellView: View {
    let cell: PreviewValue
    let rowIndex: Int
    let isExcluded: Bool

    var body: some View {
        Text(cell.displayString)
            .font(.system(size: 11, design: .monospaced))
            .foregroundColor(isExcluded ? .secondary.opacity(0.35) : cellColor)
            .lineLimit(1)
            .truncationMode(.tail)
            .padding(.horizontal, 10)
            .frame(width: 148, height: 34, alignment: .leading)
            .background(isExcluded ? Color.primary.opacity(0.01) : (rowIndex % 2 == 0 ? Color.primary.opacity(0.015) : Color.clear))
            .border(Color.primary.opacity(0.04), width: 0.5)
    }

    private var cellColor: Color {
        switch cell {
        case .string:  return .secondary
        case .number:  return Color(hue: 0.6, saturation: 0.6, brightness: 0.9)
        case .boolean: return .purple
        case .null:    return .gray.opacity(0.4)
        }
    }
}

// MARK: - Filter Popover Views

extension PreviewTableView {
    @ViewBuilder
    private func filterPopoverView(for col: String) -> some View {
        let colType = preview.columnTypes?[col] ?? "categorical"
        let categories = preview.columnCategories?[col] ?? []
        
        VStack(alignment: .leading, spacing: 12) {
            Text("Filter Column: \(col)")
                .font(.headline)
            if colType == "numeric" {
                NumericFilterView(
                    column: col,
                    config: $config,
                    onApply: {
                        filteringColumn = nil
                        onRefreshPreview?()
                    }
                )
            } else if colType == "text" {
                TextFilterView(
                    column: col,
                    config: $config,
                    onApply: {
                        filteringColumn = nil
                        onRefreshPreview?()
                    }
                )
            } else {
                CategoryFilterView(
                    column: col,
                    categories: categories,
                    config: $config,
                    onApply: {
                        filteringColumn = nil
                        onRefreshPreview?()
                    }
                )
            }
        }
        .padding()
        .frame(width: 280)
    }
}

struct NumericFilterView: View {
    let column: String
    @Binding var config: AnalysisConfig
    let onApply: () -> Void
    
    @State private var selectedOp: String = "less_than"
    @State private var thresholdStr: String = ""
    
    init(column: String, config: Binding<AnalysisConfig>, onApply: @escaping () -> Void) {
        self.column = column
        self._config = config
        self.onApply = onApply
        
        // Find existing numeric filter
        if let act = config.wrappedValue.cleaningActions.first(where: { $0.column == column && ($0.actionType.hasPrefix("remove_less_than:") || $0.actionType.hasPrefix("remove_greater_than:") || $0.actionType.hasPrefix("remove_equals:")) }) {
            if act.actionType.hasPrefix("remove_less_than:") {
                _selectedOp = State(initialValue: "less_than")
                _thresholdStr = State(initialValue: String(act.actionType.dropFirst("remove_less_than:".count)))
            } else if act.actionType.hasPrefix("remove_greater_than:") {
                _selectedOp = State(initialValue: "greater_than")
                _thresholdStr = State(initialValue: String(act.actionType.dropFirst("remove_greater_than:".count)))
            } else if act.actionType.hasPrefix("remove_equals:") {
                _selectedOp = State(initialValue: "equals")
                _thresholdStr = State(initialValue: String(act.actionType.dropFirst("remove_equals:".count)))
            }
        }
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Picker("Remove rows where value is:", selection: $selectedOp) {
                Text("Less than (<)").tag("less_than")
                Text("Greater than (>)").tag("greater_than")
                Text("Equals (=)").tag("equals")
            }
            .pickerStyle(.radioGroup)
            
            TextField("Value", text: $thresholdStr)
                .textFieldStyle(.roundedBorder)
            
            HStack {
                Button("Clear Filter") {
                    config.cleaningActions = config.cleaningActions.filter { !($0.column == column && ($0.actionType.hasPrefix("remove_less_than:") || $0.actionType.hasPrefix("remove_greater_than:") || $0.actionType.hasPrefix("remove_equals:"))) }
                    onApply()
                }
                .buttonStyle(.bordered)
                
                Spacer()
                
                Button("Apply") {
                    if let val = Double(thresholdStr) {
                        config.cleaningActions = config.cleaningActions.filter { !($0.column == column && ($0.actionType.hasPrefix("remove_less_than:") || $0.actionType.hasPrefix("remove_greater_than:") || $0.actionType.hasPrefix("remove_equals:"))) }
                        let actType = "remove_\(selectedOp):\(val)"
                        config.cleaningActions.insert(CleaningAction(column: column, actionType: actType))
                        onApply()
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(Double(thresholdStr) == nil)
            }
            .padding(.top, 8)
        }
    }
}

struct TextFilterView: View {
    let column: String
    @Binding var config: AnalysisConfig
    let onApply: () -> Void
    
    @State private var containsStr: String = ""
    
    init(column: String, config: Binding<AnalysisConfig>, onApply: @escaping () -> Void) {
        self.column = column
        self._config = config
        self.onApply = onApply
        
        // Find existing text filter
        if let act = config.wrappedValue.cleaningActions.first(where: { $0.column == column && $0.actionType.hasPrefix("remove_contains:") }) {
            _containsStr = State(initialValue: String(act.actionType.dropFirst("remove_contains:".count)))
        }
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Remove rows where text contains substring:")
                .font(.caption)
            
            TextField("Substring", text: $containsStr)
                .textFieldStyle(.roundedBorder)
            
            HStack {
                Button("Clear Filter") {
                    config.cleaningActions = config.cleaningActions.filter { !($0.column == column && $0.actionType.hasPrefix("remove_contains:")) }
                    onApply()
                }
                .buttonStyle(.bordered)
                
                Spacer()
                
                Button("Apply") {
                    config.cleaningActions = config.cleaningActions.filter { !($0.column == column && $0.actionType.hasPrefix("remove_contains:")) }
                    if !containsStr.isEmpty {
                        config.cleaningActions.insert(CleaningAction(column: column, actionType: "remove_contains:\(containsStr)"))
                    }
                    onApply()
                }
                .buttonStyle(.borderedProminent)
            }
            .padding(.top, 8)
        }
    }
}

struct CategoryFilterView: View {
    let column: String
    let categories: [String]
    @Binding var config: AnalysisConfig
    let onApply: () -> Void
    
    @State private var excluded: Set<String> = []
    
    init(column: String, categories: [String], config: Binding<AnalysisConfig>, onApply: @escaping () -> Void) {
        self.column = column
        self.categories = categories
        self._config = config
        self.onApply = onApply
        
        // Find existing category filter
        if let act = config.wrappedValue.cleaningActions.first(where: { $0.column == column && $0.actionType.hasPrefix("exclude_categories:") }) {
            let valsStr = act.actionType.dropFirst("exclude_categories:".count)
            let excludedList = valsStr.split(separator: "|").map(String.init)
            _excluded = State(initialValue: Set(excludedList))
        }
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Exclude selected categories:")
                .font(.caption)
                .foregroundColor(.secondary)
            
            if categories.isEmpty {
                Text("No category options loaded.")
                    .italic()
                    .font(.caption)
                    .foregroundColor(.secondary)
            } else {
                ScrollView {
                    VStack(alignment: .leading, spacing: 6) {
                        ForEach(categories, id: \.self) { cat in
                            Toggle(isOn: Binding(
                                get: { excluded.contains(cat) },
                                set: { val in
                                    if val {
                                        excluded.insert(cat)
                                    } else {
                                        excluded.remove(cat)
                                    }
                                }
                            )) {
                                Text(cat)
                                    .font(.system(size: 11))
                            }
                            .toggleStyle(.checkbox)
                        }
                    }
                }
                .frame(maxHeight: 180)
            }
            
            HStack {
                Button("Clear Filter") {
                    config.cleaningActions = config.cleaningActions.filter { !($0.column == column && $0.actionType.hasPrefix("exclude_categories:")) }
                    onApply()
                }
                .buttonStyle(.bordered)
                
                Spacer()
                
                Button("Apply") {
                    config.cleaningActions = config.cleaningActions.filter { !($0.column == column && $0.actionType.hasPrefix("exclude_categories:")) }
                    if !excluded.isEmpty {
                        let valStr = excluded.joined(separator: "|")
                        config.cleaningActions.insert(CleaningAction(column: column, actionType: "exclude_categories:\(valStr)"))
                    }
                    onApply()
                }
                .buttonStyle(.borderedProminent)
            }
            .padding(.top, 8)
        }
    }
}


// MARK: - Smart Sparkline Distribution View

struct SparklineView: View {
    let values: [PreviewValue]
    let isExcluded: Bool
    
    var body: some View {
        let numericValues: [Double] = values.compactMap { val in
            switch val {
            case .number(let d): return d
            case .string(let s): return Double(s)
            default: return nil
            }
        }
        
        HStack(spacing: 1.5) {
            if !numericValues.isEmpty {
                let minVal = numericValues.min() ?? 0.0
                let maxVal = numericValues.max() ?? 1.0
                let range = maxVal - minVal
                
                let binCount = 10
                let bins: [Int] = {
                    var counts = Array(repeating: 0, count: binCount)
                    for val in numericValues {
                        let pct = range > 0 ? (val - minVal) / range : 0.5
                        let index = min(max(Int(pct * Double(binCount)), 0), binCount - 1)
                        counts[index] += 1
                    }
                    return counts
                }()
                
                let maxCount = bins.max() ?? 1
                
                ForEach(0..<binCount, id: \.self) { i in
                    let heightFactor = maxCount > 0 ? CGFloat(bins[i]) / CGFloat(maxCount) : 0.0
                    RoundedRectangle(cornerRadius: 1)
                        .fill(isExcluded ? Color.secondary.opacity(0.2) : Color.blue.opacity(0.6))
                        .frame(width: 4, height: max(2, 16 * heightFactor))
                }
            } else {
                let stringValues: [String] = values.compactMap { val in
                    switch val {
                    case .string(let s): return s.trimmingCharacters(in: .whitespacesAndNewlines)
                    case .number(let d): return String(d)
                    case .boolean(let b): return String(b)
                    case .null: return ""
                    }
                }.filter { !$0.isEmpty }
                
                let frequencies: [String: Int] = {
                    var counts = [String: Int]()
                    for val in stringValues {
                        counts[val, default: 0] += 1
                    }
                    return counts
                }()
                
                let sortedFreqs = frequencies.values.sorted(by: >).prefix(10)
                let maxFreq = sortedFreqs.max() ?? 1
                
                ForEach(0..<10, id: \.self) { i in
                    let freq = i < sortedFreqs.count ? sortedFreqs[i] : 0
                    let heightFactor = maxFreq > 0 ? CGFloat(freq) / CGFloat(maxFreq) : 0.0
                    RoundedRectangle(cornerRadius: 1)
                        .fill(isExcluded ? Color.secondary.opacity(0.2) : Color.purple.opacity(0.5))
                        .frame(width: 4, height: max(2, 16 * heightFactor))
                }
            }
        }
        .frame(height: 18)
    }
}
