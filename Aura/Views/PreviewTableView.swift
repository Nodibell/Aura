import SwiftUI

// MARK: - Preview Table with Type Selector + Column Selection

struct PreviewTableView: View {
    let preview: DatasetPreview

    @Binding var config: AnalysisConfig
    var onPreviewFileRequested: ((String) -> Void)? = nil

    private let colWidth: CGFloat = 148

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
                    .padding(.horizontal, 20)
                    .padding(.top, 16)
                    .padding(.bottom, 12)

                Divider().background(Color.primary.opacity(0.06))

                // ── Dataset Type Selector ──────────────────────────────────────
                datasetTypeSelector
                    .padding(.horizontal, 20)
                    .padding(.vertical, 12)

                Divider().background(Color.primary.opacity(0.06))

                // ── Smart Sampling Toggle ──────────────────────────────────────
                smartSamplingSection
                    .padding(.horizontal, 20)
                    .padding(.vertical, 12)

                if let available = preview.availableFiles, available.count > 1 {
                    Divider().background(Color.primary.opacity(0.06))
                    
                    availableFilesPicker(available)
                        .padding(.horizontal, 20)
                        .padding(.vertical, 12)
                }

                // Optional: time-column picker for Time Series
                if config.datasetType == .timeSeries {
                    Divider().background(Color.primary.opacity(0.06))
                    
                    timeColumnPicker
                        .padding(.horizontal, 20)
                        .padding(.vertical, 12)
                }

                // ── Exclusion Banner ────────────────────────────────────────────
                if !config.excludedColumns.isEmpty {
                    Divider().background(Color.primary.opacity(0.06))
                    
                    exclusionBanner
                        .padding(.horizontal, 20)
                        .padding(.vertical, 12)
                }

                Divider().background(Color.primary.opacity(0.06))

                // ── Data Table (Horizontal Scroll Only) ───────────────────────────
                ScrollView(.horizontal, showsIndicators: true) {
                    VStack(alignment: .leading, spacing: 0) {
                        tableHeader
                            .padding(.horizontal, 20)
                        
                        // FIX: Removed the nested vertical ScrollView.
                        // The table now flows perfectly with the master page scroll.
                        LazyVStack(alignment: .leading, spacing: 0) {
                            tableRows
                        }
                        .padding(.horizontal, 20)
                    }
                }
                .padding(.top, 16)
                .padding(.bottom, 32)
            }
        }
    }

    // MARK: - Header Section

    private var headerSection: some View {
        HStack(alignment: .center, spacing: 12) {
            VStack(alignment: .leading, spacing: 3) {
                Text("Dataset Preview")
                    .font(.title3.bold())
                if let total = preview.totalRows {
                    Text("First \(preview.previewRows.count) rows of \(formatNumber(total)) total • \(preview.columns.count) columns")
                        .font(.caption)
                        .foregroundColor(.secondary)
                } else {
                    Text("First \(preview.previewRows.count) rows • \(preview.columns.count) columns")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }

            Spacer()

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

    // MARK: - Exclusion Banner

    private var exclusionBanner: some View {
        HStack(spacing: 8) {
            Image(systemName: "eye.slash.fill")
                .foregroundColor(.orange)
                .font(.system(size: 11))
            Text("\(config.excludedColumns.count) column\(config.excludedColumns.count == 1 ? "" : "s") excluded from analysis")
                .font(.system(size: 11, weight: .semibold))
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

    // MARK: - Table Header Row

    private var tableHeader: some View {
        HStack(spacing: 0) {
            ForEach(preview.columns, id: \.self) { col in
                let isExcluded = config.excludedColumns.contains(col)
                let isTarget = isColumnTarget(col)
                
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
                                    .font(.system(size: 12, weight: .semibold))
                                    .foregroundColor(isExcluded ? .secondary.opacity(0.4) : .blue)
                                    .frame(width: 20, height: 20, alignment: .leading)
                                    .contentShape(Rectangle())
                            }
                            .buttonStyle(.plain)
                            .help(isExcluded ? "Include column in analysis" : "Exclude column from analysis")
                        }
                    }
                    
                    Text(col)
                        .font(.system(size: 11, weight: .bold, design: .rounded))
                        .foregroundColor(isExcluded ? .secondary.opacity(0.5) : .primary)
                        .lineLimit(1)
                }
                .padding(.horizontal, 10)
                .frame(width: colWidth, height: 48, alignment: .leading)
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
                
                Picker("", selection: Binding<String>(
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
                )) {
                    Text("Train").tag("train")
                    if config.testFilePath != nil {
                        Text("Test").tag("test")
                    }
                    if config.validationFilePath != nil {
                        Text("Validation").tag("val")
                    }
                }
                .pickerStyle(.segmented)
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
