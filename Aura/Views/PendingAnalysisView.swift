import SwiftUI

struct PendingAnalysisView: View {
    let page: AnalysisPage
    let onRunAnalysis: () -> Void
    let onCancel: () -> Void
    var onPreviewFileRequested: ((String) -> Void)? = nil
    var onDatasetTypeChanged: (() -> Void)? = nil
    var onRefreshPreview: (() -> Void)? = nil

    var body: some View {
        VStack(spacing: 0) {
            if page.isAnalyzing {
                loadingViewForAnalysis
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if page.isPreloading {
                loadingView(
                    title: page.progressMessage.isEmpty ? "Loading dataset preview…" : page.progressMessage,
                    subtitle: "Downloading and parsing the file format…",
                    fraction: (page.progressFraction > 0.0 || !page.progressMessage.isEmpty) ? page.progressFraction : nil
                )
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if let error = page.errorMessage {
                errorView(error: error)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                // 1. Control Bar at the top (unless it is a raw data-only preview)
                if !page.isDataOnly {
                    controlBar
                    Divider().background(Color.primary.opacity(0.06))
                    
                    datasetTypeSelector
                    Divider().background(Color.primary.opacity(0.06))
                    
                    smartSamplingSection
                    Divider().background(Color.primary.opacity(0.06))
                } else {
                    dataOnlyHeader
                    Divider().background(Color.primary.opacity(0.06))
                }
                
                // Multi-file Dataset Source Picker
                if let preview = page.previewResult, let available = preview.availableFiles, available.count > 1 {
                    availableFilesPicker(available)
                    Divider().background(Color.primary.opacity(0.06))
                }
                
                // 2. Preview Table or Image Grid
                if let preview = page.previewResult {
                    if (page.analysisConfig.datasetType == .image || page.analysisConfig.datasetType == .objectDetection) && !(preview.previewImages?.isEmpty ?? true) {
                        imageGridView
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                    } else {
                        PreviewTableView(
                            preview: preview,
                            config: Binding(
                                get: { page.analysisConfig },
                                set: { page.analysisConfig = $0 }
                            ),
                            onPreviewFileRequested: { path in
                                onPreviewFileRequested?(path)
                            },
                            onRefreshPreview: {
                                onRefreshPreview?()
                            },
                            isSidebar: false
                        )
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                    }
                } else {
                    noPreviewPlaceholder
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
        }
        .background(Color(nsColor: .windowBackgroundColor))
    }

    // MARK: - Control Bar
    
    private var controlBar: some View {
        HStack(spacing: 16) {
            // Icon & Title & Info
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    Image(systemName: page.analysisConfig.datasetType.icon)
                        .font(.system(size: 13, weight: .bold))
                        .foregroundColor(.purple)
                    Text(page.title)
                        .font(Theme.Font.sectionTitle)
                        .foregroundColor(.primary)
                }
                
                Text(dimensionsString)
                    .font(Theme.Font.caption)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            // Target Column Picker
            HStack(spacing: 6) {
                Text("Target:")
                    .font(Theme.Font.captionBold)
                    .foregroundColor(.secondary)
                
                Menu {
                    Button("Auto-detect target") {
                        page.analysisConfig.targetColumn = ""
                        page.analysisConfig.taskTypeOverride = .auto
                    }
                    Button("None (Clustering)") {
                        page.analysisConfig.targetColumn = ""
                        page.analysisConfig.taskTypeOverride = .clustering
                    }
                    Divider()
                    ForEach(availableColumns, id: \.self) { col in
                        Button(col) {
                            page.analysisConfig.targetColumn = col
                            page.analysisConfig.excludedColumns.remove(col)
                            if page.analysisConfig.taskTypeOverride == .clustering {
                                page.analysisConfig.taskTypeOverride = .auto
                            }
                        }
                    }
                } label: {
                    HStack(spacing: 4) {
                        let selectedLabel = page.analysisConfig.taskTypeOverride == .clustering ? "None (Clustering)" : (page.analysisConfig.targetColumn.isEmpty ? "Auto-detect target" : page.analysisConfig.targetColumn)
                        Text(selectedLabel)
                            .font(Theme.Font.controlLabel)
                        Image(systemName: "chevron.up.chevron.down")
                            .font(.system(size: 8))
                    }
                    .foregroundColor(.primary)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 5)
                    .background(Color.primary.opacity(0.04))
                    .cornerRadius(Theme.Layout.controlCornerRadius)
                    .overlay(RoundedRectangle(cornerRadius: Theme.Layout.controlCornerRadius).stroke(Color.primary.opacity(0.08), lineWidth: 1))
                }
                .menuStyle(.borderlessButton)
            }
            
            // Task Type Segmented Picker
            HStack(spacing: 6) {
                Text("Type:")
                    .font(Theme.Font.captionBold)
                    .foregroundColor(.secondary)
                
                Picker("", selection: Binding(
                    get: { page.analysisConfig.taskTypeOverride },
                    set: { page.analysisConfig.taskTypeOverride = $0 }
                )) {
                    Image(systemName: TaskTypeOverride.auto.icon)
                        .tag(TaskTypeOverride.auto)
                        .help(TaskTypeOverride.auto.shortLabel)
                    Image(systemName: TaskTypeOverride.regression.icon)
                        .tag(TaskTypeOverride.regression)
                        .help(TaskTypeOverride.regression.shortLabel)
                    Image(systemName: TaskTypeOverride.classification.icon)
                        .tag(TaskTypeOverride.classification)
                        .help(TaskTypeOverride.classification.shortLabel)
                    Image(systemName: TaskTypeOverride.clustering.icon)
                        .tag(TaskTypeOverride.clustering)
                        .help(TaskTypeOverride.clustering.shortLabel)
                }
                .pickerStyle(.segmented)
                .frame(width: 170)
            }
            
            // Smart Sampling Toggle
            Toggle(isOn: Binding(
                get: { page.analysisConfig.smartSample },
                set: { page.analysisConfig.smartSample = $0 }
            )) {
                Text("Sample")
                    .font(Theme.Font.captionBold)
            }
            .toggleStyle(.checkbox)
            
            Divider().frame(height: 18)
            
            // Action Buttons
            HStack(spacing: 8) {
                Button("Cancel", action: onCancel)
                    .buttonStyle(.bordered)
                    .controlSize(.regular)
                
                Button(action: onRunAnalysis) {
                    HStack(spacing: 5) {
                        Image(systemName: "play.fill")
                        Text("Run Analysis")
                            .fontWeight(.bold)
                    }
                }
                .buttonStyle(.borderedProminent)
                .tint(Theme.Color.primaryAction)
                .controlSize(.regular)
            }
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 10)
        .background(Color(nsColor: .controlBackgroundColor))
    }
    
    private var dataOnlyHeader: some View {
        HStack {
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    Image(systemName: page.analysisConfig.datasetType.icon)
                        .font(.system(size: 13, weight: .bold))
                        .foregroundColor(Theme.Color.primaryAction)
                    Text(page.title)
                        .font(Theme.Font.sectionTitle)
                        .foregroundColor(.primary)
                }
                
                Text(dimensionsString)
                    .font(Theme.Font.caption)
                    .foregroundColor(.secondary)
            }
            Spacer()
            
            Button("Close", action: onCancel)
                .buttonStyle(.bordered)
                .controlSize(.regular)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 10)
        .background(Color(nsColor: .controlBackgroundColor))
    }

    // MARK: - Multi-file Dataset Source Picker

    @ViewBuilder
    private func availableFilesPicker(_ files: [String]) -> some View {
        VStack(alignment: .leading, spacing: 8) {
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
                                page.analysisConfig.trainFilePath = file
                                onPreviewFileRequested?(file)
                            }
                        }
                    } label: {
                        HStack {
                            let selectedFilename = page.analysisConfig.trainFilePath.map { URL(fileURLWithPath: $0).lastPathComponent } ?? URL(fileURLWithPath: files.first ?? "").lastPathComponent
                            
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
                            let wasPreviewingTest = (page.previewResult?.localPath == page.analysisConfig.testFilePath)
                            page.analysisConfig.testFilePath = nil
                            
                            if wasPreviewingTest {
                                let fallback = page.analysisConfig.trainFilePath ?? files.first ?? ""
                                onPreviewFileRequested?(fallback)
                            }
                        }
                        Divider()
                        ForEach(files, id: \.self) { file in
                            if file != (page.analysisConfig.trainFilePath ?? files.first) {
                                let filename = URL(fileURLWithPath: file).lastPathComponent
                                Button(filename) {
                                    page.analysisConfig.testFilePath = file
                                    onPreviewFileRequested?(file)
                                }
                            }
                        }
                    } label: {
                        HStack {
                            let selectedFilename = page.analysisConfig.testFilePath.map { URL(fileURLWithPath: $0).lastPathComponent } ?? "None / Auto-split"
                            Text(selectedFilename)
                                .font(.system(size: 12))
                                .foregroundColor(page.analysisConfig.testFilePath == nil ? .secondary : .primary)
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
                            let wasPreviewingVal = (page.previewResult?.localPath == page.analysisConfig.validationFilePath)
                            page.analysisConfig.validationFilePath = nil
                            
                            if wasPreviewingVal {
                                let fallback = page.analysisConfig.trainFilePath ?? files.first ?? ""
                                onPreviewFileRequested?(fallback)
                            }
                        }
                        Divider()
                        ForEach(files, id: \.self) { file in
                            if file != (page.analysisConfig.trainFilePath ?? files.first) && file != page.analysisConfig.testFilePath {
                                let filename = URL(fileURLWithPath: file).lastPathComponent
                                Button(filename) {
                                    page.analysisConfig.validationFilePath = file
                                    onPreviewFileRequested?(file)
                                }
                            }
                        }
                    } label: {
                        HStack {
                            let selectedFilename = page.analysisConfig.validationFilePath.map { URL(fileURLWithPath: $0).lastPathComponent } ?? "None / Auto-split"
                            Text(selectedFilename)
                                .font(.system(size: 12))
                                .foregroundColor(page.analysisConfig.validationFilePath == nil ? .secondary : .primary)
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
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 12)
        .background(Color.primary.opacity(0.01))
    }

    // MARK: - Image Grid View

    private var imageGridView: some View {
        ScrollView {
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 180, maximum: 240), spacing: 16)], spacing: 16) {
                if let images = page.previewResult?.previewImages, !images.isEmpty {
                    ForEach(images) { img in
                        VStack(alignment: .leading, spacing: 6) {
                            ZStack {
                                RoundedRectangle(cornerRadius: 8)
                                    .fill(Color.primary.opacity(0.02))
                                
                                if let data = Data(base64Encoded: img.b64Data),
                                   let nsImage = NSImage(data: data) {
                                    Image(nsImage: nsImage)
                                        .resizable()
                                        .aspectRatio(contentMode: .fit)
                                        .cornerRadius(6)
                                        .padding(4)
                                } else {
                                    Image(systemName: "photo")
                                        .font(.system(size: 24))
                                        .foregroundColor(.secondary)
                                }
                            }
                            .frame(height: 150)
                            .overlay(
                                RoundedRectangle(cornerRadius: 8)
                                    .stroke(Color.primary.opacity(0.06), lineWidth: 1)
                            )
                            
                            Text(img.name)
                                .font(.system(size: 11, weight: .bold, design: .rounded))
                                .foregroundColor(.primary)
                                .lineLimit(1)
                            
                            Text(img.label)
                                .font(.system(size: 10))
                                .foregroundColor(.secondary)
                                .lineLimit(1)
                        }
                        .padding(10)
                        .background(Color(nsColor: .controlBackgroundColor))
                        .cornerRadius(10)
                        .overlay(
                            RoundedRectangle(cornerRadius: 10)
                                .stroke(Color.primary.opacity(0.08), lineWidth: 1)
                        )
                        .shadow(color: Color.black.opacity(0.015), radius: 2, y: 1)
                    }
                }
            }
            .padding(20)
        }
    }

    // MARK: - Helper Views

    private var noPreviewPlaceholder: some View {
        VStack(spacing: 12) {
            Image(systemName: "tablecells.badge.ellipsis")
                .font(.system(size: 40))
                .foregroundColor(.secondary)
            Text("No Preview Available")
                .font(.headline)
            Text("Verify the file format or try loading the dataset again.")
                .font(.subheadline)
                .foregroundColor(.secondary)
        }
    }

    private var loadingViewForAnalysis: some View {
        VStack(spacing: 18) {
            if page.progressFraction > 0 {
                VStack(spacing: 8) {
                    ProgressView(value: page.progressFraction)
                        .progressViewStyle(.linear)
                        .tint(LinearGradient(colors: [.purple, .indigo], startPoint: .leading, endPoint: .trailing))
                        .frame(width: 280)
                    
                    Text(String(format: "%.0f%%", page.progressFraction * 100))
                        .font(.system(size: 10, weight: .bold, design: .rounded))
                        .foregroundColor(.purple)
                }
                .padding(.bottom, 8)
            } else {
                ProgressView()
                    .progressViewStyle(.circular)
                    .controlSize(.regular)
                    .padding(.bottom, 8)
            }
            
            Text(page.progressMessage.isEmpty ? "Running analysis pipeline…" : page.progressMessage)
                .font(Theme.Font.sectionTitle)
                .foregroundColor(.primary)
            Text("Fitting ML models and generating charts…")
                .font(Theme.Font.caption)
                .foregroundColor(.secondary)
            
            if !page.completedStages.isEmpty {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Completed Stages:")
                        .font(Theme.Font.captionBold)
                        .foregroundColor(.primary)
                    
                    ScrollView {
                        VStack(alignment: .leading, spacing: 6) {
                            ForEach(page.completedStages) { stage in
                                HStack(spacing: 8) {
                                    Image(systemName: "checkmark.circle.fill")
                                        .foregroundColor(.green)
                                        .font(Theme.Font.caption)
                                    Text(stage.message)
                                        .font(Theme.Font.caption)
                                        .foregroundColor(.secondary)
                                    Spacer()
                                    Text(String(format: "%.1fs", stage.elapsed))
                                        .font(.system(size: 10, weight: .semibold, design: .monospaced))
                                        .foregroundColor(.indigo)
                                }
                            }
                        }
                        .padding(.horizontal, 4)
                    }
                    .frame(height: min(CGFloat(page.completedStages.count) * 20, 100))
                }
                .frame(width: 280)
                .padding(.top, 8)
            }
            
            Button(action: {
                Task {
                    await PythonRunner.shared.cancelActiveAnalysis()
                }
            }) {
                HStack(spacing: 6) {
                    Image(systemName: "xmark.circle.fill")
                    Text("Cancel Analysis")
                }
                .font(.system(size: 11, weight: .semibold))
                .foregroundColor(.red)
                .padding(.horizontal, 14)
                .padding(.vertical, 7)
                .background(Color.red.opacity(0.08))
                .cornerRadius(8)
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(Color.red.opacity(0.25), lineWidth: 1)
                )
            }
            .buttonStyle(.plain)
            .padding(.top, 8)
        }
    }

    private func loadingView(title: String, subtitle: String, fraction: Double? = nil) -> some View {
        VStack(spacing: 18) {
            if let fraction = fraction {
                VStack(spacing: 8) {
                    ProgressView(value: fraction)
                        .progressViewStyle(.linear)
                        .tint(LinearGradient(colors: [.purple, .indigo], startPoint: .leading, endPoint: .trailing))
                        .frame(width: 280)
                    
                    Text(String(format: "%.0f%%", fraction * 100))
                        .font(.system(size: 10, weight: .bold, design: .rounded))
                        .foregroundColor(.purple)
                }
                .padding(.bottom, 8)
            } else {
                ProgressView()
                    .progressViewStyle(.circular)
                    .controlSize(.regular)
                    .padding(.bottom, 8)
            }
            Text(title).font(Theme.Font.sectionTitle).foregroundColor(.primary)
            Text(subtitle).font(Theme.Font.caption).foregroundColor(.secondary)
            
            Button(action: onCancel) {
                HStack(spacing: 6) {
                    Image(systemName: "xmark.circle.fill")
                    Text("Cancel Preloading")
                }
                .font(Theme.Font.captionBold)
                .foregroundColor(.red)
                .padding(.horizontal, 14)
                .padding(.vertical, 7)
                .background(Color.red.opacity(0.08))
                .cornerRadius(8)
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(Color.red.opacity(0.25), lineWidth: 1)
                )
            }
            .buttonStyle(.plain)
            .padding(.top, 8)
        }
    }
    
    private func errorView(error: String) -> some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                HStack {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.title).foregroundColor(.red)
                    Text("Preloading Failed").font(.title).fontWeight(.bold)
                }
                Text("Failed to preload dataset. Diagnostic details:")
                    .font(.body).foregroundColor(.secondary)
                
                Text(error)
                    .font(.system(.body, design: .monospaced))
                    .foregroundColor(.red)
                    .padding(12)
                    .background(Color.red.opacity(0.05))
                    .cornerRadius(8)
                    .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.red.opacity(0.15), lineWidth: 1))
                
                HStack(spacing: 12) {
                    Button("Dismiss", action: onCancel)
                        .buttonStyle(.bordered)
                        .controlSize(.regular)
                }
            }
            .padding(40)
        }
    }

    private var dimensionsString: String {
        guard let preview = page.previewResult else { return "--" }
        if let total = preview.totalRows {
            return "\(preview.columns.count) cols × \(total) rows"
        }
        return "\(preview.columns.count) cols × \(preview.previewRows.count)+ rows"
    }
    
    private var availableColumns: [String] {
        if !page.trainColumns.isEmpty {
            return page.trainColumns
        }
        if let preview = page.previewResult {
            return preview.columns
        }
        if let result = page.result {
            return result.columns
        }
        return []
    }
    
    private var datasetTypeSelector: some View {
        VStack(alignment: .leading, spacing: 6) {
            Label("Dataset Type", systemImage: "square.stack.3d.up")
                .font(.system(size: 10, weight: .bold, design: .rounded))
                .foregroundColor(.secondary.opacity(0.7))
                .tracking(0.5)

            HStack(spacing: 8) {
                ForEach(DatasetType.allCases) { dtype in
                    TypePill(
                        type: dtype,
                        isSelected: page.analysisConfig.datasetType == dtype
                    ) {
                        let oldType = page.analysisConfig.datasetType
                        withAnimation(.easeInOut(duration: 0.18)) {
                            page.analysisConfig.datasetType = dtype
                        }
                        if oldType != dtype {
                            onDatasetTypeChanged?()
                        }
                    }
                }
            }

            Text(page.analysisConfig.datasetType.description)
                .font(.system(size: 11))
                .foregroundColor(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 8)
    }

    private var smartSamplingSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Label("Smart Sampling (Large Datasets)", systemImage: "sparkles.rectangle.stack")
                .font(.system(size: 10, weight: .bold, design: .rounded))
                .foregroundColor(.secondary.opacity(0.7))
                .tracking(0.5)
            
            HStack(spacing: 12) {
                Toggle(isOn: Binding(
                    get: { page.analysisConfig.smartSample },
                    set: { page.analysisConfig.smartSample = $0 }
                )) {
                    Text("Enable Smart Sampling")
                        .font(.system(size: 12, weight: .semibold))
                }
                .toggleStyle(.checkbox)
                
                Text("Automatically sample classification/regression datasets down to 100,000 rows to prevent memory limits and speed up fitting.")
                    .font(.system(size: 11))
                    .foregroundColor(.secondary)
            }
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 8)
    }
}

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
            .cornerRadius(8)
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .stroke(isSelected ? Color.clear : Color.primary.opacity(0.08), lineWidth: 1)
            )
        }
        .buttonStyle(.plain)
    }
}
