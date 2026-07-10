import SwiftUI

struct DataCleaningView: View {
    let result: AnalysisResult
    @Binding var config: AnalysisConfig
    let onRunAnalysis: () -> Void
    
    @State private var hoverColumn: String? = nil
    @State private var activeCleaningTab = 0
    
    @State private var lineageNodes: [REPLService.LineageNode] = []
    @State private var activeStateId: Int = 0
    @State private var isRollingBack = false
    @State private var lineageError: String? = nil


    var body: some View {
        VStack(spacing: 0) {
            // Header
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 8) {
                    Image(systemName: "slider.horizontal.3")
                        .foregroundColor(.purple)
                        .font(.title2)
                    Text("Interactive Data Cleaning")
                        .font(.title2)
                        .fontWeight(.bold)
                    
                    Spacer()
                    
                    if let activeIndex = lineageNodes.firstIndex(where: { $0.id == activeStateId }) {
                        HStack(spacing: 8) {
                            if activeIndex > 0 {
                                let previousStateId = lineageNodes[activeIndex - 1].id
                                Button {
                                    rollbackToState(previousStateId)
                                } label: {
                                    Label("Undo", systemImage: "arrow.uturn.backward")
                                        .font(.subheadline)
                                        .fontWeight(.semibold)
                                }
                                .disabled(isRollingBack)
                                .buttonStyle(.bordered)
                                .tint(Theme.Color.caution)
                                .keyboardShortcut("z", modifiers: [.command])
                                .help("Roll back to the previous cleaning state (⌘Z)")
                            }
                            
                            if activeIndex < lineageNodes.count - 1 {
                                let nextStateId = lineageNodes[activeIndex + 1].id
                                Button {
                                    rollbackToState(nextStateId)
                                } label: {
                                    Label("Redo", systemImage: "arrow.uturn.forward")
                                        .font(.subheadline)
                                        .fontWeight(.semibold)
                                }
                                .disabled(isRollingBack)
                                .buttonStyle(.bordered)
                                .tint(Theme.Color.success)
                                .keyboardShortcut("z", modifiers: [.command, .shift])
                                .help("Restore the next cleaning state (⇧⌘Z)")
                            }
                        }
                    }
                }
                Text("Select custom imputation, outlier treatment, and encoding actions per column. The target column '\(result.targetColumn)' cannot be modified.")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding()
            .background(Color.primary.opacity(0.01))
            
            Divider()
            CustomSegmentedPicker(
                selection: $activeCleaningTab,
                items: [
                    ("Column Cleaning", 0),
                    ("Time-Travel Lineage", 1),
                    ("Custom Plugins", 2)
                ]
            )
            .padding(.horizontal)
            .padding(.vertical, 8)
            
            Divider()

            if activeCleaningTab == 0 {
                ScrollView {
                VStack(spacing: 12) {
                    let sortedCols = result.columns.sorted()
                    ForEach(sortedCols, id: \.self) { col in
                        let isTarget = (col == result.targetColumn)
                        let colProfile = result.profiling?.columns[col]
                        let colType = colProfile?.type.lowercased() ?? "numeric"
                        let missingCount = colProfile?.missing ?? 0
                        let isExcluded = config.excludedColumns.contains(col)
                        
                        HStack(alignment: .top, spacing: 12) {
                            // Checkbox to exclude/include columns
                            if isTarget {
                                Image(systemName: "checkmark.square.fill")
                                    .foregroundColor(.purple)
                                    .font(.system(size: 14))
                                    .padding(.top, 2)
                            } else {
                                Button {
                                    withAnimation {
                                        if isExcluded {
                                            config.excludedColumns.remove(col)
                                        } else {
                                            config.excludedColumns.insert(col)
                                        }
                                    }
                                } label: {
                                    Image(systemName: isExcluded ? "square" : "checkmark.square.fill")
                                        .foregroundColor(isExcluded ? .secondary.opacity(0.8) : .blue)
                                        .font(.system(size: 14))
                                }
                                .buttonStyle(.plain)
                                .padding(.top, 2)
                            }
                            
                            HStack(alignment: .top, spacing: 16) {
                                // Column Metadata
                                VStack(alignment: .leading, spacing: 4) {
                                    HStack(spacing: 6) {
                                        Image(systemName: iconForType(colType))
                                            .foregroundColor(isTarget ? .purple : (isExcluded ? .secondary : .secondary))
                                            .font(.caption)
                                        Text(col)
                                            .font(.subheadline)
                                            .fontWeight(.bold)
                                            .foregroundColor(isTarget ? .purple : (isExcluded ? .secondary : .primary))
                                            .lineLimit(1)
                                    }
                                    
                                    HStack(spacing: 8) {
                                        Text(colType.capitalized)
                                            .font(.system(size: 10, weight: .semibold))
                                            .foregroundColor(.secondary)
                                        if missingCount > 0 {
                                            Text("\(missingCount) missing")
                                                .font(.system(size: 10, weight: .bold))
                                                .foregroundColor(isExcluded ? .orange.opacity(0.5) : .orange)
                                        } else {
                                            Text("Clean")
                                                .font(.system(size: 10))
                                                .foregroundColor(.secondary.opacity(0.8))
                                        }
                                    }
                                    
                                    if !isTarget {
                                        TextField("Rename column...", text: Binding(
                                            get: { getRename(for: col) },
                                            set: { setRename($0, for: col) }
                                        ))
                                        .textFieldStyle(.plain)
                                        .font(.system(size: 10))
                                        .padding(.horizontal, 6)
                                        .padding(.vertical, 3)
                                        .background(Color.primary.opacity(0.04))
                                        .cornerRadius(4)
                                        .frame(maxWidth: 160)
                                        .disabled(isExcluded)
                                    }
                                }
                                .frame(width: 180, alignment: .leading)
                                .opacity(isExcluded ? 0.6 : 1.0)
                                
                                Spacer()
                                
                                // Cleaning Actions Controls
                                if isTarget {
                                    Text("Target Column (Exempt)")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                        .italic()
                                        .frame(maxWidth: .infinity, alignment: .trailing)
                                        .padding(.vertical, 8)
                                } else {
                                    HStack(spacing: 12) {
                                        // 1. Imputation Picker
                                        VStack(alignment: .leading, spacing: 2) {
                                            Text("Imputation")
                                                .font(.system(size: 9, weight: .bold))
                                                .foregroundColor(.secondary)
                                            
                                            let availableImputes = imputationOptions(for: colType)
                                            Menu {
                                                ForEach(availableImputes) { opt in
                                                    Button {
                                                        setImputation(opt, for: col)
                                                    } label: {
                                                        Text(opt.label)
                                                    }
                                                }
                                            } label: {
                                                let current = getImputation(for: col)
                                                Text(current.label)
                                                    .font(.caption)
                                                    .fontWeight(current != .none ? .bold : .regular)
                                                    .foregroundColor(current != .none ? .blue : .primary)
                                            }
                                            .menuStyle(.borderlessButton)
                                            .frame(width: 120, alignment: .leading)
                                        }
                                        
                                        // 2. Outlier Treatment Picker
                                        if colType == "numeric" {
                                            VStack(alignment: .leading, spacing: 2) {
                                                Text("Outlier Treatment")
                                                    .font(.system(size: 9, weight: .bold))
                                                    .foregroundColor(.secondary)
                                                
                                                Menu {
                                                    ForEach(OutlierOption.allCases) { opt in
                                                        Button {
                                                            setOutlier(opt, for: col)
                                                        } label: {
                                                            Text(opt.label)
                                                        }
                                                    }
                                                } label: {
                                                    let current = getOutlier(for: col)
                                                    Text(current.label)
                                                        .font(.caption)
                                                        .fontWeight(current != .none ? .bold : .regular)
                                                        .foregroundColor(current != .none ? .orange : .primary)
                                                }
                                                .menuStyle(.borderlessButton)
                                                .frame(width: 130, alignment: .leading)
                                            }
                                        }
                                        
                                        // 3. Encoding Picker (Categorical only)
                                        if colType == "categorical" {
                                            VStack(alignment: .leading, spacing: 2) {
                                                Text("Encoding")
                                                    .font(.system(size: 9, weight: .bold))
                                                    .foregroundColor(.secondary)
                                                
                                                Menu {
                                                    ForEach(EncodingOption.allCases) { opt in
                                                        Button {
                                                            setEncoding(opt, for: col)
                                                        } label: {
                                                            Text(opt.label)
                                                        }
                                                    }
                                                } label: {
                                                    let current = getEncoding(for: col)
                                                    Text(current.label)
                                                        .font(.caption)
                                                        .fontWeight(current != .none ? .bold : .regular)
                                                        .foregroundColor(current != .none ? .green : .primary)
                                                }
                                                .menuStyle(.borderlessButton)
                                                .frame(width: 120, alignment: .leading)
                                            }
                                        }
                                        
                                        // 4. Feature Engineering Picker
                                        VStack(alignment: .leading, spacing: 2) {
                                            Text("Feature Engineering")
                                                .font(.system(size: 9, weight: .bold))
                                                .foregroundColor(.secondary)
                                            
                                            Menu {
                                                Button("None") {
                                                    clearFeatureEngineering(for: col)
                                                }
                                                
                                                if colType == "numeric" {
                                                    Button("Log Transform (log1p)") {
                                                        setFeatureEngineering("transform_log", for: col)
                                                    }
                                                    Button("Power Transform (Square)") {
                                                        setFeatureEngineering("transform_power", for: col)
                                                    }
                                                    
                                                    Menu("Interaction with...") {
                                                        let otherNumericCols = result.columns.filter { $0 != col && $0 != result.targetColumn && (result.profiling?.columns[$0]?.type.lowercased() == "numeric" || $0 != result.targetColumn) }
                                                        ForEach(otherNumericCols, id: \.self) { otherCol in
                                                            Button(otherCol) {
                                                                setFeatureEngineering("transform_interaction:\(otherCol)", for: col)
                                                            }
                                                        }
                                                    }
                                                } else {
                                                    Button("Extract Date Parts") {
                                                        setFeatureEngineering("transform_date", for: col)
                                                    }
                                                }
                                            } label: {
                                                let current = getFeatureEngineeringLabel(for: col)
                                                Text(current)
                                                    .font(.caption)
                                                    .fontWeight(current != "None" ? .bold : .regular)
                                                    .foregroundColor(current != "None" ? .purple : .primary)
                                            }
                                            .menuStyle(.borderlessButton)
                                            .frame(width: 145, alignment: .leading)
                                        }
                                    }
                                    .opacity(isExcluded ? 0.25 : 1.0)
                                    .disabled(isExcluded)
                                }
                            }
                        }
                        .padding(12)
                        .background(Color.primary.opacity(isExcluded ? 0.005 : (hoverColumn == col ? 0.04 : 0.01)))
                        .cornerRadius(8)
                        .overlay(
                            RoundedRectangle(cornerRadius: 8)
                                .stroke(Color.primary.opacity(isExcluded ? 0.02 : (hoverColumn == col ? 0.08 : 0.03)), lineWidth: 1)
                        )
                        .onHover { isHover in
                            if !isExcluded {
                                withAnimation(.easeInOut(duration: 0.15)) {
                                    hoverColumn = isHover ? col : nil
                                }
                            } else {
                                hoverColumn = nil
                            }
                        }
                    }
                }
                .padding()
            }
            } else if activeCleaningTab == 1 {
                lineageTab
            } else {
                pluginsTab
            }
            
            Divider()
            
            // Footer run action banner
            HStack {
                if !config.cleaningActions.isEmpty {
                    Text("\(config.cleaningActions.count) cleaning action(s) selected.")
                        .font(.subheadline)
                        .foregroundColor(.purple)
                        .fontWeight(.bold)
                } else {
                    Text("No cleaning actions selected. Default configurations will be used.")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
                
                Button(action: onRunAnalysis) {
                    HStack(spacing: 6) {
                        Image(systemName: "play.fill")
                        Text("Apply Actions & Re-run Analysis")
                    }
                    .font(.subheadline)
                    .fontWeight(.bold)
                    .foregroundColor(.white)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 8)
                    .background(
                        LinearGradient(gradient: Gradient(colors: [.purple, .blue]), startPoint: .leading, endPoint: .trailing)
                    )
                    .cornerRadius(8)
                }
                .buttonStyle(.plain)
            }
            .padding()
            .background(Color.primary.opacity(0.015))
        }
        .onAppear {
            loadLineage()
        }
    }
    
    // MARK: - Helpers
    
    private func iconForType(_ type: String) -> String {
        switch type {
        case "numeric":     return "number"
        case "categorical": return "tag"
        case "text":        return "text.alignleft"
        default:            return "questionmark.circle"
        }
    }
    
    private func imputationOptions(for type: String) -> [ImputationOption] {
        if type == "numeric" {
            return [.none, .mean, .median, .mode, .knn, .mice]
        } else {
            return [.none, .mode] // Categorical/Text can only impute mode or none
        }
    }
    
    private func getImputation(for col: String) -> ImputationOption {
        if config.cleaningActions.contains(where: { $0.column == col && $0.actionType == "impute_mean" }) { return .mean }
        if config.cleaningActions.contains(where: { $0.column == col && $0.actionType == "impute_median" }) { return .median }
        if config.cleaningActions.contains(where: { $0.column == col && $0.actionType == "impute_mode" }) { return .mode }
        if config.cleaningActions.contains(where: { $0.column == col && $0.actionType == "impute_knn" }) { return .knn }
        if config.cleaningActions.contains(where: { $0.column == col && $0.actionType == "impute_mice" }) { return .mice }
        return .none
    }
    
    private func setImputation(_ option: ImputationOption, for col: String) {
        let imputes = ["impute_mean", "impute_median", "impute_mode", "impute_knn", "impute_mice"]
        config.cleaningActions = config.cleaningActions.filter { !($0.column == col && imputes.contains($0.actionType)) }
        if option != .none {
            config.cleaningActions.insert(CleaningAction(column: col, actionType: option.rawValue))
        }
    }
    
    private func getOutlier(for col: String) -> OutlierOption {
        if config.cleaningActions.contains(where: { $0.column == col && $0.actionType == "clip_outliers" }) { return .capIqr }
        if config.cleaningActions.contains(where: { $0.column == col && $0.actionType == "drop_outliers" }) { return .dropIqr }
        if config.cleaningActions.contains(where: { $0.column == col && $0.actionType == "isolation_forest" }) { return .isolationForest }
        return .none
    }
    
    private func setOutlier(_ option: OutlierOption, for col: String) {
        let outliers = ["clip_outliers", "drop_outliers", "isolation_forest"]
        config.cleaningActions = config.cleaningActions.filter { !($0.column == col && outliers.contains($0.actionType)) }
        if option != .none {
            config.cleaningActions.insert(CleaningAction(column: col, actionType: option.rawValue))
        }
    }
    
    private func getEncoding(for col: String) -> EncodingOption {
        if config.cleaningActions.contains(where: { $0.column == col && $0.actionType == "one_hot_encode" }) { return .oneHot }
        if config.cleaningActions.contains(where: { $0.column == col && $0.actionType == "target_encode" }) { return .target }
        return .none
    }
    
    private func setEncoding(_ option: EncodingOption, for col: String) {
        let encodings = ["one_hot_encode", "target_encode"]
        config.cleaningActions = config.cleaningActions.filter { !($0.column == col && encodings.contains($0.actionType)) }
        if option != .none {
            config.cleaningActions.insert(CleaningAction(column: col, actionType: option.rawValue))
        }
    }
    
    private func getFeatureEngineeringLabel(for col: String) -> String {
        if config.cleaningActions.contains(where: { $0.column == col && $0.actionType == "transform_log" }) { return "Log Transform" }
        if config.cleaningActions.contains(where: { $0.column == col && $0.actionType == "transform_power" }) { return "Power Transform" }
        if let interactionAct = config.cleaningActions.first(where: { $0.column == col && $0.actionType.hasPrefix("transform_interaction:") }) {
            let otherCol = interactionAct.actionType.dropFirst("transform_interaction:".count)
            return "Interaction (\(otherCol))"
        }
        if config.cleaningActions.contains(where: { $0.column == col && $0.actionType == "transform_date" }) { return "Extract Date" }
        return "None"
    }
    
    private func setFeatureEngineering(_ actionType: String, for col: String) {
        config.cleaningActions = config.cleaningActions.filter { !($0.column == col && $0.actionType.hasPrefix("transform_")) }
        config.cleaningActions.insert(CleaningAction(column: col, actionType: actionType))
    }
    
    private func clearFeatureEngineering(for col: String) {
        config.cleaningActions = config.cleaningActions.filter { !($0.column == col && $0.actionType.hasPrefix("transform_")) }
    }
    
    private func getRename(for col: String) -> String {
        if let renameAct = config.cleaningActions.first(where: { $0.column == col && $0.actionType.hasPrefix("rename:") }) {
            return String(renameAct.actionType.dropFirst("rename:".count))
        }
        return ""
    }
    
    private func setRename(_ newName: String, for col: String) {
        config.cleaningActions = config.cleaningActions.filter { !($0.column == col && $0.actionType.hasPrefix("rename:")) }
        let trimmed = newName.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty {
            config.cleaningActions.insert(CleaningAction(column: col, actionType: "rename:\(trimmed)"))
        }
    }

    // MARK: - Lineage Time-Travel tab view
    
    private var lineageTab: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                HStack {
                    VStack(alignment: .leading, spacing: 3) {
                        Text("Dataset State Lineage")
                            .font(.headline)
                        Text("Track mutations and roll back to any previous state in the pipeline history.")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    Spacer()
                    Button(action: loadLineage) {
                        Label("Refresh", systemImage: "arrow.clockwise")
                    }
                    .disabled(isRollingBack)
                }
                .padding(.bottom, 4)
                
                if lineageNodes.isEmpty {
                    VStack(spacing: 8) {
                        Image(systemName: "clock.arrow.circlepath")
                            .font(.system(size: 28))
                            .foregroundColor(.secondary)
                        Text("No lineage states recorded yet.")
                            .foregroundColor(.secondary)
                            .font(.subheadline)
                    }
                    .frame(maxWidth: .infinity, minHeight: 180)
                    .background(Color.primary.opacity(0.02))
                    .cornerRadius(8)
                } else {
                    VStack(alignment: .leading, spacing: 0) {
                        ForEach(Array(lineageNodes.enumerated()), id: \.element.id) { index, node in
                            HStack(alignment: .top, spacing: 12) {
                                VStack(spacing: 4) {
                                    Circle()
                                        .fill(node.id == activeStateId ? Color.purple : Color.blue)
                                        .frame(width: 10, height: 10)
                                    if index < lineageNodes.count - 1 {
                                        Rectangle()
                                            .fill(Color.primary.opacity(0.15))
                                            .frame(width: 2, height: 40)
                                    }
                                }
                                .padding(.top, 4)
                                
                                VStack(alignment: .leading, spacing: 4) {
                                    HStack {
                                        Text(node.description)
                                            .fontWeight(.semibold)
                                            .foregroundColor(node.id == activeStateId ? .purple : .primary)
                                            .font(.subheadline)
                                        
                                        if node.id == activeStateId {
                                            Text("Active State")
                                                .font(.system(size: 9, weight: .bold))
                                                .foregroundColor(.purple)
                                                .padding(.horizontal, 6)
                                                .padding(.vertical, 2)
                                                .background(Color.purple.opacity(0.15))
                                                .cornerRadius(4)
                                        }
                                    }
                                    
                                    Text("Shape: \(node.shape)")
                                        .font(.caption2)
                                        .foregroundColor(.secondary)
                                    
                                    if node.id != activeStateId {
                                        Button("Rollback to this state") {
                                            rollbackToState(node.id)
                                        }
                                        .buttonStyle(.bordered)
                                        .font(.caption)
                                        .tint(Theme.Color.caution)
                                        .disabled(isRollingBack)
                                        .padding(.top, 4)
                                    }
                                }
                                
                                Spacer()
                            }
                            .padding(.vertical, 8)
                        }
                    }
                    .padding()
                    .background(Color.primary.opacity(0.03))
                    .cornerRadius(12)
                }
                
                if let err = lineageError {
                    Text(err)
                        .foregroundColor(.red)
                        .font(.caption)
                }
            }
            .padding()
        }
        .onAppear {
            loadLineage()
        }
    }
    
    private func loadLineage() {
        Task {
            do {
                let nodes = try await REPLService.shared.getLineage()
                await MainActor.run {
                    self.lineageNodes = nodes
                    if let last = nodes.last {
                        self.activeStateId = last.id
                    }
                }
            } catch {
                self.lineageError = "Failed to load lineage: \(error.localizedDescription)"
            }
        }
    }
    
    private func rollbackToState(_ id: Int) {
        isRollingBack = true
        lineageError = nil
        Task {
            do {
                let result = try await REPLService.shared.rollback(stateId: id)
                let nodes = try await REPLService.shared.getLineage()
                await MainActor.run {
                    self.lineageNodes = nodes
                    self.activeStateId = result.activeState
                    self.isRollingBack = false
                    onRunAnalysis()
                }
            } catch {
                await MainActor.run {
                    self.lineageError = "Rollback failed: \(error.localizedDescription)"
                    self.isRollingBack = false
                }
            }
        }
    }

    // MARK: - Plugins tab view
    
    @State private var plugins: [REPLService.PluginInfo] = []
    @State private var parameterValues: [String: Double] = [:]
    @State private var pluginsError: String? = nil
    @State private var isRunningPlugin = false
    
    private var pluginsTab: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                HStack {
                    VStack(alignment: .leading, spacing: 3) {
                        Text("Custom Transform Plugins")
                            .font(.headline)
                        Text("Write Python transform scripts in ~/Documents/Aura/Plugins and dynamically configure parameters.")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    Spacer()
                    
                    Button {
                        let home = FileManager.default.homeDirectoryForCurrentUser
                        let pluginsDir = home.appendingPathComponent("Documents/Aura/Plugins")
                        try? FileManager.default.createDirectory(at: pluginsDir, withIntermediateDirectories: true)
                        NSWorkspace.shared.selectFile(nil, inFileViewerRootedAtPath: pluginsDir.path)
                    } label: {
                        Label("Open Folder", systemImage: "folder")
                    }
                    
                    Button(action: loadPlugins) {
                        Label("Scan Folder", systemImage: "arrow.clockwise")
                    }
                }
                .padding(.bottom, 4)
                
                if plugins.isEmpty {
                    VStack(spacing: 8) {
                        Image(systemName: "puzzlepiece")
                            .font(.system(size: 28))
                            .foregroundColor(.secondary)
                        Text("No plugins found.")
                            .foregroundColor(.secondary)
                            .font(.subheadline)
                        Text("Add Python files with docstring schemas to ~/Documents/Aura/Plugins")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                        
                        Button("Open Plugins Folder") {
                            let home = FileManager.default.homeDirectoryForCurrentUser
                            let pluginsDir = home.appendingPathComponent("Documents/Aura/Plugins")
                            try? FileManager.default.createDirectory(at: pluginsDir, withIntermediateDirectories: true)
                            NSWorkspace.shared.selectFile(nil, inFileViewerRootedAtPath: pluginsDir.path)
                        }
                        .buttonStyle(.bordered)
                        .padding(.top, 6)
                    }
                    .frame(maxWidth: .infinity, minHeight: 190)
                    .background(Color.primary.opacity(0.02))
                    .cornerRadius(8)
                } else {
                    VStack(spacing: 16) {
                        ForEach(plugins) { plugin in
                            VStack(alignment: .leading, spacing: 12) {
                                HStack {
                                    Image(systemName: "puzzlepiece.extension.fill")
                                        .foregroundColor(.blue)
                                    Text(plugin.name)
                                        .font(.headline)
                                    Spacer()
                                    
                                    Button("Apply Transform") {
                                        applyPlugin(plugin)
                                    }
                                    .buttonStyle(.borderedProminent)
                                    .tint(Theme.Color.info)
                                    .disabled(isRunningPlugin)
                                }
                                
                                Text(plugin.description)
                                    .font(.subheadline)
                                    .foregroundColor(.secondary)
                                
                                Divider()
                                
                                ForEach(plugin.parameters, id: \.name) { param in
                                    let key = "\(plugin.id)-\(param.name)"
                                    let binding = Binding<Double>(
                                        get: { parameterValues[key] ?? param.default },
                                        set: { parameterValues[key] = $0 }
                                    )
                                    
                                    HStack {
                                        Text(param.name.replacingOccurrences(of: "_", with: " ").capitalized)
                                            .font(.caption)
                                            .fontWeight(.medium)
                                            .frame(width: 140, alignment: .leading)
                                        
                                        if param.type == "toggle" {
                                            Toggle("", isOn: Binding<Bool>(
                                                get: { binding.wrappedValue > 0.5 },
                                                set: { binding.wrappedValue = $0 ? 1.0 : 0.0 }
                                            ))
                                            .labelsHidden()
                                        } else {
                                            Slider(value: binding, in: (param.min ?? 0.0)...(param.max ?? 1.0))
                                            Text(String(format: "%.2f", binding.wrappedValue))
                                                .font(.system(.caption, design: .monospaced))
                                                .foregroundColor(.secondary)
                                                .frame(width: 45, alignment: .trailing)
                                        }
                                    }
                                }
                            }
                            .padding()
                            .background(Color.primary.opacity(0.025))
                            .cornerRadius(12)
                            .overlay(
                                RoundedRectangle(cornerRadius: 12)
                                    .stroke(Color.primary.opacity(0.06), lineWidth: 1)
                            )
                        }
                    }


                }
                
                if let err = pluginsError {
                    Text(err)
                        .foregroundColor(.red)
                        .font(.caption)
                }
            }
            .padding()
        }
        .onAppear {
            loadPlugins()
        }
    }
    
    private func loadPlugins() {
        Task {
            do {
                let list = try await REPLService.shared.getPlugins()
                await MainActor.run {
                    self.plugins = list
                    for p in list {
                        for param in p.parameters {
                            let key = "\(p.id)-\(param.name)"
                            if self.parameterValues[key] == nil {
                                self.parameterValues[key] = param.default
                            }
                        }
                    }
                }
            } catch {
                self.pluginsError = "Failed to load plugins: \(error.localizedDescription)"
            }
        }
    }
    
    private func applyPlugin(_ plugin: REPLService.PluginInfo) {
        isRunningPlugin = true
        pluginsError = nil
        
        var paramsParts: [String] = []
        for param in plugin.parameters {
            let key = "\(plugin.id)-\(param.name)"
            let val = parameterValues[key] ?? param.default
            paramsParts.append("\(param.name)=\(val)")
        }
        
        let execCode = """
import os
import importlib.util
home = os.path.expanduser("~")
plugins_dir = os.path.join(home, "Documents", "Aura", "Plugins")
plugin_path = os.path.join(plugins_dir, "\(plugin.id).py")
spec = importlib.util.spec_from_file_location("\(plugin.id)", plugin_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
df = module.transform(df, \(paramsParts.joined(separator: ", ")))
"""
        
        Task {
            do {
                let result = try await REPLService.shared.execute(execCode)
                await MainActor.run {
                    self.isRunningPlugin = false
                    if let err = result.error, !err.isEmpty {
                        self.pluginsError = "Plugin Error: \(err)"
                    } else {
                        onRunAnalysis()
                    }
                }
            } catch {
                await MainActor.run {
                    self.pluginsError = "Execution failed: \(error.localizedDescription)"
                    self.isRunningPlugin = false
                }
            }
        }
    }
}

