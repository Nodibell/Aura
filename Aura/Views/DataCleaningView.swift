import SwiftUI

struct DataCleaningView: View {
    let result: AnalysisResult
    @Binding var config: AnalysisConfig
    let onRunAnalysis: () -> Void
    
    @State private var hoverColumn: String? = nil

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
                }
                Text("Select custom imputation, outlier treatment, and encoding actions per column. The target column '\(result.targetColumn)' cannot be modified.")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding()
            .background(Color.primary.opacity(0.01))
            
            Divider()

            // Columns List
            ScrollView {
                VStack(spacing: 12) {
                    let sortedCols = result.columns.sorted()
                    ForEach(sortedCols, id: \.self) { col in
                        let isTarget = (col == result.targetColumn)
                        let colProfile = result.profiling?.columns[col]
                        let colType = colProfile?.type.lowercased() ?? "numeric"
                        let missingCount = colProfile?.missing ?? 0
                        
                        HStack(alignment: .top, spacing: 16) {
                            // Column Metadata
                            VStack(alignment: .leading, spacing: 4) {
                                HStack(spacing: 6) {
                                    Image(systemName: iconForType(colType))
                                        .foregroundColor(isTarget ? .purple : .secondary)
                                        .font(.caption)
                                    Text(col)
                                        .font(.subheadline)
                                        .fontWeight(.bold)
                                        .foregroundColor(isTarget ? .purple : .primary)
                                        .lineLimit(1)
                                }
                                
                                HStack(spacing: 8) {
                                    Text(colType.capitalized)
                                        .font(.system(size: 10, weight: .semibold))
                                        .foregroundColor(.secondary)
                                    if missingCount > 0 {
                                        Text("\(missingCount) missing")
                                            .font(.system(size: 10, weight: .bold))
                                            .foregroundColor(.orange)
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
                                }
                            }
                            .frame(width: 180, alignment: .leading)
                            
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
                                                    HStack {
                                                        Text(opt.label)
                                                        if getImputation(for: col) == opt {
                                                            Image(systemName: "checkmark")
                                                        }
                                                    }
                                                }
                                            }
                                        } label: {
                                            let current = getImputation(for: col)
                                            Text(current.label)
                                                .font(.caption)
                                                .fontWeight(current != .none ? .bold : .regular)
                                                .foregroundColor(current != .none ? .purple : .primary)
                                        }
                                        .menuStyle(.borderlessButton)
                                        .frame(width: 130, alignment: .leading)
                                    }
                                    
                                    // 2. Outlier Picker (Numeric only)
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
                                                        HStack {
                                                            Text(opt.label)
                                                            if getOutlier(for: col) == opt {
                                                                Image(systemName: "checkmark")
                                                            }
                                                        }
                                                    }
                                                }
                                            } label: {
                                                let current = getOutlier(for: col)
                                                Text(current.label)
                                                    .font(.caption)
                                                    .fontWeight(current != .none ? .bold : .regular)
                                                    .foregroundColor(current != .none ? .purple : .primary)
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
                                            
                                            HStack(spacing: 4) {
                                                Menu {
                                                    ForEach(EncodingOption.allCases) { opt in
                                                        Button {
                                                            setEncoding(opt, for: col)
                                                        } label: {
                                                            HStack {
                                                                Text(opt.label)
                                                                if getEncoding(for: col) == opt {
                                                                    Image(systemName: "checkmark")
                                                                }
                                                            }
                                                        }
                                                    }
                                                } label: {
                                                    let current = getEncoding(for: col)
                                                    Text(current.label)
                                                        .font(.caption)
                                                        .fontWeight(current != .none ? .bold : .regular)
                                                        .foregroundColor(current != .none ? .purple : .primary)
                                                }
                                                .menuStyle(.borderlessButton)
                                                .frame(width: 130, alignment: .leading)
                                                
                                                if getEncoding(for: col) == .target {
                                                    Image(systemName: "exclamationmark.triangle.fill")
                                                        .foregroundColor(.orange)
                                                        .font(.caption2)
                                                        .help("Warning: Target encoding has a high risk of target leakage/overfitting if validation splits are not managed carefully.")
                                                }
                                            }
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
                            }
                        }
                        .padding(12)
                        .background(Color.primary.opacity(hoverColumn == col ? 0.04 : 0.01))
                        .cornerRadius(8)
                        .overlay(
                            RoundedRectangle(cornerRadius: 8)
                                .stroke(Color.primary.opacity(hoverColumn == col ? 0.08 : 0.03), lineWidth: 1)
                        )
                        .onHover { isHover in
                            withAnimation(.easeInOut(duration: 0.15)) {
                                hoverColumn = isHover ? col : nil
                            }
                        }
                    }
                }
                .padding()
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
}
