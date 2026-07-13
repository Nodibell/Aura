import SwiftUI
import Charts

struct SummaryView: View {
    let result: AnalysisResult
    @Binding var config: AnalysisConfig
    @Binding var activeModelName: String?
    let onRunAnalysis: () -> Void
    let onExportModelAndCode: () -> Void
    let onAskAI: (String) -> Void
    let onScheduleAnalysis: () -> Void

    @State private var selectedLeaderboardSortMetric: String = "default"
    @State private var showInspector: Bool = true

    // MARK: - Helper Methods for interactive cleaning actions
    
    private func actionsFor(recommendation rec: CleaningRecommendation) -> [CleaningAction] {
        let col = rec.column
        let issueLower = rec.issue.lowercased()
        
        let isNumeric: Bool = {
            if let colProfile = result.profiling?.columns[col] {
                return colProfile.type.lowercased() == "numeric"
            }
            return false
        }()
        
        if issueLower.contains("missing") || issueLower.contains("null") {
            if rec.impact.lowercased() == "high" {
                if isNumeric {
                    return [
                        CleaningAction(column: col, actionType: "drop"),
                        CleaningAction(column: col, actionType: "impute_median"),
                        CleaningAction(column: col, actionType: "impute_mean"),
                        CleaningAction(column: col, actionType: "impute_knn"),
                        CleaningAction(column: col, actionType: "impute_mice")
                    ]
                } else {
                    return [
                        CleaningAction(column: col, actionType: "drop"),
                        CleaningAction(column: col, actionType: "impute_mode")
                    ]
                }
            } else {
                if isNumeric {
                    return [
                        CleaningAction(column: col, actionType: "impute_median"),
                        CleaningAction(column: col, actionType: "impute_mean"),
                        CleaningAction(column: col, actionType: "impute_knn"),
                        CleaningAction(column: col, actionType: "impute_mice"),
                        CleaningAction(column: col, actionType: "drop")
                    ]
                } else {
                    return [
                        CleaningAction(column: col, actionType: "impute_mode"),
                        CleaningAction(column: col, actionType: "drop")
                    ]
                }
            }
        } else if issueLower.contains("outlier") {
            return [
                CleaningAction(column: col, actionType: "clip_outliers"),
                CleaningAction(column: col, actionType: "drop_outliers"),
                CleaningAction(column: col, actionType: "isolation_forest")
            ]
        } else if issueLower.contains("constant") || issueLower.contains("cardinality") {
            return [
                CleaningAction(column: col, actionType: "drop")
            ]
        }
        
        return [
            CleaningAction(column: col, actionType: "drop")
        ]
    }
    
    private func labelFor(actionType: String) -> String {
        switch actionType {
        case "drop": return "Drop Column"
        case "impute_mean": return "Impute Mean"
        case "impute_median": return "Impute Median"
        case "impute_mode": return "Impute Mode"
        case "impute_knn": return "Impute KNN"
        case "impute_mice": return "Impute MICE"
        case "clip_outliers": return "Cap/Floor IQR"
        case "drop_outliers": return "Drop Outliers (IQR)"
        case "isolation_forest": return "Isolation Forest"
        default: return actionType.replacingOccurrences(of: "_", with: " ").capitalized
        }
    }
    
    private func isApplied(_ action: CleaningAction) -> Bool {
        config.cleaningActions.contains(action)
    }
    
    private func toggleAction(_ action: CleaningAction) {
        if isApplied(action) {
            config.cleaningActions.remove(action)
        } else {
            let imputes = ["impute_mean", "impute_median", "impute_mode", "impute_knn", "impute_mice"]
            let outliers = ["clip_outliers", "drop_outliers", "isolation_forest"]
            
            if imputes.contains(action.actionType) {
                config.cleaningActions = config.cleaningActions.filter { !($0.column == action.column && (imputes.contains($0.actionType) || $0.actionType == "drop")) }
            } else if outliers.contains(action.actionType) {
                config.cleaningActions = config.cleaningActions.filter { !($0.column == action.column && outliers.contains($0.actionType)) }
            } else if action.actionType == "drop" {
                config.cleaningActions = config.cleaningActions.filter { $0.column != action.column }
            }
            config.cleaningActions.insert(action)
        }
    }

    var body: some View {
        ScrollViewReader { proxy in
            VStack(spacing: 0) {
                // Sticky Anchor Header Bar
                HStack(spacing: 12) {
                    Label("Jump to:", systemImage: "arrow.right.circle.fill")
                        .font(.system(size: 11, weight: .bold))
                        .foregroundColor(.secondary)
                        .padding(.trailing, 4)
                    
                    AnchorButton(title: "Overview", iconName: "doc.text") {
                        withAnimation { proxy.scrollTo("overview", anchor: .top) }
                    }
                    
                    if let warnings = result.dataLeakageWarnings, !warnings.isEmpty {
                        AnchorButton(title: "Warnings", iconName: "exclamationmark.triangle") {
                            withAnimation { proxy.scrollTo("warnings", anchor: .top) }
                        }
                    }
                    
                    AnchorButton(title: "Model Leaderboard", iconName: "trophy") {
                        withAnimation { proxy.scrollTo("leaderboard", anchor: .top) }
                    }
                    
                    if result.profiling != nil {
                        AnchorButton(title: "Data Profiling", iconName: "chart.bar") {
                            withAnimation { proxy.scrollTo("profiling", anchor: .top) }
                        }
                    }
                    
                    AnchorButton(title: "Missing Values", iconName: "questionmark.circle") {
                        withAnimation { proxy.scrollTo("missing", anchor: .top) }
                    }
                    
                    Spacer()
                    
                    // Collapsible Inspector Toggle
                    Button {
                        withAnimation(.spring(response: 0.3, dampingFraction: 0.8)) {
                            showInspector.toggle()
                        }
                    } label: {
                        HStack(spacing: 5) {
                            Image(systemName: "sidebar.right")
                                .foregroundColor(showInspector ? .purple : .secondary)
                            Text("Inspector")
                                .font(.caption)
                                .foregroundColor(showInspector ? .purple : .secondary)
                        }
                    }
                    .buttonStyle(.plain)
                    .help(showInspector ? "Hide Model Inspector" : "Show Model Inspector")
                }
                .padding(.horizontal, 20)
                .padding(.vertical, 12)
                .background(Color(NSColor.controlBackgroundColor))
                
                Divider()
                
                HStack(spacing: 0) {
                    ScrollView {
                        VStack(alignment: .leading, spacing: 24) {
                            // MARK: — Hero Banner
                            heroBanner
                                .id("overview")

                            // MARK: - Data Leakage Warnings (D7)
                            dataLeakageBanner
                                .id("warnings")

                            // MARK: - Auto-Cleaning Recommendations (D8)
                            cleaningRecommendationsSection

                            // MARK: — Stat Cards
                            LazyVGrid(columns: [GridItem(.adaptive(minimum: 150))], spacing: 14) {
                                let rowSubtitle: String? = {
                                    if let orig = result.originalRowCount, let sampled = result.sampledRowCount, orig > sampled {
                                        return "Sampled from \(formatNumber(orig)) (Smart)"
                                    }
                                    return nil
                                }()
                                StatCard(title: "Rows", value: formatNumber(result.rowCount),
                                         subtitle: rowSubtitle, iconName: "tablecells", color: .blue)
                                StatCard(title: "Columns", value: "\(result.colCount)",
                                         subtitle: "\(result.numericColCount) numeric · \(result.categoricalColCount) categorical\(result.textColCount > 0 ? " · \(result.textColCount) text" : "")",
                                         iconName: "square.split.2x2", color: .purple)
                                StatCard(title: "Missing Cells", value: "\(totalMissingCells())",
                                         subtitle: totalMissingCells() == 0 ? "Clean dataset ✓" : "across \(columnsMissingCount()) columns",
                                         iconName: "questionmark.folder.fill", color: .orange)
                            }
                            
                            // Visual Scorecard Block (Gauges)
                            HStack(spacing: 14) {
                                GaugeScorecard(
                                    title: "Best Model Score",
                                    value: result.metrics.score,
                                    label: result.metrics.scoreType,
                                    color: .green
                                )
                                .frame(maxWidth: .infinity)
                                
                                if let valMetrics = result.valMetrics {
                                    GaugeScorecard(
                                        title: "Validation Score",
                                        value: valMetrics.score,
                                        label: valMetrics.scoreType,
                                        color: .blue
                                    )
                                    .frame(maxWidth: .infinity)
                                }
                            }
                            
                            // MARK: — Model Leaderboard
                            modelLeaderboard
                                .id("leaderboard")

                            // MARK: - Validation Heatmap (Confusion Matrix)
                            if let cm = result.confusionMatrix {
                                ConfusionMatrixView(matrix: cm, title: "Confusion Matrix (Test Set)")
                            }
                            if let valCm = result.valConfusionMatrix {
                                ConfusionMatrixView(matrix: valCm, title: "Confusion Matrix (Validation Set)")
                            }

                            // MARK: - Data Profiling
                            if let profiling = result.profiling {
                                DataProfilingSection(profiling: profiling)
                                    .id("profiling")
                            }

                            // MARK: — Missing Values
                            missingValuesSection
                                .id("missing")
                        }
                        .padding(12)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    
                    if showInspector {
                        Divider()
                        ModelInspectorSidebar(
                            result: result,
                            activeModelName: $activeModelName,
                            showInspector: $showInspector
                        )
                        .frame(width: 300)
                        .transition(.move(edge: .trailing))
                    }
                }
            }
        }
    }

    // MARK: - Hero Banner

    private var heroBanner: some View {
        HStack(alignment: .top, spacing: 16) {
            VStack(alignment: .leading, spacing: 10) {
                HStack(spacing: 8) {
                    // Task type badge
                    Text(result.taskType.uppercased())
                        .font(.system(size: 10, weight: .bold))
                        .foregroundColor(.white)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(
                            Capsule().fill(
                                result.taskType == "classification" ? Color.blue.opacity(0.8) :
                                result.taskType == "object_detection" ? Color.purple.opacity(0.8) :
                                Color.orange.opacity(0.8)
                            )
                        )

                    Text("Target: \(result.targetColumn)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                MarkdownMessageView(content: result.summary)
            }

            Spacer()

            HStack(spacing: 8) {
                Button {
                    onAskAI("Summarize the key findings from this EDA in 3-5 bullet points. Include dataset size, task type, best model performance, and the most important features or correlations.")
                } label: {
                    Label("Ask AI", systemImage: "sparkles")
                        .font(.caption)
                        .fontWeight(.semibold)
                }
                .buttonStyle(.borderedProminent)
                .tint(Color.purple.opacity(0.8))
                .help("Pre-fill the AI chat with a summary question")
                
                Button {
                    onScheduleAnalysis()
                } label: {
                    Label("Schedule...", systemImage: "clock")
                        .font(.caption)
                        .fontWeight(.semibold)
                }
                .buttonStyle(.bordered)
                .help("Set up a cron-like schedule for this dataset analysis")
            }
        }
        .padding(20)
        .background(
            LinearGradient(
                colors: [Color.blue.opacity(0.08), Color.purple.opacity(0.05)],
                startPoint: .topLeading, endPoint: .bottomTrailing
            )
        )
        .cornerRadius(16)
        .overlay(RoundedRectangle(cornerRadius: 16).stroke(Color.blue.opacity(0.15), lineWidth: 1))
    }

    // MARK: - Data Leakage Warnings Banner
    @ViewBuilder
    private var dataLeakageBanner: some View {
        if let warnings = result.dataLeakageWarnings, !warnings.isEmpty {
            VStack(alignment: .leading, spacing: 12) {
                HStack(spacing: 8) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundColor(.red)
                        .font(.headline)
                    Text("Potential Data Leakage Warning")
                        .font(.headline)
                        .foregroundColor(.red)
                }
                
                VStack(alignment: .leading, spacing: 6) {
                    ForEach(warnings, id: \.self) { warning in
                        HStack(alignment: .top, spacing: 6) {
                            Image(systemName: "arrow.right.circle.fill")
                                .font(Theme.Font.caption)
                                .foregroundColor(.red.opacity(0.8))
                                .padding(.top, 3)
                            Text(warning)
                                .font(.subheadline)
                                .foregroundColor(.primary)
                        }
                    }
                }
            }
            .padding(16)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.red.opacity(0.08))
            .cornerRadius(12)
            .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color.red.opacity(0.2), lineWidth: 1))
        }
    }

    // MARK: - Auto-Cleaning Recommendations Panel
    @ViewBuilder
    private var cleaningRecommendationsSection: some View {
        if let recs = result.cleaningRecommendations, !recs.isEmpty {
            VStack(alignment: .leading, spacing: 14) {
                HStack(spacing: 8) {
                    Image(systemName: "wand.and.stars")
                        .foregroundColor(.purple)
                        .font(.title3)
                    Text("Auto-Cleaning Recommendations")
                        .font(.title3)
                        .fontWeight(.bold)
                }
                
                VStack(spacing: 10) {
                    ForEach(recs) { rec in
                        let impactColor: Color = rec.impact.lowercased() == "high" ? .red : (rec.impact.lowercased() == "medium" ? .orange : .blue)
                        let actions = actionsFor(recommendation: rec)
                        
                        HStack(alignment: .center, spacing: 12) {
                            // Left badge / icon
                            VStack(spacing: 4) {
                                Image(systemName: "doc.text.fill.viewfinder")
                                    .font(.title3)
                                    .foregroundColor(.purple.opacity(0.8))
                                
                                Text(rec.impact.uppercased())
                                    .font(.system(size: 8, weight: .bold))
                                    .foregroundColor(.white)
                                    .padding(.horizontal, 6)
                                    .padding(.vertical, 2)
                                    .background(Capsule().fill(impactColor))
                            }
                            .frame(width: 50)
                            
                            VStack(alignment: .leading, spacing: 4) {
                                Text(rec.column)
                                    .font(.subheadline)
                                    .fontWeight(.bold)
                                    .foregroundColor(.primary)
                                
                                Text(rec.issue)
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                
                                Text(rec.recommendation)
                                    .font(.subheadline)
                                    .foregroundColor(.primary.opacity(0.95))
                                    .padding(.top, 2)
                            }
                            
                            Spacer()
                            
                            // Interactive Actions
                            if !actions.isEmpty {
                                if actions.count == 1, let action = actions.first {
                                    let applied = isApplied(action)
                                    Button(action: { toggleAction(action) }) {
                                        HStack(spacing: 4) {
                                            if applied {
                                                Image(systemName: "checkmark.circle.fill")
                                                    .foregroundColor(.green)
                                                Text("Applied")
                                                    .foregroundColor(.green)
                                            } else {
                                                Text("Apply")
                                            }
                                        }
                                        .font(.caption)
                                        .fontWeight(.semibold)
                                        .padding(.horizontal, 10)
                                        .padding(.vertical, 6)
                                        .background(applied ? Color.green.opacity(0.15) : Color.purple.opacity(0.15))
                                        .cornerRadius(6)
                                    }
                                    .buttonStyle(.plain)
                                } else {
                                    let activeAction = actions.first(where: { isApplied($0) })
                                    Menu {
                                        ForEach(actions) { action in
                                            Button {
                                                toggleAction(action)
                                            } label: {
                                                HStack {
                                                    Text(labelFor(actionType: action.actionType))
                                                    if isApplied(action) {
                                                        Image(systemName: "checkmark")
                                                    }
                                                }
                                            }
                                        }
                                    } label: {
                                        HStack(spacing: 4) {
                                            if let active = activeAction {
                                                Image(systemName: "checkmark.circle.fill")
                                                    .foregroundColor(.green)
                                                Text(labelFor(actionType: active.actionType))
                                                    .foregroundColor(.green)
                                            } else {
                                                Text("Select Action")
                                                Image(systemName: "chevron.down")
                                            }
                                        }
                                        .font(.caption)
                                        .fontWeight(.semibold)
                                        .padding(.horizontal, 10)
                                        .padding(.vertical, 6)
                                        .background(activeAction != nil ? Color.green.opacity(0.15) : Color.purple.opacity(0.15))
                                        .cornerRadius(6)
                                    }
                                    .menuStyle(.borderlessButton)
                                }
                            }
                        }
                        .padding(12)
                        .background(Color.primary.opacity(0.02))
                        .cornerRadius(10)
                        .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color.primary.opacity(0.05), lineWidth: 1))
                    }
                }
                
                // Pending Actions floating/bottom badge
                if !config.cleaningActions.isEmpty {
                    HStack {
                        Image(systemName: "wand.and.stars.inverse")
                            .foregroundColor(.purple)
                        Text("\(config.cleaningActions.count) cleaning action(s) selected.")
                            .font(.subheadline)
                            .fontWeight(.bold)
                        Spacer()
                        Button(action: onRunAnalysis) {
                            Label("Run Analysis to Apply", systemImage: "play.fill")
                                .foregroundColor(.white)
                                .padding(.horizontal, 12)
                                .padding(.vertical, 6)
                                .background(LinearGradient(colors: [.purple, .indigo], startPoint: .leading, endPoint: .trailing))
                                .cornerRadius(6)
                        }
                        .buttonStyle(.plain)
                    }
                    .padding(12)
                    .background(Color.purple.opacity(0.08))
                    .cornerRadius(10)
                    .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color.purple.opacity(0.25), lineWidth: 1))
                    .padding(.top, 6)
                }
            }
        }
    }

    // MARK: - Model Leaderboard

    private var modelLeaderboard: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 12) {
                Text("Model Leaderboard")
                    .font(.title3)
                    .fontWeight(.bold)
                Spacer()
                
                Button {
                    onExportModelAndCode()
                } label: {
                    Label("Export Model & Code", systemImage: "square.and.arrow.down")
                        .font(.caption)
                }
                .buttonStyle(.bordered)
                
                Button {
                    let prompt = "Compare these models trained on this dataset: \(result.modelsCompared.map { "\($0.name) (\($0.metric): \(String(format: "%.4f", $0.score)))" }.joined(separator: " vs ")). Which should I choose and why?"
                    onAskAI(prompt)
                } label: {
                    Label("Ask AI", systemImage: "sparkles")
                        .font(.caption)
                }
                .buttonStyle(.bordered)
            }

            let isRegression = result.taskType.lowercased().contains("regress")
            
            HStack {
                Text("Sort By Metric:")
                    .font(Theme.Font.captionBold)
                    .foregroundColor(.secondary)
                
                Picker("", selection: $selectedLeaderboardSortMetric) {
                    Text(isRegression ? "R² Score" : "Accuracy").tag("default")
                    if isRegression {
                        Text("MSE").tag("mse")
                        Text("RMSE").tag("rmse")
                        Text("MAE").tag("mae")
                    } else {
                        Text("F1 Score").tag("f1")
                        Text("Precision").tag("precision")
                        Text("Recall").tag("recall")
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .frame(maxWidth: 320)
            }
            .padding(4)

            VStack(spacing: 10) {
                let sortedList = sortedModels(result.modelsCompared, isRegression: isRegression)
                let scores = sortedList.map { displayScore(for: $0).value }
                let maxVal = scores.max() ?? 1.0
                let minVal = scores.min() ?? 0.0
                let range = maxVal - minVal
                let isLowerBetter = ["mse", "rmse", "mae"].contains(selectedLeaderboardSortMetric)

                ForEach(sortedList) { model in
                    let isWinner = model.name == result.metrics.model
                    let isActive = (activeModelName == model.name) || (activeModelName == nil && isWinner)
                    let displayInfo = displayScore(for: model)
                    
                    let pct: Double = {
                        if range > 0 {
                            if isLowerBetter {
                                return min(max((maxVal - displayInfo.value) / range, 0), 1)
                            } else {
                                return min(max((displayInfo.value - minVal) / range, 0), 1)
                            }
                        }
                        return 1.0
                    }()
                    
                    let scoreColor: Color = isActive ? (displayInfo.value >= 0 || isLowerBetter ? .green : .orange) : .primary
                    let barColor: Color = isActive ? (displayInfo.value >= 0 || isLowerBetter ? .green : .orange) : .blue.opacity(0.4)
                    let strokeColor: Color = isActive ? Color.purple : Color.primary.opacity(0.08)
                    let bgColor: Color = isActive ? Color.purple.opacity(0.06) : Color(nsColor: .controlBackgroundColor)
                    
                    Button {
                        activeModelName = model.name
                    } label: {
                        HStack(spacing: 12) {
                            // Medal
                            ZStack {
                                Circle()
                                    .fill(isWinner ? Color.yellow.opacity(0.2) : Color.primary.opacity(0.04))
                                    .frame(width: 32, height: 32)
                                Image(systemName: isWinner ? "trophy.fill" : "medal")
                                    .font(.system(size: 14))
                                    .foregroundColor(isWinner ? .yellow : .secondary)
                            }

                            // Name
                            VStack(alignment: .leading, spacing: 2) {
                                HStack(spacing: 6) {
                                    Text(model.name)
                                        .font(.subheadline)
                                        .fontWeight(isActive ? .bold : .regular)
                                        .foregroundColor(isActive ? .purple : .primary)
                                    
                                    if isActive {
                                        Text("Active")
                                            .font(.system(size: 8, weight: .bold))
                                            .foregroundColor(.white)
                                            .padding(.horizontal, 5)
                                            .padding(.vertical, 1)
                                            .background(Color.purple)
                                            .cornerRadius(4)
                                    }
                                }
                                Text(displayInfo.label)
                                    .font(.caption2)
                                    .foregroundColor(.secondary)
                            }

                            Spacer()

                            // Score bar
                            ZStack(alignment: .leading) {
                                RoundedRectangle(cornerRadius: 4)
                                    .fill(Color.primary.opacity(0.06))
                                    .frame(width: 100, height: 6)
                                RoundedRectangle(cornerRadius: 4)
                                    .fill(barColor)
                                    .frame(width: max(4, 100 * pct), height: 6)
                            }

                            Text(formatMetricValue(displayInfo.value))
                                .font(.system(.body, design: .monospaced))
                                .fontWeight(isActive ? .bold : .regular)
                                .foregroundColor(scoreColor)
                                .lineLimit(1)
                                .minimumScaleFactor(0.7)
                                .frame(width: 90, alignment: .trailing)
                        }
                        .padding(12)
                        .background(bgColor)
                        .cornerRadius(10)
                        .overlay(
                            RoundedRectangle(cornerRadius: 10)
                                .stroke(strokeColor, lineWidth: isActive ? 2.0 : 1.0)
                        )
                    }
                    .buttonStyle(.plain)
                }
            }.padding(4)
            
            // Phase B: Cross-Validation & Baseline details
            if let cvMean = result.cvMean {
                VStack(alignment: .leading, spacing: 10) {
                    Divider()
                        .padding(.vertical, 4)
                    
                    Text("Validation & Reliability Statistics")
                        .font(.subheadline)
                        .fontWeight(.bold)
                        .foregroundColor(.primary)
                    
                    HStack(spacing: 16) {
                        // 5-Fold Cross Validation card
                        VStack(alignment: .leading, spacing: 4) {
                            Text("5-Fold Cross-Validation")
                                .font(.system(size: 10))
                                .foregroundColor(.secondary)
                            HStack(alignment: .lastTextBaseline, spacing: 4) {
                                Text(formatMetricValue(cvMean))
                                    .font(.system(size: 16, weight: .bold, design: .rounded))
                                    .foregroundColor(.green)
                                    .lineLimit(1)
                                    .minimumScaleFactor(0.8)
                                if let cvStd = result.cvStd {
                                    Text(String(format: "(±%.4f)", cvStd))
                                        .font(.system(size: 10))
                                        .foregroundColor(.secondary)
                                        .lineLimit(1)
                                        .minimumScaleFactor(0.8)
                                }
                            }
                        }
                        .padding(10)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color.primary.opacity(0.02))
                        .cornerRadius(8)
                        .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.primary.opacity(0.05), lineWidth: 1))
                        
                        // Baseline Model card
                        if let dummy = result.dummyBaselineScore {
                            VStack(alignment: .leading, spacing: 4) {
                                Text("Baseline Model (Dummy)")
                                    .font(.system(size: 10))
                                    .foregroundColor(.secondary)
                                HStack(alignment: .lastTextBaseline, spacing: 4) {
                                    Text(formatMetricValue(dummy))
                                        .font(.system(size: 16, weight: .bold, design: .rounded))
                                        .foregroundColor(.orange)
                                        .lineLimit(1)
                                        .minimumScaleFactor(0.8)
                                    
                                    let diff = result.metrics.score - dummy
                                    let diffPct = dummy > 0 ? (diff / dummy) * 100 : 0
                                    Text(String(format: "(+%0.1f%%)", diffPct))
                                        .font(.system(size: 10, weight: .bold))
                                        .foregroundColor(.green)
                                        .lineLimit(1)
                                }
                            }
                            .padding(10)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(Color.primary.opacity(0.02))
                            .cornerRadius(8)
                            .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.primary.opacity(0.05), lineWidth: 1))
                        }
                    }
                }
                .padding(.top, 4)
            }
        }
    }

    // MARK: - Missing Values Section

    private var missingValuesSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Missing Values Per Column")
                .font(.title3)
                .fontWeight(.bold)

            let nonZeroPairs = Array(result.missingValues.filter { $0.value > 0 }.sorted { $0.value > $1.value }.prefix(10))

            if nonZeroPairs.isEmpty {
                HStack(spacing: 10) {
                    Image(systemName: "checkmark.circle.fill").foregroundColor(.green)
                    Text("No missing values found — clean dataset!")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                .padding(16)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color.green.opacity(0.04))
                .cornerRadius(10)
                .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color.green.opacity(0.15)))
            } else {
                // Bar chart for missing values
                Chart {
                    ForEach(nonZeroPairs, id: \.key) { pair in
                        BarMark(
                            x: .value("Missing", pair.value),
                            y: .value("Column", pair.key)
                        )
                        .foregroundStyle(
                            LinearGradient(colors: [.orange, .red.opacity(0.7)],
                                           startPoint: .leading, endPoint: .trailing)
                        )
                        .cornerRadius(4)
                        .annotation(position: .trailing) {
                            Text("\(pair.value)")
                                .font(.caption2)
                                .foregroundColor(.secondary)
                        }
                    }
                }
                .chartXAxisLabel("Missing Count")
                .frame(height: CGFloat(max(nonZeroPairs.count * 44, 145)))
                .padding(16)
                .background(Color.primary.opacity(0.02))
                .cornerRadius(12)
                .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color.primary.opacity(0.05)))
            }
        }
    }

    // MARK: - Helpers

    private func totalMissingCells() -> Int { result.missingValues.values.reduce(0, +) }
    private func columnsMissingCount() -> Int { result.missingValues.filter { $0.value > 0 }.count }

    private func formatNumber(_ n: Int) -> String {
        let formatter = NumberFormatter()
        formatter.numberStyle = .decimal
        return formatter.string(from: NSNumber(value: n)) ?? "\(n)"
    }
    
    private func formatMetricValue(_ value: Double) -> String {
        let absVal = abs(value)
        if absVal == 0 {
            return "0.0000"
        } else if absVal >= 1_000_000 {
            return String(format: "%.4e", value)
        } else if absVal >= 100 {
            return String(format: "%.2f", value)
        } else {
            return String(format: "%.4f", value)
        }
    }
    
    private func sortedModels(_ models: [ModelComparison], isRegression: Bool) -> [ModelComparison] {
        switch selectedLeaderboardSortMetric {
        case "mse":
            return models.sorted { ($0.mse ?? Double.infinity) < ($1.mse ?? Double.infinity) }
        case "rmse":
            return models.sorted { ($0.rmse ?? Double.infinity) < ($1.rmse ?? Double.infinity) }
        case "mae":
            return models.sorted { ($0.mae ?? Double.infinity) < ($1.mae ?? Double.infinity) }
        case "f1":
            return models.sorted { ($0.f1 ?? -Double.infinity) > ($1.f1 ?? -Double.infinity) }
        case "precision":
            return models.sorted { ($0.precision ?? -Double.infinity) > ($1.precision ?? -Double.infinity) }
        case "recall":
            return models.sorted { ($0.recall ?? -Double.infinity) > ($1.recall ?? -Double.infinity) }
        default:
            return models.sorted { $0.score > $1.score }
        }
    }
    
    private func displayScore(for model: ModelComparison) -> (value: Double, label: String) {
        switch selectedLeaderboardSortMetric {
        case "mse":
            return (model.mse ?? 0.0, "MSE")
        case "rmse":
            return (model.rmse ?? 0.0, "RMSE")
        case "mae":
            return (model.mae ?? 0.0, "MAE")
        case "f1":
            return (model.f1 ?? 0.0, "F1 Score")
        case "precision":
            return (model.precision ?? 0.0, "Precision")
        case "recall":
            return (model.recall ?? 0.0, "Recall")
        default:
            return (model.score, model.metric)
        }
    }
}

// MARK: - StatCard

struct StatCard: View {
    let title: String
    let value: String
    let subtitle: String?
    let iconName: String
    let color: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: iconName)
                    .font(.title2)
                    .foregroundStyle(LinearGradient(colors: [color, color.opacity(0.6)],
                                                    startPoint: .topLeading, endPoint: .bottomTrailing))
                Spacer()
            }
            VStack(alignment: .leading, spacing: 4) {
                Text(value)
                    .font(.system(size: 20, weight: .bold, design: .rounded))
                    .lineLimit(1)
                    .minimumScaleFactor(0.7)
                Text(title)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            if let subtitle = subtitle {
                Text(subtitle)
                    .font(.system(size: 10))
                    .foregroundColor(.secondary)
                    .lineLimit(2)
            } else {
                Text(" ").font(.system(size: 10))
            }
        }
        .padding(16)
        .frame(minWidth: 140)
        .background(
            ZStack {
                Color(nsColor: .controlBackgroundColor)
                LinearGradient(colors: [color.opacity(0.06), color.opacity(0.02)],
                               startPoint: .topLeading, endPoint: .bottomTrailing)
            }
        )
        .cornerRadius(14)
        .overlay(RoundedRectangle(cornerRadius: 14).stroke(color.opacity(0.12), lineWidth: 1))
        .shadow(color: Color.black.opacity(0.04), radius: 6, x: 0, y: 2)
    }
}

// MARK: - Confusion Matrix View

struct ConfusionMatrixView: View {
    let matrix: ConfusionMatrixData
    var title: String = "Confusion Matrix (Test Set)"
    
    // Safety cap to prevent AttributeGraph crashes on datasets with thousands of classes
    private let maxDisplayClasses = 30
    
    var body: some View {
        let isTruncated = matrix.labels.count > maxDisplayClasses
        let displayLabels = isTruncated ? Array(matrix.labels.prefix(maxDisplayClasses)) : matrix.labels
        let displayRows = isTruncated ? Array(matrix.values.prefix(maxDisplayClasses)) : matrix.values
        
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .bottom, spacing: 8) {
                Text(title)
                    .font(.headline)
                if isTruncated {
                    Text("(Showing top \(maxDisplayClasses) of \(matrix.labels.count) classes)")
                        .font(.caption)
                        .foregroundColor(.orange)
                }
            }
            
            ScrollView(.horizontal, showsIndicators: true) {
                HStack(spacing: 12) {
                    // Row labels (True labels) on the left
                    VStack(alignment: .trailing, spacing: 8) {
                        Text("True")
                            .font(.system(size: 9, weight: .bold))
                            .foregroundColor(.secondary)
                            .padding(.bottom, 10)
                        
                        ForEach(displayLabels, id: \.self) { label in
                            Text(label)
                                .font(.system(size: 11, weight: .bold))
                                .lineLimit(1)
                                .minimumScaleFactor(0.75)
                                .frame(height: 36)
                        }
                    }
                    
                    // Matrix Grid
                    VStack(alignment: .leading, spacing: 8) {
                        // Column labels (Predicted labels) on top
                        HStack(spacing: 8) {
                            ForEach(displayLabels, id: \.self) { label in
                                Text(label)
                                    .font(.system(size: 11, weight: .bold))
                                    .lineLimit(1)
                                    .minimumScaleFactor(0.75)
                                    .frame(width: 65, alignment: .center)
                            }
                        }
                        
                        // Grid cells
                        ForEach(0..<displayRows.count, id: \.self) { rowIdx in
                            let fullRow = displayRows[rowIdx]
                            let displayRow = isTruncated ? Array(fullRow.prefix(maxDisplayClasses)) : fullRow
                            let rowSum = Double(fullRow.reduce(0, +)) // Keep the full row sum so color ratios remain accurate
                            
                            HStack(spacing: 8) {
                                ForEach(0..<displayRow.count, id: \.self) { colIdx in
                                    let val = displayRow[colIdx]
                                    let ratio = rowSum > 0 ? Double(val) / rowSum : 0.0
                                    
                                    Text("\(val)")
                                        .font(.system(.body, design: .rounded))
                                        .fontWeight(.bold)
                                        .foregroundColor(textColor(ratio: ratio))
                                        .frame(width: 65, height: 36)
                                        .background(
                                            RoundedRectangle(cornerRadius: 6)
                                                .fill(cellColor(rowIdx: rowIdx, colIdx: colIdx, ratio: ratio))
                                        )
                                        .help("True: \(displayLabels[rowIdx]), Predicted: \(displayLabels[colIdx])")
                                }
                            }
                        }
                        
                        Text("Predicted")
                            .font(.system(size: 9, weight: .bold))
                            .foregroundColor(.secondary)
                            .frame(maxWidth: .infinity, alignment: .center)
                            .padding(.top, 4)
                    }
                }
            }
        }
        .padding(16)
        .background(Color(nsColor: .controlBackgroundColor))
        .cornerRadius(12)
        .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color.primary.opacity(0.07)))
        .shadow(color: Color.black.opacity(0.04), radius: 6, x: 0, y: 2)
    }
    
    private func cellColor(rowIdx: Int, colIdx: Int, ratio: Double) -> Color {
        if rowIdx == colIdx {
            return Color.green.opacity(max(0.1, ratio * 0.8))
        } else {
            return Color.red.opacity(max(0.05, ratio * 0.8))
        }
    }
    
    private func textColor(ratio: Double) -> Color {
        return ratio > 0.4 ? .white : .primary
    }
}

// MARK: - Data Profiling Section

struct DataProfilingSection: View {
    let profiling: DataProfiling
    @State private var expandedColumns: Set<String> = []
    
    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Text("Data Profiling & Quality")
                    .font(.title3)
                    .fontWeight(.bold)
                Spacer()
                if profiling.duplicateRows > 0 {
                    HStack(spacing: 4) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundColor(.orange)
                        Text("\(profiling.duplicateRows) Duplicate Rows")
                            .font(.caption)
                            .foregroundColor(.orange)
                            .fontWeight(.semibold)
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Color.orange.opacity(0.1))
                    .cornerRadius(6)
                } else {
                    HStack(spacing: 4) {
                        Image(systemName: "checkmark.shield.fill")
                            .foregroundColor(.green)
                        Text("No Duplicates")
                            .font(.caption)
                            .foregroundColor(.green)
                    }
                }
            }
            
            VStack(spacing: 0) {
                // Table Header
                HStack {
                    Text("Column Name")
                        .font(.caption)
                        .fontWeight(.bold)
                        .frame(width: 180, alignment: .leading)
                    Spacer()
                    Text("Type")
                        .font(.caption)
                        .fontWeight(.bold)
                        .frame(width: 100, alignment: .leading)
                    Spacer()
                    Text("Missing")
                        .font(.caption)
                        .fontWeight(.bold)
                        .frame(width: 80, alignment: .trailing)
                    Spacer()
                    Text("Unique")
                        .font(.caption)
                        .fontWeight(.bold)
                        .frame(width: 80, alignment: .trailing)
                }
                .foregroundColor(.secondary)
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                
                Divider()
                
                // Rows
                ForEach(profiling.columns.sorted(by: { $0.key < $1.key }), id: \.key) { colName, colProfile in
                    let isExpanded = expandedColumns.contains(colName)
                    
                    VStack(spacing: 0) {
                        Button {
                            withAnimation(.easeInOut(duration: 0.2)) {
                                if isExpanded {
                                    expandedColumns.remove(colName)
                                } else {
                                    expandedColumns.insert(colName)
                                }
                            }
                        } label: {
                            HStack {
                                HStack(spacing: 6) {
                                    Image(systemName: isExpanded ? "chevron.down" : "chevron.right")
                                        .font(.caption2)
                                        .foregroundColor(.secondary)
                                    Text(colName)
                                        .font(.subheadline)
                                        .fontWeight(.semibold)
                                        .foregroundColor(.primary)
                                }
                                .frame(width: 180, alignment: .leading)
                                
                                Spacer()
                                
                                Text(colProfile.type.capitalized)
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                    .frame(width: 100, alignment: .leading)
                                
                                Spacer()
                                
                                Text("\(colProfile.missing)")
                                    .font(.system(.subheadline, design: .monospaced))
                                    .foregroundColor(colProfile.missing > 0 ? .orange : .secondary)
                                    .frame(width: 80, alignment: .trailing)
                                
                                Spacer()
                                
                                Text("\(colProfile.nunique)")
                                    .font(.system(.subheadline, design: .monospaced))
                                    .foregroundColor(.primary)
                                    .frame(width: 80, alignment: .trailing)
                            }
                            .padding(.horizontal, 12)
                            .padding(.vertical, 10)
                            .background(Color.primary.opacity(isExpanded ? 0.03 : 0))
                        }
                        .buttonStyle(.plain)
                        
                        if isExpanded {
                            VStack(alignment: .leading, spacing: 8) {
                                if let stats = colProfile.stats {
                                    let isText = colProfile.type.lowercased() == "text"
                                    // Numerical or Text Stats Grid
                                    LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible()), GridItem(.flexible()), GridItem(.flexible())], spacing: 8) {
                                        StatMiniBox(label: isText ? "Mean Length" : "Mean", value: isText ? String(format: "%.0f", stats.mean) : String(format: "%.4f", stats.mean))
                                        StatMiniBox(label: isText ? "Std Dev Length" : "Std Dev", value: isText ? String(format: "%.0f", stats.std) : String(format: "%.4f", stats.std))
                                        StatMiniBox(label: isText ? "Min Length" : "Min", value: isText ? String(format: "%.0f", stats.min) : String(format: "%.4f", stats.min))
                                        StatMiniBox(label: isText ? "Max Length" : "Max", value: isText ? String(format: "%.0f", stats.max) : String(format: "%.4f", stats.max))
                                        StatMiniBox(label: isText ? "25% Length (Q1)" : "25% (Q1)", value: isText ? String(format: "%.0f", stats.p25) : String(format: "%.4f", stats.p25))
                                        StatMiniBox(label: isText ? "Median Length" : "50% (Med)", value: isText ? String(format: "%.0f", stats.p50) : String(format: "%.4f", stats.p50))
                                        StatMiniBox(label: isText ? "75% Length (Q3)" : "75% (Q3)", value: isText ? String(format: "%.0f", stats.p75) : String(format: "%.4f", stats.p75))
                                    }
                                } else if let topCats = colProfile.topCategories {
                                    // Categorical top categories list
                                    VStack(alignment: .leading, spacing: 4) {
                                        if topCats.count > 1 {
                                            Text("Top Categories:")
                                                .font(.caption)
                                                .foregroundColor(.secondary)
                                                .fontWeight(.bold)
                                        }
                                        ForEach(topCats) { cat in
                                            HStack {
                                                Text(cat.value.isEmpty ? "(empty)" : cat.value)
                                                    .font(.caption)
                                                    .lineLimit(1)
                                                Spacer()
                                                Text("\(cat.count) count")
                                                    .font(.caption)
                                                    .foregroundColor(.secondary)
                                            }
                                        }
                                    }
                                }
                            }
                            .padding(.leading, 30)
                            .padding(.trailing, 12)
                            .padding(.bottom, 12)
                            .background(Color.primary.opacity(0.03))
                        }
                        
                        Divider()
                    }
                }
            }
            .background(Color(nsColor: .controlBackgroundColor))
            .cornerRadius(12)
            .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color.primary.opacity(0.07), lineWidth: 1))
            .shadow(color: Color.black.opacity(0.04), radius: 6, x: 0, y: 2)
        }
    }
}

// MARK: - Stat Mini Box

struct StatMiniBox: View {
    let label: String
    let value: String
    
    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label)
                .font(.system(size: 9))
                .foregroundColor(.secondary)
            Text(value)
                .font(.system(size: 11, design: .monospaced))
                .fontWeight(.bold)
                .foregroundColor(.primary)
                .lineLimit(1)
                .minimumScaleFactor(0.8)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 6)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.primary.opacity(0.02))
        .cornerRadius(6)
        .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.primary.opacity(0.04), lineWidth: 0.5))
    }
}

struct AnchorButton: View {
    let title: String
    let iconName: String?
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
                    .font(Theme.Font.captionBold)
            }
            .foregroundColor(isHovered ? .white : .secondary)
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(
                Capsule()
                    .fill(isHovered ? Color.accentColor : Color.primary.opacity(0.04))
            )
            .overlay(
                Capsule()
                    .stroke(isHovered ? Color.accentColor : Color.primary.opacity(0.08), lineWidth: 1)
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


// MARK: - Gauge Scorecard

struct GaugeScorecard: View {
    let title: String
    let value: Double
    let label: String
    let color: Color
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(title)
                    .font(.caption)
                    .fontWeight(.bold)
                    .foregroundColor(.secondary)
                Spacer()
                Image(systemName: "gauge")
                    .font(.caption)
                    .foregroundColor(color)
            }
            
            HStack(spacing: 12) {
                let displayVal = max(0, min(value, 1.0))
                
                Gauge(value: displayVal, in: 0...1) {
                    Text("")
                } currentValueLabel: {
                    Text(String(format: "%.3f", value))
                        .font(.system(.title3, design: .rounded))
                        .fontWeight(.bold)
                        .foregroundColor(color)
                }
                .gaugeStyle(.accessoryCircular)
                .tint(color.gradient)
                .frame(width: 50, height: 50)
                
                VStack(alignment: .leading, spacing: 2) {
                    Text(String(format: "%.1f%%", displayVal * 100))
                        .font(.subheadline)
                        .fontWeight(.bold)
                        .foregroundColor(.primary)
                    Text(label)
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
            }
        }
        .padding(14)
        .background(Color(nsColor: .controlBackgroundColor))
        .cornerRadius(12)
        .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color.primary.opacity(0.06), lineWidth: 1))
    }
}


// MARK: - Model Inspector Sidebar

struct ModelInspectorSidebar: View {
    let result: AnalysisResult
    @Binding var activeModelName: String?
    @Binding var showInspector: Bool
    
    var body: some View {
        let isRegression = result.taskType.lowercased().contains("regress")
        let activeName = activeModelName ?? result.metrics.model
        let activeModel = result.modelsCompared.first(where: { $0.name == activeName })
        
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 8) {
                Image(systemName: "sidebar.right")
                    .foregroundColor(.purple)
                Text("Model Inspector")
                    .font(.headline)
                Spacer()
                Button {
                    withAnimation { showInspector = false }
                } label: {
                    Image(systemName: "xmark")
                        .foregroundColor(.secondary)
                }
                .buttonStyle(.plain)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 14)
            .background(Color.primary.opacity(0.02))
            
            Divider()
            
            if let model = activeModel {
                ScrollView {
                    VStack(alignment: .leading, spacing: 18) {
                        VStack(alignment: .leading, spacing: 4) {
                            Text(model.name)
                                .font(.title3)
                                .fontWeight(.bold)
                                .foregroundColor(.purple)
                            
                            Text(model.name == result.metrics.model ? "Optimal Pipeline Winner" : "Candidate Model")
                                .font(.caption)
                                .fontWeight(.semibold)
                                .foregroundColor(model.name == result.metrics.model ? .green : .secondary)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(model.name == result.metrics.model ? Color.green.opacity(0.15) : Color.primary.opacity(0.05))
                                .cornerRadius(4)
                        }
                        
                        Divider()
                        
                        VStack(alignment: .leading, spacing: 12) {
                            Text("Validation Metrics")
                                .font(.system(size: 11, weight: .bold))
                                .foregroundColor(.secondary)
                                .textCase(.uppercase)
                            
                            let isLowerBetter = ["mse", "rmse", "mae"].contains(model.metric.lowercased())
                            let metricVal = model.score
                            
                            VStack(spacing: 8) {
                                HStack {
                                    Text(model.metric)
                                        .font(.subheadline)
                                        .fontWeight(.bold)
                                    Spacer()
                                    Text(String(format: "%.4f", metricVal))
                                        .font(.system(.body, design: .monospaced))
                                        .fontWeight(.bold)
                                }
                                
                                if !isLowerBetter {
                                    Gauge(value: max(0, min(metricVal, 1.0)), in: 0...1) {
                                        Text("")
                                    }
                                    .tint(Color.blue.gradient)
                                } else {
                                    ProgressView(value: 1.0)
                                        .tint(.red)
                                }
                            }
                            .padding(10)
                            .background(Color.primary.opacity(0.02))
                            .cornerRadius(8)
                            
                            if isRegression {
                                if let mse = model.mse { MetricRow(label: "MSE", value: mse) }
                                if let rmse = model.rmse { MetricRow(label: "RMSE", value: rmse) }
                                if let mae = model.mae { MetricRow(label: "MAE", value: mae) }
                            } else {
                                if let f1 = model.f1 { MetricRow(label: "F1 Score", value: f1) }
                                if let prec = model.precision { MetricRow(label: "Precision", value: prec) }
                                if let recall = model.recall { MetricRow(label: "Recall", value: recall) }
                            }
                        }
                        
                        Divider()
                        
                        VStack(alignment: .leading, spacing: 10) {
                            Text("Optimal Parameters")
                                .font(.system(size: 11, weight: .bold))
                                .foregroundColor(.secondary)
                                .textCase(.uppercase)
                            
                            if let params = model.params, !params.isEmpty {
                                ForEach(params.keys.sorted(), id: \.self) { key in
                                    HStack {
                                        Text(key)
                                            .font(.caption)
                                            .foregroundColor(.secondary)
                                        Spacer()
                                        Text(params[key] ?? "")
                                            .font(.system(.caption, design: .monospaced))
                                            .fontWeight(.semibold)
                                            .foregroundColor(.primary)
                                    }
                                    .padding(.vertical, 2)
                                }
                            } else {
                                Text("Standard baseline parameters used.")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                    .italic()
                            }
                        }
                        
                        Divider()
                        
                        if activeModelName != model.name && model.name != result.metrics.model {
                            Button {
                                activeModelName = model.name
                            } label: {
                                HStack {
                                    Image(systemName: "checkmark.circle")
                                    Text("Set as Active Pipeline Model")
                                        .fontWeight(.bold)
                                }
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 8)
                                .background(Color.blue)
                                .foregroundColor(.white)
                                .cornerRadius(8)
                            }
                            .buttonStyle(.plain)
                            .padding(.top, 10)
                        } else if model.name == result.metrics.model && activeModelName != nil {
                            Button {
                                activeModelName = nil
                            } label: {
                                HStack {
                                    Image(systemName: "arrow.counterclockwise")
                                    Text("Reset to Baseline Winner")
                                        .fontWeight(.bold)
                                }
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 8)
                                .background(Color.primary.opacity(0.05))
                                .foregroundColor(.primary)
                                .cornerRadius(8)
                            }
                            .buttonStyle(.plain)
                            .padding(.top, 10)
                        }
                    }
                    .padding(16)
                }
            } else {
                VStack {
                    Spacer()
                    Text("Select a model in the leaderboard to inspect details.")
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                        .padding(20)
                    Spacer()
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .background(Color(NSColor.controlBackgroundColor))
    }
}

struct MetricRow: View {
    let label: String
    let value: Double
    
    var body: some View {
        HStack {
            Text(label)
                .font(.caption)
                .foregroundColor(.secondary)
            Spacer()
            Text(String(format: "%.4f", value))
                .font(.system(.caption, design: .monospaced))
                .fontWeight(.bold)
        }
        .padding(.vertical, 1)
    }
}
