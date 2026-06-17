import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @State private var selectedFileURL: URL? = nil
    @State private var datasetURLInput: String = ""
    @State private var isAnalyzing = false
    @State private var isPreloading = false
    @State private var previewResult: DatasetPreview? = nil
    @State private var result: AnalysisResult? = nil
    @State private var errorMessage: String? = nil
    @State private var selectedTab: String = "Summary"
    @State private var fileDetails: String = ""
    @State private var showAIPanel: Bool = false
    @State private var showRenameAlert = false
    @State private var renameText = ""
    @State private var itemToRename: HistoryItem? = nil
    @State private var showExportSheet: Bool = false
    @State private var showModelExportSheet: Bool = false
    @State private var showURLInputAlert = false
    @State private var urlInputText = ""
    @State private var selectedTargetName = ""
    @AppStorage("Aura_hasSeenOnboarding") private var hasSeenOnboarding = false
    @State private var showOnboarding = false

    /// Holds all user configuration from the Preview screen (type, excluded rows, time col)
    @State private var analysisConfig: AnalysisConfig = AnalysisConfig()
    
    @State private var trainColumns: [String] = []
    @State private var selectedDataTab: String = "train"
    
    @State private var progressFraction: Double = 0.0
    @State private var progressMessage: String = ""

    @State private var chatViewModel = ChatViewModel()
    private let ollamaStatus = OllamaStatusChecker.shared
    private let historyService = AnalysisHistoryService.shared

    private var navigationTitleText: String {
        if let name = selectedFileURL?.lastPathComponent { return name }
        if !datasetURLInput.isEmpty { return "Web Dataset" }
        return "Aura Dashboard"
    }

    var body: some View {
        // 1. Native macOS Sidebar Architecture
        NavigationSplitView {
            sidebarContent
                .navigationSplitViewColumnWidth(min: 240, ideal: 260, max: 320)
        } detail: {
            // 2. Main Content Canvas & AI Panel
            HStack(spacing: 0) {
                mainContent
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(Color(nsColor: .windowBackgroundColor))
                
                // Custom Right-Side Panel
                if showAIPanel && result != nil {
                    Divider()
                    AIChatPanel(
                        viewModel: chatViewModel,
                        ollamaStatus: ollamaStatus,
                        analysisResult: result?.resultForTarget(selectedTargetName)
                    )
                    .frame(width: 325)
                    .background(Color(nsColor: .windowBackgroundColor))
                    .transition(.move(edge: .trailing))
                }
            }
            // 4. Native macOS Toolbar & Titles
            .navigationTitle(navigationTitleText)
            .navigationSubtitle(result != nil ? "Analysis Complete" : (previewResult != nil ? "Dataset Preview" : ""))
            .toolbar {
                ToolbarItemGroup(placement: .primaryAction) {
                    
                    // Close Dataset Button
                    if result != nil || previewResult != nil {
                        Button(action: clearSelection) {
                            Image(systemName: "xmark.circle")
                                .foregroundColor(.secondary)
                        }
                        .help("Close current dataset")
                    }
                    
                    // Export Button
                    if result != nil {
                        Divider()
                        Button {
                            showExportSheet = true
                        } label: {
                            Image(systemName: "square.and.arrow.up")
                        }
                        .help("Export analysis report as Markdown")
                        .keyboardShortcut("e", modifiers: [.command, .shift])
                        
                        // AI Panel Toggle
                        Button {
                            withAnimation(.spring(response: 0.3, dampingFraction: 0.8)) {
                                showAIPanel.toggle()
                            }
                        } label: {
                            Image(systemName: showAIPanel ? "sidebar.right" : "sparkles")
                                .foregroundColor(showAIPanel ? .purple : .primary)
                        }
                        .help(showAIPanel ? "Hide AI panel" : "Show AI Analyst panel")
                        .keyboardShortcut("a", modifiers: [.command, .shift])
                    }
                }
            }
        }
        .colorScheme(.dark)
        .alert("Rename Analysis", isPresented: $showRenameAlert) {
            TextField("Name", text: $renameText)
            Button("Cancel", role: .cancel) { }
            Button("Rename") {
                if let item = itemToRename {
                    historyService.renameItem(item, to: renameText)
                }
            }
        } message: {
            Text("Enter a descriptive name for this analysis.")
        }
        .alert("Import Dataset from URL", isPresented: $showURLInputAlert) {
            TextField("https://...", text: $urlInputText)
            Button("Cancel", role: .cancel) { }
            Button("Import") {
                let trimmed = urlInputText.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty {
                    datasetURLInput = trimmed
                    selectedFileURL = nil
                    errorMessage = nil
                    result = nil
                    previewResult = nil
                    
                    // Reset config for URL dataset
                    analysisConfig = AnalysisConfig()
                    trainColumns = []
                    
                    fetchPreview(for: trimmed)
                }
            }
        } message: {
            Text("Enter a direct link to a CSV, TSV, Parquet file, or a Kaggle/HuggingFace dataset URL.")
        }
        .sheet(isPresented: $showExportSheet) {
            if let analysisResult = result {
                ExportReportSheet(result: analysisResult.resultForTarget(selectedTargetName), isPresented: $showExportSheet)
            }
        }
        .sheet(isPresented: $showModelExportSheet) {
            ModelExportSheet(
                config: $analysisConfig,
                isPresented: $showModelExportSheet,
                onRunExport: { runEDA() }
            )
        }
        .sheet(isPresented: $showOnboarding) {
            OnboardingView(isPresented: $showOnboarding)
        }
        .onAppear {
            if !hasSeenOnboarding {
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.6) {
                    showOnboarding = true
                    hasSeenOnboarding = true
                }
            }
        }
    }

    // MARK: - Sidebar

    private var sidebarContent: some View {
        // FIX: Ensuring the outer VStack wraps BOTH the ScrollView and the Footer correctly
        VStack(spacing: 0) {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    // App Branding Title
                    HStack(spacing: 10) {
                        ZStack {
                            RoundedRectangle(cornerRadius: 8)
                                .fill(LinearGradient(colors: [.purple.opacity(0.2), .indigo.opacity(0.1)], startPoint: .topLeading, endPoint: .bottomTrailing))
                                .frame(width: 30, height: 30)
                            
                            Image(systemName: "chart.bar.doc.horizontal.fill")
                                .font(.system(size: 15, weight: .bold))
                                .foregroundStyle(LinearGradient(colors: [.purple, .indigo], startPoint: .topLeading, endPoint: .bottomTrailing))
                        }
                        
                        VStack(alignment: .leading, spacing: 1) {
                            Text("Aura")
                                .font(.system(size: 15, weight: .bold, design: .rounded))
                                .foregroundColor(.primary)
                            Text("AI-Powered Analytics")
                                .font(.system(size: 9))
                                .foregroundColor(.secondary)
                        }
                        Spacer()
                    }
                    .padding(.horizontal, 16)
                    .padding(.top, 16)

                    // 1. Dataset Source Card
                    SidebarCard(title: "Dataset Source") {
                        VStack(spacing: 10) {
                            if let fileURL = selectedFileURL {
                                datasetChip(
                                    icon: "doc.text.fill", color: .purple,
                                    title: fileURL.lastPathComponent,
                                    subtitle: fileDetails
                                )
                            } else if !datasetURLInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                                datasetChip(
                                    icon: "link.circle.fill", color: .orange,
                                    title: datasetURLInput,
                                    subtitle: getURLProviderName(datasetURLInput)
                                )
                            } else {
                                Text("No active dataset")
                                    .font(.system(size: 11))
                                    .foregroundColor(.secondary)
                                    .padding(.top, 8)
                            }
                            
                            Menu {
                                Button(action: selectFileManually) {
                                    Label("Load Local File...", systemImage: "doc.badge.plus")
                                }
                                Button {
                                    urlInputText = datasetURLInput
                                    showURLInputAlert = true
                                } label: {
                                    Label("Paste Dataset URL...", systemImage: "link.badge.plus")
                                }
                            } label: {
                                HStack(spacing: 4) {
                                    Image(systemName: "square.and.arrow.down.on.square")
                                    Text("Import Dataset...")
                                }
                                .font(.system(size: 11, weight: .semibold))
                                .foregroundColor(.purple)
                            }
                            .menuStyle(.borderlessButton)
                            .padding(.bottom, 8)
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.horizontal, 10)
                    }

                    // 1.5. Test Dataset Card
                    if !cannotRun {
                        SidebarCard(title: "Test Dataset (Optional)") {
                            if let testPath = analysisConfig.testFilePath, !testPath.isEmpty {
                                let testURL = URL(fileURLWithPath: testPath)
                                HStack {
                                    Image(systemName: "doc.text.fill").foregroundColor(.purple)
                                    VStack(alignment: .leading, spacing: 2) {
                                        Text(testURL.lastPathComponent)
                                            .font(.body)
                                            .fontWeight(.semibold)
                                            .lineLimit(1)
                                        Text("Separate validation set")
                                            .font(.caption2)
                                            .foregroundColor(.secondary)
                                    }
                                    Spacer()
                                    Button(action: {
                                        analysisConfig.testFilePath = nil
                                    }) {
                                        Image(systemName: "xmark.circle.fill").foregroundColor(.secondary)
                                    }
                                    .buttonStyle(.plain)
                                }
                                .padding(10)
                                .background(Color.white.opacity(0.02))
                                .cornerRadius(10)
                                .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color.white.opacity(0.06)))
                            } else {
                                VStack(spacing: 10) {
                                    Text("No test dataset selected")
                                        .font(.system(size: 11))
                                        .foregroundColor(.secondary)
                                    
                                    Button(action: selectTestFileManually) {
                                        HStack(spacing: 4) {
                                            Image(systemName: "doc.badge.plus")
                                            Text("Load Test File...")
                                        }
                                        .font(.system(size: 11, weight: .semibold))
                                        .foregroundColor(.purple)
                                    }
                                    .buttonStyle(.plain)
                                }
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 16)
                            }
                        }
                    }

                    // 2. Target Column Configuration Card
                    SidebarCard(title: "Target Column(s)") {
                        VStack(alignment: .leading, spacing: 6) {
                            let columns = !trainColumns.isEmpty ? trainColumns : (previewResult?.columns ?? result?.columns ?? [])
                            if !columns.isEmpty {
                                if analysisConfig.datasetType == .timeSeries {
                                    VStack(alignment: .leading, spacing: 6) {
                                        Text("Select targets to forecast:")
                                            .font(.system(size: 10, weight: .semibold))
                                            .foregroundColor(.secondary)
                                        
                                        ForEach(columns, id: \.self) { col in
                                            if col != analysisConfig.timeColumn {
                                                Button {
                                                    if analysisConfig.targetColumns.contains(col) {
                                                        analysisConfig.targetColumns.removeAll(where: { $0 == col })
                                                    } else {
                                                        analysisConfig.targetColumns.append(col)
                                                        analysisConfig.excludedColumns.remove(col)
                                                    }
                                                } label: {
                                                    HStack {
                                                        Image(systemName: analysisConfig.targetColumns.contains(col) ? "checkmark.square.fill" : "square")
                                                            .foregroundColor(analysisConfig.targetColumns.contains(col) ? .purple : .secondary)
                                                        Text(col)
                                                            .font(.system(size: 11))
                                                            .foregroundColor(.primary)
                                                            .lineLimit(1)
                                                        Spacer()
                                                    }
                                                    .contentShape(Rectangle())
                                                }
                                                .buttonStyle(.plain)
                                                .padding(.vertical, 1)
                                            }
                                        }
                                    }
                                    .padding(.vertical, 4)
                                } else {
                                    Menu {
                                        Button("Auto-detect target") {
                                            analysisConfig.targetColumn = ""
                                        }
                                        Divider()
                                        ForEach(columns, id: \.self) { col in
                                            Button(col) {
                                                analysisConfig.targetColumn = col
                                                analysisConfig.excludedColumns.remove(col)
                                            }
                                        }
                                    } label: {
                                        HStack {
                                            Text(analysisConfig.targetColumn.isEmpty ? "Auto-detect target" : analysisConfig.targetColumn)
                                                .font(.system(size: 12))
                                                .foregroundColor(analysisConfig.targetColumn.isEmpty ? .secondary : .primary)
                                                .lineLimit(1)
                                            Spacer()
                                            Image(systemName: "chevron.up.chevron.down")
                                                .font(.system(size: 9))
                                                .foregroundColor(.secondary)
                                        }
                                        .padding(.horizontal, 10)
                                        .padding(.vertical, 6)
                                        .background(Color.white.opacity(0.03))
                                        .cornerRadius(6)
                                        .contentShape(Rectangle())
                                    }
                                    .menuStyle(.borderlessButton)
                                }
                            } else {
                                TextField("Auto-detect target", text: .constant(analysisConfig.targetColumn.isEmpty ? "Auto-detect target" : analysisConfig.targetColumn))
                                    .textFieldStyle(.plain)
                                    .font(.system(size: 12))
                                    .padding(.horizontal, 10)
                                    .padding(.vertical, 6)
                                    .background(Color.white.opacity(0.03))
                                    .cornerRadius(6)
                                    .disabled(true)
                            }
                            
                            Text(analysisConfig.datasetType == .timeSeries ? "Select target variables to forecast." : "Guides machine learning model tasks.")
                                .font(.system(size: 9))
                                .foregroundColor(.secondary.opacity(0.6))
                        }
                        .padding(10)
                    }

                    // 3. Main Analyze Button CTA
                    Button(action: runEDA) {
                        HStack(spacing: 6) {
                            if isAnalyzing {
                                ProgressView().controlSize(.small).padding(.trailing, 2)
                            } else {
                                Image(systemName: "play.fill")
                                    .font(.system(size: 11))
                            }
                            Text(isAnalyzing ? "Analyzing…" : "Run Analysis Pipeline")
                                .font(.system(size: 12, weight: .bold, design: .rounded))
                        }
                        .foregroundColor(.white)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 11)
                        .background(
                            (cannotRun || isAnalyzing || isPreloading)
                            ? AnyShapeStyle(Color.white.opacity(0.05))
                            : AnyShapeStyle(LinearGradient(colors: [.purple, .indigo], startPoint: .leading, endPoint: .trailing))
                        )
                        .cornerRadius(8)
                        .shadow(color: (cannotRun || isAnalyzing || isPreloading) ? .clear : .purple.opacity(0.2), radius: 6, x: 0, y: 2)
                    }
                    .disabled(cannotRun || isAnalyzing || isPreloading)
                    .buttonStyle(.plain)
                    .padding(.horizontal, 16)

                    // 4. Sidebar Sample Datasets Card
                    SidebarCard(title: "Sample Datasets") {
                        VStack(spacing: 0) {
                            Button { loadSampleDataset(named: "house_prices.csv") } label: {
                                HStack {
                                    Image(systemName: "house.fill").foregroundColor(.purple)
                                        .font(.system(size: 11))
                                    Text("House Prices")
                                        .font(.system(size: 12))
                                    Spacer()
                                    Image(systemName: "chevron.right")
                                        .font(.system(size: 8))
                                        .foregroundColor(.secondary.opacity(0.5))
                                }
                                .padding(.vertical, 8)
                                .padding(.horizontal, 12)
                                .contentShape(Rectangle())
                            }
                            .buttonStyle(.plain)
                            
                            Divider().background(Color.white.opacity(0.05)).padding(.horizontal, 8)
                            
                            Button { loadSampleDataset(named: "iris.csv") } label: {
                                HStack {
                                    Image(systemName: "leaf.fill").foregroundColor(.green)
                                        .font(.system(size: 11))
                                    Text("Iris Flowers")
                                        .font(.system(size: 12))
                                    Spacer()
                                    Image(systemName: "chevron.right")
                                        .font(.system(size: 8))
                                        .foregroundColor(.secondary.opacity(0.5))
                                }
                                .padding(.vertical, 8)
                                .padding(.horizontal, 12)
                                .contentShape(Rectangle())
                            }
                            .buttonStyle(.plain)
                            
                            Divider().background(Color.white.opacity(0.05)).padding(.horizontal, 8)
                            
                            Button { loadSampleDataset(named: "airline_passengers.csv") } label: {
                                HStack {
                                    Image(systemName: "chart.line.uptrend.xyaxis").foregroundColor(.blue)
                                        .font(.system(size: 11))
                                    Text("Airline Passengers")
                                        .font(.system(size: 12))
                                    Spacer()
                                    Image(systemName: "chevron.right")
                                        .font(.system(size: 8))
                                        .foregroundColor(.secondary.opacity(0.5))
                                }
                                .padding(.vertical, 8)
                                .padding(.horizontal, 12)
                                .contentShape(Rectangle())
                            }
                            .buttonStyle(.plain)
                            
                            Divider().background(Color.white.opacity(0.05)).padding(.horizontal, 8)
                            
                            Button { loadSampleDataset(named: "movie_reviews.csv") } label: {
                                HStack {
                                    Image(systemName: "text.bubble").foregroundColor(.green)
                                        .font(.system(size: 11))
                                    Text("Movie Reviews")
                                        .font(.system(size: 12))
                                    Spacer()
                                    Image(systemName: "chevron.right")
                                        .font(.system(size: 8))
                                        .foregroundColor(.secondary.opacity(0.5))
                                }
                                .padding(.vertical, 8)
                                .padding(.horizontal, 12)
                                .contentShape(Rectangle())
                            }
                            .buttonStyle(.plain)
                            
                            Divider().background(Color.white.opacity(0.05)).padding(.horizontal, 8)
                            
                            Button { loadSampleDataset(named: "mnist_mini.npz") } label: {
                                HStack {
                                    Image(systemName: "photo.stack").foregroundColor(.orange)
                                        .font(.system(size: 11))
                                    Text("MNIST Mini (NPZ)")
                                        .font(.system(size: 12))
                                    Spacer()
                                    Image(systemName: "chevron.right")
                                        .font(.system(size: 8))
                                        .foregroundColor(.secondary.opacity(0.5))
                                }
                                .padding(.vertical, 8)
                                .padding(.horizontal, 12)
                                .contentShape(Rectangle())
                            }
                            .buttonStyle(.plain)
                        }
                    }

                    // 4.5 Recent Analyses Card
                    if !historyService.items.isEmpty {
                        SidebarCard(title: "Recent Analyses") {
                            VStack(spacing: 0) {
                                ForEach(historyService.items.prefix(4)) { item in
                                    Button { loadHistoryItem(item) } label: {
                                        HStack(spacing: 10) {
                                            RoundedRectangle(cornerRadius: 1.5)
                                                .fill(item.taskType.map { getTaskColor($0) } ?? .secondary.opacity(0.4))
                                                .frame(width: 3, height: 28)
                                            
                                            VStack(alignment: .leading, spacing: 2) {
                                                Text(item.datasetName)
                                                    .font(.system(size: 11, weight: .semibold, design: .rounded))
                                                    .foregroundColor(.primary)
                                                    .lineLimit(1)
                                                
                                                HStack(spacing: 4) {
                                                    if let task = item.taskType {
                                                        Text(getTaskShortLabel(task))
                                                            .font(.system(size: 8, weight: .bold))
                                                            .foregroundColor(getTaskColor(task))
                                                    }
                                                    Text("•")
                                                        .font(.system(size: 8))
                                                        .foregroundColor(.secondary.opacity(0.5))
                                                    Text(item.timestamp, style: .relative)
                                                        .font(.system(size: 8))
                                                        .foregroundColor(.secondary)
                                                }
                                                
                                                if let model = item.bestModel, let score = item.bestScore {
                                                    Text("\(model) (\(String(format: "%.2f", score)))")
                                                        .font(.system(size: 8))
                                                        .foregroundColor(.secondary.opacity(0.8))
                                                        .lineLimit(1)
                                                } else if let rows = item.rowCount, let cols = item.colCount {
                                                    Text("\(rows) rows • \(cols) cols")
                                                        .font(.system(size: 8))
                                                        .foregroundColor(.secondary.opacity(0.8))
                                                }
                                            }
                                            Spacer()
                                            Image(systemName: "chevron.right")
                                                .font(.system(size: 8))
                                                .foregroundColor(.secondary.opacity(0.5))
                                        }
                                        .padding(.vertical, 6)
                                        .padding(.horizontal, 10)
                                        .contentShape(Rectangle())
                                    }
                                    .buttonStyle(.plain)
                                    .contextMenu {
                                        Button {
                                            renameText = item.datasetName
                                            itemToRename = item
                                            showRenameAlert = true
                                        } label: {
                                            Label("Rename Analysis...", systemImage: "pencil")
                                        }
                                        Button(role: .destructive) {
                                            historyService.deleteItem(item)
                                        } label: {
                                            Label("Delete Analysis", systemImage: "trash")
                                        }
                                    }
                                    
                                    if item.id != historyService.items.prefix(4).last?.id {
                                        Divider().background(Color.white.opacity(0.05)).padding(.horizontal, 8)
                                    }
                                }
                            }
                        }
                    }
                    Spacer(minLength: 20)
                }
            }
            
            // 5. Sidebar Footer Panel pinned to bottom
            VStack(spacing: 10) {
                Divider().background(Color.white.opacity(0.06))
                
                HStack(spacing: 6) {
                    Circle()
                        .fill(ollamaStatus.isAvailable ? Color.green : Color.secondary.opacity(0.4))
                        .frame(width: 7, height: 7)
                        .shadow(color: ollamaStatus.isAvailable ? Color.green.opacity(0.4) : Color.clear, radius: 2)
                    
                    Text(ollamaStatus.isAvailable ? "AI Analyst Ready" : "Ollama Offline")
                        .font(.system(size: 11, design: .rounded))
                        .foregroundColor(.secondary)
                    
                    Spacer()
                }
                .padding(.horizontal, 16)
                
                SettingsLink {
                    HStack {
                        Image(systemName: "gearshape.fill")
                        Text("Configure Settings...")
                        Spacer()
                    }
                    .font(.system(size: 11))
                    .foregroundColor(.secondary)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 4)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            }
            .padding(.top, 10)
            .padding(.bottom, 12)
            .background(Color(nsColor: .windowBackgroundColor))
        } // <-- End of outer VStack wrapper
    }

    // MARK: - Main Detail Content

    @ViewBuilder
    private var mainContent: some View {
        ZStack {
            if isAnalyzing {
                loadingView(
                    title: progressMessage.isEmpty ? "Running analysis pipeline…" : progressMessage,
                    subtitle: "Fitting ML models and generating charts…",
                    fraction: progressFraction,
                    onCancel: { PythonRunner.shared.cancelActiveAnalysis() }
                )

            } else if isPreloading {
                loadingView(
                    title: "Loading dataset preview…",
                    subtitle: "Downloading and parsing the file format…",
                    onCancel: { PythonRunner.shared.cancelActiveAnalysis() }
                )

            } else if let error = errorMessage {
                errorView(error: error)

            } else if let analysisResult = result {
                VStack(spacing: 0) {
                    // Tab bar
                    Picker("View", selection: $selectedTab) {
                        Text("Summary").tag("Summary")
                        Text("Charts").tag("Charts")
                        Text("Correlations").tag("Correlations")
                        Text("Data").tag("Data")
                    }
                    .pickerStyle(.segmented)
                    .padding()

                    // Target Selector Pill Bar for Multi-Target Time Series
                    if let targetsMap = analysisResult.targets, !targetsMap.isEmpty {
                        VStack(spacing: 8) {
                            HStack(spacing: 8) {
                                Text("Forecast Target:")
                                    .font(.system(size: 11, weight: .bold))
                                    .foregroundColor(.secondary)
                                ScrollView(.horizontal, showsIndicators: false) {
                                    HStack(spacing: 6) {
                                        ForEach(targetsMap.keys.sorted(), id: \.self) { targetName in
                                            Button {
                                                withAnimation(.spring(response: 0.2, dampingFraction: 0.8)) {
                                                    selectedTargetName = targetName
                                                }
                                            } label: {
                                                Text(targetName)
                                                    .font(.system(size: 10, weight: .semibold))
                                                    .padding(.horizontal, 10)
                                                    .padding(.vertical, 5)
                                                    .background(selectedTargetName == targetName ? Color.purple : Color.white.opacity(0.05))
                                                    .foregroundColor(selectedTargetName == targetName ? .white : .secondary)
                                                    .cornerRadius(12)
                                            }
                                            .buttonStyle(.plain)
                                        }
                                    }
                                }
                            }
                            .padding(.horizontal, 16)
                            .padding(.bottom, 10)
                            
                            Divider()
                        }
                    } else {
                        Divider()
                    }

                    ZStack {
                        switch selectedTab {
                        case "Summary":
                            SummaryView(
                                result: analysisResult.resultForTarget(selectedTargetName),
                                config: $analysisConfig,
                                onRunAnalysis: { runEDA() },
                                onExportModelAndCode: { showModelExportSheet = true },
                                onAskAI: { prompt in
                                    sendToChat(prompt)
                                }
                            )
                        case "Charts":
                            ChartsListView(result: analysisResult.resultForTarget(selectedTargetName)) { prompt in
                                sendToChat(prompt)
                            }
                        case "Correlations":
                            CorrelationMatrixView(result: analysisResult.resultForTarget(selectedTargetName)) { prompt in
                                sendToChat(prompt)
                            }
                        case "Data":
                            VStack(spacing: 0) {
                                if analysisResult.testFullPreview != nil || analysisResult.valFullPreview != nil {
                                    HStack {
                                        Spacer()
                                        Picker("Select Table:", selection: $selectedDataTab) {
                                            Text("Train").tag("train")
                                            if analysisResult.testFullPreview != nil {
                                                Text("Test").tag("test")
                                            }
                                            if analysisResult.valFullPreview != nil {
                                                Text("Validation").tag("val")
                                            }
                                        }
                                        .pickerStyle(.segmented)
                                        .frame(width: 280)
                                        .padding(.horizontal)
                                        .padding(.vertical, 8)
                                        Spacer()
                                    }
                                    Divider().background(Color.white.opacity(0.06))
                                }
                                
                                let activePreview: FullTablePreview? = {
                                    if selectedDataTab == "test", let testFp = analysisResult.testFullPreview {
                                        return testFp
                                    }
                                    if selectedDataTab == "val", let valFp = analysisResult.valFullPreview {
                                        return valFp
                                    }
                                    return analysisResult.fullPreview
                                }()
                                
                                if let fp = activePreview {
                                    FullTableView(preview: fp)
                                } else {
                                    VStack(spacing: 12) {
                                        Image(systemName: "tablecells")
                                            .font(.system(size: 40))
                                            .foregroundColor(.secondary)
                                        Text("Full table not available for this analysis.")
                                            .foregroundColor(.secondary)
                                        Text("Re-run the analysis to include table data.")
                                            .font(.caption)
                                            .foregroundColor(.secondary.opacity(0.6))
                                    }
                                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                                }
                            }
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                        default:
                            Text("Select a tab")
                        }
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                }

            // FIX: Using VStack so the GeometryReader calculates exactly what's left!
            } else if let preview = previewResult {
                VStack(spacing: 0) {
                    PreviewTableView(
                        preview: preview,
                        config: $analysisConfig,
                        onPreviewFileRequested: { path in
                            fetchPreview(for: path)
                        }
                    )
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    
                    previewActionBar
                        .background(Color(nsColor: .windowBackgroundColor)) // Blocks scrolling content
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)

            } else {
                DragDropView(
                    onFileDropped: { url in loadDroppedFile(url) },
                    onSelectFileManually: { selectFileManually() },
                    onURLSubmitted: { urlString in
                        datasetURLInput = urlString
                        selectedFileURL = nil
                        fetchPreview(for: urlString)
                    },
                    onSampleSelected: { name in
                        loadSampleDataset(named: name)
                    },
                    recentAnalyses: historyService.items,
                    onRecentSelected: { item in
                        loadHistoryItem(item)
                    },
                    onRename: { item in
                        renameText = item.datasetName
                        itemToRename = item
                        showRenameAlert = true
                    },
                    onDelete: { item in
                        historyService.deleteItem(item)
                    }
                )
            }
        }
    }

    // MARK: - Reusable Subviews

    private func datasetChip(icon: String, color: Color, title: String, subtitle: String) -> some View {
        HStack {
            Image(systemName: icon).foregroundColor(color)
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.body)
                    .fontWeight(.semibold)
                    .lineLimit(1)
                Text(subtitle)
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
            Spacer()
            Button(action: clearSelection) {
                Image(systemName: "xmark.circle.fill").foregroundColor(.secondary)
            }
            .buttonStyle(.plain)
        }
        .padding(10)
        .background(Color.white.opacity(0.02))
        .cornerRadius(10)
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color.white.opacity(0.06)))
    }

    private func loadingView(title: String, subtitle: String, fraction: Double? = nil, onCancel: (() -> Void)? = nil) -> some View {
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
                ProgressView().scaleEffect(1.4)
                    .padding(.bottom, 8)
            }
            Text(title).font(.system(size: 14, weight: .bold, design: .rounded)).foregroundColor(.primary)
            Text(subtitle).font(.system(size: 11)).foregroundColor(.secondary)
            
            if let onCancel = onCancel {
                Button(action: onCancel) {
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
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func errorView(error: String) -> some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                HStack {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.title).foregroundColor(.red)
                    Text("Operation Failed").font(.title).fontWeight(.bold)
                }
                Text("The Python subprocess failed. Diagnostic details:")
                    .font(.body).foregroundColor(.secondary)

                Text(error)
                    .font(.system(.body, design: .monospaced))
                    .foregroundColor(.red)
                    .padding(16)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color.red.opacity(0.05))
                    .cornerRadius(12)
                    .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color.red.opacity(0.2)))

                HStack(spacing: 12) {
                    Button("Go Back") { clearSelection() }.buttonStyle(.bordered)
                    SettingsLink { Text("Open Settings…") }.buttonStyle(.borderedProminent)
                }
            }
            .padding(40)
        }
    }

    private var previewActionBar: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text("Dataset Loaded Successfully").fontWeight(.semibold)
                Text("Customize the target column in the sidebar, then run analysis.")
                    .font(.caption2).foregroundColor(.secondary)
            }
            Spacer()
            Button(action: runEDA) {
                HStack {
                    Image(systemName: "play.fill")
                    Text("Run Full Analysis").fontWeight(.bold)
                }
                .foregroundColor(.white)
                .padding(.horizontal, 16).padding(.vertical, 8)
                .background(LinearGradient(colors: [.purple, .indigo], startPoint: .leading, endPoint: .trailing))
                .cornerRadius(8)
            }
            .buttonStyle(.plain)
        }
        .padding(16)
        .background(Color.white.opacity(0.02))
        .overlay(Divider(), alignment: .top)
    }

    // MARK: - Helpers

    private var cannotRun: Bool {
        selectedFileURL == nil && datasetURLInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private func getTaskShortLabel(_ task: String) -> String {
        switch task.lowercased() {
        case "regression": return "Reg"
        case "classification": return "Clf"
        case "time_series": return "TS"
        case "nlp": return "NLP"
        case "image": return "Img"
        default: return task
        }
    }

    private func getTaskColor(_ task: String) -> Color {
        switch task.lowercased() {
        case "regression": return .purple
        case "classification": return .indigo
        case "time_series": return .blue
        case "nlp": return .green
        case "image": return .orange
        default: return .secondary
        }
    }

    private func sendToChat(_ prompt: String) {
        if !showAIPanel {
            showAIPanel = true
        }
        let model = UserDefaults.standard.string(forKey: "Aura_OllamaModel")
            ?? ollamaStatus.availableModels.first?.name
            ?? "llama3.2"
        let temp = UserDefaults.standard.double(forKey: "Aura_OllamaTemp")
        let tokens = UserDefaults.standard.integer(forKey: "Aura_OllamaMaxTokens")
        chatViewModel.sendMessage(prompt, model: model,
                                  temperature: temp == 0 ? 0.3 : temp,
                                  maxTokens: tokens == 0 ? 2048 : tokens)
    }

    // MARK: - File Handling

    private func loadDroppedFile(_ url: URL) {
        selectedFileURL = url
        datasetURLInput = ""
        errorMessage = nil
        result = nil
        previewResult = nil
        
        // Reset configuration and columns for the new dataset
        analysisConfig = AnalysisConfig()
        analysisConfig.trainFilePath = url.path
        trainColumns = []
        
        if let size = try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize {
            let fmt = ByteCountFormatter()
            fmt.countStyle = .file
            fileDetails = fmt.string(fromByteCount: Int64(size))
            
            // Automatically enable smart sample for files > 20MB
            if size > 20 * 1024 * 1024 {
                analysisConfig.smartSample = true
            }
        } else {
            fileDetails = "Unknown size"
        }
        fetchPreview(for: url.path)
    }

    private func selectFileManually() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        let fileTypes = ["csv", "tsv", "parquet", "npz"]
        panel.allowedContentTypes = fileTypes.compactMap { UTType(filenameExtension: $0) }
        if panel.runModal() == .OK, let url = panel.url {
            loadDroppedFile(url)
        }
    }

    private func selectTestFileManually() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        let fileTypes = ["csv", "tsv", "parquet", "npz"]
        panel.allowedContentTypes = fileTypes.compactMap { UTType(filenameExtension: $0) }
        if panel.runModal() == .OK, let url = panel.url {
            analysisConfig.testFilePath = url.path
        }
    }

    private func loadSampleDataset(named name: String) {
        let workspacePath = "/Users/oleksiichumak/Developer/Xcode.projects/Aura/sample_data/\(name)"
        let fileURL = URL(fileURLWithPath: workspacePath)
        if FileManager.default.fileExists(atPath: fileURL.path) {
            loadDroppedFile(fileURL)
        } else if let bundleURL = Bundle.main.url(forResource: name, withExtension: nil) {
            loadDroppedFile(bundleURL)
        } else {
            errorMessage = "Sample dataset '\(name)' could not be found."
        }
    }

    private func clearSelection() {
        selectedFileURL = nil
        datasetURLInput = ""
        previewResult = nil
        result = nil
        errorMessage = nil
        trainColumns = []
        selectedDataTab = "train"
        analysisConfig = AnalysisConfig()
        chatViewModel.clearConversation()
    }

    private func getURLProviderName(_ urlString: String) -> String {
        let lower = urlString.lowercased()
        if lower.contains("kaggle.com") {
            if lower.contains("/code/") || lower.contains("/kernels/") {
                return "Kaggle Notebook Output"
            }
            return "Kaggle Dataset"
        }
        if lower.contains("huggingface.co") { return "Hugging Face Dataset" }
        return "Generic Web Dataset"
    }

    private func fetchPreview(for pathOrURL: String) {
        withAnimation {
            isPreloading = true
            errorMessage = nil
            previewResult = nil
            result = nil
        }
        PythonRunner.shared.runPreview(csvPathOrURL: pathOrURL) { response in
            DispatchQueue.main.async {
                withAnimation {
                    self.isPreloading = false
                    switch response {
                    case .success(let previewData):
                        self.previewResult = previewData
                        if self.analysisConfig.trainFilePath == nil {
                            self.analysisConfig.trainFilePath = previewData.localPath
                        }
                        if previewData.localPath == self.analysisConfig.trainFilePath {
                            self.trainColumns = previewData.columns
                        }
                        if let inferred = previewData.inferredDatasetType,
                           let type = DatasetType(rawValue: inferred) {
                            self.analysisConfig.datasetType = type
                        }
                        
                        // Automate test and validation pre-selection
                        if let available = previewData.availableFiles {
                            if self.analysisConfig.testFilePath == nil {
                                if let testFile = available.first(where: { $0.lowercased().contains("test") }) {
                                    self.analysisConfig.testFilePath = testFile
                                }
                            }
                            if self.analysisConfig.validationFilePath == nil {
                                if let valFile = available.first(where: { $0.lowercased().contains("val") || $0.lowercased().contains("valid") }) {
                                    self.analysisConfig.validationFilePath = valFile
                                }
                            }
                        }
                    case .failure(let error):
                        if (error as NSError).code != -999 {
                            self.errorMessage = error.localizedDescription
                        }
                    }
                }
            }
        }
    }

    private func runEDA() {
        let csvPath: String
        if let trainPath = analysisConfig.trainFilePath {
            csvPath = trainPath
        } else if let preview = previewResult {
            csvPath = preview.localPath
        } else if let fileURL = selectedFileURL {
            csvPath = fileURL.path
        } else {
            let urlStr = datasetURLInput.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !urlStr.isEmpty else { return }
            csvPath = urlStr
        }

        withAnimation {
            isAnalyzing = true
            progressFraction = 0.0
            progressMessage = "Starting analysis..."
            errorMessage = nil
            result = nil
        }

        let targetParam: String?
        if analysisConfig.datasetType == .timeSeries && !analysisConfig.targetColumns.isEmpty {
            targetParam = analysisConfig.targetColumns.joined(separator: ",")
        } else {
            let target = analysisConfig.targetColumn.trimmingCharacters(in: .whitespacesAndNewlines)
            targetParam = target.isEmpty ? nil : target
        }

        var finalConfig = analysisConfig
        if let targetParam = targetParam {
            let targetsList = targetParam.split(separator: ",").map(String.init)
            for t in targetsList {
                finalConfig.excludedColumns.remove(t)
            }
        }

        let originalSource = selectedFileURL?.path ?? datasetURLInput.trimmingCharacters(in: .whitespacesAndNewlines)
        
        PythonRunner.shared.runAnalysis(
            csvPath: csvPath,
            targetColumn: targetParam,
            config: finalConfig,
            progress: { frac, msg in
                Task { @MainActor in
                    withAnimation(.easeInOut(duration: 0.15)) {
                        self.progressFraction = frac
                        self.progressMessage = msg
                    }
                }
            }
        ) { response in
            DispatchQueue.main.async {
                withAnimation {
                    self.isAnalyzing = false
                    switch response {
                    case .success(let data):
                        self.result = data
                        if let targetsMap = data.targets, !targetsMap.isEmpty {
                            self.selectedTargetName = targetsMap.keys.sorted().first ?? data.targetColumn
                            self.analysisConfig.targetColumns = Array(targetsMap.keys).sorted()
                        } else {
                            self.selectedTargetName = data.targetColumn
                            self.analysisConfig.targetColumn = data.targetColumn
                        }
                        self.historyService.saveAnalysis(result: data, datasetPath: csvPath, targetColumn: targetParam, originalSource: originalSource)
                        self.chatViewModel.injectContext(data)
                        if self.ollamaStatus.isAvailable { self.showAIPanel = true }
                    case .failure(let error):
                        if (error as NSError).code != -999 {
                            self.errorMessage = error.localizedDescription
                        }
                    }
                }
            }
        }
    }

    private func loadHistoryItem(_ item: HistoryItem) {
        withAnimation {
            self.errorMessage = nil
            self.isAnalyzing = false
            self.isPreloading = false
            
            // Reconstruct the base configuration for this history item
            var newConfig = AnalysisConfig()
            newConfig.trainFilePath = item.datasetPath
            
            if item.datasetPath.hasPrefix("http://") || item.datasetPath.hasPrefix("https://") {
                self.datasetURLInput = item.datasetPath
                self.selectedFileURL = nil
            } else {
                self.selectedFileURL = URL(fileURLWithPath: item.datasetPath)
                self.datasetURLInput = ""
                if let size = try? selectedFileURL?.resourceValues(forKeys: [.fileSizeKey]).fileSize {
                    let fmt = ByteCountFormatter()
                    fmt.countStyle = .file
                    self.fileDetails = fmt.string(fromByteCount: Int64(size))
                } else {
                    self.fileDetails = "Local File"
                }
            }
            
            if let loadedResult = historyService.loadAnalysisResult(item: item) {
                self.result = loadedResult
                self.trainColumns = loadedResult.columns
                self.chatViewModel.injectContext(loadedResult)
                
                if let targetsMap = loadedResult.targets, !targetsMap.isEmpty {
                    self.selectedTargetName = targetsMap.keys.sorted().first ?? loadedResult.targetColumn
                    newConfig.targetColumns = Array(targetsMap.keys).sorted()
                } else {
                    self.selectedTargetName = loadedResult.targetColumn
                    newConfig.targetColumn = loadedResult.targetColumn
                }
                
                // Reconstruct a DatasetPreview from the final AnalysisResult
                let previewRows = (loadedResult.fullPreview?.rows.prefix(15) ?? []).map { row in
                    row.map { PreviewValue.string($0) }
                }
                
                let datasetTypeStr: String
                switch loadedResult.taskType.lowercased() {
                case "classification", "regression":
                    datasetTypeStr = "tabular"
                default:
                    datasetTypeStr = loadedResult.taskType.lowercased()
                }
                
                self.previewResult = DatasetPreview(
                    columns: loadedResult.columns,
                    previewRows: previewRows,
                    localPath: item.datasetPath,
                    error: nil,
                    inferredDatasetType: datasetTypeStr,
                    availableFiles: [item.datasetPath],
                    totalRows: loadedResult.rowCount
                )
                
                if let type = DatasetType(rawValue: datasetTypeStr) {
                    newConfig.datasetType = type
                }
                
                self.analysisConfig = newConfig
                if self.ollamaStatus.isAvailable { self.showAIPanel = true }
                self.selectedTab = "Summary"
            } else {
                newConfig.targetColumn = item.targetColumn ?? ""
                self.analysisConfig = newConfig
                self.errorMessage = "Could not load saved analysis result from disk."
            }
        }
    }
}

struct SidebarCard<Content: View>: View {
    let title: String
    let content: Content
    
    init(title: String, @ViewBuilder content: () -> Content) {
        self.title = title
        self.content = content()
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title.uppercased())
                .font(.system(size: 9, weight: .bold, design: .rounded))
                .foregroundColor(.secondary.opacity(0.6))
                .tracking(1.0)
                .padding(.leading, 4)
            
            VStack(spacing: 0) {
                content
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.white.opacity(0.02))
            .cornerRadius(8)
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .stroke(Color.white.opacity(0.04), lineWidth: 1)
            )
        }
        .padding(.horizontal, 16)
    }
}
