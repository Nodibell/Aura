import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @State private var viewModel = DashboardViewModel()
    @State private var showHistoryBrowser = false
    @State private var showCommandPalette = false
    @State private var expandedDatasets: Set<String> = []
    @AppStorage("Aura_hasSeenOnboarding") private var hasSeenOnboarding = false
    @AppStorage("Aura_Appearance") private var appearanceMode = "System"
    
    private var navigationTitleText: String {
        if let name = viewModel.selectedFileURL?.lastPathComponent { return name }
        if !viewModel.datasetURLInput.isEmpty { return "Web Dataset" }
        return "Aura Dashboard"
    }

    var body: some View {
        ZStack {
            // 1. Native macOS Sidebar Architecture
            NavigationSplitView {
                sidebarContent
                    .navigationSplitViewColumnWidth(min: 240, ideal: 260, max: 320)
            } detail: {
                // 2. Main Content Canvas & AI Panel
                VStack(spacing: 0) {
                    if !viewModel.openPages.isEmpty {
                        TabsHeaderView(viewModel: viewModel)
                    }
                    
                    HStack(spacing: 0) {
                        mainContent
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                            .background(Color(nsColor: .windowBackgroundColor))
                        
                        // Custom Right-Side Panel
                        if viewModel.showAIPanel && viewModel.result != nil {
                            Divider()
                            AIChatPanel(
                                viewModel: viewModel.chatViewModel,
                                ollamaStatus: viewModel.ollamaStatus,
                                analysisResult: viewModel.result?.resultForTarget(viewModel.selectedTargetName)
                            )
                            .frame(width: 325)
                            .background(Color(nsColor: .windowBackgroundColor))
                            .transition(.move(edge: .trailing))
                        }
                    }
                }
                // 4. Native macOS Toolbar & Titles
                .navigationTitle(navigationTitleText)
                .navigationSubtitle(viewModel.result != nil ? "Analysis Complete" : (viewModel.previewResult != nil ? "Dataset Preview" : ""))
                .toolbar {
                    ToolbarItemGroup(placement: .primaryAction) {
                        
                        // Close Dataset Button
                        if viewModel.result != nil || viewModel.previewResult != nil {
                            Button(action: viewModel.clearSelection) {
                                Image(systemName: "xmark.circle")
                                    .foregroundColor(.secondary)
                            }
                            .help("Close current dataset")
                        }
                        
                        // Export Button
                        if viewModel.result != nil {
                            Divider()
                            Button {
                                viewModel.showExportSheet = true
                            } label: {
                                Image(systemName: "square.and.arrow.up")
                            }
                            .help("Export analysis report as Markdown")
                            .keyboardShortcut("e", modifiers: [.command, .shift])
                            
                            // AI Panel Toggle
                            let aiIconName = viewModel.showAIPanel ? "sidebar.right" : "sparkles"
                            let aiIconColor = viewModel.showAIPanel ? Color.purple : Color.primary
                            Button {
                                withAnimation(.spring(response: 0.3, dampingFraction: 0.8)) {
                                    viewModel.showAIPanel.toggle()
                                }
                            } label: {
                                Image(systemName: aiIconName)
                                    .foregroundColor(aiIconColor)
                            }
                            .help(viewModel.showAIPanel ? "Hide AI panel" : "Show AI Analyst panel")
                            .keyboardShortcut("a", modifiers: [.command, .shift])
                        }
                    }
                }
            }
            
            if showCommandPalette {
                Color.black.opacity(0.15)
                    .edgesIgnoringSafeArea(.all)
                    .onTapGesture {
                        showCommandPalette = false
                    }
                    .transition(.opacity)
                
                CommandPaletteView(
                    isPresented: $showCommandPalette,
                    viewModel: viewModel,
                    onNavigateTab: { tabName in
                        viewModel.selectedTab = tabName
                    },
                    onShowHistory: {
                        showHistoryBrowser = true
                    }
                )
                .transition(.scale(scale: 0.95).combined(with: .opacity))
                .zIndex(100)
            }
        }
        .preferredColorScheme(appearanceMode == "Dark" ? .dark : (appearanceMode == "Light" ? .light : nil))
        .background {
            Group {
                Button("") {
                    viewModel.runEDA()
                }
                .keyboardShortcut("r", modifiers: .command)
                
                Button("") {
                    let pathOrURL = viewModel.selectedFileURL?.path ?? viewModel.datasetURLInput.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !pathOrURL.isEmpty {
                        viewModel.fetchPreview(for: pathOrURL)
                    }
                }
                .keyboardShortcut(.return, modifiers: .command)
                
                Button("") {
                    showHistoryBrowser = true
                }
                .keyboardShortcut("y", modifiers: .command)
                
                Button("") {
                    showCommandPalette.toggle()
                }
                .keyboardShortcut("k", modifiers: .command)
            }
            .opacity(0)
            .frame(width: 0, height: 0)
        }
        .alert("Rename Analysis", isPresented: $viewModel.showRenameAlert) {
            TextField("Name", text: $viewModel.renameText)
            Button("Cancel", role: .cancel) { }
            Button("Rename") {
                if let item = viewModel.itemToRename {
                    viewModel.historyService.renameItem(item, to: viewModel.renameText)
                }
            }
        } message: {
            Text("Enter a descriptive name for this analysis.")
        }
        .alert("Import Dataset from URL", isPresented: $viewModel.showURLInputAlert) {
            TextField("https://...", text: $viewModel.urlInputText)
            Button("Cancel", role: .cancel) { }
            Button("Import") {
                let trimmed = viewModel.urlInputText.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty {
                    viewModel.datasetURLInput = trimmed
                    viewModel.selectedFileURL = nil
                    viewModel.errorMessage = nil
                    viewModel.result = nil
                    viewModel.previewResult = nil
                    
                    // Reset config for URL dataset
                    viewModel.analysisConfig = AnalysisConfig()
                    viewModel.trainColumns = []
                    
                    viewModel.fetchPreview(for: trimmed)
                }
            }
        } message: {
            Text("Enter a direct link to a CSV, TSV, Parquet file, or a Kaggle/HuggingFace dataset URL.")
        }
        .sheet(isPresented: $viewModel.showExportSheet) {
            if let analysisResult = viewModel.result {
                ExportReportSheet(
                    result: analysisResult.resultForTarget(viewModel.selectedTargetName),
                    csvPath: viewModel.analysisConfig.trainFilePath ?? viewModel.previewResult?.localPath ?? viewModel.selectedFileURL?.path ?? "",
                    config: viewModel.analysisConfig,
                    isPresented: $viewModel.showExportSheet
                )
            }
        }
        .sheet(isPresented: $showHistoryBrowser) {
            HistoryBrowserView(
                isPresented: $showHistoryBrowser,
                historyService: viewModel.historyService,
                onSelect: { item in
                    viewModel.loadHistoryItem(item)
                },
                onRename: { item in
                    viewModel.renameText = item.datasetName
                    viewModel.itemToRename = item
                    viewModel.showRenameAlert = true
                },
                onDelete: { item in
                    viewModel.historyService.deleteItem(item)
                }
            )
        }
        .sheet(isPresented: $viewModel.showModelExportSheet) {
            ModelExportSheet(
                config: $viewModel.analysisConfig,
                isPresented: $viewModel.showModelExportSheet,
                onRunExport: { _ in viewModel.runEDA() }
            )
        }
        .sheet(isPresented: $viewModel.showOnboarding) {
            OnboardingView(isPresented: $viewModel.showOnboarding)
        }
        .sheet(isPresented: $viewModel.showDatabaseSheet) {
            DatabaseConnectionSheet(
                isPresented: $viewModel.showDatabaseSheet,
                onImportSuccess: { csvPath, rowCount, columns in
                    self.viewModel.fileDetails = "Database Query (\(rowCount) rows)"
                    self.viewModel.selectedFileURL = URL(fileURLWithPath: csvPath)
                    self.viewModel.datasetURLInput = ""
                    self.viewModel.analysisConfig = AnalysisConfig()
                    self.viewModel.analysisConfig.trainFilePath = csvPath
                    self.viewModel.trainColumns = columns
                    viewModel.fetchPreview(for: csvPath)
                }
            )
        }
        .sheet(isPresented: $viewModel.showSchedulerSheet) {
            AnalysisSchedulerSheet(
                isPresented: $viewModel.showSchedulerSheet,
                currentDatasetPath: viewModel.analysisConfig.trainFilePath,
                currentTargetColumn: viewModel.analysisConfig.targetColumn.isEmpty ? nil : viewModel.analysisConfig.targetColumn,
                currentDatasetType: viewModel.analysisConfig.datasetType,
                currentConfig: viewModel.analysisConfig
            )
        }
        .sheet(isPresented: $viewModel.showMergeSheet) {
            FileMergeSheet(
                file1Path: viewModel.mergeFile1Path,
                file2Path: viewModel.mergeFile2Path,
                isPresented: $viewModel.showMergeSheet,
                onMergeCompleted: { mergedPath in
                    viewModel.loadDroppedFile(URL(fileURLWithPath: mergedPath))
                }
            )
        }
        .onAppear {
            // Check for the explicit launch argument we set in the test
            if ProcessInfo.processInfo.arguments.contains("-UITesting") {
                hasSeenOnboarding = true
                return
            }
            
            if !hasSeenOnboarding {
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.6) {
                    viewModel.showOnboarding = true
                    hasSeenOnboarding = true
                }
            }
        }
        .frame(minWidth: 1000, idealWidth: 1000, minHeight: 700, idealHeight: 700)
    }

    // MARK: - Sidebar

    private var sidebarContent: some View {
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


                    // ── Sample Datasets ───────────────────────────────────────────
                    DisclosureGroup("Sample Datasets") {
                        VStack(spacing: 0) {
                            Button { viewModel.loadSampleDataset(named: "house_prices.csv") } label: {
                                HStack {
                                    Image(systemName: "house.fill").foregroundColor(.purple)
                                        .font(.system(size: 11))
                                    Text("House Prices")
                                        .font(.system(size: 11))
                                    Spacer()
                                    Image(systemName: "chevron.right")
                                        .font(.system(size: 8))
                                        .foregroundColor(.secondary.opacity(0.5))
                                }
                                .padding(.vertical, 6)
                                .padding(.horizontal, 8)
                                .contentShape(Rectangle())
                            }
                            .buttonStyle(.plain)
                            
                            Divider().background(Color.primary.opacity(0.05))
                            
                            Button { viewModel.loadSampleDataset(named: "iris.csv") } label: {
                                HStack {
                                    Image(systemName: "leaf.fill").foregroundColor(.green)
                                        .font(.system(size: 11))
                                    Text("Iris Flowers")
                                        .font(.system(size: 11))
                                    Spacer()
                                    Image(systemName: "chevron.right")
                                        .font(.system(size: 8))
                                        .foregroundColor(.secondary.opacity(0.5))
                                }
                                .padding(.vertical, 6)
                                .padding(.horizontal, 8)
                                .contentShape(Rectangle())
                            }
                            .buttonStyle(.plain)
                            
                            Divider().background(Color.primary.opacity(0.05))
                            
                            Button { viewModel.loadSampleDataset(named: "airline_passengers.csv") } label: {
                                HStack {
                                    Image(systemName: "chart.line.uptrend.xyaxis").foregroundColor(.blue)
                                        .font(.system(size: 11))
                                    Text("Airline Passengers")
                                        .font(.system(size: 11))
                                    Spacer()
                                    Image(systemName: "chevron.right")
                                        .font(.system(size: 8))
                                        .foregroundColor(.secondary.opacity(0.5))
                                }
                                .padding(.vertical, 6)
                                .padding(.horizontal, 8)
                                .contentShape(Rectangle())
                            }
                            .buttonStyle(.plain)
                            
                            Divider().background(Color.primary.opacity(0.05))
                            
                            Button { viewModel.loadSampleDataset(named: "movie_reviews.csv") } label: {
                                HStack {
                                    Image(systemName: "text.bubble").foregroundColor(.green)
                                        .font(.system(size: 11))
                                    Text("Movie Reviews")
                                        .font(.system(size: 11))
                                    Spacer()
                                    Image(systemName: "chevron.right")
                                        .font(.system(size: 8))
                                        .foregroundColor(.secondary.opacity(0.5))
                                }
                                .padding(.vertical, 6)
                                .padding(.horizontal, 8)
                                .contentShape(Rectangle())
                            }
                            .buttonStyle(.plain)
                            
                            Divider().background(Color.primary.opacity(0.05))
                            
                            Button { viewModel.loadSampleDataset(named: "mnist_mini.npz") } label: {
                                HStack {
                                    Image(systemName: "photo.stack").foregroundColor(.orange)
                                        .font(.system(size: 11))
                                    Text("MNIST Mini (NPZ)")
                                        .font(.system(size: 11))
                                    Spacer()
                                    Image(systemName: "chevron.right")
                                        .font(.system(size: 8))
                                        .foregroundColor(.secondary.opacity(0.5))
                                }
                                .padding(.vertical, 6)
                                .padding(.horizontal, 8)
                                .contentShape(Rectangle())
                            }
                            .buttonStyle(.plain)
                            
                            Divider().background(Color.primary.opacity(0.05))
                            
                            Button { viewModel.loadSampleDataset(named: "drone_dataset") } label: {
                                HStack {
                                    Image(systemName: "viewfinder.rectangular").foregroundColor(.red)
                                        .font(.system(size: 11))
                                    Text("Drone Detection")
                                        .font(.system(size: 11))
                                    Spacer()
                                    Image(systemName: "chevron.right")
                                        .font(.system(size: 8))
                                        .foregroundColor(.secondary.opacity(0.5))
                                }
                                .padding(.vertical, 6)
                                .padding(.horizontal, 8)
                                .contentShape(Rectangle())
                            }
                            .buttonStyle(.plain)
                        }
                        .padding(.top, 4)
                    }
                    .font(.system(size: 11, weight: .bold))
                    .padding(.horizontal, 16)

                    // ── Datasets (Grouped Hierarchy) ──────────────────────────────
                    if !viewModel.historyService.items.isEmpty {
                        DisclosureGroup("Datasets", isExpanded: Binding<Bool>(
                            get: { expandedDatasets.contains("ROOT_DATASETS") },
                            set: { isExpanded in
                                if isExpanded {
                                    expandedDatasets.insert("ROOT_DATASETS")
                                } else {
                                    expandedDatasets.remove("ROOT_DATASETS")
                                }
                            }
                        )) {
                            VStack(alignment: .leading, spacing: 8) {
                                ForEach(viewModel.groupedDatasets) { group in
                                    let isExpanded = Binding<Bool>(
                                        get: { expandedDatasets.contains(group.name) },
                                        set: { isAdding in
                                            if isAdding {
                                                expandedDatasets.insert(group.name)
                                            } else {
                                                expandedDatasets.remove(group.name)
                                            }
                                        }
                                    )
                                    
                                    DisclosureGroup(isExpanded: isExpanded) {
                                        VStack(spacing: 0) {
                                            // 1. Shared Data View
                                            Button {
                                                if let latestRun = group.runs.first {
                                                    viewModel.loadHistoryItem(latestRun, isPreview: true, isDataOnly: true)
                                                }
                                            } label: {
                                                HStack {
                                                    Image(systemName: "grid")
                                                        .foregroundColor(.secondary)
                                                        .font(.system(size: 11))
                                                    Text("Data")
                                                        .font(.system(size: 11))
                                                        .foregroundColor(.primary)
                                                    Spacer()
                                                    Text("shared")
                                                        .font(.system(size: 9))
                                                        .foregroundColor(.secondary.opacity(0.6))
                                                }
                                                .padding(.vertical, 6)
                                                .padding(.horizontal, 8)
                                                .contentShape(Rectangle())
                                            }
                                            .buttonStyle(.plain)
                                            
                                            Divider().background(Color.primary.opacity(0.04))
                                            
                                            // 2. Individual Runs
                                            ForEach(group.runs) { item in
                                                Button {
                                                    viewModel.loadHistoryItem(item, isPreview: true, isDataOnly: false)
                                                } label: {
                                                    HStack(spacing: 8) {
                                                        RoundedRectangle(cornerRadius: 1.5)
                                                            .fill(item.uiColor)
                                                            .frame(width: 3, height: 24)
                                                        
                                                        VStack(alignment: .leading, spacing: 1) {
                                                            HStack(spacing: 4) {
                                                                if item.isPinned ?? false {
                                                                    Image(systemName: "star.fill")
                                                                        .font(.system(size: 7))
                                                                        .foregroundColor(.yellow)
                                                                }
                                                                Text(item.shortLabel)
                                                                    .font(.system(size: 11, weight: .bold))
                                                                    .foregroundColor(item.uiColor)
                                                            }
                                                            
                                                            Text(item.timestamp, style: .date)
                                                                .font(.system(size: 8))
                                                                .foregroundColor(.secondary)
                                                        }
                                                        Spacer()
                                                        
                                                        if let model = item.bestModel, let score = item.bestScore {
                                                            Text("\(model) (\(String(format: "%.2f", score)))")
                                                                .font(.system(size: 9))
                                                                .foregroundColor(.secondary)
                                                        }
                                                    }
                                                    .padding(.vertical, 5)
                                                    .padding(.horizontal, 6)
                                                    .contentShape(Rectangle())
                                                }
                                                .buttonStyle(.plain)
                                                .contextMenu {
                                                    Button {
                                                        viewModel.renameText = item.datasetName
                                                        viewModel.itemToRename = item
                                                        viewModel.showRenameAlert = true
                                                    } label: {
                                                        Label("Rename Analysis...", systemImage: "pencil")
                                                    }
                                                    Button {
                                                        viewModel.historyService.togglePinItem(item)
                                                    } label: {
                                                        Label((item.isPinned ?? false) ? "Unpin Analysis" : "Pin Analysis", systemImage: "star")
                                                    }
                                                    Button(role: .destructive) {
                                                        viewModel.historyService.deleteItem(item)
                                                    } label: {
                                                        Label("Delete Analysis", systemImage: "trash")
                                                    }
                                                }
                                                
                                                if item.id != group.runs.last?.id {
                                                    Divider().background(Color.primary.opacity(0.04))
                                                }
                                            }
                                        }
                                        .padding(.top, 4)
                                    } label: {
                                        HStack {
                                            Image(systemName: "tablecells")
                                                .foregroundColor(.purple)
                                                .font(.system(size: 11))
                                            Text(group.name)
                                                .font(.system(size: 11, weight: .semibold))
                                                .foregroundColor(.primary)
                                                .lineLimit(1)
                                            Spacer()
                                        }
                                    }
                                }
                            }
                            .padding(.top, 4)
                        }
                        .font(.system(size: 11, weight: .bold))
                        .padding(.horizontal, 16)
                        .onAppear {
                            expandedDatasets.insert("ROOT_DATASETS")
                        }
                    }
                    
                    Spacer(minLength: 20)
                }
            }
            
            // 5. Sidebar Footer Panel pinned to bottom
            VStack(spacing: 10) {
                Divider().background(Color.primary.opacity(0.06))
                
                HStack(spacing: 6) {
                    Circle()
                        .fill(viewModel.ollamaStatus.isAvailable ? Color.green : Color.secondary.opacity(0.4))
                        .frame(width: 7, height: 7)
                        .shadow(color: viewModel.ollamaStatus.isAvailable ? Color.green.opacity(0.4) : Color.clear, radius: 2)
                    
                    Text(viewModel.ollamaStatus.isAvailable ? "AI Analyst Ready" : "Ollama Offline")
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
                .accessibilityIdentifier("configureSettingsButton")
                
                Button {
                    viewModel.showSchedulerSheet = true
                } label: {
                    HStack {
                        Image(systemName: "clock.fill")
                        Text("Manage Schedules...")
                        Spacer()
                    }
                    .font(.system(size: 11))
                    .foregroundColor(.secondary)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 4)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .accessibilityIdentifier("manageSchedulesButton")
            }
            .padding(.top, 10)
            .padding(.bottom, 12)
            .background(Color(nsColor: .windowBackgroundColor))
        }
    }

    // MARK: - Main Detail Content

    @ViewBuilder
    private var mainContent: some View {
        ZStack {
            if viewModel.openPages.isEmpty {
                DragDropView(
                    onFileDropped: { urls in viewModel.handleDroppedFiles(urls) },
                    onSelectFileManually: { viewModel.selectFileManually() },
                    onImportFromDatabase: { viewModel.showDatabaseSheet = true },
                    onURLSubmitted: { urlString in
                        viewModel.datasetURLInput = urlString
                        viewModel.selectedFileURL = nil
                        viewModel.fetchPreview(for: urlString)
                    },
                    onSampleSelected: { name in
                        viewModel.loadSampleDataset(named: name)
                    },
                    recentAnalyses: viewModel.historyService.items,
                    onRecentSelected: { item in
                        viewModel.loadHistoryItem(item)
                    },
                    onRename: { item in
                        viewModel.renameText = item.datasetName
                        viewModel.itemToRename = item
                        viewModel.showRenameAlert = true
                    },
                    onDelete: { item in
                        viewModel.historyService.deleteItem(item)
                    }
                )
            } else if let activePage = viewModel.activePage, activePage.isDataOnly {
                if let preview = activePage.previewResult {
                    VStack(spacing: 0) {
                        PreviewTableView(
                            preview: preview,
                            config: Binding(
                                get: { activePage.analysisConfig },
                                set: { activePage.analysisConfig = $0 }
                            ),
                            onPreviewFileRequested: { path in
                                viewModel.fetchPreview(for: path, page: activePage)
                            },
                            isSidebar: false
                        )
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    loadingView(
                        title: "Loading data preview...",
                        subtitle: "Parsing the dataset schema..."
                    )
                }
            } else if viewModel.isAnalyzing {
                loadingView(
                    title: viewModel.progressMessage.isEmpty ? "Running analysis pipeline…" : viewModel.progressMessage,
                    subtitle: "Fitting ML models and generating charts…",
                    fraction: viewModel.progressFraction,
                    onCancel: { Task { await PythonRunner.shared.cancelActiveAnalysis() } }
                )

            } else if viewModel.isPreloading {
                loadingView(
                    title: viewModel.progressMessage.isEmpty ? "Loading dataset preview…" : viewModel.progressMessage,
                    subtitle: "Downloading and parsing the file format…",
                    fraction: (viewModel.progressFraction > 0.0 || !viewModel.progressMessage.isEmpty) ? viewModel.progressFraction : nil,
                    onCancel: { Task { await PythonRunner.shared.cancelActiveAnalysis() } }
                )

            } else if let error = viewModel.errorMessage {
                errorView(error: error)

            } else if let analysisResult = viewModel.result {
                VStack(spacing: 0) {
                    // Tab bar
                    HStack {
                        CustomSegmentedPicker(
                            selection: $viewModel.selectedTab,
                            items: [
                                ("Summary", "Summary"),
                                ("Charts", "Charts"),
                                ("Correlations", "Correlations"),
                                ("Data", "Data"),
                                ("Cleaning", "Cleaning"),
                                ("Diff", "Diff")
                            ] + (viewModel.result?.taskType != "clustering" ? [("Predict", "Predict")] : [])
                        )
                        
                        Spacer()
                        
                        Button(action: {
                            withAnimation {
                                viewModel.activePage?.result = nil
                            }
                        }) {
                            Label("Reanalyze", systemImage: "arrow.counterclockwise")
                                .font(.system(size: 11, weight: .bold))
                                .foregroundColor(.purple)
                                .padding(.horizontal, 10)
                                .padding(.vertical, 5)
                                .background(Color.purple.opacity(0.08))
                                .cornerRadius(6)
                        }
                        .buttonStyle(.plain)
                        .padding(.trailing, 16)
                    }
                    .padding(.horizontal, 16)
                    .padding(.vertical, 8)
                    .background(Color.primary.opacity(0.015))


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
                                                    viewModel.selectedTargetName = targetName
                                                }
                                            } label: {
                                                Text(targetName)
                                                    .font(.system(size: 10, weight: .semibold))
                                                    .padding(.horizontal, 10)
                                                    .padding(.vertical, 5)
                                                    .background(viewModel.selectedTargetName == targetName ? Color.purple : Color.primary.opacity(0.05))
                                                    .foregroundColor(viewModel.selectedTargetName == targetName ? .white : .secondary)
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
                        switch viewModel.selectedTab {
                        case "Summary":
                            SummaryView(
                                result: analysisResult.resultForTarget(viewModel.selectedTargetName),
                                config: $viewModel.analysisConfig,
                                activeModelName: $viewModel.activeModelName,
                                onRunAnalysis: { viewModel.runEDA() },
                                onExportModelAndCode: { viewModel.showModelExportSheet = true },
                                onAskAI: { prompt in
                                    sendToChat(prompt)
                                },
                                onScheduleAnalysis: {
                                    viewModel.showSchedulerSheet = true
                                }
                            )
                        case "Charts":
                            ChartsListView(result: analysisResult.resultForTarget(viewModel.selectedTargetName)) { prompt in
                                sendToChat(prompt)
                            }
                        case "Correlations":
                            CorrelationMatrixView(result: analysisResult.resultForTarget(viewModel.selectedTargetName)) { prompt in
                                sendToChat(prompt)
                            }
                        case "Data":
                            VStack(spacing: 0) {
                                if analysisResult.testFullPreview != nil || analysisResult.valFullPreview != nil {
                                    HStack {
                                        Spacer()
                                        let dataItems: [(String, String)] = {
                                            var list = [("Train", "train")]
                                            if analysisResult.testFullPreview != nil {
                                                list.append(("Test", "test"))
                                            }
                                            if analysisResult.valFullPreview != nil {
                                                list.append(("Validation", "val"))
                                            }
                                            return list
                                        }()
                                        CustomSegmentedPicker(
                                            selection: $viewModel.selectedDataTab,
                                            items: dataItems
                                        )
                                        .frame(width: 280)
                                        .padding(.horizontal)
                                        .padding(.vertical, 8)
                                        Spacer()
                                    }
                                    Divider().background(Color.primary.opacity(0.06))
                                }
                                
                                let activePreview: FullTablePreview? = {
                                    if viewModel.selectedDataTab == "test", let testFp = analysisResult.testFullPreview {
                                        return testFp
                                    }
                                    if viewModel.selectedDataTab == "val", let valFp = analysisResult.valFullPreview {
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
                        case "Cleaning":
                            DataCleaningView(
                                result: analysisResult,
                                config: $viewModel.analysisConfig,
                                onRunAnalysis: { viewModel.runEDA() }
                            )
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                        case "Diff":
                            AnalysisDiffView(
                                currentResult: analysisResult,
                                currentHistoryItemId: viewModel.currentHistoryItemId
                            )
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                        case "Predict":
                            PredictionTabView(
                                result: analysisResult.resultForTarget(viewModel.selectedTargetName),
                                csvPath: viewModel.analysisConfig.trainFilePath ?? viewModel.previewResult?.localPath ?? viewModel.selectedFileURL?.path ?? viewModel.datasetURLInput.trimmingCharacters(in: .whitespacesAndNewlines),
                                config: viewModel.analysisConfig,
                                activeModelName: viewModel.activeModelName
                            )
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                        default:
                            Text("Select a tab")
                        }
                    }
                }
            } else if let activePage = viewModel.activePage {
                PendingAnalysisView(
                    page: activePage,
                    onRunAnalysis: {
                        viewModel.runEDA()
                    },
                    onCancel: {
                        withAnimation {
                            viewModel.closePage(id: activePage.id)
                        }
                    }
                )
            } else {
                Text("No active content")
            }
        }
    }

    // MARK: - Reusable Subviews

    private func datasetChip(icon: String, color: Color, title: String, subtitle: String) -> some View {
        HStack(spacing: 8) {
            Image(systemName: icon)
                .font(.system(size: 14))
                .foregroundColor(color)
            
            VStack(alignment: .leading, spacing: 1) {
                Text(title)
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundColor(.primary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Text(subtitle)
                    .font(.system(size: 9))
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            Button(action: viewModel.clearSelection) {
                Image(systemName: "xmark.circle.fill")
                    .foregroundColor(.secondary.opacity(0.8))
                    .font(.system(size: 11))
            }
            .buttonStyle(.plain)
            .help("Clear selection")
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
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
                NativeProgressView(controlSize: .regular)
                    .padding(.bottom, 8)
            }
            Text(title).font(.system(size: 14, weight: .bold, design: .rounded)).foregroundColor(.primary)
            Text(subtitle).font(.system(size: 11)).foregroundColor(.secondary)
            
            if !viewModel.completedStages.isEmpty {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Completed Stages:")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundColor(.primary)
                    
                    ScrollView {
                        VStack(alignment: .leading, spacing: 6) {
                            ForEach(viewModel.completedStages) { stage in
                                HStack(spacing: 8) {
                                    Image(systemName: "checkmark.circle.fill")
                                        .foregroundColor(.green)
                                        .font(.system(size: 11))
                                    Text(stage.message)
                                        .font(.system(size: 11))
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
                    .frame(height: min(CGFloat(viewModel.completedStages.count) * 20, 100))
                }
                .frame(width: 280)
                .padding(.top, 8)
            }
            
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
                    Button("Go Back") { viewModel.clearSelection() }.buttonStyle(.bordered)
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
            Button(action: viewModel.runEDA) {
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
        .background(Color.primary.opacity(0.02))
        .overlay(Divider(), alignment: .top)
    }

    // MARK: - Helpers

    private var cannotRun: Bool {
        viewModel.selectedFileURL == nil && viewModel.datasetURLInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private func getTaskShortLabel(_ task: String) -> String {
        switch task.lowercased() {
        case "regression":        return "Regr"
        case "classification":    return "Clsf"
        case "clustering":        return "Clst"
        case "forecast":          return "TSFC"
        case "nlp":               return "NLP"
        case "image":             return "Img"
        case "object_detection":  return "ObDt"
        default: return task
        }
    }

    private func getTaskColor(_ task: String) -> Color {
        switch task.lowercased() {
        case "regression":        return .purple
        case "classification":    return .indigo
        case "forecast":          return .blue
        case "nlp":               return .green
        case "image":             return .orange
        case "object_detection":  return .red
        case "clustering":        return .yellow
        default: return .secondary
        }
    }

    private func sendToChat(_ prompt: String) {
        if !viewModel.showAIPanel {
            viewModel.showAIPanel = true
        }
        let model = UserDefaults.standard.string(forKey: "Aura_OllamaModel")
            ?? viewModel.ollamaStatus.availableModels.first?.name
            ?? "llama3.2"
        let temp = UserDefaults.standard.double(forKey: "Aura_OllamaTemp")
        let tokens = UserDefaults.standard.integer(forKey: "Aura_OllamaMaxTokens")
        viewModel.chatViewModel.sendMessage(prompt, model: model,
                                  temperature: temp == 0 ? 0.3 : temp,
                                  maxTokens: tokens == 0 ? 2048 : tokens)
    }

    } // struct ContentView

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
            .background(Color.primary.opacity(0.02))
            .cornerRadius(8)
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .stroke(Color.primary.opacity(0.04), lineWidth: 1)
            )
        }
        .padding(.horizontal, 16)
    }
}

struct TabsHeaderView: View {
    let viewModel: DashboardViewModel
    
    var body: some View {
        VStack(spacing: 0) {
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 0) {
                    ForEach(viewModel.openPages) { page in
                        let isActive = page.id == viewModel.activePageId
                        
                        HStack(spacing: 8) {
                            Image(systemName: page.analysisConfig.datasetType.icon)
                                .font(.system(size: 11))
                                .foregroundColor(isActive ? Color.purple : Color.secondary)
                            
                            Group {
                                if page.isPreview {
                                    Text(page.title)
                                        .font(.system(size: 11, weight: isActive ? .semibold : .regular))
                                        .italic()
                                        .foregroundColor(isActive ? .primary : .secondary)
                                        .lineLimit(1)
                                } else {
                                    Text(page.title)
                                        .font(.system(size: 11, weight: isActive ? .semibold : .regular))
                                        .foregroundColor(isActive ? .primary : .secondary)
                                        .lineLimit(1)
                                }
                            }
                            
                            Button(action: {
                                withAnimation {
                                    viewModel.closePage(id: page.id)
                                }
                            }) {
                                Image(systemName: "xmark")
                                    .font(.system(size: 8, weight: .bold))
                                    .foregroundColor(.secondary)
                                    .padding(4)
                                    .contentShape(Rectangle())
                            }
                            .buttonStyle(.plain)
                        }
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                        .background(isActive ? Color(nsColor: .windowBackgroundColor) : Color.primary.opacity(0.015))
                        .overlay(
                            Rectangle()
                                .fill(isActive ? Color.purple : Color.clear)
                                .frame(height: 2),
                            alignment: .bottom
                        )
                        .contentShape(Rectangle())
                        .onTapGesture(count: 2) {
                            withAnimation {
                                page.isPreview = false
                            }
                        }
                        .onTapGesture {
                            withAnimation {
                                viewModel.activePageId = page.id
                            }
                        }
                        
                        Divider()
                            .frame(height: 28)
                    }
                }
            }
            .frame(height: 32)
            .background(Color.primary.opacity(0.01))
            
            Divider()
        }
    }
}

