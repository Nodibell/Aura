import SwiftUI

struct SettingsView: View {
    @State private var selectedTab = 0
    private var appLogger = AppLogger.shared
    
    @AppStorage("Aura_Appearance") private var appearanceMode = "System"

    // Python Runtime State
    @State private var pythonPath = ""
    @State private var validationMessage = ""
    @State private var isValid = false
    @State private var isChecking = false

    // Local API Server State
    @State private var serverStatus: ServerStatus = .stopped
    @State private var serverPID: Int32? = nil
    @State private var serverErrorMsg: String? = nil
    @State private var isPerformingServerAction = false
    
    // Cloud Offloading (Hybrid Mode)
    @AppStorage("Aura_HybridMode") private var hybridMode = false
    @AppStorage("Aura_RemoteServerURL") private var remoteServerURL = "http://127.0.0.1:11435"

    
    private let statusTimer = Timer.publish(every: 2, on: .main, in: .common).autoconnect()
    
    enum ServerStatus {
        case stopped
        case running
        case starting
        case stopping
    }

    // API Credentials State
    @State private var kaggleUsername = ""
    @State private var kaggleKey = ""
    @State private var hfToken = ""

    // AI Settings State
    @State private var llmProvider = LLMProvider.ollama
    @State private var openAIKey = ""
    @State private var claudeKey = ""
    @State private var ollamaBaseURL = ""
    @State private var openAIModel = "gpt-4o-mini"
    @State private var claudeModel = "claude-3-5-haiku-latest"
    @State private var ollamaModel = ""
    @State private var ollamaTemp: Double = 0.3
    @State private var ollamaMaxTokens: Int = 2048
    @State private var isPullingModel = false
    @State private var pullModelName = ""
    @State private var pullStatus = ""
    @Environment(\.dismiss) private var dismiss

    // Cache Cleaner State
    @State private var cachePath = ""
    @State private var cacheSizeBytes: Int64 = 0
    @State private var cacheFileCount = 0
    @State private var isCleaningCache = false
    @State private var cleaningCacheError: String? = nil
    
    private var cacheSizeFormatted: String {
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useAll]
        formatter.countStyle = .file
        return formatter.string(fromByteCount: cacheSizeBytes)
    }

    var body: some View {
        HStack(spacing: 0) {
            // Sidebar
            VStack(alignment: .leading, spacing: 0) {
                VStack(alignment: .leading, spacing: 4) {
                    sidebarButton(title: "Python Runtime", icon: "terminal", tag: 0)
                    sidebarButton(title: "Local Server", icon: "network", tag: 1)
                    sidebarButton(title: "API Credentials", icon: "key.fill", tag: 2)
                    sidebarButton(title: "AI / LLM", icon: "sparkles", tag: 3)
                    sidebarButton(title: "System Logs", icon: "doc.text.magnifyingglass", tag: 4)
                    sidebarButton(title: "Appearance", icon: "paintpalette", tag: 5)
                    sidebarButton(title: "Cache Cleaner", icon: "trash", tag: 6)
                }
                .padding(.top, 16)
                .padding(.horizontal, 8)
                
                Spacer()
            }
            .frame(width: 180)
            .background(Color.primary.opacity(0.02))
            
            Divider()
            
            // Detail Content
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    switch selectedTab {
                    case 0:
                        pythonRuntimeTab
                    case 1:
                        localServerTab
                    case 2:
                        apiCredentialsTab
                    case 3:
                        aiTab
                    case 4:
                        logsTab
                    case 5:
                        appearanceTab
                    case 6:
                        cacheCleanerTab
                    default:
                        EmptyView()
                    }
                }
                .padding(24)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .background(Color(NSColor.windowBackgroundColor))
        }
        .frame(width: 820, height: 560)
        .onAppear {
            pythonPath = PythonRunner.shared.resolvePythonPath()
            verifyPath()
            updateServerStatus()
            kaggleUsername = KeychainService.shared.getSecureString(forKey: "Aura_KaggleUsername") ?? ""
            kaggleKey = KeychainService.shared.getSecureString(forKey: "Aura_KaggleKey") ?? ""
            hfToken = KeychainService.shared.getSecureString(forKey: "Aura_HFToken") ?? ""
            
            let providerStr = UserDefaults.standard.string(forKey: "Aura_LLMProvider") ?? "Ollama"
            llmProvider = LLMProvider(rawValue: providerStr) ?? .ollama
            ollamaBaseURL = UserDefaults.standard.string(forKey: "Aura_OllamaBaseURL") ?? "http://localhost:11434"
            openAIKey = KeychainService.shared.getSecureString(forKey: "Aura_OpenAIKey") ?? ""
            claudeKey = KeychainService.shared.getSecureString(forKey: "Aura_ClaudeKey") ?? ""
            openAIModel = UserDefaults.standard.string(forKey: "Aura_OpenAIModel") ?? "gpt-4o-mini"
            claudeModel = UserDefaults.standard.string(forKey: "Aura_ClaudeModel") ?? "claude-3-5-haiku-latest"
            
            ollamaModel = UserDefaults.standard.string(forKey: "Aura_OllamaModel") ?? ""
            let temp = UserDefaults.standard.double(forKey: "Aura_OllamaTemp")
            ollamaTemp = temp == 0 ? 0.3 : temp
            let tokens = UserDefaults.standard.integer(forKey: "Aura_OllamaMaxTokens")
            ollamaMaxTokens = tokens == 0 ? 2048 : tokens
        }
        .onChange(of: kaggleUsername) { _, newValue in
            _ = KeychainService.shared.save(newValue, forKey: "Aura_KaggleUsername")
        }
        .onChange(of: kaggleKey) { _, newValue in
            _ = KeychainService.shared.save(newValue, forKey: "Aura_KaggleKey")
        }
        .onChange(of: hfToken) { _, newValue in
            _ = KeychainService.shared.save(newValue, forKey: "Aura_HFToken")
        }
        .onChange(of: llmProvider) { _, newValue in
            UserDefaults.standard.set(newValue.rawValue, forKey: "Aura_LLMProvider")
        }
        .onChange(of: ollamaBaseURL) { _, newValue in
            UserDefaults.standard.set(newValue, forKey: "Aura_OllamaBaseURL")
        }
        .onChange(of: openAIKey) { _, newValue in
            _ = KeychainService.shared.save(newValue, forKey: "Aura_OpenAIKey")
        }
        .onChange(of: claudeKey) { _, newValue in
            _ = KeychainService.shared.save(newValue, forKey: "Aura_ClaudeKey")
        }
        .onChange(of: openAIModel) { _, newValue in
            UserDefaults.standard.set(newValue, forKey: "Aura_OpenAIModel")
        }
        .onChange(of: claudeModel) { _, newValue in
            UserDefaults.standard.set(newValue, forKey: "Aura_ClaudeModel")
        }
        .onReceive(statusTimer) { _ in
            updateServerStatus()
        }
    }
    
    private func verifyPath() {
        isChecking = true
        let pathToCheck = pythonPath.trimmingCharacters(in: .whitespacesAndNewlines)
        
        DispatchQueue.global(qos: .userInitiated).async {
            let pathExists = FileManager.default.fileExists(atPath: pathToCheck)
            
            if !pathExists {
                DispatchQueue.main.async {
                    self.isValid = false
                    self.validationMessage = "No file exists at the specified path."
                    self.isChecking = false
                }
                return
            }
            
            let works = PythonRunner.shared.verifyPythonEnvironment(at: pathToCheck)
            
            DispatchQueue.main.async {
                self.isChecking = false
                if works {
                    self.isValid = true
                    self.validationMessage = "Successfully connected. Required packages (pandas, numpy, scikit-learn, torch) and Apple Silicon GPU (MPS) support are verified."
                    PythonRunner.shared.setCustomPythonPath(pathToCheck)
                } else {
                    self.isValid = false
                    self.validationMessage = "Python binary located, but environment is missing required packages or Apple Silicon GPU (MPS) support is unavailable. Run: 'pip install pandas scikit-learn numpy torch' inside this environment."
                }
            }
        }
    }
    
    private func selectPythonBinary() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.showsHiddenFiles = true
        
        if panel.runModal() == .OK {
            if let url = panel.url {
                pythonPath = url.path
                verifyPath()
            }
        }
    }
    
    private func autoDetectPath() {
        isChecking = true
        PythonRunner.shared.resetPythonPath()
        let resolved = PythonRunner.shared.resolvePythonPath()
        pythonPath = resolved
        verifyPath()
    }

    // MARK: - AI Tab

    private var aiTab: some View {
        let ollamaChecker = OllamaStatusChecker.shared
        return ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                // Header
                VStack(alignment: .leading, spacing: 4) {
                    Text("AI / LLM Settings")
                        .font(.title2)
                        .fontWeight(.bold)
                    Text("Configure the AI Analyst (Local Ollama or Cloud APIs)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding(.bottom, 4)
                
                // Provider Picker
                VStack(alignment: .leading, spacing: 6) {
                    CustomSegmentedPicker(
                        selection: $llmProvider,
                        items: LLMProvider.allCases.map { ($0.rawValue, $0) }
                    )
                }

                Divider()

                if llmProvider == .ollama {
                    // Ollama Customizable Endpoint URL
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Ollama API Base URL").font(.headline)
                        HStack(spacing: 12) {
                            TextField("Enter Ollama API url", text: $ollamaBaseURL)
                                .textFieldStyle(.roundedBorder)
                                .font(.system(.body, design: .monospaced))
                            
                            Button("Test") {
                                Task { await ollamaChecker.refresh() }
                            }
                        }
                    }
                    
                    // Ollama Status
                    HStack(spacing: 10) {
                        Circle()
                            .fill(ollamaChecker.isAvailable ? Color.green : Color.red)
                            .frame(width: 10, height: 10)
                        Text(ollamaChecker.isAvailable
                             ? "Ollama is running — \(ollamaChecker.availableModels.count) model(s) installed"
                             : "Ollama is not running (start with: ollama serve)")
                            .font(.subheadline)
                            .foregroundColor(ollamaChecker.isAvailable ? .green : .red)
                        Spacer()
                        Button("Refresh") {
                            Task { await ollamaChecker.refresh() }
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                    }
                    .padding(12)
                    .background(ollamaChecker.isAvailable ? Color.green.opacity(0.05) : Color.red.opacity(0.05))
                    .cornerRadius(10)
                    .overlay(RoundedRectangle(cornerRadius: 10)
                        .stroke(ollamaChecker.isAvailable ? Color.green.opacity(0.2) : Color.red.opacity(0.2)))

                    // Default Model
                    if ollamaChecker.isAvailable && !ollamaChecker.availableModels.isEmpty {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Default Model").font(.headline)
                            Picker("Model", selection: $ollamaModel) {
                                ForEach(ollamaChecker.availableModels) { m in
                                    Text(m.name).tag(m.name)
                                }
                            }
                            .pickerStyle(.menu)
                            .onChange(of: ollamaModel) { _, v in
                                UserDefaults.standard.set(v, forKey: "Aura_OllamaModel")
                            }
                        }
                    }

                    // Pull a model
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Pull a New Model").font(.headline)
                        HStack {
                            TextField("e.g. llama3.2, qwen2.5:7b", text: $pullModelName)
                                .textFieldStyle(.roundedBorder)
                            Button("Pull") {
                                guard !pullModelName.isEmpty else { return }
                                isPullingModel = true
                                pullStatus = "Pulling \(pullModelName)..."
                                let name = pullModelName
                                DispatchQueue.global().async {
                                    let proc = Process()
                                    proc.executableURL = URL(fileURLWithPath: "/usr/bin/env")
                                    proc.arguments = ["ollama", "pull", name]
                                    try? proc.run()
                                    proc.waitUntilExit()
                                    DispatchQueue.main.async {
                                        isPullingModel = false
                                        pullStatus = proc.terminationStatus == 0
                                            ? "✓ Successfully pulled \(name)"
                                            : "✗ Failed to pull \(name). Check model name."
                                        Task { await ollamaChecker.refresh() }
                                    }
                                }
                            }
                            .disabled(pullModelName.isEmpty || isPullingModel)
                            .buttonStyle(.borderedProminent)
                        }
                        if !pullStatus.isEmpty {
                            Text(pullStatus)
                                .font(.caption)
                                .foregroundColor(pullStatus.hasPrefix("✓") ? .green : .red)
                        }
                    }
                } else if llmProvider == .openAI {
                    // OpenAI Configs
                    VStack(alignment: .leading, spacing: 10) {
                        Text("OpenAI API Configuration").font(.headline).foregroundColor(.blue)
                        
                        Text("API Key:")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                        SecureField("sk-...", text: $openAIKey)
                            .textFieldStyle(.roundedBorder)
                        
                        Text("Default Model:")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                        Picker("", selection: $openAIModel) {
                            Text("gpt-4o-mini").tag("gpt-4o-mini")
                            Text("gpt-4o").tag("gpt-4o")
                            Text("o1-mini").tag("o1-mini")
                        }
                        .pickerStyle(.menu)
                        
                        Text("API key is securely saved in the macOS Keychain.")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                } else if llmProvider == .claude {
                    // Claude Configs
                    VStack(alignment: .leading, spacing: 10) {
                        Text("Claude API Configuration").font(.headline).foregroundColor(.orange)
                        
                        Text("API Key:")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                        SecureField("sk-ant-...", text: $claudeKey)
                            .textFieldStyle(.roundedBorder)
                        
                        Text("Default Model:")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                        Picker("", selection: $claudeModel) {
                            Text("claude-3-5-haiku-latest").tag("claude-3-5-haiku-latest")
                            Text("claude-3-5-sonnet-latest").tag("claude-3-5-sonnet-latest")
                        }
                        .pickerStyle(.menu)
                        
                        Text("API key is securely saved in the macOS Keychain.")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                }

                Divider()

                // Temperature
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Text("Temperature").font(.headline)
                        Spacer()
                        Text(String(format: "%.2f", ollamaTemp))
                            .font(.system(.body, design: .monospaced))
                            .foregroundColor(.secondary)
                    }
                    Slider(value: $ollamaTemp, in: 0.0...1.0, step: 0.05)
                        .onChange(of: ollamaTemp) { _, v in
                            UserDefaults.standard.set(v, forKey: "Aura_OllamaTemp")
                        }
                    Text("Lower = more deterministic (recommended: 0.3 for analytical tasks)")
                        .font(.caption2).foregroundColor(.secondary)
                }

                // Max Tokens
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Text("Max Tokens").font(.headline)
                        Spacer()
                        Text("\(ollamaMaxTokens)")
                            .font(.system(.body, design: .monospaced))
                            .foregroundColor(.secondary)
                    }
                    Slider(value: Binding(
                        get: { Double(ollamaMaxTokens) },
                        set: { ollamaMaxTokens = Int($0) }
                    ), in: 512...4096, step: 256)
                    .onChange(of: ollamaMaxTokens) { _, v in
                        UserDefaults.standard.set(v, forKey: "Aura_OllamaMaxTokens")
                    }
                    Text("Maximum tokens generated per response (2048 recommended)")
                        .font(.caption2).foregroundColor(.secondary)
                }
            }
            .padding(24)
        }
    }

    private var logsTab: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("System Logs")
                        .font(.title2)
                        .fontWeight(.bold)
                    Text("Diagnostic logs for debugging Python subprocesses and app lifecycle")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                Spacer()
                
                Button("Copy All") {
                    let text = appLogger.getRawLogs()
                    let pasteboard = NSPasteboard.general
                    pasteboard.clearContents()
                    pasteboard.setString(text, forType: .string)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                
                Button("Clear") {
                    appLogger.clearLogs()
                }
                .buttonStyle(.bordered)
                .foregroundColor(.red)
                .controlSize(.small)
            }
            
            // Logs Terminal-like view
            ScrollViewReader { proxy in
                ScrollView(.vertical, showsIndicators: true) {
                    LazyVStack(alignment: .leading, spacing: 4) {
                        ForEach(appLogger.logs) { entry in
                            HStack(alignment: .top, spacing: 6) {
                                Text(entry.formattedString)
                                    .font(.system(size: 10, design: .monospaced))
                                    .foregroundColor(colorForLogLevel(entry.level))
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .textSelection(.enabled)
                            }
                            .id(entry.id)
                        }
                    }
                    .padding(8)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Color(red: 0.10, green: 0.10, blue: 0.12))
                .cornerRadius(10)
                .overlay(
                    RoundedRectangle(cornerRadius: 10)
                        .stroke(Color.black.opacity(0.5), lineWidth: 1)
                )
                .onChange(of: appLogger.logs.count) { _, _ in
                    if let last = appLogger.logs.last {
                        withAnimation {
                            proxy.scrollTo(last.id, anchor: .bottom)
                        }
                    }
                }
            }
            
            HStack {
                if let logURL = appLogger.getLogFileURL() {
                    Button(action: {
                        NSWorkspace.shared.selectFile(logURL.path, inFileViewerRootedAtPath: "")
                    }) {
                        Label("Reveal Log File in Finder", systemImage: "folder")
                    }
                    .buttonStyle(.link)
                    .font(.caption)
                }
                Spacer()
            }
        }
        .padding(24)
    }
    
    private func colorForLogLevel(_ level: AppLogger.LogLevel) -> Color {
        switch level {
        case .debug:
            return Color(red: 0.35, green: 0.85, blue: 0.45)   // terminal green
        case .info:
            return Color(red: 0.85, green: 0.85, blue: 0.90)   // soft white
        case .warning:
            return .orange
        case .error:
            return Color(red: 1.0, green: 0.38, blue: 0.38)    // bright red
        }
    }

    // MARK: - Local Server Helpers & Views

    private var statusTitle: String {
        switch serverStatus {
        case .stopped: return "Stopped"
        case .running: return "Running"
        case .starting: return "Starting..."
        case .stopping: return "Stopping..."
        }
    }
    
    private var statusIndicatorBadge: some View {
        let color: Color
        switch serverStatus {
        case .stopped: color = .red
        case .running: color = .green
        case .starting, .stopping: color = .orange
        }
        
        return Circle()
            .fill(color)
            .frame(width: 14, height: 14)
            .shadow(color: color.opacity(0.4), radius: 3)
    }

    private var localServerTab: some View {
        VStack(alignment: .leading, spacing: 20) {
            // Header
            VStack(alignment: .leading, spacing: 4) {
                Text("Local API Server")
                    .font(.title2)
                    .fontWeight(.bold)
                Text("Manage the background microservice for ML operations")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            .padding(.bottom, 8)
            
            // Server Status Box
            VStack(spacing: 0) {
                HStack(spacing: 16) {
                    // Status Badge Indicator
                    statusIndicatorBadge
                    
                    VStack(alignment: .leading, spacing: 4) {
                        Text(statusTitle)
                            .fontWeight(.semibold)
                            .font(.headline)
                        
                        Text("Address: http://127.0.0.1:11435")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    
                    Spacer()
                    
                    if let pid = serverPID {
                        Text("PID: \(pid)")
                            .font(.system(.caption, design: .monospaced))
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(Color.secondary.opacity(0.1))
                            .cornerRadius(4)
                    }
                }
                .padding(16)
                
                Divider()
                
                // Details row
                HStack {
                    Text("Python Path:")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text(pythonPath.isEmpty ? "Not configured" : pythonPath)
                        .font(.system(.caption, design: .monospaced))
                        .foregroundColor(.primary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Spacer()
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 12)
                .background(Color.secondary.opacity(0.03))
            }
            .background(Color(NSColor.controlBackgroundColor))
            .cornerRadius(12)
            .overlay(
                RoundedRectangle(cornerRadius: 12)
                    .stroke(Color.secondary.opacity(0.15), lineWidth: 1)
            )
            
            // Error Message (if any)
            if let errorMsg = serverErrorMsg {
                HStack(spacing: 12) {
                    Image(systemName: "exclamationmark.octagon.fill")
                        .foregroundColor(.red)
                        .font(.title3)
                    
                    Text(errorMsg)
                        .font(.caption)
                        .foregroundColor(.red)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .padding(12)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color.red.opacity(0.05))
                .cornerRadius(8)
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(Color.red.opacity(0.2), lineWidth: 1)
                )
            }
            
            // Action Buttons Card
            VStack(alignment: .leading, spacing: 12) {
                Text("Lifecycle Operations")
                    .font(.headline)
                
                HStack(spacing: 12) {
                    // Start Button
                    Button(action: startLocalServer) {
                        Label("Start", systemImage: "play.fill")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(Theme.Color.success)
                    .disabled(serverStatus == .running || serverStatus == .starting || serverStatus == .stopping || isPerformingServerAction)
                    
                    // Stop Button
                    Button(action: stopLocalServer) {
                        Label("Stop", systemImage: "stop.fill")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(Theme.Color.destructive)
                    .disabled(serverStatus == .stopped || serverStatus == .starting || serverStatus == .stopping || isPerformingServerAction)
                    
                    // Restart Button
                    Button(action: restartLocalServer) {
                        Label("Restart", systemImage: "arrow.clockwise")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                    .disabled(serverStatus == .starting || serverStatus == .stopping || isPerformingServerAction)
                }
            }
            .padding()
            .background(Color(NSColor.controlBackgroundColor))
            .cornerRadius(12)
            .overlay(
                RoundedRectangle(cornerRadius: 12)
                    .stroke(Color.secondary.opacity(0.1), lineWidth: 1)
            )
            
            // Hybrid Mode / Cloud Compute Offloading Card
            VStack(alignment: .leading, spacing: 12) {
                Text("Cloud Compute Offloading (Hybrid Mode)")
                    .font(.headline)
                
                Toggle("Enable Hybrid Mode (Offload to Cloud)", isOn: $hybridMode)
                    .toggleStyle(.checkbox)
                
                if hybridMode {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Remote Server Base URL")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        TextField("http://remote-gpu-server-ip:11435", text: $remoteServerURL)
                            .textFieldStyle(.roundedBorder)
                            .font(.system(.body, design: .monospaced))
                    }
                }
            }
            .padding()
            .background(Color(NSColor.controlBackgroundColor))
            .cornerRadius(12)
            .overlay(
                RoundedRectangle(cornerRadius: 12)
                    .stroke(Color.secondary.opacity(0.1), lineWidth: 1)
            )

            Spacer()

            
            // Info Note
            Text("The local API server handles dataset previewing, profiling, DB queries, data cleaning, and model training asynchronously. If it stops, Aura will attempt to restart it automatically on the next ML request.")
                .font(.caption2)
                .foregroundColor(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(24)
    }

    private func updateServerStatus() {
        Task {
            let running = await PythonRunner.shared.isServerRunning()
            let pid = PythonRunner.shared.getServerPID()
            await MainActor.run {
                if !isPerformingServerAction {
                    self.serverStatus = running ? .running : .stopped
                    self.serverPID = pid
                }
            }
        }
    }

    private func startLocalServer() {
        isPerformingServerAction = true
        serverStatus = .starting
        serverErrorMsg = nil
        Task {
            do {
                try await PythonRunner.shared.startServerManual()
                let pid = PythonRunner.shared.getServerPID()
                await MainActor.run {
                    self.serverStatus = .running
                    self.serverPID = pid
                    self.isPerformingServerAction = false
                }
            } catch {
                await MainActor.run {
                    self.serverStatus = .stopped
                    self.serverPID = nil
                    self.serverErrorMsg = error.localizedDescription
                    self.isPerformingServerAction = false
                }
            }
        }
    }
    
    private func stopLocalServer() {
        isPerformingServerAction = true
        serverStatus = .stopping
        serverErrorMsg = nil
        Task {
            await PythonRunner.shared.stopServerManual()
            try? await Task.sleep(nanoseconds: 300_000_000)
            await MainActor.run {
                self.serverStatus = .stopped
                self.serverPID = nil
                self.isPerformingServerAction = false
            }
        }
    }
    
    private func restartLocalServer() {
        isPerformingServerAction = true
        serverStatus = .stopping
        serverErrorMsg = nil
        Task {
            do {
                try await PythonRunner.shared.restartServerManual()
                let pid = PythonRunner.shared.getServerPID()
                await MainActor.run {
                    self.serverStatus = .running
                    self.serverPID = pid
                    self.isPerformingServerAction = false
                }
            } catch {
                await MainActor.run {
                    self.serverStatus = .stopped
                    self.serverPID = nil
                    self.serverErrorMsg = error.localizedDescription
                    self.isPerformingServerAction = false
                }
            }
        }
    }

    // MARK: - Tab Views (Sidebar navigation)

    private var pythonRuntimeTab: some View {
        VStack(alignment: .leading, spacing: 20) {
            // Header
            VStack(alignment: .leading, spacing: 4) {
                Text("Settings & Diagnostics")
                    .font(.title2)
                    .fontWeight(.bold)
                Text("Configure local Python runtime environment")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            .padding(.bottom, 8)
            
            // Python executable settings
            VStack(alignment: .leading, spacing: 8) {
                Text("Python 3 Executable Path")
                    .font(.headline)
                
                HStack(spacing: 12) {
                    TextField("Enter path to python3 binary", text: $pythonPath)
                        .textFieldStyle(.roundedBorder)
                        .font(.system(.body, design: .monospaced))
                    
                    Button("Browse...") {
                        selectPythonBinary()
                    }
                    
                    Button("Verify") {
                        verifyPath()
                    }
                    .disabled(pythonPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || isChecking)
                }
                
                Text("Ensure the path points to a python3 environment containing: pandas, numpy, scikit-learn, torch (with MPS support).")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
            
            Divider()
            
            // Diagnostic Status
            VStack(alignment: .leading, spacing: 12) {
                Text("Verification Result")
                    .font(.headline)
                
                if isChecking {
                    HStack(spacing: 8) {
                        NativeProgressView(controlSize: .small)
                        Text("Testing python environment...")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }
                    .padding(.vertical, 8)
                } else {
                    HStack(alignment: .top, spacing: 12) {
                        Image(systemName: isValid ? "checkmark.seal.fill" : "exclamationmark.octagon.fill")
                            .font(.title2)
                            .foregroundColor(isValid ? .green : .orange)
                        
                        VStack(alignment: .leading, spacing: 4) {
                            Text(isValid ? "Environment OK" : "Configuration Warning")
                                .fontWeight(.bold)
                                .foregroundColor(isValid ? .green : .orange)
                            
                            Text(validationMessage)
                                .font(.caption)
                                .foregroundColor(.secondary)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }
                    .padding(16)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(
                        RoundedRectangle(cornerRadius: 12)
                            .fill(isValid ? Color.green.opacity(0.04) : Color.orange.opacity(0.04))
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 12)
                            .stroke(isValid ? Color.green.opacity(0.15) : Color.orange.opacity(0.15), lineWidth: 1)
                    )
                }
            }
            
            Spacer()
            
            // Actions footer
            HStack {
                Button(action: autoDetectPath) {
                    Label("Auto-detect Python", systemImage: "sparkles")
                }
                .disabled(isChecking)
                
                Spacer()
                
                if !isValid && !isChecking {
                    Text("Tip: Run 'pip install pandas scikit-learn' in terminal")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
            }
        }
    }

    private var apiCredentialsTab: some View {
        VStack(alignment: .leading, spacing: 20) {
            // Header
            VStack(alignment: .leading, spacing: 4) {
                Text("API Credentials")
                    .font(.title2)
                    .fontWeight(.bold)
                Text("Configure optional credentials for dataset downloads")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            .padding(.bottom, 8)
            
            // Kaggle section
            VStack(alignment: .leading, spacing: 10) {
                Text("Kaggle API Credentials")
                    .font(.headline)
                    .foregroundColor(.blue)
                
                Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 8) {
                    GridRow {
                        Text("Username:")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                            .frame(width: 80, alignment: .trailing)
                        TextField("Kaggle Username", text: $kaggleUsername)
                            .textFieldStyle(.roundedBorder)
                    }
                    GridRow {
                        Text("API Key:")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                            .frame(width: 80, alignment: .trailing)
                        SecureField("Kaggle API Key", text: $kaggleKey)
                            .textFieldStyle(.roundedBorder)
                    }
                }
                
                Text("Required to download datasets from Kaggle. Find yours in: Kaggle Profile > Account > Create New Token.")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            
            Divider()
            
            // Hugging Face section
            VStack(alignment: .leading, spacing: 10) {
                Text("Hugging Face Credentials")
                    .font(.headline)
                    .foregroundColor(.orange)
                
                Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 8) {
                    GridRow {
                        Text("User Token:")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                            .frame(width: 80, alignment: .trailing)
                        SecureField("HF User Token (hf_...)", text: $hfToken)
                            .textFieldStyle(.roundedBorder)
                    }
                }
                
                Text("Only needed for private datasets. Generate at: Hugging Face > Settings > Access Tokens.")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            
            Spacer()
        }
    }

    private var appearanceTab: some View {
        VStack(alignment: .leading, spacing: 20) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Appearance Settings")
                    .font(.title2)
                    .fontWeight(.bold)
                Text("Configure your user interface appearance preference.")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            .padding(.bottom, 8)
            
            VStack(alignment: .leading, spacing: 12) {
                CustomSegmentedPicker(
                    selection: $appearanceMode,
                    items: [
                        ("System", "System"),
                        ("Light", "Light"),
                        ("Dark", "Dark")
                    ]
                )
                .frame(width: 250)
            }
            .padding()
            .background(Color(NSColor.controlBackgroundColor))
            .cornerRadius(8)
            
            Spacer()
        }
    }

    private func sidebarButton(title: String, icon: String, tag: Int) -> some View {
        Button(action: {
            selectedTab = tag
        }) {
            HStack(spacing: 8) {
                Image(systemName: icon)
                    .font(.body)
                    .foregroundColor(selectedTab == tag ? .white : .primary.opacity(0.8))
                    .frame(width: 20, alignment: .center)
                Text(title)
                    .font(.body)
                    .foregroundColor(selectedTab == tag ? .white : .primary.opacity(0.8))
                Spacer()
            }
            .padding(.vertical, 6)
            .padding(.horizontal, 10)
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .fill(selectedTab == tag ? Color.accentColor : Color.clear)
            )
        }
        .buttonStyle(.plain)
    }

    private var cacheCleanerTab: some View {
        VStack(alignment: .leading, spacing: 20) {
            // Header
            VStack(alignment: .leading, spacing: 4) {
                Text("Cache Cleaner")
                    .font(.title2)
                    .fontWeight(.bold)
                Text("Manage and clear downloaded remote datasets cache")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            .padding(.bottom, 8)
            
            // Cache Information Card
            VStack(alignment: .leading, spacing: 12) {
                Text("Cache Information")
                    .font(.headline)
                
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("Cache Folder:")
                            .foregroundColor(.secondary)
                        Text(cachePath)
                            .font(.system(.body, design: .monospaced))
                            .lineLimit(1)
                            .truncationMode(.middle)
                    }
                    
                    HStack {
                        Text("Total Size:")
                            .foregroundColor(.secondary)
                        Text(cacheSizeFormatted)
                            .fontWeight(.semibold)
                    }
                    
                    HStack {
                        Text("Cached Files:")
                            .foregroundColor(.secondary)
                        Text("\(cacheFileCount) datasets")
                    }
                }
                .padding()
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color.primary.opacity(0.02))
                .cornerRadius(12)
                .overlay(
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(Color.primary.opacity(0.06), lineWidth: 1)
                )
            }
            
            Divider()
            
            // Actions
            HStack(spacing: 12) {
                Button(action: performClearCache) {
                    Label(isCleaningCache ? "Clearing..." : "Clear Cache Now", systemImage: "trash.fill")
                }
                .disabled(isCleaningCache || cacheFileCount == 0)
                
                Button(action: loadCacheInfo) {
                    Label("Refresh", systemImage: "arrow.clockwise")
                }
                .disabled(isCleaningCache)
            }
            
            if let err = cleaningCacheError {
                Text(err)
                    .foregroundColor(.red)
                    .font(.caption)
            }
        }
        .onAppear {
            loadCacheInfo()
        }
    }

    private func loadCacheInfo() {
        Task {
            do {
                let info = try await PythonRunner.shared.getCacheInfo()
                await MainActor.run {
                    self.cachePath = info.path
                    self.cacheSizeBytes = info.sizeBytes
                    self.cacheFileCount = info.fileCount
                }
            } catch {
                appLogger.error("Failed to load cache info: \(error.localizedDescription)")
            }
        }
    }
    
    private func performClearCache() {
        isCleaningCache = true
        cleaningCacheError = nil
        Task {
            do {
                try await PythonRunner.shared.cleanCache()
                let info = try await PythonRunner.shared.getCacheInfo()
                await MainActor.run {
                    self.cachePath = info.path
                    self.cacheSizeBytes = info.sizeBytes
                    self.cacheFileCount = info.fileCount
                    self.isCleaningCache = false
                }
            } catch {
                await MainActor.run {
                    self.cleaningCacheError = "Failed to clear cache: \(error.localizedDescription)"
                    self.isCleaningCache = false
                }
            }
        }
    }
}


#Preview {
    SettingsView()
}
