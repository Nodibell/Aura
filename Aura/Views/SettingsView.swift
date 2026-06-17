import SwiftUI

struct SettingsView: View {
    @State private var selectedTab = 0
    private var appLogger = AppLogger.shared

    // Python Runtime State
    @State private var pythonPath = ""
    @State private var validationMessage = ""
    @State private var isValid = false
    @State private var isChecking = false

    // API Credentials State
    @State private var kaggleUsername = ""
    @State private var kaggleKey = ""
    @State private var hfToken = ""

    // AI Settings State
    @State private var ollamaModel = ""
    @State private var ollamaTemp: Double = 0.3
    @State private var ollamaMaxTokens: Int = 2048
    @State private var isPullingModel = false
    @State private var pullModelName = ""
    @State private var pullStatus = ""
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        TabView(selection: $selectedTab) {
            // TAB 1: Python Runtime Settings
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
                    
                    Text("Ensure the path points to a python3 environment containing: pandas, numpy, scikit-learn.")
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
                            ProgressView()
                                .controlSize(.small)
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
            .padding(24)
            .tabItem {
                Label("Python Runtime", systemImage: "terminal")
            }
            .tag(0)
            
            // TAB 2: API Credentials
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
            .padding(24)
            .tabItem {
                Label("API Credentials", systemImage: "key.fill")
            }
            .tag(1)

            // TAB 3: AI / Local LLM
            aiTab
                .tabItem {
                    Label("AI / LLM", systemImage: "sparkles")
                }
                .tag(2)

            // TAB 4: System Logs
            logsTab
                .tabItem {
                    Label("System Logs", systemImage: "doc.text.magnifyingglass")
                }
                .tag(3)
        }
        .frame(width: 600, height: 520)
        .onAppear {
            pythonPath = PythonRunner.shared.resolvePythonPath()
            verifyPath()
            kaggleUsername = KeychainService.shared.getSecureString(forKey: "Aura_KaggleUsername") ?? ""
            kaggleKey = KeychainService.shared.getSecureString(forKey: "Aura_KaggleKey") ?? ""
            hfToken = KeychainService.shared.getSecureString(forKey: "Aura_HFToken") ?? ""
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
                    self.validationMessage = "Successfully connected. Required packages (pandas, numpy, scikit-learn) are imported correctly."
                    PythonRunner.shared.setCustomPythonPath(pathToCheck)
                } else {
                    self.isValid = false
                    self.validationMessage = "Python binary located, but environment is missing required packages. Run: 'pip install pandas scikit-learn numpy' inside this environment."
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
        return VStack(alignment: .leading, spacing: 20) {
            // Header
            VStack(alignment: .leading, spacing: 4) {
                Text("AI / Local LLM")
                    .font(.title2)
                    .fontWeight(.bold)
                Text("Configure the Ollama-powered AI Analyst")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            .padding(.bottom, 4)

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

            Divider()

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

            Spacer()
        }
        .padding(24)
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
                .background(Color.black.opacity(0.4))
                .cornerRadius(8)
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(Color.secondary.opacity(0.2), lineWidth: 1)
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
            return .green.opacity(0.8)
        case .info:
            return .white.opacity(0.9)
        case .warning:
            return .orange
        case .error:
            return .red
        }
    }
}


#Preview {
    SettingsView()
}
