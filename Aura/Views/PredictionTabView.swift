import SwiftUI

struct PredictionTabView: View {
    let result: AnalysisResult
    let csvPath: String
    let config: AnalysisConfig
    let activeModelName: String?
    
    @State private var selectedPredictionMode = 0 // 0 = Single, 1 = Batch
    
    // Single Prediction States
    @State private var inputValues: [String: Any] = [:]
    @State private var isPredicting = false
    @State private var predictionResult: PredictionResult? = nil
    @State private var errorMessage: String? = nil
    @State private var showIdentifierColumns = false
    
    // Batch Prediction States
    @State private var selectedBatchFile: URL? = nil
    @State private var isRunningBatch = false
    @State private var batchResult: BatchPredictionResult? = nil
    @State private var batchError: String? = nil
    
    var features: [String] {
        result.columns.filter { feature in
            feature != result.targetColumn &&
            !config.excludedColumns.contains(feature) &&
            (showIdentifierColumns || result.profiling?.columns[feature]?.type != "identifier")
        }
    }
    
    private var hasIdentifierColumns: Bool {
        result.columns.contains { feature in
            result.profiling?.columns[feature]?.type == "identifier"
        }
    }
    
    var body: some View {
        VStack(spacing: 0) {
            // Mode Selector Header
            HStack {
                Picker("Prediction Mode", selection: $selectedPredictionMode) {
                    Text("Single Prediction").tag(0)
                    Text("Batch CSV Inference").tag(1)
                }
                .pickerStyle(.palette)
                .frame(width: 400)
                
                Spacer()
                
                Text("Active Model: \(activeModelName ?? result.metrics.model)")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 4)
                    .background(Color.primary.opacity(0.04))
                    .cornerRadius(6)
            }
            .padding()
            .background(Color(NSColor.controlBackgroundColor))
            
            Divider()
            
            if selectedPredictionMode == 0 {
                // Single Predict Mode Layout
                HStack(spacing: 0) {
                    // Inputs Panel
                    ScrollView {
                        VStack(alignment: .leading, spacing: 20) {
                            Text("Interactive Feature Inputs")
                                .font(.title3)
                                .fontWeight(.bold)
                            
                            Text("Modify feature values to run live inference on the active model.")
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                            
                            if hasIdentifierColumns {
                                Divider()
                                Toggle("Show Identifier Columns", isOn: $showIdentifierColumns)
                                    .toggleStyle(.checkbox)
                                    .help("Include ID or key columns that were excluded from predictive modeling.")
                            }
                            
                            Divider()
                            
                            if features.isEmpty {
                                Text("No features available for prediction.")
                                    .foregroundColor(.secondary)
                            } else {
                                VStack(spacing: 16) {
                                    ForEach(features, id: \.self) { feature in
                                        featureInputView(for: feature)
                                    }
                                }
                            }
                            
                            Spacer(minLength: 20)
                        }
                        .padding()
                    }
                    .frame(minWidth: 350, maxWidth: 450)
                    
                    Divider()
                    
                    // Results Panel
                    VStack(spacing: 0) {
                        if isPredicting {
                            VStack(spacing: 12) {
                                NativeProgressView(controlSize: .regular)
                                Text("Running Inference...")
                                    .foregroundColor(.secondary)
                            }
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                        } else if let err = errorMessage {
                            VStack(spacing: 12) {
                                Image(systemName: "exclamationmark.triangle.fill")
                                    .font(.largeTitle)
                                    .foregroundColor(.red)
                                Text("Prediction Error")
                                    .font(.headline)
                                Text(err)
                                    .multilineTextAlignment(.center)
                                    .font(.subheadline)
                                    .foregroundColor(.secondary)
                                    .padding(.horizontal)
                            }
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                        } else if let pred = predictionResult {
                            ScrollView {
                                VStack(alignment: .leading, spacing: 24) {
                                    // Primary Prediction Card
                                    VStack(alignment: .leading, spacing: 8) {
                                        Text("Predicted Value")
                                            .font(.subheadline)
                                            .fontWeight(.semibold)
                                            .foregroundColor(.secondary)
                                        
                                        Text(String(describing: pred.prediction))
                                            .font(.system(size: 40, weight: .bold, design: .rounded))
                                            .foregroundColor(.purple)
                                    }
                                    .padding()
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .background(Color.purple.opacity(0.08))
                                    .cornerRadius(12)
                                    .overlay(
                                        RoundedRectangle(cornerRadius: 12)
                                            .stroke(Color.purple.opacity(0.2), lineWidth: 1)
                                    )
                                    
                                    // Probabilities (for classification)
                                    if let probs = pred.probabilities, !probs.isEmpty {
                                        VStack(alignment: .leading, spacing: 12) {
                                            Text("Class Probabilities")
                                                .font(.headline)
                                            
                                            let sortedProbs = probs.sorted { $0.value > $1.value }
                                            ForEach(sortedProbs, id: \.key) { className, prob in
                                                VStack(alignment: .leading, spacing: 4) {
                                                    HStack {
                                                        Text(className)
                                                            .font(.subheadline)
                                                        Spacer()
                                                        Text(String(format: "%.1f%%", prob * 100))
                                                            .font(.subheadline)
                                                            .fontWeight(.bold)
                                                    }
                                                    
                                                    GeometryReader { geo in
                                                        ZStack(alignment: .leading) {
                                                            Capsule()
                                                                .fill(Color.primary.opacity(0.06))
                                                                .frame(height: 6)
                                                            Capsule()
                                                                .fill(className == String(describing: pred.prediction) ? Color.purple : Color.blue)
                                                                .frame(width: geo.size.width * CGFloat(prob), height: 6)
                                                        }
                                                    }
                                                    .frame(height: 6)
                                                }
                                            }
                                        }
                                        .padding()
                                        .background(Color.primary.opacity(0.02))
                                        .cornerRadius(12)
                                    }
                                    
                                    // Metadata
                                    VStack(alignment: .leading, spacing: 8) {
                                        Text("Inference Metadata")
                                            .font(.headline)
                                        
                                        HStack {
                                            Text("Model Used")
                                            Spacer()
                                            Text(activeModelName ?? result.metrics.model)
                                                .foregroundColor(.secondary)
                                        }
                                        .font(.subheadline)
                                        
                                        HStack {
                                            Text("Time Elapsed")
                                            Spacer()
                                            Text(String(format: "%.2f ms", (pred.timeElapsed ?? 0.0) * 1000))
                                                .foregroundColor(.secondary)
                                        }
                                        .font(.subheadline)
                                    }
                                    .padding()
                                    .background(Color.primary.opacity(0.02))
                                    .cornerRadius(12)
                                    
                                    HStack(alignment: .center) {
                                        Spacer()
                                        Button {
                                            runPrediction()
                                        } label: {
                                            Text("Run Inference")
                                        }
                                        .buttonStyle(.borderedProminent)
                                        .tint(.purple)
                                        .padding(.top, 8)
                                        Spacer()
                                    }
                                    
                                }
                                .padding()
                            }
                        } else {
                            VStack(spacing: 12) {
                                Image(systemName: "play.circle")
                                    .font(.system(size: 48))
                                    .foregroundColor(.secondary)
                                Text("Ready for Inference")
                                    .font(.headline)
                                    .foregroundColor(.secondary)
                                Text("Fill out the feature fields on the left and click 'Run Inference'.")
                                    .font(.subheadline)
                                    .foregroundColor(.secondary)
                                    .multilineTextAlignment(.center)
                                    .padding(.horizontal)
                                
                                Button {
                                    runPrediction()
                                } label: {
                                    Text("Run Inference")
                                }
                                .buttonStyle(.borderedProminent)
                                .tint(.purple)
                                .padding(.top, 8)
                            }
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                        }
                    }
                    .background(Color(NSColor.underPageBackgroundColor))
                }
            } else {
                // Batch CSV Inference Layout
                HStack(spacing: 0) {
                    // Left Input Configuration
                    VStack(alignment: .leading, spacing: 20) {
                        Text("Batch CSV Selection")
                            .font(.title3)
                            .fontWeight(.bold)
                        
                        Text("Provide a CSV file with columns matching the expected features. Predictions will be appended as a new column, along with probabilities for classifier models.")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                        
                        Divider()
                        
                        // Drag & Drop Area / Select File Button
                        VStack(spacing: 16) {
                            if let file = selectedBatchFile {
                                Image(systemName: "doc.text.fill")
                                    .font(.system(size: 48))
                                    .foregroundColor(.purple)
                                
                                VStack(spacing: 4) {
                                    Text(file.lastPathComponent)
                                        .font(.headline)
                                        .lineLimit(1)
                                    Text(file.path)
                                        .font(.caption2)
                                        .foregroundColor(.secondary)
                                        .lineLimit(2)
                                        .multilineTextAlignment(.center)
                                }
                                
                                Button("Choose Another File...") {
                                    selectBatchFile()
                                }
                                .buttonStyle(.bordered)
                            } else {
                                Image(systemName: "arrow.down.doc")
                                    .font(.system(size: 48))
                                    .foregroundColor(.secondary)
                                
                                Text("Drag & Drop CSV file here or click Browse")
                                    .font(.subheadline)
                                    .foregroundColor(.secondary)
                                
                                Button("Browse Files...") {
                                    selectBatchFile()
                                }
                                .buttonStyle(.borderedProminent)
                            }
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 32)
                        .background(Color.primary.opacity(0.02))
                        .cornerRadius(12)
                        .overlay(
                            RoundedRectangle(cornerRadius: 12)
                                .stroke(Color.primary.opacity(0.1), style: StrokeStyle(lineWidth: 1.5, lineCap: .round, lineJoin: .round, dash: [4, 4]))
                        )
                        .onDrop(of: ["public.file-url"], isTargeted: nil) { providers in
                            if let first = providers.first {
                                _ = first.loadObject(ofClass: URL.self) { url, _ in
                                    if let url = url, url.pathExtension.lowercased() == "csv" {
                                        DispatchQueue.main.async {
                                            self.selectedBatchFile = url
                                            self.batchResult = nil
                                            self.batchError = nil
                                        }
                                    }
                                }
                                return true
                            }
                            return false
                        }
                        
                        Spacer()
                        
                        // Run Button
                        Button {
                            runBatchPrediction()
                        } label: {
                            HStack {
                                Spacer()
                                if isRunningBatch {
                                    ProgressView().controlSize(.small)
                                        .padding(.trailing, 4)
                                    Text("Processing Batch...")
                                } else {
                                    Text("Run Batch Prediction")
                                }
                                Spacer()
                            }
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(.purple)
                        .controlSize(.large)
                        .disabled(selectedBatchFile == nil || isRunningBatch)
                    }
                    .padding()
                    .frame(width: 400)
                    
                    Divider()
                    
                    // Right Output Panel
                    VStack {
                        if isRunningBatch {
                            VStack(spacing: 16) {
                                NativeProgressView(controlSize: .regular)
                                Text("Running batch inference via server...")
                                    .font(.headline)
                                Text("Processing rows and appending model predictions...")
                                    .font(.subheadline)
                                    .foregroundColor(.secondary)
                            }
                        } else if let err = batchError {
                            VStack(spacing: 12) {
                                Image(systemName: "exclamationmark.triangle.fill")
                                    .font(.largeTitle)
                                    .foregroundColor(.red)
                                Text("Batch Inference Failed")
                                    .font(.headline)
                                Text(err)
                                    .font(.subheadline)
                                    .foregroundColor(.secondary)
                                    .multilineTextAlignment(.center)
                                    .padding()
                            }
                        } else if let res = batchResult {
                            VStack(spacing: 24) {
                                Image(systemName: "checkmark.circle.fill")
                                    .font(.system(size: 64))
                                    .foregroundColor(.green)
                                
                                VStack(spacing: 6) {
                                    Text("Batch Predictions Completed!")
                                        .font(.title2)
                                        .fontWeight(.bold)
                                    Text("Successfully ran active model predictions on all rows.")
                                        .font(.subheadline)
                                        .foregroundColor(.secondary)
                                }
                                
                                VStack(alignment: .leading, spacing: 12) {
                                    HStack {
                                        Text("Processed Rows")
                                        Spacer()
                                        Text("\(res.rowCount) rows")
                                            .fontWeight(.bold)
                                    }
                                    
                                    HStack {
                                        Text("Output File")
                                        Spacer()
                                        Text(URL(fileURLWithPath: res.outputPath).lastPathComponent)
                                            .fontWeight(.semibold)
                                            .lineLimit(1)
                                    }
                                    
                                    HStack {
                                        Text("Saved Path")
                                        Spacer()
                                        Text(res.outputPath)
                                            .font(.caption2)
                                            .foregroundColor(.secondary)
                                            .lineLimit(2)
                                            .multilineTextAlignment(.trailing)
                                    }
                                }
                                .padding()
                                .background(Color.primary.opacity(0.03))
                                .cornerRadius(12)
                                .frame(maxWidth: 480)
                                
                                HStack(spacing: 16) {
                                    Button("Show in Finder") {
                                        let url = URL(fileURLWithPath: res.outputPath)
                                        NSWorkspace.shared.activateFileViewerSelecting([url])
                                    }
                                    .buttonStyle(.borderedProminent)
                                    
                                    Button("Open Output File") {
                                        let url = URL(fileURLWithPath: res.outputPath)
                                        NSWorkspace.shared.open(url)
                                    }
                                    .buttonStyle(.bordered)
                                }
                            }
                            .padding()
                        } else {
                            VStack(spacing: 16) {
                                Image(systemName: "doc.text.magnifyingglass")
                                    .font(.system(size: 64))
                                    .foregroundColor(.secondary.opacity(0.5))
                                Text("Inference Results Console")
                                    .font(.headline)
                                    .foregroundColor(.secondary)
                                Text("Select an input file on the left and start batch execution to generate a predicted dataset CSV.")
                                    .font(.subheadline)
                                    .foregroundColor(.secondary)
                                    .multilineTextAlignment(.center)
                                    .padding(.horizontal)
                            }
                        }
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(Color(NSColor.underPageBackgroundColor))
                }
            }
        }
        .onAppear {
            initializeDefaultValues()
        }
    }
    
    // MARK: - Input Controls Renderers
    
    @ViewBuilder
    private func featureInputView(for feature: String) -> some View {
        let profile = result.profiling?.columns[feature]
        let colType = profile?.type.lowercased() ?? "numeric"
        
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(feature)
                    .font(.subheadline)
                    .fontWeight(.semibold)
                Spacer()
                Text(colType.capitalized)
                    .font(.system(size: 9, weight: .bold))
                    .foregroundColor(.secondary)
            }
            
            if colType == "numeric" {
                let stats = profile?.stats
                let minVal = stats?.min ?? 0.0
                let maxVal = stats?.max ?? 100.0
                let currentVal = (inputValues[feature] as? Double) ?? 0.0
                let range = minVal < maxVal ? minVal...maxVal : minVal...(minVal + 1.0)
                
                HStack(spacing: 12) {
                    Slider(value: Binding<Double>(
                        get: { currentVal },
                        set: { inputValues[feature] = $0 }
                    ), in: range)
                    
                    TextField("", value: Binding<Double>(
                        get: { currentVal },
                        set: { inputValues[feature] = $0 }
                    ), formatter: NumberFormatter())
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 80)
                }
            } else if colType == "categorical" {
                let categories = profile?.topCategories ?? []
                let selection = (inputValues[feature] as? String) ?? ""
                
                Picker("", selection: Binding<String>(
                    get: { selection },
                    set: { inputValues[feature] = $0 }
                )) {
                    ForEach(categories, id: \.value) { cat in
                        Text(cat.value).tag(cat.value)
                    }
                }
                .pickerStyle(.menu)
            } else if colType == "datetime" {
                let selection = (inputValues[feature] as? Date) ?? Date()
                
                DatePicker("", selection: Binding<Date>(
                    get: { selection },
                    set: { inputValues[feature] = $0 }
                ))
                .datePickerStyle(.field)
            } else {
                let text = (inputValues[feature] as? String) ?? ""
                
                TextField("Enter text value...", text: Binding<String>(
                    get: { text },
                    set: { inputValues[feature] = $0 }
                ))
                .textFieldStyle(.roundedBorder)
            }
        }
        .padding(10)
        .background(Color.primary.opacity(0.02))
        .cornerRadius(8)
    }
    
    // MARK: - Handlers & File Methods
    
    private func selectBatchFile() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.commaSeparatedText, .tabSeparatedText, .data]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.title = "Select Input CSV for Batch Prediction"
        
        if panel.runModal() == .OK, let url = panel.url {
            self.selectedBatchFile = url
            self.batchResult = nil
            self.batchError = nil
        }
    }
    
    private func initializeDefaultValues() {
        for feature in features {
            let profile = result.profiling?.columns[feature]
            let colType = profile?.type.lowercased() ?? "numeric"
            
            if colType == "numeric", let stats = profile?.stats {
                let isInteger = profile?.type.lowercased() == "integer"
                inputValues[feature] = isInteger ? stats.p50.rounded() : stats.p50
            } else if let categories = profile?.topCategories, !categories.isEmpty {
                inputValues[feature] = categories.first?.value ?? ""
            } else if colType == "datetime" {
                inputValues[feature] = Date()
            } else {
                inputValues[feature] = ""
            }
        }
    }
    
    private func runPrediction() {
        guard !isPredicting else { return }
        isPredicting = true
        errorMessage = nil
        
        var path: String
        if let modelPath = config.modelExportPath, !modelPath.isEmpty {
            path = modelPath
        } else {
            let csvURL = URL(fileURLWithPath: csvPath)
            if csvPath.starts(with: "http://") || csvPath.starts(with: "https://") {
                let tempDir = FileManager.default.temporaryDirectory
                let baseName = csvURL.deletingPathExtension().lastPathComponent
                path = tempDir.appendingPathComponent("\(baseName)_model.joblib").path
            } else {
                let folder = csvURL.deletingLastPathComponent()
                let baseName = csvURL.deletingPathExtension().lastPathComponent
                path = folder.appendingPathComponent("\(baseName)_model.joblib").path
            }
        }
        
        if let activeModel = activeModelName, activeModel != result.metrics.model {
            let url = URL(fileURLWithPath: path)
            let baseName = url.deletingPathExtension().lastPathComponent
            let folder = url.deletingLastPathComponent()
            let safeName = activeModel.replacingOccurrences(of: " ", with: "_")
                                    .replacingOccurrences(of: "(", with: "")
                                    .replacingOccurrences(of: ")", with: "")
                                    .replacingOccurrences(of: "=", with: "")
                                    .replacingOccurrences(of: ",", with: "")
            path = folder.appendingPathComponent("\(baseName)_\(safeName).joblib").path
        }
        
        var serializedInputs: [String: Any] = [:]
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd HH:mm:ss"
        
        for feature in features {
            if let val = inputValues[feature] {
                if let dateVal = val as? Date {
                    serializedInputs[feature] = dateFormatter.string(from: dateVal)
                } else {
                    serializedInputs[feature] = val
                }
            }
        }
        
        Task {
            do {
                let startTime = Date()
                var response = try await PythonRunner.shared.runInference(modelPath: path, inputData: serializedInputs)
                let elapsed = Date().timeIntervalSince(startTime)
                response.timeElapsed = elapsed
                await MainActor.run {
                    self.predictionResult = response
                    self.isPredicting = false
                }
            } catch {
                await MainActor.run {
                    self.errorMessage = error.localizedDescription
                    self.isPredicting = false
                }
            }
        }
    }
    
    private func runBatchPrediction() {
        guard let inputFile = selectedBatchFile, !isRunningBatch else { return }
        
        var path: String
        if let modelPath = config.modelExportPath, !modelPath.isEmpty {
            path = modelPath
        } else {
            let csvURL = URL(fileURLWithPath: csvPath)
            if csvPath.starts(with: "http://") || csvPath.starts(with: "https://") {
                let tempDir = FileManager.default.temporaryDirectory
                let baseName = csvURL.deletingPathExtension().lastPathComponent
                path = tempDir.appendingPathComponent("\(baseName)_model.joblib").path
            } else {
                let folder = csvURL.deletingLastPathComponent()
                let baseName = csvURL.deletingPathExtension().lastPathComponent
                path = folder.appendingPathComponent("\(baseName)_model.joblib").path
            }
        }
        
        if let activeModel = activeModelName, activeModel != result.metrics.model {
            let url = URL(fileURLWithPath: path)
            let baseName = url.deletingPathExtension().lastPathComponent
            let folder = url.deletingLastPathComponent()
            let safeName = activeModel.replacingOccurrences(of: " ", with: "_")
                                    .replacingOccurrences(of: "(", with: "")
                                    .replacingOccurrences(of: ")", with: "")
                                    .replacingOccurrences(of: "=", with: "")
                                    .replacingOccurrences(of: ",", with: "")
            path = folder.appendingPathComponent("\(baseName)_\(safeName).joblib").path
        }
        
        let savePanel = NSSavePanel()
        savePanel.allowedContentTypes = [.commaSeparatedText]
        savePanel.nameFieldStringValue = inputFile.deletingPathExtension().lastPathComponent + "_predicted.csv"
        savePanel.title = "Save Predicted Dataset"
        
        if savePanel.runModal() == .OK, let outputFile = savePanel.url {
            isRunningBatch = true
            batchError = nil
            batchResult = nil
            
            Task {
                do {
                    let response = try await PythonRunner.shared.runBatchInference(
                        modelPath: path,
                        inputFilePath: inputFile.path,
                        outputFilePath: outputFile.path
                    )
                    await MainActor.run {
                        self.batchResult = response
                        self.isRunningBatch = false
                    }
                } catch {
                    await MainActor.run {
                        self.batchError = error.localizedDescription
                        self.isRunningBatch = false
                    }
                }
            }
        }
    }
}
