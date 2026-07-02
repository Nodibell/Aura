import SwiftUI

struct PredictionTabView: View {
    let result: AnalysisResult
    let csvPath: String
    let config: AnalysisConfig
    
    @State private var inputValues: [String: Any] = [:]
    @State private var isPredicting = false
    @State private var predictionResult: PredictionResult? = nil
    @State private var errorMessage: String? = nil
    @State private var showIdentifierColumns = false
    
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
        HStack(spacing: 0) {
            // Inputs Panel
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    Text("Interactive Feature Inputs")
                        .font(.title2)
                        .fontWeight(.bold)
                    
                    Text("Modify feature values to run live inference on the best performing model (\(result.metrics.model)).")
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
                                
                                Text(pred.prediction.displayString)
                                    .font(.system(size: 36, weight: .bold, design: .rounded))
                                    .foregroundColor(.accentColor)
                                    .padding(.vertical, 4)
                                
                                if let probs = pred.probabilities, let currentProb = probs[pred.prediction.displayString] {
                                    Text("Confidence: \(String(format: "%.1f%%", currentProb * 100))")
                                        .font(.subheadline)
                                        .foregroundColor(.secondary)
                                }
                            }
                            .padding()
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(
                                RoundedRectangle(cornerRadius: 16, style: .continuous)
                                    .fill(Color(NSColor.controlBackgroundColor))
                                    .shadow(color: Color.black.opacity(0.05), radius: 8, x: 0, y: 4)
                            )
                            
                            // Class Probabilities if classification
                            if let probs = pred.probabilities {
                                VStack(alignment: .leading, spacing: 12) {
                                    Text("Class Probabilities")
                                        .font(.headline)
                                    
                                    let sortedProbs = probs.sorted(by: { $0.value > $1.value })
                                    ForEach(sortedProbs, id: \.key) { className, prob in
                                        VStack(alignment: .leading, spacing: 4) {
                                            HStack {
                                                Text(className)
                                                    .font(.subheadline)
                                                    .fontWeight(.medium)
                                                Spacer()
                                                Text(String(format: "%.1f%%", prob * 100))
                                                    .font(.subheadline)
                                                    .foregroundColor(.secondary)
                                            }
                                            
                                            GeometryReader { geo in
                                                ZStack(alignment: .leading) {
                                                    Capsule()
                                                        .fill(Color(NSColor.gridColor))
                                                        .frame(height: 8)
                                                    
                                                    Capsule()
                                                        .fill(className == pred.prediction.displayString ? Color.accentColor : Color.secondary)
                                                        .frame(width: geo.size.width * CGFloat(prob), height: 8)
                                                }
                                            }
                                            .frame(height: 8)
                                        }
                                    }
                                }
                                .padding()
                                .background(
                                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                                        .fill(Color(NSColor.controlBackgroundColor))
                                )
                            }
                            
                            Spacer()
                        }
                        .padding(24)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    VStack(spacing: 12) {
                        Image(systemName: "play.fill")
                            .font(.largeTitle)
                            .foregroundColor(.secondary)
                        Text("Ready for Inference")
                            .font(.headline)
                        Text("Fill in the features on the left and click 'Predict' to compute a prediction.")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
                
                Divider()
                
                // Bottom predict action
                HStack {
                    Spacer()
                    Button(action: runPrediction) {
                        HStack {
                            Image(systemName: "cpu")
                            Text("Predict")
                        }
                        .font(.headline)
                        .padding(.horizontal, 24)
                        .padding(.vertical, 8)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(isPredicting || features.isEmpty)
                }
                .padding()
                .background(Color(NSColor.windowBackgroundColor))
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(Color(NSColor.underPageBackgroundColor))
        }
        .onAppear {
            initializeInputs()
        }
    }
    
    // MARK: - Input Binding Helper Views
    
    private func isDateField(_ feature: String) -> Bool {
        let profile = result.profiling?.columns[feature]
        if let profile = profile {
            if profile.type == "datetime" {
                return true
            }
            if profile.type == "categorical" || profile.type == "numeric" {
                return false
            }
        }
        
        let lowerName = feature.lowercased()
        if lowerName.contains("date") || 
           lowerName.contains("time") || 
           lowerName.contains("timestamp") ||
           lowerName.contains("year") ||
           lowerName.contains("month") ||
           lowerName.contains("day") ||
           lowerName.contains("period") ||
           lowerName == "ds" {
            return true
        }
        
        // If the first category can be successfully parsed as a date, treat as date field
        if let firstCat = profile?.topCategories?.first?.value,
           parseDateString(firstCat) != nil {
            return true
        }
        
        return false
    }
    
    private func parseDateString(_ str: String) -> Date? {
        let cleanStr = str.components(separatedBy: " (").first ?? str
        let formatters = [
            "yyyy-MM-dd HH:mm:ss",
            "yyyy-MM-dd",
            "MM/dd/yyyy",
            "yyyy-MM-dd'T'HH:mm:ss'Z'",
            "yyyy-MM-dd'T'HH:mm:ss.SSSZ"
        ].map { fmt -> DateFormatter in
            let df = DateFormatter()
            df.dateFormat = fmt
            return df
        }
        for formatter in formatters {
            if let date = formatter.date(from: cleanStr) {
                return date
            }
        }
        return nil
    }

    @ViewBuilder
    private func featureInputView(for feature: String) -> some View {
        let profile = result.profiling?.columns[feature]
        let isNumeric = profile?.type == "numeric" || (profile?.type == "identifier" && profile?.stats != nil)
        
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(feature)
                    .font(.subheadline)
                    .fontWeight(.bold)
                    .foregroundColor(.primary)
                Spacer()
                if let missing = profile?.missing, missing > 0 {
                    Text("\(missing)% nulls")
                        .font(.caption2)
                        .foregroundColor(.orange)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Capsule().fill(Color.orange.opacity(0.1)))
                }
            }
            
            if isDateField(feature) {
                dateInputView(feature: feature)
            } else if isNumeric, let stats = profile?.stats {
                numericInputView(feature: feature, stats: stats)
            } else if let categories = profile?.topCategories, !categories.isEmpty {
                categoricalInputView(feature: feature, categories: categories)
            } else {
                standardTextInputView(feature: feature)
            }
        }
        .padding(12)
        .background(Color.primary.opacity(0.025))
        .cornerRadius(10)
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .stroke(Color.primary.opacity(0.06), lineWidth: 1)
        )

    }
    
    @ViewBuilder
    private func dateInputView(feature: String) -> some View {
        let currentDate = Binding<Date>(
            get: { (inputValues[feature] as? Date) ?? Date() },
            set: { inputValues[feature] = $0 }
        )
        
        DatePicker("", selection: currentDate)
            .labelsHidden()
            .datePickerStyle(.compact)
    }
    
    @ViewBuilder
    private func numericInputView(feature: String, stats: NumericStats) -> some View {
        let isInteger = result.profiling?.columns[feature]?.isInteger ?? false
        let currentVal = Binding<Double>(
            get: { 
                let rawVal = (inputValues[feature] as? Double) ?? stats.p50
                return isInteger ? rawVal.rounded() : rawVal
            },
            set: { 
                inputValues[feature] = isInteger ? $0.rounded() : $0
            }
        )
        
        VStack(spacing: 6) {
            HStack {
                if isInteger {
                    Slider(value: currentVal, in: stats.min...stats.max, step: 1)
                } else {
                    Slider(value: currentVal, in: stats.min...stats.max)
                }
                
                TextField("", value: currentVal, format: isInteger ? .number.precision(.fractionLength(0)) : .number)
                    .textFieldStyle(RoundedBorderTextFieldStyle())
                    .frame(width: 70)
                    .multilineTextAlignment(.trailing)
            }
            
            HStack {
                Text(isInteger ? "Min: \(Int(stats.min))" : String(format: "Min: %.2f", stats.min))
                    .font(.caption2)
                    .foregroundColor(.secondary)
                Spacer()
                Text(isInteger ? "Max: \(Int(stats.max))" : String(format: "Max: %.2f", stats.max))
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
        }
    }
    
    @ViewBuilder
    private func categoricalInputView(feature: String, categories: [TopCategory]) -> some View {
        let currentVal = Binding<String>(
            get: { (inputValues[feature] as? String) ?? categories.first?.value ?? "" },
            set: { inputValues[feature] = $0 }
        )
        
        Picker("", selection: currentVal) {
            ForEach(categories) { cat in
                Text("\(cat.value) (\(cat.count))").tag(cat.value)
            }
        }
        .pickerStyle(MenuPickerStyle())
    }
    
    @ViewBuilder
    private func standardTextInputView(feature: String) -> some View {
        let currentVal = Binding<String>(
            get: { (inputValues[feature] as? String) ?? "" },
            set: { inputValues[feature] = $0 }
        )
        
        TextField("Enter value...", text: currentVal)
            .textFieldStyle(RoundedBorderTextFieldStyle())
    }
    
    // MARK: - Logic Helpers
    
    private func initializeInputs() {
        guard inputValues.isEmpty else { return }
        for feature in result.columns where feature != result.targetColumn {
            let profile = result.profiling?.columns[feature]
            if isDateField(feature) {
                if let firstCat = profile?.topCategories?.first?.value,
                   let parsedDate = parseDateString(firstCat) {
                    inputValues[feature] = parsedDate
                } else {
                    inputValues[feature] = Date()
                }
            } else if let stats = profile?.stats, (profile?.type == "numeric" || profile?.type == "identifier") {
                let isInteger = profile?.isInteger ?? false
                inputValues[feature] = isInteger ? stats.p50.rounded() : stats.p50
            } else if let categories = profile?.topCategories, !categories.isEmpty {
                inputValues[feature] = categories.first?.value ?? ""
            } else {
                inputValues[feature] = ""
            }
        }
    }
    
    private func runPrediction() {
        guard !isPredicting else { return }
        isPredicting = true
        errorMessage = nil
        
        let path: String
        if let modelPath = config.modelExportPath, !modelPath.isEmpty {
            path = modelPath
        } else {
            // Reconstruct the default path alongside the CSV
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
        
        // Prepare inputs (convert Date to formatted string, ONLY for active features)
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
                let response = try await PythonRunner.shared.runInference(modelPath: path, inputData: serializedInputs)
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
}
