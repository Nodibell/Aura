import SwiftUI

struct PredictionTabView: View {
    let result: AnalysisResult
    let csvPath: String
    let config: AnalysisConfig
    
    @State private var inputValues: [String: Any] = [:]
    @State private var isPredicting = false
    @State private var predictionResult: PredictionResult? = nil
    @State private var errorMessage: String? = nil
    
    var features: [String] {
        result.columns.filter { $0 != result.targetColumn && !config.excludedColumns.contains($0) }
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
                        ProgressView()
                            .progressViewStyle(CircularProgressViewStyle())
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
    
    @ViewBuilder
    private func featureInputView(for feature: String) -> some View {
        let profile = result.profiling?.columns[feature]
        let isNumeric = profile?.type == "numeric"
        
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(feature)
                    .font(.body)
                    .fontWeight(.medium)
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
            
            if isNumeric, let stats = profile?.stats {
                numericInputView(feature: feature, stats: stats)
            } else if let categories = profile?.topCategories, !categories.isEmpty {
                categoricalInputView(feature: feature, categories: categories)
            } else {
                standardTextInputView(feature: feature)
            }
        }
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(Color(NSColor.controlBackgroundColor).opacity(0.5))
        )
    }
    
    @ViewBuilder
    private func numericInputView(feature: String, stats: NumericStats) -> some View {
        let currentVal = Binding<Double>(
            get: { (inputValues[feature] as? Double) ?? stats.p50 },
            set: { inputValues[feature] = $0 }
        )
        
        VStack(spacing: 6) {
            HStack {
                Slider(value: currentVal, in: stats.min...stats.max)
                
                TextField("", value: currentVal, format: .number)
                    .textFieldStyle(RoundedBorderTextFieldStyle())
                    .frame(width: 70)
                    .multilineTextAlignment(.trailing)
            }
            
            HStack {
                Text(String(format: "Min: %.2f", stats.min))
                    .font(.caption2)
                    .foregroundColor(.secondary)
                Spacer()
                Text(String(format: "Max: %.2f", stats.max))
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
        for feature in features {
            let profile = result.profiling?.columns[feature]
            if profile?.type == "numeric", let stats = profile?.stats {
                inputValues[feature] = stats.p50
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
        
        Task {
            do {
                let response = try await PythonRunner.shared.runInference(modelPath: path, inputData: inputValues)
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
