import SwiftUI

struct ModelExportSheet: View {
    @Binding var config: AnalysisConfig
    @Binding var isPresented: Bool
    let onRunExport: () -> Void

    @State private var selectedFolderURL: URL? = nil
    @State private var modelFilename: String = "best_model.joblib"
    @State private var codeFilename: String = "reproduce_pipeline.py"
    @State private var errorMessage: String? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Export Trained Model & Code")
                        .font(.title3.bold())
                    Text("Save scikit-learn pipeline (.joblib) and reproduction script (.py)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                Spacer()
                Button { isPresented = false } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.title3)
                        .foregroundColor(.secondary)
                }
                .buttonStyle(.plain)
            }
            .padding(20)

            Divider().background(Color.white.opacity(0.07))

            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    // Explanation
                    VStack(alignment: .leading, spacing: 8) {
                        Label("How it works", systemImage: "info.circle")
                            .font(.system(size: 11, weight: .bold, design: .rounded))
                            .foregroundColor(.secondary.opacity(0.7))
                            .tracking(0.3)
                        
                        Text("Aura will fit the best model pipeline on your dataset and serialize it using the 'joblib' library. A corresponding Python script will be generated containing the exact steps (imputation, scaling, encoding, training) to replicate the pipeline locally.")
                            .font(.system(size: 11))
                            .foregroundColor(.secondary)
                            .lineSpacing(3)
                            .padding(12)
                            .background(Color.white.opacity(0.02))
                            .cornerRadius(8)
                            .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.white.opacity(0.05)))
                    }

                    // Directory selector
                    VStack(alignment: .leading, spacing: 8) {
                        Label("Export Destination Folder", systemImage: "folder")
                            .font(.system(size: 11, weight: .bold, design: .rounded))
                            .foregroundColor(.secondary.opacity(0.7))
                            .tracking(0.3)

                        HStack(spacing: 12) {
                            if let url = selectedFolderURL {
                                HStack {
                                    Image(systemName: "folder.fill").foregroundColor(.yellow)
                                    Text(url.path)
                                        .font(.system(size: 11, design: .monospaced))
                                        .lineLimit(1)
                                        .truncationMode(.middle)
                                }
                                .padding(8)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .background(Color.white.opacity(0.03))
                                .cornerRadius(6)
                            } else {
                                Text("No folder selected")
                                    .font(.system(size: 11))
                                    .foregroundColor(.secondary)
                                    .padding(8)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .background(Color.white.opacity(0.03))
                                    .cornerRadius(6)
                            }

                            Button("Browse...") {
                                selectFolder()
                            }
                        }
                    }

                    // Filename customisation
                    if selectedFolderURL != nil {
                        VStack(alignment: .leading, spacing: 12) {
                            Label("Output Filenames", systemImage: "doc.on.doc")
                                .font(.system(size: 11, weight: .bold, design: .rounded))
                                .foregroundColor(.secondary.opacity(0.7))
                                .tracking(0.3)
                            
                            HStack(spacing: 12) {
                                VStack(alignment: .leading, spacing: 4) {
                                    Text("Model Pipeline (.joblib)")
                                        .font(.caption2)
                                        .foregroundColor(.secondary)
                                    TextField("Model Filename", text: $modelFilename)
                                        .textFieldStyle(.roundedBorder)
                                        .font(.system(size: 11, design: .monospaced))
                                }
                                
                                VStack(alignment: .leading, spacing: 4) {
                                    Text("Reproduction Script (.py)")
                                        .font(.caption2)
                                        .foregroundColor(.secondary)
                                    TextField("Code Filename", text: $codeFilename)
                                        .textFieldStyle(.roundedBorder)
                                        .font(.system(size: 11, design: .monospaced))
                                }
                            }
                        }
                    }

                    if let err = errorMessage {
                        Label(err, systemImage: "exclamationmark.triangle.fill")
                            .font(.caption)
                            .foregroundColor(.red)
                            .padding(10)
                            .background(Color.red.opacity(0.07))
                            .cornerRadius(8)
                    }
                }
                .padding(20)
            }

            Divider().background(Color.white.opacity(0.07))

            // Action Buttons
            HStack(spacing: 12) {
                Button("Cancel") { isPresented = false }
                    .buttonStyle(.bordered)

                Spacer()

                Button {
                    executeExport()
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: "square.and.arrow.down")
                        Text("Run & Export")
                            .fontWeight(.semibold)
                    }
                    .foregroundColor(.white)
                    .padding(.horizontal, 18)
                    .padding(.vertical, 9)
                    .background(
                        selectedFolderURL == nil
                            ? AnyShapeStyle(Color.white.opacity(0.08))
                            : AnyShapeStyle(LinearGradient(colors: [.purple, .indigo], startPoint: .leading, endPoint: .trailing))
                    )
                    .cornerRadius(9)
                }
                .buttonStyle(.plain)
                .disabled(selectedFolderURL == nil)
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 14)
        }
        .frame(width: 480, height: 420)
        .background(Color(white: 0.08))
        .cornerRadius(16)
        .colorScheme(.dark)
    }

    private func selectFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.canCreateDirectories = true
        panel.title = "Select Export Destination Folder"
        if panel.runModal() == .OK {
            selectedFolderURL = panel.url
        }
    }

    private func executeExport() {
        guard let folder = selectedFolderURL else { return }
        
        let mFilename = modelFilename.trimmingCharacters(in: .whitespacesAndNewlines)
        let cFilename = codeFilename.trimmingCharacters(in: .whitespacesAndNewlines)
        
        guard !mFilename.isEmpty && !cFilename.isEmpty else {
            errorMessage = "Filenames cannot be empty."
            return
        }
        
        let modelURL = folder.appendingPathComponent(mFilename.hasSuffix(".joblib") ? mFilename : "\(mFilename).joblib")
        let codeURL = folder.appendingPathComponent(cFilename.hasSuffix(".py") ? cFilename : "\(cFilename).py")
        
        config.modelExportPath = modelURL.path
        config.codeExportPath = codeURL.path
        
        isPresented = false
        
        // Trigger re-run analysis
        onRunExport()
    }
}
