import SwiftUI

struct FileMergeSheet: View {
    let file1Path: String
    let file2Path: String
    @Binding var isPresented: Bool
    let onMergeCompleted: (String) -> Void
    
    @State private var file1Columns: [String] = []
    @State private var file2Columns: [String] = []
    
    @State private var key1: String = ""
    @State private var key2: String = ""
    @State private var joinType: String = "inner"
    
    @State private var isLoadingColumns: Bool = false
    @State private var isMerging: Bool = false
    @State private var errorMessage: String? = nil
    @State private var statusMessage: String = ""
    
    let joinTypes = [
        ("inner", "Inner Join (Intersection)"),
        ("left", "Left Join (Keep all left)"),
        ("right", "Right Join (Keep all right)"),
        ("outer", "Outer Join (Union)")
    ]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Merge Datasets")
                        .font(.title3.bold())
                    Text("Combine two tables on corresponding join keys")
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
            
            Divider().background(Color.primary.opacity(0.07))
            
            if isLoadingColumns {
                VStack(spacing: 12) {
                    ProgressView()
                    Text("Loading columns from files...")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollView {
                    VStack(alignment: .leading, spacing: 20) {
                        
                        // File 1
                        VStack(alignment: .leading, spacing: 6) {
                            Text("File 1 (Left Table)")
                                .font(.system(size: 11, weight: .bold))
                                .foregroundColor(.purple)
                            Text(URL(fileURLWithPath: file1Path).lastPathComponent)
                                .font(.subheadline)
                                .lineLimit(1)
                                .foregroundColor(.primary)
                            
                            HStack {
                                Text("Join Key:")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                Picker("", selection: $key1) {
                                    if key1.isEmpty {
                                        Text("Select Key...").tag("")
                                    }
                                    ForEach(file1Columns, id: \.self) { col in
                                        Text(col).tag(col)
                                    }
                                }
                                .pickerStyle(.menu)
                            }
                        }
                        .padding(12)
                        .background(Color.primary.opacity(0.03))
                        .cornerRadius(10)
                        
                        // File 2
                        VStack(alignment: .leading, spacing: 6) {
                            Text("File 2 (Right Table)")
                                .font(.system(size: 11, weight: .bold))
                                .foregroundColor(.blue)
                            Text(URL(fileURLWithPath: file2Path).lastPathComponent)
                                .font(.subheadline)
                                .lineLimit(1)
                                .foregroundColor(.primary)
                            
                            HStack {
                                Text("Join Key:")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                Picker("", selection: $key2) {
                                    if key2.isEmpty {
                                        Text("Select Key...").tag("")
                                    }
                                    ForEach(file2Columns, id: \.self) { col in
                                        Text(col).tag(col)
                                    }
                                }
                                .pickerStyle(.menu)
                            }
                        }
                        .padding(12)
                        .background(Color.primary.opacity(0.03))
                        .cornerRadius(10)
                        
                        // Join Type
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Join Strategy")
                                .font(.system(size: 11, weight: .bold))
                                .foregroundColor(.secondary)
                            
                            Picker("", selection: $joinType) {
                                ForEach(joinTypes, id: \.0) { item in
                                    Text(item.1).tag(item.0)
                                }
                            }
                            .pickerStyle(.radioGroup)
                        }
                        .padding(12)
                        .background(Color.primary.opacity(0.03))
                        .cornerRadius(10)
                        
                        if let err = errorMessage {
                            Label(err, systemImage: "exclamationmark.triangle.fill")
                                .font(.caption)
                                .foregroundColor(.red)
                                .padding(10)
                                .background(Color.red.opacity(0.07))
                                .cornerRadius(8)
                        }
                        
                        if isMerging {
                            HStack(spacing: 8) {
                                ProgressView().controlSize(.small)
                                Text(statusMessage.isEmpty ? "Merging tables..." : statusMessage)
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        }
                    }
                    .padding(20)
                }
            }
            
            Divider().background(Color.primary.opacity(0.07))
            
            // Footer Action buttons
            HStack {
                Button("Cancel") { isPresented = false }
                    .buttonStyle(.bordered)
                    .disabled(isMerging)
                
                Spacer()
                
                Button {
                    Task { await performMerge() }
                } label: {
                    HStack(spacing: 6) {
                        if isMerging {
                            ProgressView().controlSize(.small)
                        } else {
                            Image(systemName: "plus.square.on.square")
                        }
                        Text("Merge & Open")
                            .fontWeight(.semibold)
                    }
                    .foregroundColor(.white)
                    .padding(.horizontal, 18)
                    .padding(.vertical, 9)
                    .background(
                        isMerging || key1.isEmpty || key2.isEmpty
                        ? AnyShapeStyle(Color.primary.opacity(0.08))
                        : AnyShapeStyle(LinearGradient(colors: [.purple, .blue], startPoint: .leading, endPoint: .trailing))
                    )
                    .cornerRadius(9)
                }
                .buttonStyle(.plain)
                .disabled(isMerging || key1.isEmpty || key2.isEmpty)
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 14)
        }
        .frame(width: 450, height: 500)
        .background(Color(nsColor: .windowBackgroundColor))
        .cornerRadius(16)
        
        .onAppear {
            loadColumns()
        }
    }
    
    private func loadColumns() {
        isLoadingColumns = true
        errorMessage = nil
        
        Task {
            let runner = PythonRunner.shared
            
            // Load columns for File 1
            let res1 = await withCheckedContinuation { continuation in
                Task {
                    await runner.runPreview(csvPathOrURL: file1Path, progress: { _, _ in }) { result in
                        continuation.resume(returning: result)
                    }
                }
            }
            
            // Load columns for File 2
            let res2 = await withCheckedContinuation { continuation in
                Task {
                    await runner.runPreview(csvPathOrURL: file2Path, progress: { _, _ in }) { result in
                        continuation.resume(returning: result)
                    }
                }
            }
            
            await MainActor.run {
                switch (res1, res2) {
                case (.success(let preview1), .success(let preview2)):
                    self.file1Columns = preview1.columns
                    self.file2Columns = preview2.columns
                    
                    // Auto-select common keys if they exist
                    if let firstCommon = preview1.columns.first(where: { preview2.columns.contains($0) }) {
                        self.key1 = firstCommon
                        self.key2 = firstCommon
                    } else {
                        self.key1 = preview1.columns.first ?? ""
                        self.key2 = preview2.columns.first ?? ""
                    }
                    
                case (.failure(let err), _):
                    self.errorMessage = "Failed to load File 1 columns: \(err.localizedDescription)"
                case (_, .failure(let err)):
                    self.errorMessage = "Failed to load File 2 columns: \(err.localizedDescription)"
                }
                self.isLoadingColumns = false
            }
        }
    }
    
    @MainActor
    private func performMerge() async {
        isMerging = true
        errorMessage = nil
        statusMessage = "Merging datasets..."
        
        let parentDir = URL(fileURLWithPath: file1Path).deletingLastPathComponent()
        let name1 = URL(fileURLWithPath: file1Path).deletingPathExtension().lastPathComponent
        let name2 = URL(fileURLWithPath: file2Path).deletingPathExtension().lastPathComponent
        let outputMergePath = parentDir.appendingPathComponent("merged_\(name1)_with_\(name2).csv").path
        
        do {
            let runner = PythonRunner.shared
            let (rowCount, _) = try await runner.runMerge(
                file1: file1Path,
                file2: file2Path,
                key1: key1,
                key2: key2,
                joinType: joinType,
                outputMergePath: outputMergePath
            )
            
            statusMessage = "Completed! Joined into \(rowCount) rows."
            onMergeCompleted(outputMergePath)
            isPresented = false
        } catch {
            errorMessage = "Merge failed: \(error.localizedDescription)"
        }
        isMerging = false
    }
}
