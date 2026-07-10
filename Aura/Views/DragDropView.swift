import SwiftUI

struct DragDropView: View {
    let onFileDropped: ([URL]) -> Void
    let onSelectFileManually: () -> Void
    let onImportFromDatabase: () -> Void
    let onURLSubmitted: (String) -> Void
    let onSampleSelected: (String) -> Void
    let recentAnalyses: [HistoryItem]
    let onRecentSelected: (HistoryItem) -> Void
    let onRename: (HistoryItem) -> Void
    let onDelete: (HistoryItem) -> Void
    
    @State private var isDraggingOver = false
    @State private var urlInput = ""
    @State private var isUrlHovered = false
    @State private var animateAurora = false
    
    var body: some View {
        ZStack {
            // ── Background Aurora Glow ──────────────────────────────────────
            GeometryReader { geo in
                ZStack {
                    // Dark theme window background base
                    Theme.Color.background
                    
                    // Left Purple Glow
                    Circle()
                        .fill(Theme.Color.purple.opacity(animateAurora ? 0.15 : 0.08))
                        .frame(width: 450, height: 450)
                        .blur(radius: 90)
                        .offset(x: animateAurora ? -100 : -150, y: animateAurora ? -80 : -120)
                    
                    // Right Blue/Indigo Glow
                    Circle()
                        .fill(Theme.Color.indigo.opacity(animateAurora ? 0.12 : 0.06))
                        .frame(width: 500, height: 500)
                        .blur(radius: 100)
                        .offset(x: animateAurora ? 120 : 180, y: animateAurora ? 100 : 150)
                }
                .frame(width: geo.size.width, height: geo.size.height)
            }
            .ignoresSafeArea()
            .onAppear {
                withAnimation(.easeInOut(duration: 8).repeatForever(autoreverses: true)) {
                    animateAurora = true
                }
            }
            
            // ── Main Layout Scrollable Content ──────────────────────────────
            ScrollView(.vertical, showsIndicators: false) {
                VStack(spacing: 36) {
                    
                    // 1. Brand Header Block
                    VStack(spacing: 12) {
                        ZStack {
                            Circle()
                                .fill(LinearGradient(colors: [Theme.Color.purple.opacity(0.2), Theme.Color.indigo.opacity(0.1)], startPoint: .topLeading, endPoint: .bottomTrailing))
                                .frame(width: 72, height: 72)
                                .blur(radius: 4)
                            
                            Image(systemName: "chart.bar.doc.horizontal.fill")
                                .font(Theme.Font.brand(size: 34, weight: .bold))
                                .foregroundStyle(
                                    LinearGradient(
                                        colors: [Theme.Color.purple, Theme.Color.indigo, .blue],
                                        startPoint: .topLeading,
                                        endPoint: .bottomTrailing
                                    )
                                )
                                .shadow(color: Theme.Color.purple.opacity(0.3), radius: 8, x: 0, y: 4)
                        }
                        .padding(.top, 24)
                        
                        Text("Aura")
                            .font(Theme.Font.brand(size: 32, weight: .bold))
                            .foregroundColor(.primary)
                        
                        Text("Instantly preview, analyze, and generate AI insights for any dataset.")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                            .multilineTextAlignment(.center)
                            .frame(maxWidth: 420)
                    }
                    
                    // 2. Interactive Drag & Drop Area
                    VStack(spacing: 0) {
                        ZStack {
                            // Glassmorphic background
                            RoundedRectangle(cornerRadius: Theme.Layout.cornerRadius * 2)
                                .fill(.ultraThinMaterial)
                                .shadow(color: Color.black.opacity(0.15), radius: 10, x: 0, y: 5)
                            
                            // Highlight on Drag
                            RoundedRectangle(cornerRadius: Theme.Layout.cornerRadius * 2)
                                .fill(isDraggingOver ? Theme.Color.purple.opacity(0.06) : Color.clear)
                            
                            // Dashed stroke border
                            RoundedRectangle(cornerRadius: Theme.Layout.cornerRadius * 2)
                                .strokeBorder(
                                    LinearGradient(
                                        colors: isDraggingOver
                                            ? [Theme.Color.purple, Theme.Color.indigo, .blue]
                                            : [.secondary.opacity(0.2), .secondary.opacity(0.08)],
                                        startPoint: .topLeading,
                                        endPoint: .bottomTrailing
                                    ),
                                    style: StrokeStyle(lineWidth: isDraggingOver ? 2.5 : 1.5, dash: [8, 4])
                                )
                            
                            // Drop zone inner contents
                            VStack(spacing: 18) {
                                Image(systemName: "arrow.down.doc.fill")
                                    .font(Theme.Font.brand(size: 44))
                                    .foregroundStyle(
                                        LinearGradient(
                                            colors: isDraggingOver ? [Theme.Color.purple, Theme.Color.indigo] : [.secondary.opacity(0.6), .secondary.opacity(0.3)],
                                            startPoint: .topLeading,
                                            endPoint: .bottomTrailing
                                        )
                                    )
                                    .scaleEffect(isDraggingOver ? 1.12 : 1.0)
                                    .rotationEffect(.degrees(isDraggingOver ? 360 : 0))
                                    .animation(.spring(response: 0.4, dampingFraction: 0.6), value: isDraggingOver)
                                
                                VStack(spacing: 6) {
                                    Text("Drag & Drop CSV File Here")
                                        .font(.system(.headline, design: .rounded))
                                        .foregroundColor(.primary)
                                    
                                    Text("Supports local CSV files up to 100MB")
                                        .font(.subheadline)
                                        .foregroundColor(.secondary)
                                }
                                
                                HStack(spacing: 12) {
                                    Button(action: onSelectFileManually) {
                                        HStack(spacing: 6) {
                                            Image(systemName: "folder.fill")
                                            Text("Browse Files...")
                                        }
                                        .fontWeight(.semibold)
                                        .foregroundColor(.white)
                                        .padding(.horizontal, 16)
                                        .padding(.vertical, 10)
                                        .background(
                                            LinearGradient(colors: [.purple, .indigo], startPoint: .leading, endPoint: .trailing)
                                        )
                                        .cornerRadius(10)
                                    }
                                    .buttonStyle(.plain)
                                    .shadow(color: Color.purple.opacity(0.25), radius: 6, x: 0, y: 3)
                                    
                                    Button(action: onImportFromDatabase) {
                                        HStack(spacing: 6) {
                                            Image(systemName: "server.rack")
                                            Text("Import from DB...")
                                        }
                                        .fontWeight(.semibold)
                                        .foregroundColor(.white)
                                        .padding(.horizontal, 16)
                                        .padding(.vertical, 10)
                                        .background(
                                            LinearGradient(colors: [.indigo, .blue], startPoint: .leading, endPoint: .trailing)
                                        )
                                        .cornerRadius(10)
                                    }
                                    .buttonStyle(.plain)
                                    .accessibilityIdentifier("importFromDbButton")
                                    .shadow(color: Color.blue.opacity(0.25), radius: 6, x: 0, y: 3)
                                }
                            }
                            .padding(40)
                        }
                        .frame(maxWidth: 540, minHeight: 260)
                        .dropDestination(for: URL.self) { items, location in
                            guard !items.isEmpty else { return false }
                            let validItems = items.filter { url in
                                let ext = url.pathExtension.lowercased()
                                var isDir: ObjCBool = false
                                let exists = FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir)
                                return exists && (isDir.boolValue || ["csv", "tsv", "parquet", "xlsx", "xls", "npz"].contains(ext))
                            }
                            guard !validItems.isEmpty else { return false }
                            onFileDropped(validItems)
                            return true
                        } isTargeted: { targeted in
                            withAnimation(.easeInOut(duration: 0.2)) {
                                isDraggingOver = targeted
                            }
                        }
                    }
                    
                    // 3. Web URL Section
                    VStack(spacing: 14) {
                        HStack(spacing: 6) {
                            Text("— OR CONNECT DATASET VIA URL —")
                                .font(.system(size: 10, weight: .bold, design: .rounded))
                                .foregroundColor(.secondary)
                                .tracking(1.5)
                        }
                        
                        HStack(spacing: 12) {
                            HStack(spacing: 8) {
                                // Dynamic provider icon based on input URL
                                Image(systemName: getURLIconName(urlInput))
                                    .foregroundColor(getURLIconColor(urlInput))
                                    .font(.system(size: 16))
                                    .frame(width: 24, height: 24)
                                    .background(Color.primary.opacity(0.03))
                                    .cornerRadius(6)
                                
                                TextField("Kaggle, Hugging Face, or direct CSV URL...", text: $urlInput)
                                    .textFieldStyle(.plain)
                                    .font(.subheadline)
                                    .onSubmit {
                                        let cleaned = urlInput.trimmingCharacters(in: .whitespacesAndNewlines)
                                        if !cleaned.isEmpty { onURLSubmitted(cleaned) }
                                    }
                            }
                            .padding(.horizontal, 12)
                            .padding(.vertical, 8)
                            .background(Color(nsColor: .controlBackgroundColor))
                            .cornerRadius(10)
                            .overlay(
                                RoundedRectangle(cornerRadius: 10)
                                    .stroke(isUrlHovered ? Color.purple.opacity(0.4) : Color.primary.opacity(0.06), lineWidth: 1)
                            )
                            .onHover { hovering in
                                withAnimation(.easeOut(duration: 0.15)) {
                                    isUrlHovered = hovering
                                }
                            }
                            .frame(maxWidth: 420)
                            
                            Button(action: {
                                let cleaned = urlInput.trimmingCharacters(in: .whitespacesAndNewlines)
                                onURLSubmitted(cleaned)
                            }) {
                                Text("Connect")
                                    .fontWeight(.bold)
                                    .foregroundColor(.white)
                                    .padding(.horizontal, 18)
                                    .padding(.vertical, 8)
                                    .background(
                                        urlInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                                        ? AnyShapeStyle(Color.gray.opacity(0.2))
                                        : AnyShapeStyle(LinearGradient(colors: [.indigo, .purple], startPoint: .leading, endPoint: .trailing))
                                    )
                                    .cornerRadius(10)
                                    .shadow(color: urlInput.isEmpty ? .clear : .purple.opacity(0.2), radius: 4, x: 0, y: 2)
                            }
                            .buttonStyle(.plain)
                            .disabled(urlInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                        }
                        .frame(maxWidth: 540)
                    }
                    
                    // 3.5 Recent Analyses Section (Horizontal Carousel)
                    if !recentAnalyses.isEmpty {
                        VStack(spacing: 14) {
                            Text("RECENT ANALYSES")
                                .font(.system(size: 10, weight: .bold, design: .rounded))
                                .foregroundColor(.secondary)
                                .tracking(1.5)
                            
                            ScrollView(.horizontal, showsIndicators: false) {
                                HStack(spacing: 14) {
                                    ForEach(recentAnalyses.prefix(4)) { item in
                                        recentAnalysisCard(item: item)
                                    }
                                }
                                .padding(.horizontal, 4)
                                .padding(.vertical, 6)
                            }
                            .frame(maxWidth: 540)
                        }
                    }
                    
                    // 4. Sample Datasets Section
                    VStack(spacing: 16) {
                        Text("QUICK START SAMPLES")
                            .font(.system(size: 10, weight: .bold, design: .rounded))
                            .foregroundColor(.secondary)
                            .tracking(1.5)
                        
                        LazyVGrid(columns: [GridItem(.adaptive(minimum: 250), spacing: 16)], spacing: 14) {
                            sampleCard(
                                name: "House Prices",
                                filename: "house_prices.csv",
                                icon: "house.fill",
                                color: .purple,
                                desc: "Regression target, missing values, numerical columns."
                            )
                            
                            sampleCard(
                                name: "Iris Flowers",
                                filename: "iris.csv",
                                icon: "leaf.fill",
                                color: .green,
                                desc: "Classification target, 3 species, balanced classes."
                            )
                            
                            sampleCard(
                                name: "Airline Passengers",
                                filename: "airline_passengers.csv",
                                icon: "chart.line.uptrend.xyaxis",
                                color: .blue,
                                desc: "Time Series forecasting, trend, seasonality, lag context."
                            )
                            
                            sampleCard(
                                name: "Movie Reviews",
                                filename: "movie_reviews.csv",
                                icon: "text.bubble",
                                color: .green,
                                desc: "Text / NLP target, TF-IDF weights, sentiment classification."
                            )
                            
                            sampleCard(
                                name: "MNIST Mini",
                                filename: "mnist_mini.npz",
                                icon: "photo.stack",
                                color: .orange,
                                desc: "Image dataset, pixel grids, classification accuracy."
                            )
                            
                            sampleCard(
                                name: "Drone Detection",
                                filename: "drone_dataset",
                                icon: "viewfinder.rectangular",
                                color: .indigo,
                                desc: "YOLO dataset, object detection, bounding-box crops classification."
                            )
                        }
                        .frame(maxWidth: 1350)
                    }
                    .padding(.bottom, 32)
                }
                .padding(.horizontal, 24)
            }
        }
    }
    
    // MARK: - Subviews & Helpers
    
    @ViewBuilder
    private func sampleCard(name: String, filename: String, icon: String, color: Color, desc: String) -> some View {
        Button(action: { onSampleSelected(filename) }) {
            HStack(alignment: .top, spacing: 14) {
                ZStack {
                    RoundedRectangle(cornerRadius: 10)
                        .fill(color.opacity(0.12))
                        .frame(width: 42, height: 42)
                    
                    Image(systemName: icon)
                        .font(.system(size: 20))
                        .foregroundColor(color)
                }
                
                VStack(alignment: .leading, spacing: 4) {
                    Text(name)
                        .font(.system(.body, design: .rounded))
                        .fontWeight(.semibold)
                        .foregroundColor(.primary)
                    
                    Text(desc)
                        .font(.caption2)
                        .foregroundColor(.secondary)
                        .lineLimit(2)
                        .multilineTextAlignment(.leading)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer()
                
                Image(systemName: "chevron.right")
                    .font(.caption)
                    .foregroundColor(.secondary.opacity(0.5))
                    .padding(.top, 4)
            }
            .padding(14)
            .frame(width: 250, height: 74)
            .background(Color.primary.opacity(0.02))
            .cornerRadius(12)
            .overlay(
                RoundedRectangle(cornerRadius: 12)
                    .stroke(Color.primary.opacity(0.05), lineWidth: 1)
            )
        }
        .buttonStyle(SampleCardButtonStyle())
    }
    
    private func getURLIconName(_ url: String) -> String {
        let lower = url.lowercased()
        if lower.contains("kaggle.com") { return "k.circle.fill" } // Note: using k.circle.fill as placeholder or cloud
        if lower.contains("huggingface.co") { return "face.smiling.fill" }
        if lower.isEmpty { return "link" }
        return "link.circle.fill"
    }
    
    private func getURLIconColor(_ url: String) -> Color {
        let lower = url.lowercased()
        if lower.contains("kaggle.com") { return .blue }
        if lower.contains("huggingface.co") { return .yellow }
        if lower.isEmpty { return .secondary }
        return .purple
    }
    
    private func getTaskLabel(_ task: String) -> String {
        switch task.lowercased() {
        case "regression": return "REGRESSION"
        case "classification": return "CLASSIFICATION"
        case "time_series": return "TIME SERIES"
        case "nlp": return "TEXT / NLP"
        case "image": return "IMAGE"
        default: return task.uppercased()
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

    @ViewBuilder
    private func recentAnalysisCard(item: HistoryItem) -> some View {
        Button(action: { onRecentSelected(item) }) {
            VStack(alignment: .leading, spacing: 8) {
                HStack(alignment: .center) {
                    if let task = item.taskType {
                        Text(getTaskLabel(task))
                            .font(.system(size: 8, weight: .bold))
                            .foregroundColor(.white)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(getTaskColor(task).opacity(0.8))
                            .cornerRadius(4)
                    } else {
                        Image(systemName: "clock.arrow.circlepath")
                            .foregroundColor(.purple)
                            .font(Theme.Font.caption)
                    }
                    Spacer()
                    Text(item.timestamp, style: .date)
                        .font(.system(size: 8))
                        .foregroundColor(.secondary)
                }
                
                VStack(alignment: .leading, spacing: 2) {
                    Text(item.datasetName)
                        .font(.system(.caption, design: .rounded))
                        .fontWeight(.semibold)
                        .foregroundColor(.primary)
                        .lineLimit(1)
                    
                    if let rows = item.rowCount, let cols = item.colCount {
                        Text("\(rows) rows • \(cols) cols")
                            .font(.system(size: 9))
                            .foregroundColor(.secondary)
                    } else {
                        Text(item.targetColumn.map { "Target: \($0)" } ?? "Auto-detect target")
                            .font(.system(size: 9))
                            .foregroundColor(.secondary)
                            .lineLimit(1)
                    }
                }
                
                if let bestModel = item.bestModel, let bestScore = item.bestScore {
                    let scoreName = item.scoreType ?? "Score"
                    let scoreStr = String(format: "%.3f", bestScore)
                    HStack(spacing: 3) {
                        Image(systemName: "trophy.fill")
                            .font(.system(size: 8))
                            .foregroundColor(.yellow)
                        Text("\(bestModel) (\(scoreName): \(scoreStr))")
                            .font(.system(size: 8, weight: .medium))
                            .foregroundColor(.secondary)
                            .lineLimit(1)
                    }
                    .padding(.top, 2)
                }
            }
            .padding(12)
            .frame(width: 220, height: 110)
            .background(Color.primary.opacity(0.02))
            .cornerRadius(12)
            .overlay(
                RoundedRectangle(cornerRadius: 12)
                    .stroke(Color.primary.opacity(0.05), lineWidth: 1)
            )
        }
        .buttonStyle(SampleCardButtonStyle())
        .contextMenu {
            Button {
                onRename(item)
            } label: {
                Label("Rename Analysis...", systemImage: "pencil")
            }
            
            Button(role: .destructive) {
                onDelete(item)
            } label: {
                Label("Delete Analysis", systemImage: "trash")
            }
        }
    }
}

struct SampleCardButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.98 : 1.0)
            .background(configuration.isPressed ? Color.primary.opacity(0.04) : Color.clear)
            .animation(.easeOut(duration: 0.1), value: configuration.isPressed)
    }
}

#Preview {
    DragDropView(
        onFileDropped: { _ in },
        onSelectFileManually: {},
        onImportFromDatabase: {},
        onURLSubmitted: { _ in },
        onSampleSelected: { _ in },
        recentAnalyses: [
            HistoryItem(
                id: UUID(),
                datasetName: "dummy.csv",
                datasetPath: "/path/dummy.csv",
                targetColumn: "target",
                timestamp: Date(),
                resultFileName: "dummy.json",
                taskType: "classification",
                bestModel: "XGBoost",
                bestScore: 0.942,
                scoreType: "Accuracy",
                rowCount: 1200,
                colCount: 15
            )
        ],
        onRecentSelected: { _ in },
        onRename: { _ in },
        onDelete: { _ in }
    )
    .frame(width: 650, height: 600)
}
