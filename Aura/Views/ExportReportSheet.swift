import SwiftUI
import Charts

// MARK: - Export Report Sheet

struct ExportReportSheet: View {
    let result: AnalysisResult
    @Binding var isPresented: Bool

    @State private var includeCharts: Bool = true
    @State private var includeAINarrative: Bool = true
    @State private var includeStats: Bool = true
    @State private var includeTable: Bool = false
    @State private var isGenerating: Bool = false
    @State private var generationStatus: String = ""
    @State private var errorMessage: String? = nil

    private let ollamaStatus = OllamaStatusChecker.shared

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Export Analysis Report")
                        .font(.title3.bold())
                    Text("Generate a Markdown report from your analysis results")
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

                    // ── Include Options ───────────────────────────────────────
                    VStack(alignment: .leading, spacing: 10) {
                        sectionHeader("Include in Report", icon: "checklist")

                        ToggleRow(label: "AI Narrative",
                                  icon: "sparkles",
                                  color: .purple,
                                  description: "AI-generated executive summary and key insights",
                                  isOn: $includeAINarrative)
                            .opacity(ollamaStatus.isAvailable ? 1.0 : 0.4)
                            .disabled(!ollamaStatus.isAvailable)

                        if !ollamaStatus.isAvailable {
                            Label("Ollama is offline — AI narrative will be skipped", systemImage: "exclamationmark.triangle")
                                .font(.caption)
                                .foregroundColor(.orange)
                        }

                        ToggleRow(label: "Statistics Table",
                                  icon: "chart.bar.doc.horizontal",
                                  color: .blue,
                                  description: "Dataset overview, column types, missing values, model metrics",
                                  isOn: $includeStats)

                        ToggleRow(label: "Charts",
                                  icon: "chart.line.uptrend.xyaxis",
                                  color: .indigo,
                                  description: "Chart data embedded as data tables in Markdown",
                                  isOn: $includeCharts)

                        ToggleRow(label: "Full Data Sample",
                                  icon: "tablecells",
                                  color: .cyan,
                                  description: "First 50 rows of the dataset as a Markdown table",
                                  isOn: $includeTable)
                    }

                    Divider().background(Color.white.opacity(0.07))

                    // ── Dataset Summary Preview ───────────────────────────────
                    VStack(alignment: .leading, spacing: 8) {
                        sectionHeader("What Will Be Exported", icon: "doc.text.magnifyingglass")

                        HStack(spacing: 16) {
                            exportStat(value: "\(result.rowCount)", label: "Rows")
                            exportStat(value: "\(result.colCount)", label: "Columns")
                            exportStat(value: result.taskType.capitalized, label: "Task")
                            exportStat(value: String(format: "%.3f", result.metrics.score), label: result.metrics.scoreType)
                        }
                        .padding(14)
                        .background(Color.white.opacity(0.03))
                        .cornerRadius(10)
                        .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color.white.opacity(0.06), lineWidth: 1))
                    }

                    // ── Error ─────────────────────────────────────────────────
                    if let err = errorMessage {
                        Label(err, systemImage: "exclamationmark.triangle.fill")
                            .font(.caption)
                            .foregroundColor(.red)
                            .padding(10)
                            .background(Color.red.opacity(0.07))
                            .cornerRadius(8)
                    }

                    // ── Generation Status ─────────────────────────────────────
                    if isGenerating {
                        HStack(spacing: 8) {
                            ProgressView().controlSize(.small)
                            Text(generationStatus.isEmpty ? "Generating report…" : generationStatus)
                                .font(.system(size: 12))
                                .foregroundColor(.secondary)
                        }
                        .padding(12)
                        .background(Color.white.opacity(0.03))
                        .cornerRadius(8)
                    }
                }
                .padding(20)
            }

            Divider().background(Color.white.opacity(0.07))

            // ── Action Buttons ────────────────────────────────────────────────
            HStack(spacing: 12) {
                Button("Cancel") { isPresented = false }
                    .buttonStyle(.bordered)
                    .disabled(isGenerating)

                Spacer()

                Button {
                    Task { await generateAndExport() }
                } label: {
                    HStack(spacing: 6) {
                        if isGenerating {
                            ProgressView().controlSize(.small)
                        } else {
                            Image(systemName: "square.and.arrow.up")
                        }
                        Text(isGenerating ? "Generating…" : "Generate & Export")
                            .fontWeight(.semibold)
                    }
                    .foregroundColor(.white)
                    .padding(.horizontal, 18)
                    .padding(.vertical, 9)
                    .background(
                        isGenerating
                            ? AnyShapeStyle(Color.white.opacity(0.08))
                            : AnyShapeStyle(LinearGradient(colors: [.purple, .indigo], startPoint: .leading, endPoint: .trailing))
                    )
                    .cornerRadius(9)
                }
                .buttonStyle(.plain)
                .disabled(isGenerating)
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 14)
        }
        .frame(width: 480, height: 560)
        .background(Color(white: 0.08))
        .cornerRadius(16)
        .colorScheme(.dark)
    }

    // MARK: - Subviews

    private func sectionHeader(_ title: String, icon: String) -> some View {
        Label(title, systemImage: icon)
            .font(.system(size: 11, weight: .bold, design: .rounded))
            .foregroundColor(.secondary.opacity(0.7))
            .tracking(0.3)
    }

    private func exportStat(value: String, label: String) -> some View {
        VStack(spacing: 3) {
            Text(value)
                .font(.system(size: 15, weight: .bold, design: .rounded))
                .foregroundColor(.primary)
                .lineLimit(1)
            Text(label)
                .font(.system(size: 9))
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
    }

    // MARK: - Export Logic

    @MainActor
    private func generateAndExport() async {
        isGenerating = true
        errorMessage = nil

        generationStatus = "Building report structure…"
        var md = buildMarkdownReport()

        // AI Narrative
        if includeAINarrative && ollamaStatus.isAvailable {
            generationStatus = "Asking AI to generate narrative…"
            if let narrative = await generateAINarrative() {
                md = narrative + "\n\n---\n\n" + md
            }
        }

        generationStatus = "Opening save dialog…"
        await saveMarkdownFile(content: md)
        isGenerating = false
    }

    private func buildMarkdownReport() -> String {
        var md = "# Analysis Report: \(result.targetColumn)\n\n"
        md += "_Generated by Aura — \(Date().formatted(date: .long, time: .shortened))_\n\n"
        md += "---\n\n"

        if includeStats {
            md += "## 📊 Dataset Overview\n\n"
            md += "| Metric | Value |\n|---|---|\n"
            md += "| Rows | \(result.rowCount) |\n"
            md += "| Columns | \(result.colCount) |\n"
            md += "| Task Type | \(result.taskType.capitalized) |\n"
            md += "| Target Column | `\(result.targetColumn)` |\n"
            md += "| Numeric Columns | \(result.numericColCount) |\n"
            md += "| Categorical Columns | \(result.categoricalColCount) |\n\n"

            md += "### 🤖 Model Performance\n\n"
            md += "| Model | \(result.metrics.scoreType) |\n|---|---|\n"
            for model in result.modelsCompared {
                md += "| \(model.name) | \(String(format: "%.4f", model.score)) |\n"
            }
            md += "\n"

            let missingCols = result.missingValues.filter { $0.value > 0 }
            if !missingCols.isEmpty {
                md += "### ⚠️ Missing Values\n\n"
                md += "| Column | Missing Count |\n|---|---|\n"
                for (col, count) in missingCols.sorted(by: { $0.value > $1.value }) {
                    md += "| `\(col)` | \(count) |\n"
                }
                md += "\n"
            }
        }

        if includeCharts && !result.charts.isEmpty {
            md += "## 📈 Charts\n\n"
            for chart in result.charts {
                md += "### \(chart.title)\n\n"
                md += "_\(chart.xLabel) → \(chart.yLabel)_\n\n"
                md += "| \(chart.xLabel) | \(chart.yLabel) |\n|---|---|\n"
                for point in chart.data.prefix(20) {
                    let x = point.xVal ?? (point.xNum.map { String(format: "%.4f", $0) } ?? "—")
                    md += "| \(x) | \(String(format: "%.6f", point.y)) |\n"
                }
                if chart.data.count > 20 {
                    md += "\n_… \(chart.data.count - 20) more points not shown_\n"
                }
                md += "\n"
            }
        }

        if includeTable, let fp = result.fullPreview {
            md += "## 🗃️ Data Sample (first 50 rows)\n\n"
            md += "| " + fp.columns.joined(separator: " | ") + " |\n"
            md += "|" + fp.columns.map { _ in "---" }.joined(separator: "|") + "|\n"
            for row in fp.rows.prefix(50) {
                md += "| " + row.joined(separator: " | ") + " |\n"
            }
            md += "\n"
        }

        if !result.correlations.isEmpty {
            md += "## 🔗 Top Correlations\n\n"
            md += "| Feature A | Feature B | Correlation |\n|---|---|---|\n"
            for corr in result.correlations.prefix(10) {
                md += "| `\(corr.x)` | `\(corr.y)` | \(String(format: "%.3f", corr.value)) |\n"
            }
            md += "\n"
        }

        return md
    }

    private func generateAINarrative() async -> String? {
        let model = UserDefaults.standard.string(forKey: "Aura_OllamaModel")
            ?? ollamaStatus.availableModels.first?.name
            ?? "llama3.2"

        var extraContext = ""
        
        // Include Leakage Warnings
        if let warnings = result.dataLeakageWarnings, !warnings.isEmpty {
            extraContext += "- Data Leakage Warnings: \(warnings.joined(separator: "; "))\n"
        }
        
        // Include Cleaning Recommendations
        if let recs = result.cleaningRecommendations, !recs.isEmpty {
            let recsStr = recs.map { "\($0.column) (\($0.issue) -> \($0.recommendation) [Impact: \($0.impact)])" }.joined(separator: "; ")
            extraContext += "- Data Quality Issues & Recommendations: \(recsStr)\n"
        }
        
        // Include Models Compared
        let modelsStr = result.modelsCompared.map { "\($0.name) (\($0.metric): \(String(format: "%.4f", $0.score)))" }.joined(separator: ", ")
        extraContext += "- Models Compared: \(modelsStr)\n"
        
        // Include Feature Importances (from charts if available)
        if let importanceChart = result.charts.first(where: { $0.title.lowercased().contains("importance") }) {
            let topFeats = importanceChart.data.prefix(5).compactMap { pt in
                pt.xVal.map { "\($0): \(String(format: "%.4f", pt.y))" }
            }.joined(separator: ", ")
            extraContext += "- Top Feature Importances: \(topFeats)\n"
        }

        let prompt = """
        You are a Senior Data Scientist. Write a professional, highly polished data analysis review in Markdown based on the following metrics:
        - Dataset size: \(result.rowCount) rows × \(result.colCount) columns
        - Target variable: '\(result.targetColumn)' (\(result.taskType.capitalized) task)
        - Best Model: \(result.metrics.model) (\(result.metrics.scoreType): \(String(format: "%.4f", result.metrics.score)))
        \(extraContext)
        - Top correlations: \(result.correlations.prefix(5).map { "\($0.x) ↔ \($0.y) = \(String(format: "%.3f", $0.value))" }.joined(separator: ", "))
        - Missing values: \(result.missingValues.filter { $0.value > 0 }.count) columns with missing values.

        Ensure the response strictly follows this Markdown structure:
        
        ## 🧠 AI Analysis Summary
        Write 2-3 detailed paragraphs. Discuss the dataset size and task. Assess the model performance (R² / Accuracy / F1) critically: if the score is low (e.g. R² close to 0 or negative), explain that the features have very weak predictive power and cannot explain the variance in the target. Discuss whether any data quality or data leakage warnings were flagged.
        
        ## 💡 Key Findings
        Provide 3-5 specific, bulleted insights. Each bullet MUST reference exact numbers, features, correlations, or performance metrics from the data provided above.
        
        ## 📋 Recommendations
        Provide 2-4 concrete, actionable next steps. Address data quality issues, recommended column actions, feature engineering suggestions, or model improvements (e.g. testing non-linear algorithms or getting additional features if the R² is weak).
        
        Write in a professional, authoritative tone. Ground all comments in the provided data. Do not use placeholders or repeat sentences.
        """

        return await withCheckedContinuation { continuation in
            guard let url = URL(string: "http://localhost:11434/api/generate") else {
                continuation.resume(returning: nil)
                return
            }
            var request = URLRequest(url: url)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.timeoutInterval = 120

            let body: [String: Any] = [
                "model": model,
                "prompt": prompt,
                "stream": false,
                "options": ["temperature": 0.4, "num_predict": 1200]
            ]
            request.httpBody = try? JSONSerialization.data(withJSONObject: body)

            URLSession.shared.dataTask(with: request) { data, _, _ in
                if let data = data,
                   let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let text = json["response"] as? String {
                    continuation.resume(returning: text)
                } else {
                    continuation.resume(returning: nil)
                }
            }.resume()
        }
    }

    @MainActor
    private func saveMarkdownFile(content: String) async {
        let panel = NSSavePanel()
        panel.title = "Export Analysis Report"
        panel.allowedContentTypes = [.init(filenameExtension: "md")!]
        panel.nameFieldStringValue = "Aura_Report_\(result.targetColumn).md"
        panel.canCreateDirectories = true

        if panel.runModal() == .OK, let url = panel.url {
            do {
                try content.write(to: url, atomically: true, encoding: .utf8)
            } catch {
                errorMessage = "Failed to save: \(error.localizedDescription)"
            }
        }
        isPresented = false
    }
}

// MARK: - Toggle Row

private struct ToggleRow: View {
    let label: String
    let icon: String
    let color: Color
    let description: String
    @Binding var isOn: Bool

    var body: some View {
        HStack(spacing: 12) {
            ZStack {
                RoundedRectangle(cornerRadius: 7)
                    .fill(color.opacity(0.12))
                    .frame(width: 30, height: 30)
                Image(systemName: icon)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundColor(color)
            }

            VStack(alignment: .leading, spacing: 1) {
                Text(label)
                    .font(.system(size: 12, weight: .semibold))
                Text(description)
                    .font(.system(size: 10))
                    .foregroundColor(.secondary)
            }

            Spacer()

            Toggle("", isOn: $isOn)
                .toggleStyle(.switch)
                .labelsHidden()
                .tint(color)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(Color.white.opacity(isOn ? 0.04 : 0.02))
        .cornerRadius(9)
        .overlay(RoundedRectangle(cornerRadius: 9).stroke(isOn ? color.opacity(0.2) : Color.white.opacity(0.04), lineWidth: 1))
    }
}
