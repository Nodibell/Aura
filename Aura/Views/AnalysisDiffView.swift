import SwiftUI

struct AnalysisDiffView: View {
    let currentResult: AnalysisResult
    let currentHistoryItemId: UUID?
    
    @State private var selectedCompareItem: HistoryItem? = nil
    @State private var compareResult: AnalysisResult? = nil
    @State private var isLoadingCompare: Bool = false
    
    var body: some View {
        VStack(spacing: 0) {
            let currentPath = AnalysisHistoryService.shared.items.first(where: { $0.id == currentHistoryItemId })?.datasetPath ?? ""
            let comparableItems = AnalysisHistoryService.shared.items.filter { item in
                item.datasetPath == currentPath && item.id != currentHistoryItemId
            }

            
            if comparableItems.isEmpty {
                VStack(spacing: 12) {
                    Image(systemName: "square.split.2x1")
                        .font(.system(size: 40))
                        .foregroundColor(.secondary)
                    Text("No previous runs found for this dataset.")
                        .font(.headline)
                        .foregroundColor(.secondary)
                    Text("Run another analysis on this dataset with different cleaning or feature configurations to compare them side-by-side.")
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                        .frame(maxWidth: 400)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                // Header picker for run comparison
                HStack(spacing: 12) {
                    Text("Compare with previous run:")
                        .font(.subheadline)
                        .fontWeight(.bold)
                    
                    Picker("", selection: $selectedCompareItem) {
                        Text("Select a run...").tag(nil as HistoryItem?)
                        ForEach(comparableItems) { item in
                            Text(formatDate(item.timestamp) + " (\(item.bestModel ?? "Unknown") - \(item.bestScore.map { String(format: "%.4f", $0) } ?? ""))")
                                .tag(item as HistoryItem?)
                        }
                    }
                    .pickerStyle(.menu)
                    .frame(width: 350)
                    
                    Spacer()
                }
                .padding()
                .background(Color.primary.opacity(0.015))
                
                Divider()
                
                if isLoadingCompare {
                    VStack {
                        NativeProgressView(controlSize: .regular)
                        Text("Loading run details...")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else if let comp = compareResult {
                    ScrollView {
                        VStack(spacing: 20) {
                            // Top Side-by-Side Summary & Delta Badge
                            HStack(spacing: 20) {
                                // Previous Run Summary Card
                                RunSummaryCard(
                                    title: "Previous Run",
                                    model: comp.metrics.model,
                                    score: comp.metrics.score,
                                    scoreType: comp.metrics.scoreType,
                                    cvMean: comp.cvMean,
                                    rowCount: comp.rowCount,
                                    colCount: comp.colCount,
                                    color: .blue
                                )
                                
                                // Prominent Delta Badge
                                VStack(spacing: 6) {
                                    let delta = currentResult.metrics.score - comp.metrics.score
                                    let metricName = currentResult.metrics.scoreType
                                    
                                    Text("Performance Δ")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                        .fontWeight(.bold)
                                    
                                    Text(String(format: "%+.4f", delta))
                                        .font(.system(size: 28, weight: .bold, design: .rounded))
                                        .foregroundColor(delta >= 0 ? .green : .red)
                                        .padding(.horizontal, 16)
                                        .padding(.vertical, 8)
                                        .background((delta >= 0 ? Color.green : Color.red).opacity(0.12))
                                        .cornerRadius(12)
                                        .overlay(
                                            RoundedRectangle(cornerRadius: 12)
                                                .stroke((delta >= 0 ? Color.green : Color.red).opacity(0.3), lineWidth: 1.5)
                                        )
                                    
                                    Text("in \(metricName)")
                                        .font(.system(size: 10))
                                        .foregroundColor(.secondary)
                                }
                                .frame(width: 150)
                                
                                // Current Run Summary Card
                                RunSummaryCard(
                                    title: "Current Run",
                                    model: currentResult.metrics.model,
                                    score: currentResult.metrics.score,
                                    scoreType: currentResult.metrics.scoreType,
                                    cvMean: currentResult.cvMean,
                                    rowCount: currentResult.rowCount,
                                    colCount: currentResult.colCount,
                                    color: .purple
                                )
                            }
                            
                            Divider()
                            
                            // Side-by-side Top Feature Importances Comparison
                            HStack(alignment: .top, spacing: 20) {
                                VStack(alignment: .leading, spacing: 8) {
                                    Text("Previous Feature Importances")
                                        .font(.subheadline)
                                        .fontWeight(.bold)
                                        .foregroundColor(.secondary)
                                    
                                    FeatureImportancesList(result: comp)
                                }
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding(12)
                                .background(Color.primary.opacity(0.02))
                                .cornerRadius(10)
                                
                                VStack(alignment: .leading, spacing: 8) {
                                    Text("Current Feature Importances")
                                        .font(.subheadline)
                                        .fontWeight(.bold)
                                        .foregroundColor(.purple)
                                    
                                    FeatureImportancesList(result: currentResult)
                                }
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding(12)
                                .background(Color.primary.opacity(0.02))
                                .cornerRadius(10)
                            }
                            
                        }
                        .padding()
                    }
                } else {
                    VStack {
                        Image(systemName: "square.split.2x1")
                            .font(.largeTitle)
                            .foregroundColor(.secondary)
                            .padding(.bottom, 6)
                        Text("Select a run from the dropdown to compare side-by-side.")
                            .font(.headline)
                            .foregroundColor(.secondary)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
        }
        .onChange(of: selectedCompareItem) { oldValue, newValue in
            if let item = newValue {
                loadCompareResult(item)
            } else {
                compareResult = nil
            }
        }
    }
    
    private func loadCompareResult(_ item: HistoryItem) {
        isLoadingCompare = true
        Task {
            let res = await AnalysisHistoryService.shared.loadAnalysisResult(item: item)
            await MainActor.run {
                self.compareResult = res
                self.isLoadingCompare = false
            }
        }
    }
    
    private func formatDate(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateStyle = .short
        formatter.timeStyle = .short
        return formatter.string(from: date)
    }
}

struct RunSummaryCard: View {
    let title: String
    let model: String
    let score: Double
    let scoreType: String
    let cvMean: Double?
    let rowCount: Int
    let colCount: Int
    let color: Color
    
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(title)
                .font(.system(size: 11, weight: .bold))
                .foregroundColor(color)
            
            Text(model)
                .font(.headline)
                .fontWeight(.bold)
                .lineLimit(1)
            
            HStack(spacing: 20) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(String(format: "%.4f", score))
                        .font(.title3)
                        .fontWeight(.bold)
                    Text(scoreType)
                        .font(.system(size: 9))
                        .foregroundColor(.secondary)
                }
                
                if let cv = cvMean {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(String(format: "%.4f", cv))
                            .font(.title3)
                            .fontWeight(.bold)
                            .foregroundColor(.secondary)
                        Text("CV Mean")
                            .font(.system(size: 9))
                            .foregroundColor(.secondary)
                    }
                }
            }
            
            HStack(spacing: 12) {
                Text("\(rowCount) rows")
                Text("•")
                Text("\(colCount) cols")
            }
            .font(.caption)
            .foregroundColor(.secondary)
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.primary.opacity(0.03))
        .cornerRadius(12)
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(Color.primary.opacity(0.06), lineWidth: 1)
        )
    }
}

struct FeatureImportancesList: View {
    let result: AnalysisResult
    
    var body: some View {
        let importanceChart = result.charts.first(where: {
            $0.title.lowercased().contains("importance")
        })
        
        if let chart = importanceChart, !chart.data.isEmpty {
            VStack(spacing: 8) {
                ForEach(chart.data.prefix(8)) { point in
                    HStack {
                        Text(point.xVal ?? "Unknown")
                            .font(.caption)
                            .fontWeight(.medium)
                            .foregroundColor(.primary)
                            .lineLimit(1)
                        
                        Spacer()
                        
                        Text(String(format: "%.3f", point.y))
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundColor(.secondary)
                    }
                    .padding(.vertical, 2)
                }
            }
        } else {
            Text("No feature importances available.")
                .font(.caption)
                .foregroundColor(.secondary)
                .italic()
                .padding(.vertical, 10)
        }
    }
}
