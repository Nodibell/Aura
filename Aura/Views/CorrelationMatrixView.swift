import SwiftUI
import Charts

struct CorrelationMatrixView: View {
    let result: AnalysisResult
    let onAskAI: (String) -> Void

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                if result.correlations.isEmpty {
                    VStack(spacing: 12) {
                        Image(systemName: "chart.dots.scatter")
                            .font(.system(size: 40))
                            .foregroundColor(.secondary)
                        Text("No numeric correlations to display.")
                            .foregroundColor(.secondary)
                    }
                    .frame(maxWidth: .infinity, minHeight: 300)
                } else {
                    // Header + AI button
                    HStack {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Feature Correlations")
                                .font(.title2)
                                .fontWeight(.bold)
                            Text("Pearson correlation coefficient between numeric features")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        Spacer()
                        Button {
                            let top = result.correlations.prefix(5).map {
                                "\($0.x) ↔ \($0.y): \(String(format: "%.3f", $0.value))"
                            }.joined(separator: ", ")
                            onAskAI("Explain the significance of these top correlations: \(top). Which ones matter most for predicting \(result.targetColumn)?")
                        } label: {
                            Label("Ask AI", systemImage: "sparkles")
                                .font(.caption)
                        }
                        .buttonStyle(.bordered)
                        .tint(.purple)
                    }

                    // Heatmap Grid
                    correlationHeatmap

                    // Top correlations list
                    topCorrelationsList
                }
            }
            .padding(20)
        }
    }

    // MARK: - Heatmap

    private var correlationHeatmap: some View {
        let uniqueVars = uniqueVariables()
        let n = uniqueVars.count
        guard n > 0 else { return AnyView(EmptyView()) }

        return AnyView(
            VStack(alignment: .leading, spacing: 12) {
                Text("Correlation Heatmap")
                    .font(.headline)

                // Legend
                HStack(spacing: 8) {
                    Text("-1")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                    LinearGradient(colors: [.red, Color(nsColor: .systemGray).opacity(0.3), .blue], startPoint: .leading, endPoint: .trailing)
                        .frame(height: 6)
                        .cornerRadius(3)
                    Text("+1")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
                .frame(maxWidth: 200)

                ScrollView([.horizontal, .vertical]) {
                    VStack(spacing: 2) {
                        // Column labels row
                        HStack(spacing: 2) {
                            Text("")
                                .frame(width: 80, height: 60)

                            ForEach(uniqueVars, id: \.self) { colName in
                                Text(colName)
                                    .font(.system(size: 9))
                                    .foregroundColor(.secondary)
                                    .lineLimit(1)
                                    .truncationMode(.tail)
                                    .frame(width: 70, alignment: .leading)
                                    .rotationEffect(.degrees(-45), anchor: .bottomLeading)
                                    .frame(width: 44, height: 60, alignment: .bottomLeading)
                                    .offset(x: 10)
                            }
                        }

                        // Data rows
                        ForEach(uniqueVars, id: \.self) { rowVar in
                            HStack(spacing: 2) {
                                Text(rowVar)
                                    .font(.system(size: 9))
                                    .foregroundColor(.secondary)
                                    .lineLimit(1)
                                    .frame(width: 80, alignment: .trailing)

                                ForEach(uniqueVars, id: \.self) { colVar in
                                    let val = correlationValue(row: rowVar, col: colVar)
                                    let cellColor = heatmapColor(for: val)
                                    
                                    ZStack {
                                        Rectangle()
                                            .fill(cellColor)
                                            .frame(width: 44, height: 44)
                                            .cornerRadius(3)
                                        
                                        if rowVar == colVar {
                                            Text("1.0")
                                                .font(.system(size: 8, weight: .bold))
                                                .foregroundColor(.white)
                                        } else if abs(val) > 0.01 {
                                            Text(String(format: "%.2f", val))
                                                .font(.system(size: 8, weight: .medium))
                                                .foregroundColor(abs(val) > 0.5 ? .white : .primary)
                                        } else {
                                            Text("-")
                                                .font(.system(size: 8, weight: .medium))
                                                .foregroundColor(.secondary.opacity(0.5))
                                        }
                                    }
                                }
                            }
                        }
                    }
                    .padding(8)
                }
                .background(Color.primary.opacity(0.02))
                .cornerRadius(12)
                .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color.primary.opacity(0.07)))
            }
        )
    }

    // MARK: - Top Correlations List

    private var topCorrelationsList: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Top Correlations (by |r|)")
                .font(.headline)

            VStack(spacing: 8) {
                // Increased from 10 to 30 to show a much longer list of top pairs
                ForEach(result.correlations.prefix(30)) { pair in
                    HStack(spacing: 12) {
                        // Color dot
                        Circle()
                            .fill(pair.value > 0 ? Color.blue : Color.red)
                            .frame(width: 10, height: 10)

                        Text("\(pair.x)  ↔  \(pair.y)")
                            .font(.subheadline)

                        Spacer()

                        // Mini bar
                        let absVal = min(abs(pair.value), 1.0)
                        ZStack(alignment: pair.value > 0 ? .leading : .trailing) {
                            RoundedRectangle(cornerRadius: 3).fill(Color.primary.opacity(0.06)).frame(width: 80, height: 6)
                            RoundedRectangle(cornerRadius: 3)
                                .fill(pair.value > 0 ? Color.blue.gradient : Color.red.gradient)
                                .frame(width: max(4, 80 * absVal), height: 6)
                        }

                        Text(String(format: "%+.3f", pair.value))
                            .font(.system(.body, design: .monospaced))
                            .fontWeight(.semibold)
                            .foregroundColor(pair.value > 0 ? .blue : .red)
                            .frame(width: 60, alignment: .trailing)
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(Color.primary.opacity(0.02))
                    .cornerRadius(8)
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(Color.primary.opacity(0.04), lineWidth: 1)
                    )
                }

            }
        }
    }

    // MARK: - Helpers

    /// O(1) lookup table built once from result.correlations.
    /// Key format: "a|b" where a <= b lexicographically to guarantee a single canonical key.
    private var correlationLookup: [String: Double] {
        var table: [String: Double] = [:]
        table.reserveCapacity(result.correlations.count)
        for pair in result.correlations {
            let key = pair.x <= pair.y ? "\(pair.x)|\(pair.y)" : "\(pair.y)|\(pair.x)"
            table[key] = pair.value
        }
        return table
    }

    private func uniqueVariables() -> [String] {
        var seen = Set<String>()
        var uniqueVars: [String] = []
        
        // Iterate through all correlations to extract every unique feature name
        for pair in result.correlations {
            if !seen.contains(pair.x) {
                seen.insert(pair.x)
                uniqueVars.append(pair.x)
            }
            if !seen.contains(pair.y) {
                seen.insert(pair.y)
                uniqueVars.append(pair.y)
            }
        }
        
        // Cap at 40 just to prevent SwiftUI from crashing on datasets with hundreds of columns
        return Array(uniqueVars.prefix(40))
    }

    private func correlationValue(row: String, col: String) -> Double {
        if row == col { return 1.0 }
        let key = row <= col ? "\(row)|\(col)" : "\(col)|\(row)"
        return correlationLookup[key] ?? 0.0
    }

    private func heatmapColor(for value: Double) -> Color {
        if value == 1.0 { return Color.blue.opacity(0.8) }
        let absVal = Swift.min(Swift.abs(value), 1.0)
        
        if absVal < 0.01 { return Color.primary.opacity(0.04) }

        if value > 0 {
            return Color.blue.opacity(absVal * 0.8 + 0.1)
        } else {
            return Color.red.opacity(absVal * 0.8 + 0.1)
        }
    }
}
