import SwiftUI

struct ChartCard: View {
    let config: ChartConfig
    let onAskAI: (String) -> Void
    var onTapPoint: ((ChartPoint) -> Void)? = nil

    @State private var isPieView = false

    private var isTogglableToPie: Bool {
        config.type == "bar" && !config.data.isEmpty && config.data.allSatisfy { $0.xVal != nil }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(alignment: .center) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(config.title)
                        .font(.headline)
                    if !config.xLabel.isEmpty || !config.yLabel.isEmpty {
                        HStack(spacing: 6) {
                            Text(config.xLabel)
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Text("→")
                                .font(.caption2)
                                .foregroundColor(.secondary)
                            Text(config.yLabel)
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                }
                Spacer()
                
                HStack(spacing: 12) {
                    if isTogglableToPie {
                        Picker("", selection: $isPieView) {
                            Image(systemName: "chart.bar.fill").tag(false)
                            Image(systemName: "chart.pie.fill").tag(true)
                        }
                        .pickerStyle(.segmented)
                        .frame(width: 70)
                        .labelsHidden()
                    }

                    Button {
                        onAskAI(buildChartPrompt(config))
                    } label: {
                        Label("AI Insight", systemImage: "sparkles")
                            .font(.caption)
                            .fontWeight(.medium)
                    }
                    .buttonStyle(.bordered)
                    .tint(.purple)
                }
            }

            chartView
                .frame(height: (config.type == "image_grid" || config.type == "boxplot" || config.type == "wordcloud") ? 340 : 260)
        }
        .padding(20)
        .background(Color(NSColor.controlBackgroundColor))
        .cornerRadius(16)
        .overlay(RoundedRectangle(cornerRadius: 16).stroke(Color.primary.opacity(0.08), lineWidth: 1))
    }

    @ViewBuilder
    private var chartView: some View {
        if config.type == "bar" {
            if isPieView {
                PieDonutChartView(config: config)
            } else {
                BarChartView(config: config, onTapPoint: onTapPoint)
            }
        } else if config.type == "line" {
            LineChartView(config: config, onTapPoint: onTapPoint)
        } else if config.type == "scatter" {
            ScatterChartView(config: config, onTapPoint: onTapPoint)
        } else if config.type == "shap_beeswarm" {
            ShapBeeswarmView(config: config)
        } else if config.type == "pdp_ice" {
            PdpIceView(config: config)
        } else if config.type == "image_grid" {
            ImageGridView(config: config)
        } else if config.type == "boxplot" {
            BoxPlotView(config: config)
        } else if config.type == "wordcloud" {
            WordCloudView(config: config)
        } else if config.type == "ridgeline" {
            RidgelineChartView(config: config)
        } else {
            Text("Unsupported chart type: \(config.type)")
                .foregroundColor(.secondary)
        }
    }
}
