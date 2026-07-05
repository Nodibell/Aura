import SwiftUI
import Charts

struct BoxPlotView: View {
    let config: ChartConfig

    var body: some View {
        if let stats = config.boxStats {
            GeometryReader { geo in
                Chart {
                    // Box from Q1 to Q3
                    RectangleMark(
                        xStart: .value("Left", 0.8),
                        xEnd: .value("Right", 1.2),
                        yStart: .value("Q1", stats.q1),
                        yEnd: .value("Q3", stats.q3)
                    )
                    .foregroundStyle(Color.purple.opacity(0.25))
                    
                    // Bottom border of box
                    RuleMark(
                        xStart: .value("Left", 0.8),
                        xEnd: .value("Right", 1.2),
                        y: .value("Q1", stats.q1)
                    )
                    .foregroundStyle(Color.purple)
                    .lineStyle(StrokeStyle(lineWidth: 1.5))
                    
                    // Top border of box
                    RuleMark(
                        xStart: .value("Left", 0.8),
                        xEnd: .value("Right", 1.2),
                        y: .value("Q3", stats.q3)
                    )
                    .foregroundStyle(Color.purple)
                    .lineStyle(StrokeStyle(lineWidth: 1.5))

                    // Whiskers (vertical rules at x = 1.0)
                    RuleMark(
                        x: .value("X", 1.0),
                        yStart: .value("Min", stats.min),
                        yEnd: .value("Q1", stats.q1)
                    )
                    .lineStyle(StrokeStyle(lineWidth: 1.5, dash: [4, 4]))
                    .foregroundStyle(Color.purple)
                    
                    RuleMark(
                        x: .value("X", 1.0),
                        yStart: .value("Q3", stats.q3),
                        yEnd: .value("Max", stats.max)
                    )
                    .lineStyle(StrokeStyle(lineWidth: 1.5, dash: [4, 4]))
                    .foregroundStyle(Color.purple)
                    
                    // Whisker caps (horizontal rules)
                    RuleMark(
                        xStart: .value("Left", 0.9),
                        xEnd: .value("Right", 1.1),
                        y: .value("Min", stats.min)
                    )
                    .foregroundStyle(Color.purple)
                    .lineStyle(StrokeStyle(lineWidth: 1.5))
                    
                    RuleMark(
                        xStart: .value("Left", 0.9),
                        xEnd: .value("Right", 1.1),
                        y: .value("Max", stats.max)
                    )
                    .foregroundStyle(Color.purple)
                    .lineStyle(StrokeStyle(lineWidth: 1.5))
                    
                    // Median line (horizontal rule at x = 1.0)
                    RuleMark(
                        xStart: .value("Left", 0.8),
                        xEnd: .value("Right", 1.2),
                        y: .value("Median", stats.median)
                    )
                    .foregroundStyle(Color.purple)
                    .lineStyle(StrokeStyle(lineWidth: 2.5))
                    
                    // Outliers (points at x = 1.0)
                    ForEach(stats.outliers, id: \.self) { val in
                        PointMark(
                            x: .value("X", 1.0),
                            y: .value("Outlier", val)
                        )
                        .foregroundStyle(Color.red)
                        .symbolSize(20)
                    }
                }
                .chartXAxis {
                    AxisMarks(values: [1.0]) { value in
                        AxisValueLabel {
                            Text(config.yLabel)
                        }
                    }
                }
                .chartYAxis {
                    AxisMarks(position: .leading)
                }
                .chartXScale(domain: 0.5...1.5)
                .chartYScale(domain: .automatic(includesZero: false))
                .padding(.all, 8)
            }
        } else {
            Text("No box stats available.")
                .foregroundColor(.secondary)
        }
    }
}
