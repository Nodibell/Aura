import SwiftUI
import Charts

struct PdpIceView: View {
    let config: ChartConfig
    
    var body: some View {
        Chart {
            ForEach(config.data.filter { $0.series != "PDP" }) { point in
                if let xNum = point.xNum {
                    LineMark(
                        x: .value(config.xLabel, xNum),
                        y: .value(config.yLabel, point.y),
                        series: .value("Series", point.series ?? "")
                    )
                    .foregroundStyle(Color.primary.opacity(0.12))
                    .lineStyle(StrokeStyle(lineWidth: 0.8))
                }
            }
            
            ForEach(config.data.filter { $0.series == "PDP" }) { point in
                if let xNum = point.xNum {
                    LineMark(
                        x: .value(config.xLabel, xNum),
                        y: .value(config.yLabel, point.y),
                        series: .value("Series", "PDP")
                    )
                    .foregroundStyle(Color.purple)
                    .lineStyle(StrokeStyle(lineWidth: 3.0))
                }
            }
        }
        .chartXAxis {
            AxisMarks()
        }
        .chartYAxis {
            AxisMarks(position: .leading)
        }
        .chartXScale(domain: .automatic(includesZero: false))
        .chartYScale(domain: .automatic(includesZero: false))
        .chartLegend(.hidden)
    }
}
