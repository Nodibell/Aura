import SwiftUI

struct WordCloudView: View {
    let config: ChartConfig
    
    @State private var hoveredWord: String? = nil
    
    private let colors: [Color] = [
        .blue, .purple, .pink, .indigo, .teal, .orange, .cyan, .mint
    ]
    
    var body: some View {
        let maxWeight = config.data.map { $0.y }.max() ?? 1.0
        let minWeight = config.data.map { $0.y }.min() ?? 0.0
        let weightRange = max(0.0001, maxWeight - minWeight)
        
        ScrollView(.vertical, showsIndicators: true) {
            WordCloudLayout {
                ForEach(Array(config.data.enumerated()), id: \.element.id) { index, point in
                    if let word = point.xVal {
                        let normalizedWeight = (point.y - minWeight) / weightRange
                        let size = 12 + normalizedWeight * 20 // size from 12 to 32
                        
                        Text(word)
                            .font(.system(size: size, weight: .semibold, design: .rounded))
                            .foregroundColor(colors[index % colors.count])
                            .padding(.horizontal, 6)
                            .padding(.vertical, 3)
                            .background(
                                RoundedRectangle(cornerRadius: 6)
                                    .fill(colors[index % colors.count].opacity(hoveredWord == word ? 0.15 : 0.03))
                            )
                            .scaleEffect(hoveredWord == word ? 1.08 : 1.0)
                            .animation(.spring(response: 0.2, dampingFraction: 0.6), value: hoveredWord)
                            .onHover { isHovered in
                                hoveredWord = isHovered ? word : nil
                            }
                            .help(String(format: "TF-IDF Weight: %.4f", point.y))
                    }
                }
            }
            .padding(10)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color.primary.opacity(0.01))
        .cornerRadius(12)
    }
}

struct WordCloudLayout: Layout {
    var spacing: CGFloat = 8
    
    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let width = proposal.width ?? 400
        var currentX: CGFloat = 0
        var currentY: CGFloat = 0
        var lineHeight: CGFloat = 0
        var maxWidth: CGFloat = 0
        
        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if currentX + size.width > width {
                maxWidth = max(maxWidth, currentX)
                currentX = 0
                currentY += lineHeight + spacing
                lineHeight = 0
            }
            currentX += size.width + spacing
            lineHeight = max(lineHeight, size.height)
        }
        maxWidth = max(maxWidth, currentX)
        return CGSize(width: maxWidth, height: currentY + lineHeight)
    }
    
    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        var currentX: CGFloat = bounds.minX
        var currentY: CGFloat = bounds.minY
        var lineHeight: CGFloat = 0
        
        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if currentX + size.width > bounds.maxX {
                currentX = bounds.minX
                currentY += lineHeight + spacing
                lineHeight = 0
            }
            subview.place(
                at: CGPoint(x: currentX, y: currentY),
                proposal: ProposedViewSize(size)
            )
            currentX += size.width + spacing
            lineHeight = max(lineHeight, size.height)
        }
    }
}
