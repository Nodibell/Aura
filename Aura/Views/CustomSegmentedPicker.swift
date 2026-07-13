import SwiftUI

struct CustomSegmentedPicker<SelectionValue: Hashable>: View {
    @Binding var selection: SelectionValue
    let items: [(String, SelectionValue)]
    
    @Namespace private var animation
    
    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 4) {
                ForEach(items, id: \.1) { item in
                    let isSelected = selection == item.1
                    
                    Button {
                        withAnimation(.spring(response: 0.22, dampingFraction: 0.8)) {
                            selection = item.1
                        }
                    } label: {
                        Text(item.0)
                            .font(.system(size: 11, weight: isSelected ? .semibold : .medium, design: .rounded))
                            .foregroundColor(isSelected ? .white : .secondary)
                            .lineLimit(1)
                            .fixedSize()
                            .padding(.vertical, 6)
                            .padding(.horizontal, 12)
                            .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .accessibilityIdentifier("tab_\(item.0)")
                    .background {
                        if isSelected {
                            RoundedRectangle(cornerRadius: 6)
                                .fill(LinearGradient(
                                    colors: [.purple, .indigo],
                                    startPoint: .topLeading,
                                    endPoint: .bottomTrailing
                                ))
                                .matchedGeometryEffect(id: "selectedSegment", in: animation)
                        }
                    }
                }
            }
            .padding(3)
            .background(Color.primary.opacity(0.03))
            .cornerRadius(8)
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .stroke(Color.primary.opacity(0.05), lineWidth: 1)
            )
        }
        .fixedSize(horizontal: false, vertical: true)
    }
}