import SwiftUI

struct Theme {
    // MARK: - Colors
    struct Color {
        static let purple = SwiftUI.Color.purple
        static let indigo = SwiftUI.Color.indigo
        
        static let brandGradient = LinearGradient(
            colors: [SwiftUI.Color.purple, SwiftUI.Color.indigo],
            startPoint: .topLeading,
            endPoint: .bottomTrailing
        )
        
        static let background = SwiftUI.Color(nsColor: .windowBackgroundColor)
        static let cardBackground = SwiftUI.Color.primary.opacity(0.02)
        static let cardStroke = SwiftUI.Color.primary.opacity(0.04)
        static let textSecondary = SwiftUI.Color.secondary.opacity(0.6)
    }
    
    // MARK: - Fonts
    struct Font {
        static func brand(size: CGFloat, weight: SwiftUI.Font.Weight = .regular) -> SwiftUI.Font {
            SwiftUI.Font.system(size: size, weight: weight, design: .rounded)
        }
        
        static let sidebarHeader = SwiftUI.Font.system(size: 9, weight: .bold, design: .rounded)
        static let cardTitle = SwiftUI.Font.system(size: 11, weight: .semibold, design: .rounded)
        static let bodyRounded = SwiftUI.Font.system(.body, design: .rounded)
    }
    
    // MARK: - Spacing & Corner Radius
    struct Layout {
        static let cornerRadius: CGFloat = 8
        static let spacing: CGFloat = 6
        static let padding: CGFloat = 16
    }
}
