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

        // NEW — semantic roles, so intent is named instead of re-guessed per call site
        static let primaryAction = SwiftUI.Color.purple      // matches the 9 existing .tint(.purple) call sites
        static let success       = SwiftUI.Color.green        // matches DataCleaningView's "apply/confirm" tints
        static let caution       = SwiftUI.Color.orange        // matches DataCleaningView's rollback/undo tints
        static let destructive   = SwiftUI.Color.red           // matches SettingsView's destructive action
        static let info          = SwiftUI.Color.blue          // matches plugin-related actions
    }
    
    // MARK: - Fonts
    struct Font {
        static func brand(size: CGFloat, weight: SwiftUI.Font.Weight = .regular) -> SwiftUI.Font {
            SwiftUI.Font.system(size: size, weight: weight, design: .rounded)
        }
        
        static let sidebarHeader = SwiftUI.Font.system(size: 9, weight: .bold, design: .rounded)
        static let cardTitle = SwiftUI.Font.system(size: 11, weight: .semibold, design: .rounded)
        static let bodyRounded = SwiftUI.Font.system(.body, design: .rounded)

        // NEW — the sizes that actually recur across Views/ (from a grep of font(.system(size:)) usage)
        static let caption       = SwiftUI.Font.system(size: 11)
        static let captionBold   = SwiftUI.Font.system(size: 11, weight: .semibold)
        static let controlLabel  = SwiftUI.Font.system(size: 12, weight: .semibold)
        static let sectionTitle  = SwiftUI.Font.system(size: 14, weight: .bold, design: .rounded)
        static let pageTitle     = SwiftUI.Font.system(size: 15, weight: .semibold)
    }
    
    // MARK: - Spacing & Corner Radius
    struct Layout {
        static let cornerRadius: CGFloat = 8
        static let spacing: CGFloat = 6
        static let padding: CGFloat = 16
        
        static let controlCornerRadius: CGFloat = 6   // matches the Menu-label style already used in PendingAnalysisView
        static let toolbarPadding: CGFloat = 10
    }
}
