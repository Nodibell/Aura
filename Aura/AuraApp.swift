import SwiftUI

@main
struct AuraApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                .preferredColorScheme(.dark) // Dark mode by default for premium developer feel
        }
        .commands {
            TextEditingCommands()
        }
        
        Settings {
            SettingsView()
                .preferredColorScheme(.dark)
        }
        .commands {
            TextEditingCommands()
        }
    }
}
