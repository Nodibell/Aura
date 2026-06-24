import SwiftUI

@main
struct AuraApp: App {
    @AppStorage("Aura_Appearance") private var appearanceMode = "System"

    var body: some Scene {
        WindowGroup {
            ContentView()
                .preferredColorScheme(appearanceMode == "Dark" ? .dark : (appearanceMode == "Light" ? .light : nil))
        }
        .commands {
            TextEditingCommands()
        }
        
        Settings {
            SettingsView()
                .preferredColorScheme(appearanceMode == "Dark" ? .dark : (appearanceMode == "Light" ? .light : nil))
        }
        .commands {
            TextEditingCommands()
        }
    }
}
