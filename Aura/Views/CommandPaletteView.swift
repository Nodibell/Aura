import SwiftUI

struct CommandPaletteItem: Identifiable {
    let id = UUID()
    let title: String
    let subtitle: String
    let iconName: String
    let shortcut: String?
    let action: () -> Void
}

struct CommandPaletteView: View {
    @Binding var isPresented: Bool
    let viewModel: DashboardViewModel
    let onNavigateTab: (String) -> Void
    let onShowHistory: () -> Void
    
    @State private var searchText = ""
    @State private var selectedIndex = 0
    
    @AppStorage("Aura_Appearance") private var appearanceMode = "System"
    
    var body: some View {
        VStack(spacing: 0) {
            // Search Input Field
            HStack(spacing: Theme.Layout.spacing * 2) {
                Image(systemName: "magnifyingglass")
                    .font(.title3)
                    .foregroundColor(.secondary)
                
                TextField("Search commands, navigation targets, or history runs...", text: $searchText)
                    .textFieldStyle(.plain)
                    .font(Theme.Font.brand(size: 14))
                
                if !searchText.isEmpty {
                    Button {
                        searchText = ""
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundColor(.secondary)
                    }
                    .buttonStyle(.plain)
                }
                
                Text("ESC")
                    .font(.caption2)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Theme.Color.cardBackground)
                    .cornerRadius(Theme.Layout.cornerRadius - 4)
                    .foregroundColor(.secondary)
            }
            .padding(Theme.Layout.padding)
            .background(Theme.Color.background)
            
            Divider()
            
            // Commands List
            if filteredItems.isEmpty {
                VStack(spacing: 12) {
                    Image(systemName: "terminal")
                        .font(Theme.Font.brand(size: 32))
                        .foregroundColor(.secondary)
                    Text("No matching commands found.")
                        .font(Theme.Font.bodyRounded)
                        .foregroundColor(.secondary)
                }
                .frame(maxWidth: .infinity, minHeight: 240)
            } else {
                ScrollViewReader { proxy in
                    List {
                        ForEach(Array(filteredItems.enumerated()), id: \.element.id) { idx, item in
                            HStack(spacing: Theme.Layout.spacing * 2) {
                                Image(systemName: item.iconName)
                                    .font(.system(size: 14))
                                    .foregroundColor(selectedIndex == idx ? .white : .accentColor)
                                    .frame(width: 24, height: 24)
                                    .background(selectedIndex == idx ? Color.white.opacity(0.2) : Theme.Color.cardBackground)
                                    .cornerRadius(Theme.Layout.cornerRadius - 2)
                                
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(item.title)
                                        .font(Theme.Font.brand(size: 13, weight: .semibold))
                                        .foregroundColor(selectedIndex == idx ? .white : .primary)
                                    Text(item.subtitle)
                                        .font(Theme.Font.brand(size: 10))
                                        .foregroundColor(selectedIndex == idx ? .white.opacity(0.8) : .secondary)
                                }
                                
                                Spacer()
                                
                                if let shortcut = item.shortcut {
                                    Text(shortcut)
                                        .font(.system(size: 10, weight: .medium, design: .monospaced))
                                        .padding(.horizontal, 6)
                                        .padding(.vertical, 2)
                                        .background(selectedIndex == idx ? Color.white.opacity(0.25) : Theme.Color.cardStroke)
                                        .cornerRadius(Theme.Layout.cornerRadius - 4)
                                        .foregroundColor(selectedIndex == idx ? .white : .secondary)
                                }
                            }
                            .padding(.vertical, 6)
                            .padding(.horizontal, 10)
                            .background(selectedIndex == idx ? Color.accentColor : Color.clear)
                            .cornerRadius(Theme.Layout.cornerRadius)
                            .id(idx)
                            .contentShape(Rectangle())
                            .onTapGesture {
                                item.action()
                                isPresented = false
                            }
                        }
                    }
                    .listStyle(PlainListStyle())
                    .frame(minHeight: 260, maxHeight: 360)
                    .onChange(of: selectedIndex) { _, newVal in
                        withAnimation {
                            proxy.scrollTo(newVal, anchor: .center)
                        }
                    }
                }
            }
            
            Divider()
            
            // Footer help text
            HStack {
                Text("↑↓ to navigate  •  Enter to select  •  Esc to close")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                Spacer()
            }
            .padding(.vertical, 8)
            .padding(.horizontal)
            .background(Color(NSColor.windowBackgroundColor))
        }
        .frame(width: 580)
        .background(Color(NSColor.windowBackgroundColor))
        .cornerRadius(12)
        .shadow(radius: 16)
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(Color.primary.opacity(0.1), lineWidth: 1)
        )
        .background(
            // Invisible button layer to capture arrow key events
            Group {
                Button("") {
                    if selectedIndex > 0 { selectedIndex -= 1 }
                }
                .keyboardShortcut(.upArrow, modifiers: [])
                
                Button("") {
                    if selectedIndex < filteredItems.count - 1 { selectedIndex += 1 }
                }
                .keyboardShortcut(.downArrow, modifiers: [])
                
                Button("") {
                    if selectedIndex < filteredItems.count {
                        filteredItems[selectedIndex].action()
                        isPresented = false
                    }
                }
                .keyboardShortcut(.return, modifiers: [])
                
                Button("") {
                    isPresented = false
                }
                .keyboardShortcut(.escape, modifiers: [])
            }
            .opacity(0)
            .frame(width: 0, height: 0)
        )
        .onChange(of: searchText) { _, _ in
            selectedIndex = 0
        }
    }
    
    // MARK: - Filtered Commands Mappings
    
    private var allItems: [CommandPaletteItem] {
        var items: [CommandPaletteItem] = []
        
        // 1. Navigation items
        items.append(CommandPaletteItem(title: "Go to Summary", subtitle: "View leaderboard and model training summary metrics", iconName: "list.bullet.clipboard", shortcut: "Tab 1") {
            onNavigateTab("Summary")
        })
        items.append(CommandPaletteItem(title: "Go to Data Cleaning", subtitle: "Configure imputations, outlier treatments, and column actions", iconName: "slider.horizontal.3", shortcut: "Tab 2") {
            onNavigateTab("Data Cleaning")
        })
        items.append(CommandPaletteItem(title: "Go to Data Profiling", subtitle: "Examine histograms, correlation matrices and target interactions", iconName: "chart.bar", shortcut: "Tab 3") {
            onNavigateTab("Data Profiling")
        })
        items.append(CommandPaletteItem(title: "Go to Interactive Predictions", subtitle: "Run live inference or drag-and-drop batch CSV predictions", iconName: "play.circle", shortcut: "Tab 4") {
            onNavigateTab("Predict")
        })
        items.append(CommandPaletteItem(title: "Go to AI Assistant", subtitle: "Ask Ollama / local LLM for suggestions and clarifications", iconName: "bubble.left.and.bubble.right", shortcut: "Tab 5") {
            onNavigateTab("Summary")
            viewModel.showAIPanel = true
        })
        
        // 2. Action items
        items.append(CommandPaletteItem(title: "Run Pipeline Analysis", subtitle: "Trigger the scikit-learn / XGBoost training pipeline", iconName: "play.fill", shortcut: "⌘R") {
            viewModel.runEDA()
        })
        items.append(CommandPaletteItem(title: "Export Model & Code", subtitle: "Serialize model binaries and build reproduction Python code", iconName: "square.and.arrow.up", shortcut: "Export") {
            viewModel.showModelExportSheet = true
        })
        items.append(CommandPaletteItem(title: "Browse History Archive", subtitle: "Open full full-screen history tracker sheet", iconName: "clock.arrow.circlepath", shortcut: "⌘Y") {
            onShowHistory()
        })
        
        // 3. Settings items
        items.append(CommandPaletteItem(title: "Use Dark Theme", subtitle: "Set application color scheme to dark mode", iconName: "moon.fill", shortcut: nil) {
            appearanceMode = "Dark"
        })
        items.append(CommandPaletteItem(title: "Use Light Theme", subtitle: "Set application color scheme to light mode", iconName: "sun.max.fill", shortcut: nil) {
            appearanceMode = "Light"
        })
        items.append(CommandPaletteItem(title: "Use System Theme", subtitle: "Follow system-wide color appearance preference", iconName: "desktopcomputer", shortcut: nil) {
            appearanceMode = "System"
        })
        
        // 4. Quick History Runs items
        for run in viewModel.historyService.items.prefix(10) {
            items.append(CommandPaletteItem(title: "Load Run: \(run.datasetName)", subtitle: "Load results of dataset: \(run.datasetPath) (Run: \(run.timestamp.formatted()))", iconName: "doc.text.fill", shortcut: nil) {
                viewModel.loadHistoryItem(run)
            })
        }
        
        return items
    }
    
    private var filteredItems: [CommandPaletteItem] {
        if searchText.isEmpty {
            return allItems
        }
        return allItems.filter { item in
            item.title.localizedCaseInsensitiveContains(searchText) ||
            item.subtitle.localizedCaseInsensitiveContains(searchText)
        }
    }
}
