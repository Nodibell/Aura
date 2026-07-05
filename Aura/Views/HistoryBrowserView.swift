import SwiftUI

struct HistoryBrowserView: View {
    @Binding var isPresented: Bool
    let historyService: any AnalysisHistoryServiceProtocol
    let onSelect: (HistoryItem) -> Void
    let onRename: (HistoryItem) -> Void
    let onDelete: (HistoryItem) -> Void
    
    @State private var searchText = ""
    @State private var filterTaskType = "All"
    @State private var sortOption = "date_desc"
    
    var body: some View {
        VStack(spacing: 0) {
            // Header Title Bar
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Analysis History")
                        .font(.title2)
                        .fontWeight(.bold)
                    Text("Browse, search, and manage your past data analysis runs.")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                Spacer()
                
                Button("Done") {
                    isPresented = false
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.return, modifiers: [])
            }
            .padding()
            .background(Color(NSColor.controlBackgroundColor))
            
            Divider()
            
            // Search and Filter Bar
            HStack(spacing: 12) {
                // Search Field
                HStack {
                    Image(systemName: "magnifyingglass")
                        .foregroundColor(.secondary)
                    TextField("Search by dataset name...", text: $searchText)
                        .textFieldStyle(.plain)
                    if !searchText.isEmpty {
                        Button { searchText = "" } label: {
                            Image(systemName: "xmark.circle.fill")
                                .foregroundColor(.secondary)
                        }
                        .buttonStyle(.plain)
                    }
                }
                .padding(6)
                .background(Color.primary.opacity(0.04))
                .cornerRadius(6)
                .frame(maxWidth: .infinity)
                
                // Task Type Filter
                Picker("Task Type:", selection: $filterTaskType) {
                    Text("All Tasks").tag("All")
                    Text("Classification").tag("classification")
                    Text("Regression").tag("regression")
                    Text("Time Series").tag("timeseries")
                    Text("Computer Vision").tag("object_detection")
                }
                .frame(width: 200)
                
                // Sort Picker
                Picker("Sort By:", selection: $sortOption) {
                    Text("Newest").tag("date_desc")
                    Text("Oldest").tag("date_asc")
                    Text("Best Score").tag("score_desc")
                    Text("Worst Score").tag("score_asc")
                }
                .frame(width: 180)
            }
            .padding()
            .background(Color(NSColor.windowBackgroundColor))
            
            Divider()
            
            // History list
            if filteredItems.isEmpty {
                VStack(spacing: 12) {
                    Image(systemName: "doc.text.magnifyingglass")
                        .font(.system(size: 40))
                        .foregroundColor(.secondary)
                    Text("No history items found.")
                        .font(.headline)
                        .foregroundColor(.secondary)
                    Text("Try adjusting your search query or filters.")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Color(NSColor.underPageBackgroundColor))
            } else {
                List {
                    ForEach(filteredItems) { item in
                        Button {
                            onSelect(item)
                            isPresented = false
                        } label: {
                            HStack(spacing: 12) {
                                // Pinned Icon / Star
                                Button {
                                    historyService.togglePinItem(item)
                                } label: {
                                    Image(systemName: (item.isPinned ?? false) ? "star.fill" : "star")
                                        .foregroundColor((item.isPinned ?? false) ? .yellow : .secondary.opacity(0.4))
                                        .font(.system(size: 14))
                                }
                                .buttonStyle(.plain)
                                .help((item.isPinned ?? false) ? "Unpin analysis" : "Pin analysis")
                                
                                RoundedRectangle(cornerRadius: 1.5)
                                    .fill(item.uiColor)
                                    .frame(width: 3, height: 32)
                                
                                VStack(alignment: .leading, spacing: 3) {
                                    HStack {
                                        Text(item.datasetName)
                                            .font(.system(size: 13, weight: .semibold, design: .rounded))
                                            .foregroundColor(.primary)
                                            .lineLimit(1)
                                        
                                        Text(item.shortLabel)
                                            .font(.system(size: 9, weight: .bold))
                                            .foregroundColor(item.uiColor)
                                            .padding(.horizontal, 6)
                                            .padding(.vertical, 1)
                                            .background(item.uiColor.opacity(0.12))
                                            .cornerRadius(4)
                                        
                                        Spacer()
                                        
                                        Text(item.timestamp, style: .date)
                                            .font(.caption2)
                                            .foregroundColor(.secondary)
                                        Text(item.timestamp, style: .time)
                                            .font(.caption2)
                                            .foregroundColor(.secondary)
                                    }
                                    
                                    HStack(spacing: 12) {
                                        if let model = item.bestModel, let score = item.bestScore {
                                            Text("Model: \(model) (\(item.scoreType ?? "Score"): \(String(format: "%.4f", score)))")
                                                .font(.caption)
                                                .foregroundColor(.secondary)
                                        } else if let rows = item.rowCount, let cols = item.colCount {
                                            Text("Data profile: \(rows) rows × \(cols) columns")
                                                .font(.caption)
                                                .foregroundColor(.secondary)
                                        }
                                        
                                        Spacer()
                                        
                                        Text("Target: \(item.targetColumn ?? "Auto-detect")")
                                            .font(.caption2)
                                            .foregroundColor(.secondary.opacity(0.8))
                                    }
                                }
                            }
                            .padding(.vertical, 6)
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                        .contextMenu {
                            Button {
                                onRename(item)
                            } label: {
                                Label("Rename...", systemImage: "pencil")
                            }
                            Button(role: .destructive) {
                                onDelete(item)
                            } label: {
                                Label("Delete", systemImage: "trash")
                            }
                        }
                    }
                }
                .listStyle(SidebarListStyle())
            }
        }
        .frame(minWidth: 750, minHeight: 480)
    }
    
    // MARK: - Filtering and Sorting Logic
    
    private var filteredItems: [HistoryItem] {
        var items = historyService.items
        
        // 1. Search text filter
        if !searchText.isEmpty {
            items = items.filter { $0.datasetName.localizedCaseInsensitiveContains(searchText) }
        }
        
        // 2. Task type filter
        if filterTaskType != "All" {
            items = items.filter { item in
                let type = item.taskType?.lowercased() ?? ""
                if filterTaskType == "classification" {
                    return type.contains("class")
                } else if filterTaskType == "regression" {
                    return type.contains("regress")
                } else if filterTaskType == "timeseries" {
                    return type.contains("time") || type.contains("forecast")
                } else if filterTaskType == "object_detection" {
                    return type.contains("object") || type.contains("vision") || type.contains("image")
                }
                return type == filterTaskType
            }
        }
        
        // 3. Sorting
        items.sort { a, b in
            // Always float pinned items to top first
            let aPinned = a.isPinned ?? false
            let bPinned = b.isPinned ?? false
            if aPinned != bPinned {
                return aPinned // True floats to top
            }
            
            switch sortOption {
            case "date_asc":
                return a.timestamp < b.timestamp
            case "score_desc":
                let aScore = a.bestScore ?? -Double.infinity
                let bScore = b.bestScore ?? -Double.infinity
                return aScore > bScore
            case "score_asc":
                let aScore = a.bestScore ?? Double.infinity
                let bScore = b.bestScore ?? Double.infinity
                return aScore < bScore
            default: // date_desc
                return a.timestamp > b.timestamp
            }
        }
        
        return items
    }
}
