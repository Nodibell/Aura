import SwiftUI

struct PendingAnalysisView: View {
    let page: AnalysisPage
    let onRunAnalysis: () -> Void
    let onCancel: () -> Void

    var body: some View {
        ScrollView {
            VStack(spacing: 28) {
                Spacer(minLength: 20)

                // Icon & Title
                VStack(spacing: 12) {
                    ZStack {
                        Circle()
                            .fill(LinearGradient(colors: [.purple.opacity(0.12), .indigo.opacity(0.06)], startPoint: .topLeading, endPoint: .bottomTrailing))
                            .frame(width: 80, height: 80)
                        
                        Image(systemName: page.analysisConfig.datasetType.icon)
                            .font(.system(size: 36, weight: .semibold))
                            .foregroundStyle(LinearGradient(colors: [.purple, .indigo], startPoint: .topLeading, endPoint: .bottomTrailing))
                    }
                    
                    Text(page.title)
                        .font(.system(size: 20, weight: .bold, design: .rounded))
                        .foregroundColor(.primary)
                    
                    Text("AutoML Pipeline Pending")
                        .font(.system(size: 13, weight: .medium))
                        .foregroundColor(.secondary)
                }
                
                // Dataset Metadata Card
                HStack(spacing: 24) {
                    metadataItem(title: "Dimensions", value: dimensionsString)
                    Divider().frame(height: 32)
                    metadataItem(title: "Type", value: page.analysisConfig.datasetType.label)
                    Divider().frame(height: 32)
                    metadataItem(title: "Source", value: page.fileDetails)
                }
                .padding(.horizontal, 24)
                .padding(.vertical, 16)
                .background(Color.primary.opacity(0.015))
                .cornerRadius(12)
                .overlay(
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(Color.primary.opacity(0.04), lineWidth: 1)
                )
                
                // Interactive Configuration Card
                VStack(alignment: .leading, spacing: 18) {
                    HStack(spacing: 8) {
                        Image(systemName: "slider.horizontal.3")
                            .font(.system(size: 14, weight: .bold))
                            .foregroundColor(.purple)
                        Text("Configure Analysis")
                            .font(.system(size: 14, weight: .bold, design: .rounded))
                    }
                    .padding(.bottom, 4)
                    
                    Divider().background(Color.primary.opacity(0.06))
                    
                    // 1. Target Column Selector
                    VStack(alignment: .leading, spacing: 6) {
                        Text("TARGET COLUMN")
                            .font(.system(size: 9, weight: .bold))
                            .foregroundColor(.secondary.opacity(0.7))
                            .tracking(1.0)
                        
                        let columns = availableColumns
                        if !columns.isEmpty {
                            Menu {
                                Button("Auto-detect target") {
                                    page.analysisConfig.targetColumn = ""
                                    page.analysisConfig.taskTypeOverride = .auto
                                }
                                Button("None (Run Clustering)") {
                                    page.analysisConfig.targetColumn = ""
                                    page.analysisConfig.taskTypeOverride = .clustering
                                }
                                Divider()
                                ForEach(columns, id: \.self) { col in
                                    Button(col) {
                                        page.analysisConfig.targetColumn = col
                                        page.analysisConfig.excludedColumns.remove(col)
                                        if page.analysisConfig.taskTypeOverride == .clustering {
                                            page.analysisConfig.taskTypeOverride = .auto
                                        }
                                    }
                                }
                            } label: {
                                HStack {
                                    let selectedLabel = page.analysisConfig.taskTypeOverride == .clustering ? "None (Clustering)" : (page.analysisConfig.targetColumn.isEmpty ? "Auto-detect target" : page.analysisConfig.targetColumn)
                                    Text(selectedLabel)
                                        .font(.system(size: 12))
                                        .foregroundColor(page.analysisConfig.targetColumn.isEmpty && page.analysisConfig.taskTypeOverride != .clustering ? .secondary : .primary)
                                        .lineLimit(1)
                                    Spacer()
                                    Image(systemName: "chevron.up.chevron.down")
                                        .font(.system(size: 8))
                                        .foregroundColor(.secondary)
                                }
                                .padding(.horizontal, 10)
                                .padding(.vertical, 8)
                                .background(Color.primary.opacity(0.025))
                                .cornerRadius(6)
                                .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.primary.opacity(0.05), lineWidth: 1))
                            }
                            .menuStyle(.borderlessButton)
                        } else {
                            Text("No columns available")
                                .font(.system(size: 11))
                                .foregroundColor(.secondary)
                        }
                    }
                    
                    // 2. Task Type Selector
                    VStack(alignment: .leading, spacing: 6) {
                        Text("TASK TYPE")
                            .font(.system(size: 9, weight: .bold))
                            .foregroundColor(.secondary.opacity(0.7))
                            .tracking(1.0)
                        
                        HStack(spacing: 4) {
                            taskTypeButton(label: "Auto", type: .auto)
                            taskTypeButton(label: "Regr", type: .regression)
                            taskTypeButton(label: "Clsf", type: .classification)
                            taskTypeButton(label: "Clst", type: .clustering)
                        }
                        .padding(2)
                        .background(Color.primary.opacity(0.015))
                        .cornerRadius(6)
                        .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.primary.opacity(0.04), lineWidth: 1))
                    }
                    
                    // 3. Excluded Columns Editor
                    VStack(alignment: .leading, spacing: 6) {
                        Text("EXCLUDED COLUMNS")
                            .font(.system(size: 9, weight: .bold))
                            .foregroundColor(.secondary.opacity(0.7))
                            .tracking(1.0)
                        
                        let excludedList = Array(page.analysisConfig.excludedColumns).sorted()
                        
                        VStack(alignment: .leading, spacing: 6) {
                            if !excludedList.isEmpty {
                                FlowLayout(spacing: 6) {
                                    ForEach(excludedList, id: \.self) { col in
                                        HStack(spacing: 4) {
                                            Text(col)
                                                .font(.system(size: 11))
                                                .foregroundColor(.primary)
                                            Button(action: {
                                                page.analysisConfig.excludedColumns.remove(col)
                                            }) {
                                                Image(systemName: "xmark.circle.fill")
                                                    .font(.system(size: 10))
                                                    .foregroundColor(.secondary)
                                            }
                                            .buttonStyle(.plain)
                                        }
                                        .padding(.horizontal, 8)
                                        .padding(.vertical, 4)
                                        .background(Color.primary.opacity(0.03))
                                        .cornerRadius(12)
                                        .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color.primary.opacity(0.06), lineWidth: 1))
                                    }
                                }
                            }
                            
                            let addable = availableColumns.filter { !page.analysisConfig.excludedColumns.contains($0) && $0 != page.analysisConfig.targetColumn }
                            
                            if !addable.isEmpty {
                                Menu {
                                    ForEach(addable, id: \.self) { col in
                                        Button(col) {
                                            page.analysisConfig.excludedColumns.insert(col)
                                        }
                                    }
                                } label: {
                                    Label("Exclude Column...", systemImage: "plus")
                                        .font(.system(size: 10, weight: .semibold))
                                        .foregroundColor(.purple)
                                        .padding(.horizontal, 10)
                                        .padding(.vertical, 5)
                                        .background(Color.purple.opacity(0.08))
                                        .cornerRadius(6)
                                }
                                .menuStyle(.borderlessButton)
                            }
                        }
                    }
                    
                    // 4. Smart Sampling Toggle
                    Toggle(isOn: Binding(
                        get: { page.analysisConfig.smartSample },
                        set: { page.analysisConfig.smartSample = $0 }
                      )) {
                          VStack(alignment: .leading, spacing: 2) {
                              Text("Smart Sampling")
                                  .font(.system(size: 11, weight: .semibold))
                              Text("Optimizes memory for larger datasets")
                                  .font(.system(size: 9))
                                  .foregroundColor(.secondary)
                          }
                      }
                      .toggleStyle(.checkbox)
                      
                      Divider().background(Color.primary.opacity(0.06))
                      
                      // Actions in Card
                      HStack(spacing: 8) {
                          Button("Cancel", action: onCancel)
                              .buttonStyle(.bordered)
                              .controlSize(.small)
                          
                          Spacer()
                          
                          Button(action: onRunAnalysis) {
                              HStack(spacing: 6) {
                                  Image(systemName: "play.fill")
                                      .font(.system(size: 11))
                                  Text("Run Analysis Pipeline")
                                      .font(.system(size: 12, weight: .bold, design: .rounded))
                              }
                          }
                          .buttonStyle(.borderedProminent)
                          .tint(.purple)
                          .controlSize(.small)
                      }
                      .padding(.top, 4)
                  }
                  .frame(width: 420)
                  .padding(20)
                  .background(Color(nsColor: .controlBackgroundColor))
                  .cornerRadius(12)
                  .overlay(
                      RoundedRectangle(cornerRadius: 12)
                          .stroke(Color.primary.opacity(0.06), lineWidth: 1)
                  )
                  
                  Spacer(minLength: 20)
              }
              .frame(maxWidth: .infinity)
          }
          .background(Color(nsColor: .windowBackgroundColor))
      }
      
      private var dimensionsString: String {
          guard let preview = page.previewResult else { return "--" }
          if let total = preview.totalRows {
              return "\(preview.columns.count) cols × \(total) rows"
          }
          return "\(preview.columns.count) cols × \(preview.previewRows.count)+ rows"
      }
      
      private var availableColumns: [String] {
          if !page.trainColumns.isEmpty {
              return page.trainColumns
          }
          if let preview = page.previewResult {
              return preview.columns
          }
          if let result = page.result {
              return result.columns
          }
          return []
      }
      
      private func metadataItem(title: String, value: String) -> some View {
          VStack(spacing: 4) {
              Text(title.uppercased())
                  .font(.system(size: 8, weight: .bold))
                  .foregroundColor(.secondary.opacity(0.7))
                  .tracking(0.5)
              Text(value)
                  .font(.system(size: 12, weight: .semibold, design: .rounded))
                  .foregroundColor(.primary)
          }
      }
      
      private func taskTypeButton(label: String, type: TaskTypeOverride) -> some View {
          let isSelected = page.analysisConfig.taskTypeOverride == type
          return Button(action: {
              withAnimation(.easeInOut(duration: 0.15)) {
                  page.analysisConfig.taskTypeOverride = type
              }
          }) {
              Text(label)
                  .font(.system(size: 11, weight: isSelected ? .bold : .medium))
                  .foregroundColor(isSelected ? .white : .secondary)
                  .frame(maxWidth: .infinity)
                  .padding(.vertical, 6)
                  .background(isSelected ? Color.purple : Color.clear)
                  .cornerRadius(4)
          }
          .buttonStyle(.plain)
      }
}

// Simple Flow Layout helper for wrapping chips horizontally
struct FlowLayout: Layout {
    var spacing: CGFloat
    
    init(spacing: CGFloat = 6) {
        self.spacing = spacing
    }
    
    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let width = proposal.width ?? 0
        var height: CGFloat = 0
        var currentX: CGFloat = 0
        var currentY: CGFloat = 0
        var maxHeightInRow: CGFloat = 0
        
        for view in subviews {
            let size = view.sizeThatFits(.unspecified)
            if currentX + size.width > width {
                currentX = 0
                currentY += maxHeightInRow + spacing
                maxHeightInRow = 0
            }
            currentX += size.width + spacing
            maxHeightInRow = max(maxHeightInRow, size.height)
        }
        height = currentY + maxHeightInRow
        return CGSize(width: width, height: height)
    }
    
    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        var currentX: CGFloat = bounds.minX
        var currentY: CGFloat = bounds.minY
        var maxHeightInRow: CGFloat = 0
        
        for view in subviews {
            let size = view.sizeThatFits(.unspecified)
            if currentX + size.width > bounds.maxX {
                currentX = bounds.minX
                currentY += maxHeightInRow + spacing
                maxHeightInRow = 0
            }
            view.place(at: CGPoint(x: currentX, y: currentY), proposal: .unspecified)
            currentX += size.width + spacing
            maxHeightInRow = max(maxHeightInRow, size.height)
        }
    }
}
