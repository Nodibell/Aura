import SwiftUI

struct PendingAnalysisView: View {
    let page: AnalysisPage
    let onRunAnalysis: () -> Void

    var body: some View {
        VStack(spacing: 32) {
            Spacer()

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
            VStack(spacing: 12) {
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
            }
            
            // Interactive Steps Card
            VStack(alignment: .leading, spacing: 14) {
                Text("PRE-RUN CHECKLIST")
                    .font(.system(size: 9, weight: .bold))
                    .foregroundColor(.secondary.opacity(0.6))
                    .tracking(1.5)
                    .padding(.bottom, 2)
                
                checklistRow(number: 1, description: "Select dataset type (e.g. Tabular or Time Series) in the sidebar.")
                checklistRow(number: 2, description: "Choose one or more target columns to forecast/predict.")
                checklistRow(number: 3, description: "Exclude any identifier or unneeded columns from the schema.")
                checklistRow(number: 4, description: "Enable smart sampling if fitting a very large dataset.")
            }
            .frame(width: 420, alignment: .leading)
            .padding(20)
            .background(Color.primary.opacity(0.01))
            .cornerRadius(12)
            .overlay(
                RoundedRectangle(cornerRadius: 12)
                    .stroke(Color.primary.opacity(0.03), lineWidth: 1)
            )
            
            // CTA Button
            Button(action: onRunAnalysis) {
                HStack(spacing: 8) {
                    Image(systemName: "play.fill")
                        .font(.system(size: 11))
                    Text("Run Analysis Pipeline")
                        .font(.system(size: 13, weight: .bold, design: .rounded))
                }
                .foregroundColor(.white)
                .padding(.horizontal, 28)
                .padding(.vertical, 12)
                .background(
                    LinearGradient(colors: [.purple, .indigo], startPoint: .leading, endPoint: .trailing)
                )
                .cornerRadius(8)
                .shadow(color: .purple.opacity(0.15), radius: 8, x: 0, y: 3)
            }
            .buttonStyle(.plain)
            
            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(nsColor: .windowBackgroundColor))
    }
    
    private var dimensionsString: String {
        guard let preview = page.previewResult else { return "--" }
        if let total = preview.totalRows {
            return "\(preview.columns.count) cols × \(total) rows"
        }
        return "\(preview.columns.count) cols × \(preview.previewRows.count)+ rows"
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
    
    private func checklistRow(number: Int, description: String) -> some View {
        HStack(alignment: .top, spacing: 12) {
            Text("\(number)")
                .font(.system(size: 10, weight: .bold, design: .rounded))
                .foregroundColor(.purple)
                .frame(width: 18, height: 18)
                .background(Circle().fill(Color.purple.opacity(0.08)))
            
            Text(description)
                .font(.system(size: 11.5))
                .foregroundColor(.secondary)
                .lineSpacing(2)
        }
    }
}
