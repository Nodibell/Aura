import SwiftUI

struct ImageGridView: View {
    let config: ChartConfig
    @State private var selectedImage: ImageItem? = nil

    private let columns = [
        GridItem(.adaptive(minimum: 120, maximum: 160), spacing: 16)
    ]

    var body: some View {
        let items = config.images ?? []
        ScrollView {
            LazyVGrid(columns: columns, spacing: 16) {
                ForEach(items) { item in
                    VStack(alignment: .center, spacing: 8) {
                        if let platImg = platformImage(from: item.base64) {
                            #if canImport(AppKit)
                            Image(nsImage: platImg)
                                .resizable()
                                .interpolation(.none) // keeps pixel-art look (nearest neighbor)
                                .aspectRatio(contentMode: .fit)
                                .frame(width: 100, height: 100)
                                .cornerRadius(8)
                                .shadow(radius: 2)
                            #else
                            Image(uiImage: platImg)
                                .resizable()
                                .interpolation(.none)
                                .aspectRatio(contentMode: .fit)
                                .frame(width: 100, height: 100)
                                .cornerRadius(8)
                                .shadow(radius: 2)
                            #endif
                        } else {
                            RoundedRectangle(cornerRadius: 8)
                                .fill(Color.gray.opacity(0.2))
                                .frame(width: 100, height: 100)
                                .overlay(Image(systemName: "photo").foregroundColor(.secondary))
                        }
                        
                        Text(item.label)
                            .font(.caption)
                            .fontWeight(.medium)
                            .lineLimit(2)
                            .multilineTextAlignment(.center)
                            .foregroundColor(.primary)
                    }
                    .padding(8)
                    .background(Color.primary.opacity(0.03))
                    .cornerRadius(12)
                    .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color.primary.opacity(0.05), lineWidth: 1))
                    .onTapGesture {
                        selectedImage = item
                    }
                }
            }
            .padding(.top, 4)
        }
        .sheet(item: $selectedImage) { item in
            LightboxView(item: item)
        }
    }

    private func platformImage(from base64String: String) -> PlatformImage? {
        guard let data = Data(base64Encoded: base64String) else { return nil }
        return PlatformImage(data: data)
    }
}
