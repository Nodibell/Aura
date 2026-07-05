import SwiftUI

#if canImport(AppKit)
import AppKit
typealias PlatformImage = NSImage
#elseif canImport(UIKit)
import UIKit
typealias PlatformImage = UIImage
#endif

struct ImageGridView: View {
    let config: ChartConfig
    @State private var selectedImage: ImageItem? = nil

    // NEW: decoded-image cache, keyed by ImageItem.id.
    // Populated once per chart (see .task below) instead of decoding
    // base64 -> Data -> NSImage inline in `body` on every re-render.
    @State private var decodedImages: [String: PlatformImage] = [:]

    private let columns = [
        GridItem(.adaptive(minimum: 120, maximum: 160), spacing: 16)
    ]

    var body: some View {
        let items = config.images ?? []
        ScrollView {
            LazyVGrid(columns: columns, spacing: 16) {
                ForEach(items) { item in
                    imageCell(for: item)
                }
            }
            .padding(.top, 4)
        }
        .sheet(item: $selectedImage) { item in
            LightboxView(item: item)
        }
        // Re-decodes only when this chart's identity actually changes,
        // not on every SwiftUI body re-evaluation (hover, animation, etc).
        .task(id: config.id) {
            await decodeImagesIfNeeded(items)
        }
    }

    @ViewBuilder
    private func imageCell(for item: ImageItem) -> some View {
        VStack(alignment: .center, spacing: 8) {
            if let platImg = decodedImages[item.id] {
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
                    .overlay(ProgressView().controlSize(.small))
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

    /// Decodes all base64 images for this chart off the main actor, then
    /// publishes the finished dictionary once. Avoids blocking the UI thread
    /// with repeated Data(base64Encoded:) + PlatformImage(data:) calls.
    private func decodeImagesIfNeeded(_ items: [ImageItem]) async {
        guard !items.isEmpty else { return }

        let decoded: [(String, PlatformImage)] = await Task.detached(priority: .userInitiated) {
            var result: [(String, PlatformImage)] = []
            result.reserveCapacity(items.count)
            for item in items {
                guard let data = Data(base64Encoded: item.base64),
                      let image = PlatformImage(data: data) else { continue }
                result.append((item.id, image))
            }
            return result
        }.value

        var dict: [String: PlatformImage] = [:]
        dict.reserveCapacity(decoded.count)
        for (id, image) in decoded { dict[id] = image }

        await MainActor.run {
            self.decodedImages = dict
        }
    }
}
