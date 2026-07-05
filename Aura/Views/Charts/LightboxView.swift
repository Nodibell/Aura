import SwiftUI

struct LightboxView: View {
    let item: ImageItem
    @Environment(\.dismiss) var dismiss

    var body: some View {
        VStack(spacing: 16) {
            HStack {
                Text(item.label)
                    .font(.headline)
                Spacer()
                Button("Close") {
                    dismiss()
                }
                .keyboardShortcut(.cancelAction)
            }
            .padding([.top, .horizontal])

            Divider()

            Spacer()

            if let data = Data(base64Encoded: item.base64),
               let platImg = PlatformImage(data: data) {
                #if canImport(AppKit)
                Image(nsImage: platImg)
                    .resizable()
                    .interpolation(.none)
                    .aspectRatio(contentMode: .fit)
                    .frame(maxWidth: 400, maxHeight: 400)
                    .cornerRadius(8)
                    .shadow(radius: 4)
                #else
                Image(uiImage: platImg)
                    .resizable()
                    .interpolation(.none)
                    .aspectRatio(contentMode: .fit)
                    .frame(maxWidth: 400, maxHeight: 400)
                    .cornerRadius(8)
                    .shadow(radius: 4)
                #endif
            } else {
                Image(systemName: "exclamationmark.triangle")
                    .font(.largeTitle)
                    .foregroundColor(.red)
                Text("Failed to decode image data.")
            }

            Spacer()
        }
        .frame(minWidth: 450, minHeight: 480)
        .padding(.bottom)
    }
}
