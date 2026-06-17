import SwiftUI

struct OllamaSetupView: View {
    let onCheckAgain: () async -> Void
    @State private var isChecking = false

    var body: some View {
        VStack(spacing: 20) {
            Spacer()

            // Icon
            ZStack {
                Circle()
                    .fill(
                        LinearGradient(colors: [Color.orange.opacity(0.15), Color.red.opacity(0.08)],
                                       startPoint: .topLeading, endPoint: .bottomTrailing)
                    )
                    .frame(width: 72, height: 72)
                Image(systemName: "brain.head.profile")
                    .font(.system(size: 32))
                    .foregroundStyle(
                        LinearGradient(colors: [.orange, .red], startPoint: .topLeading, endPoint: .bottomTrailing)
                    )
            }

            VStack(spacing: 8) {
                Text("Ollama Not Running")
                    .font(.title3)
                    .fontWeight(.bold)

                Text("The AI Analyst requires Ollama to be installed and running locally. All analysis stays on your Mac — no data leaves your computer.")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
            }

            // Install instructions
            VStack(alignment: .leading, spacing: 10) {
                Text("Quick Setup")
                    .font(.caption)
                    .fontWeight(.bold)
                    .foregroundColor(.secondary)

                stepRow(number: "1", text: "Install Ollama", command: "brew install ollama")
                stepRow(number: "2", text: "Start the server", command: "ollama serve")
                stepRow(number: "3", text: "Pull a model", command: "ollama pull llama3.2")
            }
            .padding(16)
            .background(Color.white.opacity(0.03))
            .cornerRadius(12)
            .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color.white.opacity(0.08)))

            HStack(spacing: 12) {
                Button {
                    if let url = URL(string: "https://ollama.com") {
                        NSWorkspace.shared.open(url)
                    }
                } label: {
                    Label("ollama.com", systemImage: "arrow.up.right.square")
                        .font(.caption)
                }
                .buttonStyle(.bordered)

                Button {
                    isChecking = true
                    Task {
                        await onCheckAgain()
                        await MainActor.run { isChecking = false }
                    }
                } label: {
                    if isChecking {
                        HStack(spacing: 6) {
                            ProgressView().controlSize(.mini)
                            Text("Checking...")
                        }
                    } else {
                        Label("Check Again", systemImage: "arrow.clockwise")
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(isChecking)
            }

            Spacer()
        }
        .padding(20)
    }

    private func stepRow(number: String, text: String, command: String) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Text(number)
                .font(.system(size: 10, weight: .bold))
                .foregroundColor(.white)
                .frame(width: 18, height: 18)
                .background(Circle().fill(Color.orange.opacity(0.7)))

            VStack(alignment: .leading, spacing: 2) {
                Text(text)
                    .font(.caption)
                    .fontWeight(.semibold)
                Text(command)
                    .font(.system(.caption2, design: .monospaced))
                    .foregroundColor(.secondary)
                    .textSelection(.enabled)
            }
        }
    }
}
