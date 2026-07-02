import Foundation

// MARK: - Ollama Model Info

struct OllamaTagsResponse: Codable, Sendable {
    let models: [OllamaModelInfo]
}

struct OllamaModelInfo: Codable, Identifiable, Equatable, Sendable {
    var id: String { name }
    let name: String
    let size: Int?
}

// MARK: - Ollama Chat Request/Response

struct OllamaChatRequest: Encodable, Sendable {
    let model: String
    let messages: [OllamaChatMessage]
    let stream: Bool
    let options: OllamaOptions?
}

struct OllamaChatMessage: Encodable, Sendable {
    let role: String
    let content: String
}

struct OllamaOptions: Encodable, Sendable {
    let temperature: Double
    let num_predict: Int
}

struct OllamaChatChunk: Decodable, Sendable {
    let message: OllamaMessageChunk?
    let done: Bool
}

struct OllamaMessageChunk: Decodable, Sendable {
    let content: String
}

// MARK: - Ollama Service

final class OllamaService: Sendable {
    static let shared = OllamaService()
    
    private var baseURL: String {
        let saved = UserDefaults.standard.string(forKey: "Aura_OllamaBaseURL") ?? ""
        return saved.isEmpty ? "http://localhost:11434" : saved
    }

    // Check if Ollama is running
    func checkAvailability() async -> Bool {
        guard let url = URL(string: "\(baseURL)/api/tags") else { return false }
        do {
            var request = URLRequest(url: url)
            request.timeoutInterval = 60
            let (_, response) = try await URLSession.shared.data(for: request)
            let isAvailable = (response as? HTTPURLResponse)?.statusCode == 200
            await AppLogger.shared.info("Ollama availability checked. Active: \(isAvailable)", category: "Ollama")
            return isAvailable
        } catch {
            await AppLogger.shared.warning("Ollama availability check failed: \(error.localizedDescription)", category: "Ollama")
            return false
        }
    }

    // Fetch all installed models
    func listModels() async -> [OllamaModelInfo] {
        guard let url = URL(string: "\(baseURL)/api/tags") else { return [] }
        do {
            var request = URLRequest(url: url)
            request.timeoutInterval = 60
            let (data, _) = try await URLSession.shared.data(for: request)
            let decoded = try JSONDecoder().decode(OllamaTagsResponse.self, from: data)
            await AppLogger.shared.info("Fetched \(decoded.models.count) installed models from Ollama.", category: "Ollama")
            return decoded.models
        } catch {
            await AppLogger.shared.error("Failed to fetch Ollama models: \(error.localizedDescription)", category: "Ollama")
            return []
        }
    }

    // Stream a chat response — returns AsyncThrowingStream<String, Error>
    func streamChat(
        messages: [OllamaChatMessage],
        systemPrompt: String,
        model: String,
        temperature: Double = 0.3,
        maxTokens: Int = 2048
    ) -> AsyncThrowingStream<String, Error> {
        let baseURL = self.baseURL
        return AsyncThrowingStream { continuation in
            Task.detached { @Sendable in
                await AppLogger.shared.info("Initiating Ollama chat stream (Model: \(model), Temp: \(temperature))", category: "Ollama")
                guard let url = URL(string: "\(baseURL)/api/chat") else {
                    await AppLogger.shared.error("Ollama chat failed: Invalid URL", category: "Ollama")
                    continuation.finish(throwing: URLError(.badURL))
                    return
                }

                var request = URLRequest(url: url)
                request.httpMethod = "POST"
                request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                request.timeoutInterval = 60

                var finalMessages = messages
                let trimmedSystem = systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmedSystem.isEmpty {
                    finalMessages.insert(OllamaChatMessage(role: "system", content: trimmedSystem), at: 0)
                }

                let body = OllamaChatRequest(
                    model: model,
                    messages: finalMessages,
                    stream: true,
                    options: OllamaOptions(temperature: temperature, num_predict: maxTokens)
                )
                request.httpBody = try? JSONEncoder().encode(body)

                do {
                    let (asyncBytes, response) = try await URLSession.shared.bytes(for: request)
                    guard (response as? HTTPURLResponse)?.statusCode == 200 else {
                        await AppLogger.shared.error("Ollama chat request failed with status code \((response as? HTTPURLResponse)?.statusCode ?? 0)", category: "Ollama")
                        continuation.finish(throwing: URLError(.badServerResponse))
                        return
                    }

                    for try await line in asyncBytes.lines {
                        guard !line.isEmpty,
                               let data = line.data(using: .utf8) else { continue }
                        do {
                            let chunk = try JSONDecoder().decode(OllamaChatChunk.self, from: data)
                            if let content = chunk.message?.content, !content.isEmpty {
                                continuation.yield(content)
                            }
                            if chunk.done {
                                await AppLogger.shared.info("Ollama chat stream completed successfully", category: "Ollama")
                                continuation.finish()
                                return
                            }
                        } catch {
                            // Skip malformed lines silently
                        }
                    }
                    continuation.finish()
                } catch {
                    await AppLogger.shared.error("Ollama chat stream threw error: \(error.localizedDescription)", category: "Ollama")
                    continuation.finish(throwing: error)
                }
            }
        }
    }
}
