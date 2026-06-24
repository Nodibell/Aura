import Foundation

enum LLMProvider: String, CaseIterable, Codable, Sendable {
    case ollama = "Ollama"
    case openAI = "OpenAI"
    case claude = "Claude"
}

enum AIError: LocalizedError {
    case missingAPIKey(provider: String)
    case invalidResponse(statusCode: Int)
    
    var errorDescription: String? {
        switch self {
        case .missingAPIKey(let provider):
            return "Missing API key for \(provider). Please configure it in Settings."
        case .invalidResponse(let code):
            return "Server returned an error status code: \(code)."
        }
    }
}

// MARK: - OpenAI Decodables
private struct OpenAIStreamResponse: Codable {
    struct Choice: Codable {
        struct Delta: Codable {
            let content: String?
        }
        let delta: Delta
    }
    let choices: [Choice]
}

// MARK: - Claude Decodables
private struct ClaudeStreamResponse: Codable {
    struct Delta: Codable {
        let text: String?
    }
    let type: String
    let delta: Delta?
}

final class AIService: Sendable {
    static let shared = AIService()
    
    private init() {}
    
    func streamChat(
        messages: [OllamaChatMessage],
        systemPrompt: String,
        provider: LLMProvider,
        model: String,
        temperature: Double,
        maxTokens: Int
    ) -> AsyncThrowingStream<String, Error> {
        
        switch provider {
        case .ollama:
            return OllamaService.shared.streamChat(
                messages: messages,
                model: model,
                temperature: temperature,
                maxTokens: maxTokens
            )
            
        case .openAI:
            return streamOpenAIChat(
                messages: messages,
                systemPrompt: systemPrompt,
                model: model,
                temperature: temperature,
                maxTokens: maxTokens
            )
            
        case .claude:
            return streamClaudeChat(
                messages: messages,
                systemPrompt: systemPrompt,
                model: model,
                temperature: temperature,
                maxTokens: maxTokens
            )
        }
    }
    
    // MARK: - OpenAI API Integration
    private func streamOpenAIChat(
        messages: [OllamaChatMessage],
        systemPrompt: String,
        model: String,
        temperature: Double,
        maxTokens: Int
    ) -> AsyncThrowingStream<String, Error> {
        return AsyncThrowingStream { continuation in
            Task.detached { @Sendable in
                guard let apiKey = KeychainService.shared.getSecureString(forKey: "Aura_OpenAIKey"),
                      !apiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                    continuation.finish(throwing: AIError.missingAPIKey(provider: "OpenAI"))
                    return
                }
                
                guard let url = URL(string: "https://api.openai.com/v1/chat/completions") else {
                    continuation.finish(throwing: URLError(.badURL))
                    return
                }
                
                var request = URLRequest(url: url)
                request.httpMethod = "POST"
                request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
                request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                request.timeoutInterval = 60
                
                // Construct messages array
                var openAIMessages: [[String: String]] = []
                if !systemPrompt.isEmpty {
                    openAIMessages.append(["role": "system", "content": systemPrompt])
                }
                for msg in messages {
                    openAIMessages.append(["role": msg.role, "content": msg.content])
                }
                
                let body: [String: Any] = [
                    "model": model,
                    "messages": openAIMessages,
                    "temperature": temperature,
                    "max_tokens": maxTokens,
                    "stream": true
                ]
                
                request.httpBody = try? JSONSerialization.data(withJSONObject: body)
                
                do {
                    let (asyncBytes, response) = try await URLSession.shared.bytes(for: request)
                    let statusCode = (response as? HTTPURLResponse)?.statusCode ?? 0
                    guard statusCode == 200 else {
                        continuation.finish(throwing: AIError.invalidResponse(statusCode: statusCode))
                        return
                    }
                    
                    for try await line in asyncBytes.lines {
                        let trimmedLine = line.trimmingCharacters(in: .whitespacesAndNewlines)
                        guard !trimmedLine.isEmpty else { continue }
                        
                        if trimmedLine.hasPrefix("data: ") {
                            let jsonStr = trimmedLine.dropFirst(6).trimmingCharacters(in: .whitespacesAndNewlines)
                            if jsonStr == "[DONE]" {
                                break
                            }
                            
                            if let data = jsonStr.data(using: .utf8),
                               let chunk = try? JSONDecoder().decode(OpenAIStreamResponse.self, from: data),
                               let content = chunk.choices.first?.delta.content {
                                continuation.yield(content)
                            }
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
    
    // MARK: - Claude API Integration
    private func streamClaudeChat(
        messages: [OllamaChatMessage],
        systemPrompt: String,
        model: String,
        temperature: Double,
        maxTokens: Int
    ) -> AsyncThrowingStream<String, Error> {
        return AsyncThrowingStream { continuation in
            Task.detached { @Sendable in
                guard let apiKey = KeychainService.shared.getSecureString(forKey: "Aura_ClaudeKey"),
                      !apiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                    continuation.finish(throwing: AIError.missingAPIKey(provider: "Claude"))
                    return
                }
                
                guard let url = URL(string: "https://api.anthropic.com/v1/messages") else {
                    continuation.finish(throwing: URLError(.badURL))
                    return
                }
                
                var request = URLRequest(url: url)
                request.httpMethod = "POST"
                request.setValue(apiKey, forHTTPHeaderField: "x-api-key")
                request.setValue("2023-06-01", forHTTPHeaderField: "anthropic-version")
                request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                request.timeoutInterval = 60
                
                // Anthropic maps history messages without system prompts
                var claudeMessages: [[String: String]] = []
                for msg in messages {
                    claudeMessages.append(["role": msg.role, "content": msg.content])
                }
                
                var body: [String: Any] = [
                    "model": model,
                    "messages": claudeMessages,
                    "temperature": temperature,
                    "max_tokens": maxTokens,
                    "stream": true
                ]
                if !systemPrompt.isEmpty {
                    body["system"] = systemPrompt
                }
                
                request.httpBody = try? JSONSerialization.data(withJSONObject: body)
                
                do {
                    let (asyncBytes, response) = try await URLSession.shared.bytes(for: request)
                    let statusCode = (response as? HTTPURLResponse)?.statusCode ?? 0
                    guard statusCode == 200 else {
                        continuation.finish(throwing: AIError.invalidResponse(statusCode: statusCode))
                        return
                    }
                    
                    for try await line in asyncBytes.lines {
                        let trimmedLine = line.trimmingCharacters(in: .whitespacesAndNewlines)
                        guard !trimmedLine.isEmpty else { continue }
                        
                        if trimmedLine.hasPrefix("data: ") {
                            let jsonStr = trimmedLine.dropFirst(6).trimmingCharacters(in: .whitespacesAndNewlines)
                            
                            if let data = jsonStr.data(using: .utf8),
                               let chunk = try? JSONDecoder().decode(ClaudeStreamResponse.self, from: data) {
                                if chunk.type == "content_block_delta", let text = chunk.delta?.text {
                                    continuation.yield(text)
                                }
                            }
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
}
