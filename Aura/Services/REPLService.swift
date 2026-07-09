import Foundation

// MARK: - REPL Result Model

struct REPLResult: Decodable {
    let stdout: String
    let error: String?
    let figures: [String]   // base64-encoded PNG strings
}

// MARK: - REPLService

/// Lightweight Swift actor that calls the FastAPI /repl/exec and /repl/reset
/// endpoints backed by utils/repl_session.py.
///
/// The REPL holds a persistent Python namespace with `df` already loaded,
/// so code executed here can immediately inspect and mutate the dataset.
actor REPLService: REPLServiceProtocol {

    static let shared = REPLService()

    private let port: Int
    private var baseURL: URL { URL(string: "http://127.0.0.1:\(port)")! }
    
    // Cache parameters to enable self-healing reload
    private var activeFilePath: String?
    private var activeOllamaBaseURL: String = "http://localhost:11434"
    private var activeOllamaModel: String = "llama3.2"
    private var activeCleaningActions: String? = nil

    private init(port: Int = 11435) {
        self.port = port
    }

    // MARK: - Reset (load new dataset into REPL namespace)

    /// Call this whenever a new file is loaded.  Sets `df` in the Python REPL.
    func reset(filePath: String,
               ollamaBaseURL: String = "http://localhost:11434",
               ollamaModel: String = "llama3.2",
               cleaningActions: String? = nil) async throws {
        self.activeFilePath = filePath
        self.activeOllamaBaseURL = ollamaBaseURL
        self.activeOllamaModel = ollamaModel
        self.activeCleaningActions = cleaningActions

        let url = baseURL.appendingPathComponent("repl/reset")
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 15

        var body: [String: Any] = [
            "file_path": filePath,
            "ollama_base_url": ollamaBaseURL,
            "ollama_model": ollamaModel
        ]
        if let cleaningActions = cleaningActions {
            body["cleaning_actions"] = cleaningActions
        }
        request.httpBody = try? JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await URLSession.shared.data(for: request)
        if let http = response as? HTTPURLResponse, http.statusCode != 200 {
            let errorMsg = String(data: data, encoding: .utf8) ?? "Unknown server error"
            await AppLogger.shared.error("REPL reset failed: \(errorMsg)", category: "REPLService")
            throw NSError(domain: "REPLService", code: http.statusCode, userInfo: [NSLocalizedDescriptionKey: errorMsg])
        }
    }

    // MARK: - Execute code

    /// Execute Python `code` in the REPL sandbox.  Returns captured stdout,
    /// any error string, and a list of base64-encoded PNG figures.
    func execute(_ code: String) async throws -> REPLResult {
        let result = try await performExecute(code)
        
        // Self-healing: if server restarted or namespace cleared, reload dataset and retry
        if let err = result.error, err.contains("name 'df' is not defined"), let filePath = activeFilePath {
            try? await reset(filePath: filePath, ollamaBaseURL: activeOllamaBaseURL, ollamaModel: activeOllamaModel, cleaningActions: activeCleaningActions)
            return try await performExecute(code)
        }
        
        return result
    }
    
    private func performExecute(_ code: String) async throws -> REPLResult {
        let url = baseURL.appendingPathComponent("repl/exec")
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 60   // code execution can be slow

        let body = ["code": code]
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await URLSession.shared.data(for: request)
        if let http = response as? HTTPURLResponse, http.statusCode != 200 {
            throw URLError(.badServerResponse)
        }
        return try JSONDecoder().decode(REPLResult.self, from: data)
    }

    // MARK: - Lineage and Rollback (Time-Travel)

    struct LineageNode: Decodable, Identifiable, Hashable {
        let id: Int
        let description: String
        let shape: String
    }

    struct RollbackResult: Decodable {
        let status: String
        let activeState: Int
        let rows: Int
        let cols: Int
        
        enum CodingKeys: String, CodingKey {
            case status
            case activeState = "active_state"
            case rows
            case cols
        }
    }

    func getLineage() async throws -> [LineageNode] {
        let url = baseURL.appendingPathComponent("repl/lineage")
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.timeoutInterval = 10

        let (data, response) = try await URLSession.shared.data(for: request)
        if let http = response as? HTTPURLResponse, http.statusCode != 200 {
            throw URLError(.badServerResponse)
        }
        return try JSONDecoder().decode([LineageNode].self, from: data)
    }

    func rollback(stateId: Int) async throws -> RollbackResult {
        let url = baseURL.appendingPathComponent("repl/rollback")
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 15

        let body = ["state_id": stateId]
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await URLSession.shared.data(for: request)
        if let http = response as? HTTPURLResponse, http.statusCode != 200 {
            throw URLError(.badServerResponse)
        }
        return try JSONDecoder().decode(RollbackResult.self, from: data)
    }

    // MARK: - Plugins

    struct PluginParameter: Decodable, Hashable {
        let name: String
        let type: String // "slider" or "toggle"
        let min: Double?
        let max: Double?
        let `default`: Double
    }

    struct PluginInfo: Decodable, Identifiable, Hashable {
        let id: String
        let name: String
        let description: String
        let parameters: [PluginParameter]
    }

    func getPlugins() async throws -> [PluginInfo] {
        let url = baseURL.appendingPathComponent("plugins")
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.timeoutInterval = 10

        let (data, response) = try await URLSession.shared.data(for: request)
        if let http = response as? HTTPURLResponse, http.statusCode != 200 {
            throw URLError(.badServerResponse)
        }
        return try JSONDecoder().decode([PluginInfo].self, from: data)
    }
}


