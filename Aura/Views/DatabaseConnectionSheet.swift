import SwiftUI
import UniformTypeIdentifiers

struct DatabaseConnectionSheet: View {
    @Binding var isPresented: Bool
    let onImportSuccess: (String, Int, [String]) -> Void // CSV path, row count, columns
    
    @State private var connectionName: String = "MyConnection"
    @State private var dbType: DBType = .sqlite
    
    // SQLite
    @State private var sqlitePath: String = ""
    
    // PostgreSQL / MySQL
    @State private var host: String = "localhost"
    @State private var port: String = ""
    @State private var database: String = ""
    @State private var user: String = ""
    @State private var password: String = ""
    
    // BigQuery
    @State private var bqProject: String = ""
    @State private var bqCredsPath: String = ""
    
    // Query
    @State private var sqlQuery: String = "SELECT * FROM my_table LIMIT 100"
    
    // Status
    @State private var isExecuting = false
    @State private var statusMessage: String? = nil
    @State private var isError = false
    
    enum DBType: String, CaseIterable, Identifiable {
        case sqlite = "SQLite"
        case postgresql = "PostgreSQL"
        case mysql = "MySQL"
        case bigquery = "Google BigQuery"
        
        var id: String { self.rawValue }
        
        var defaultPort: String {
            switch self {
            case .postgresql: return "5432"
            case .mysql: return "3306"
            default: return ""
            }
        }
    }
    
    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Database Ingestion")
                        .font(.title2)
                        .fontWeight(.bold)
                    Text("Query and import datasets directly into Aura")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                Spacer()
                
                Button(action: { isPresented = false }) {
                    Image(systemName: "xmark.circle.fill")
                        .font(.title3)
                        .foregroundColor(.secondary)
                }
                .buttonStyle(.plain)
            }
            .padding()
            .background(Color.primary.opacity(0.02))
            
            Divider()
            
            ScrollView {
                VStack(spacing: 20) {
                    // Connection Profile Name
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Connection Profile Name")
                            .font(.caption)
                            .fontWeight(.semibold)
                            .foregroundColor(.secondary)
                        TextField("e.g. Production_Postgres", text: $connectionName)
                            .textFieldStyle(.roundedBorder)
                            .onChange(of: connectionName) {
                                loadConnectionFromKeychain()
                            }
                    }
                    
                    // Database Type Picker
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Database Engine")
                            .font(.caption)
                            .fontWeight(.semibold)
                            .foregroundColor(.secondary)
                        
                        Picker("", selection: $dbType) {
                            ForEach(DBType.allCases) { type in
                                Text(type.rawValue).tag(type)
                            }
                        }
                        .pickerStyle(.segmented)
                        .onChange(of: dbType) { oldValue, newValue in
                            port = newValue.defaultPort
                            loadConnectionFromKeychain()
                        }
                    }
                    
                    // Connection Config Panel
                    VStack(spacing: 12) {
                        switch dbType {
                        case .sqlite:
                            sqliteFields
                        case .postgresql, .mysql:
                            remoteDBFields
                        case .bigquery:
                            bigQueryFields
                        }
                    }
                    .padding()
                    .background(Color(nsColor: .controlBackgroundColor))
                    .cornerRadius(12)
                    .overlay(
                        RoundedRectangle(cornerRadius: 12)
                            .stroke(Color.primary.opacity(0.07), lineWidth: 1)
                    )
                    .shadow(color: Color.black.opacity(0.04), radius: 6, x: 0, y: 2)
                    
                    // SQL Query TextEditor
                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Text("SQL Query")
                                .font(.caption)
                                .fontWeight(.semibold)
                                .foregroundColor(.secondary)
                            Spacer()
                            Text("Standard SQL Syntax")
                                .font(.caption2)
                                .foregroundColor(.secondary.opacity(0.6))
                        }
                        
                        TextEditor(text: $sqlQuery)
                            .font(.system(.body, design: .monospaced))
                            .frame(height: 120)
                            .padding(4)
                            .background(Color(nsColor: .textBackgroundColor))
                            .cornerRadius(8)
                            .overlay(
                                RoundedRectangle(cornerRadius: 8)
                                    .stroke(Color.primary.opacity(0.08), lineWidth: 1)
                            )
                    }
                    
                    // Status Output
                    if let message = statusMessage {
                        HStack(spacing: 10) {
                            Image(systemName: isError ? "exclamationmark.triangle.fill" : "checkmark.circle.fill")
                                .foregroundColor(isError ? .red : .green)
                            Text(message)
                                .font(.callout)
                                .foregroundColor(isError ? .red : .green)
                            Spacer()
                        }
                        .padding()
                        .background(isError ? Color.red.opacity(0.1) : Color.green.opacity(0.1))
                        .cornerRadius(8)
                    }
                }
                .padding()
            }
            
            Divider()
            
            // Action Buttons
            HStack(spacing: 16) {
                Button(action: saveConnectionToKeychain) {
                    HStack(spacing: 6) {
                        Image(systemName: "key.fill")
                        Text("Save Credentials")
                    }
                }
                .buttonStyle(.bordered)
                .disabled(connectionName.trimmingCharacters(in: .whitespaces).isEmpty)
                
                Spacer()
                
                if isExecuting {
                    ProgressView()
                        .scaleEffect(0.8)
                        .padding(.horizontal, 10)
                } else {
                    Button(action: testQuery) {
                        Text("Test Query")
                    }
                    .buttonStyle(.bordered)
                    
                    Button(action: runImport) {
                        Text("Connect & Import")
                            .fontWeight(.semibold)
                            .foregroundColor(.white)
                            .padding(.horizontal, 16)
                            .padding(.vertical, 8)
                            .background(
                                LinearGradient(colors: [.purple, .indigo], startPoint: .leading, endPoint: .trailing)
                            )
                            .cornerRadius(8)
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding()
            .background(Color(nsColor: .windowBackgroundColor))
        }
        .frame(width: 580, height: 600)
        
        .onAppear {
            port = dbType.defaultPort
            loadConnectionFromKeychain()
        }
    }
    
    // MARK: - Subfields
    
    private var sqliteFields: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("SQLite Database File Path")
                .font(.caption)
                .fontWeight(.medium)
                .foregroundColor(.secondary)
            
            HStack {
                TextField("/path/to/database.db", text: $sqlitePath)
                    .textFieldStyle(.roundedBorder)
                
                Button("Browse...") {
                    let panel = NSOpenPanel()
                    panel.allowsMultipleSelection = false
                    panel.canChooseDirectories = false
                    panel.canChooseFiles = true
                    panel.allowedContentTypes = [UTType(filenameExtension: "db") ?? .data, UTType(filenameExtension: "sqlite") ?? .data, .database]
                    if panel.runModal() == .OK, let url = panel.url {
                        sqlitePath = url.path
                    }
                }
            }
        }
    }
    
    private var remoteDBFields: some View {
        VStack(spacing: 12) {
            HStack(spacing: 12) {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Host")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    TextField("127.0.0.1", text: $host)
                        .textFieldStyle(.roundedBorder)
                }
                
                VStack(alignment: .leading, spacing: 6) {
                    Text("Port")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    TextField(dbType.defaultPort, text: $port)
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 80)
                }
            }
            
            VStack(alignment: .leading, spacing: 6) {
                Text("Database Name")
                    .font(.caption)
                    .foregroundColor(.secondary)
                TextField("my_database", text: $database)
                    .textFieldStyle(.roundedBorder)
            }
            
            HStack(spacing: 12) {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Username")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    TextField("postgres", text: $user)
                        .textFieldStyle(.roundedBorder)
                }
                
                VStack(alignment: .leading, spacing: 6) {
                    Text("Password")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    SecureField("Required", text: $password)
                        .textFieldStyle(.roundedBorder)
                }
            }
        }
    }
    
    private var bigQueryFields: some View {
        VStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 6) {
                Text("Google Cloud Project ID")
                    .font(.caption)
                    .foregroundColor(.secondary)
                TextField("my-gcp-project", text: $bqProject)
                    .textFieldStyle(.roundedBorder)
            }
            
            VStack(alignment: .leading, spacing: 6) {
                Text("Credentials JSON File Path (Optional)")
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                HStack {
                    TextField("/path/to/service-account.json", text: $bqCredsPath)
                        .textFieldStyle(.roundedBorder)
                    
                    Button("Browse...") {
                        let panel = NSOpenPanel()
                        panel.allowsMultipleSelection = false
                        panel.canChooseDirectories = false
                        panel.canChooseFiles = true
                        panel.allowedContentTypes = [.json]
                        if panel.runModal() == .OK, let url = panel.url {
                            bqCredsPath = url.path
                        }
                    }
                }
            }
        }
    }
    
    // MARK: - Logic & Operations
    
    private func buildConnParams() -> [String: String] {
        var params: [String: String] = [:]
        switch dbType {
        case .sqlite:
            params["db_path"] = sqlitePath
        case .postgresql, .mysql:
            params["host"] = host
            params["port"] = port
            params["database"] = database
            params["user"] = user
            params["password"] = password
        case .bigquery:
            params["project"] = bqProject
            if !bqCredsPath.isEmpty {
                params["credentials_path"] = bqCredsPath
            }
        }
        return params
    }
    
    private func saveConnectionToKeychain() {
        let prefix = "Aura_DB_\(connectionName)_\(dbType.rawValue)_"
        let keychain = KeychainService.shared
        
        switch dbType {
        case .sqlite:
            _ = keychain.save(sqlitePath, forKey: prefix + "db_path")
        case .postgresql, .mysql:
            _ = keychain.save(host, forKey: prefix + "host")
            _ = keychain.save(port, forKey: prefix + "port")
            _ = keychain.save(database, forKey: prefix + "database")
            _ = keychain.save(user, forKey: prefix + "user")
            if !password.isEmpty {
                _ = keychain.save(password, forKey: prefix + "password")
            }
        case .bigquery:
            _ = keychain.save(bqProject, forKey: prefix + "project")
            _ = keychain.save(bqCredsPath, forKey: prefix + "credentials_path")
        }
        
        // Save database engine type as metadata in UserDefaults
        UserDefaults.standard.set(dbType.rawValue, forKey: "Aura_DB_Engine_\(connectionName)")
        
        isError = false
        statusMessage = "Connection details saved successfully in Keychain."
    }
    
    private func loadConnectionFromKeychain() {
        let prefix = "Aura_DB_\(connectionName)_\(dbType.rawValue)_"
        let keychain = KeychainService.shared
        
        // Reset current values
        sqlitePath = ""
        host = "localhost"
        port = dbType.defaultPort
        database = ""
        user = ""
        password = ""
        bqProject = ""
        bqCredsPath = ""
        
        switch dbType {
        case .sqlite:
            sqlitePath = keychain.load(forKey: prefix + "db_path") ?? ""
        case .postgresql, .mysql:
            host = keychain.load(forKey: prefix + "host") ?? "localhost"
            port = keychain.load(forKey: prefix + "port") ?? dbType.defaultPort
            database = keychain.load(forKey: prefix + "database") ?? ""
            user = keychain.load(forKey: prefix + "user") ?? ""
            password = keychain.load(forKey: prefix + "password") ?? ""
        case .bigquery:
            bqProject = keychain.load(forKey: prefix + "project") ?? ""
            bqCredsPath = keychain.load(forKey: prefix + "credentials_path") ?? ""
        }
    }
    
    private func testQuery() {
        isExecuting = true
        statusMessage = "Testing connection and query..."
        isError = false
        
        let typeStr = getPythonDBType()
        let params = buildConnParams()
        let query = sqlQuery
        
        // Temporary CSV path just for testing
        let tempCSV = NSTemporaryDirectory() + "aura_test_query.csv"
        
        Task {
            do {
                let (rowCount, columns) = try await PythonRunner.shared.runDatabaseQuery(
                    dbType: typeStr,
                    query: query,
                    connParams: params,
                    outputCSVPath: tempCSV
                )
                
                // Clean up temp file
                try? FileManager.default.removeItem(atPath: tempCSV)
                
                await MainActor.run {
                    self.isExecuting = false
                    self.isError = false
                    self.statusMessage = "Success! Test query returned \(rowCount) rows, \(columns.count) columns."
                }
            } catch {
                await MainActor.run {
                    self.isExecuting = false
                    self.isError = true
                    self.statusMessage = "Error: \(error.localizedDescription)"
                }
            }
        }
    }
    
    private func runImport() {
        isExecuting = true
        statusMessage = "Running query and exporting..."
        isError = false
        
        let typeStr = getPythonDBType()
        let params = buildConnParams()
        let query = sqlQuery
        
        // Create an output path in the Application Support directory so it's persisted for analysis/EDAs
        let fileManager = FileManager.default
        let appSupportDirs = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask)
        guard let appSupportURL = appSupportDirs.first else {
            isExecuting = false
            isError = true
            statusMessage = "Error: Could not locate Application Support directory."
            return
        }
        
        let auraDir = appSupportURL.appendingPathComponent("Aura", isDirectory: true)
        try? fileManager.createDirectory(at: auraDir, withIntermediateDirectories: true, attributes: nil)
        
        let outputCSV = auraDir.appendingPathComponent("query_export_\(connectionName).csv").path
        
        Task {
            do {
                let (rowCount, columns) = try await PythonRunner.shared.runDatabaseQuery(
                    dbType: typeStr,
                    query: query,
                    connParams: params,
                    outputCSVPath: outputCSV
                )
                
                await MainActor.run {
                    self.isExecuting = false
                    self.isError = false
                    self.isPresented = false
                    self.onImportSuccess(outputCSV, rowCount, columns)
                }
            } catch {
                await MainActor.run {
                    self.isExecuting = false
                    self.isError = true
                    self.statusMessage = "Error: \(error.localizedDescription)"
                }
            }
        }
    }
    
    private func getPythonDBType() -> String {
        switch dbType {
        case .sqlite: return "sqlite"
        case .postgresql: return "postgresql"
        case .mysql: return "mysql"
        case .bigquery: return "bigquery"
        }
    }
}
