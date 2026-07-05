import SwiftUI

struct AIChatPanel: View {
    @Bindable var viewModel: ChatViewModel
    let ollamaStatus: OllamaStatusChecker
    let analysisResult: AnalysisResult?

    @State private var selectedModel: String = ""
    @State private var scrollProxy: ScrollViewProxy? = nil
    @State private var activeProvider: LLMProvider = .ollama

    // Retrieve settings
    private var temperature: Double {
        UserDefaults.standard.double(forKey: "Aura_OllamaTemp").clamped(to: 0.0...1.0).nonZeroOr(0.3)
    }
    private var maxTokens: Int {
        let v = UserDefaults.standard.integer(forKey: "Aura_OllamaMaxTokens")
        return v > 0 ? v : 2048
    }

    private var isOpenAIKeySet: Bool {
        let key = KeychainService.shared.getSecureString(forKey: "Aura_OpenAIKey") ?? ""
        return !key.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }
    
    private var isClaudeKeySet: Bool {
        let key = KeychainService.shared.getSecureString(forKey: "Aura_ClaudeKey") ?? ""
        return !key.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private var activeProviderIsReady: Bool {
        switch activeProvider {
        case .ollama:
            return ollamaStatus.isAvailable
        case .openAI:
            return isOpenAIKeySet
        case .claude:
            return isClaudeKeySet
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            // Header bar
            panelHeader

            Divider()

            if activeProvider == .ollama {
                if !ollamaStatus.isAvailable {
                    // Ollama not running
                    OllamaSetupView {
                        await ollamaStatus.refresh()
                    }
                } else {
                    chatArea
                }
            } else if activeProvider == .openAI {
                if !isOpenAIKeySet {
                    CloudKeySetupView(providerName: "OpenAI")
                } else {
                    chatArea
                }
            } else if activeProvider == .claude {
                if !isClaudeKeySet {
                    CloudKeySetupView(providerName: "Claude")
                } else {
                    chatArea
                }
            }
        }
        .onAppear {
            let providerStr = UserDefaults.standard.string(forKey: "Aura_LLMProvider") ?? "Ollama"
            activeProvider = LLMProvider(rawValue: providerStr) ?? .ollama
            resolveSelectedModel()
            if let analysisResult {
                viewModel.injectContext(analysisResult)
            }
        }
        .onChange(of: ollamaStatus.availableModels) { _, models in
            resolveSelectedModel()
        }
        .onChange(of: analysisResult) { _, newResult in
            if let newResult {
                viewModel.injectContext(newResult)
            }
        }
        .onReceive(NotificationCenter.default.publisher(for: UserDefaults.didChangeNotification)) { _ in
            let providerStr = UserDefaults.standard.string(forKey: "Aura_LLMProvider") ?? "Ollama"
            let newProvider = LLMProvider(rawValue: providerStr) ?? .ollama
            if activeProvider != newProvider {
                activeProvider = newProvider
                resolveSelectedModel()
            }
        }
    }

    private var chatArea: some View {
        VStack(spacing: 0) {
            // Messages
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(spacing: 12) {
                        if viewModel.messages.isEmpty {
                            emptyState
                                .id("empty")
                        } else {
                            ForEach(viewModel.messages) { message in
                                MessageBubble(message: message)
                                    .id(message.id)
                            }
                        }
                    }
                    .padding(12)
                }
                .onAppear { scrollProxy = proxy }
                .onChange(of: viewModel.messages.count) { _, _ in
                    scrollToBottom(proxy: proxy)
                }
                .onChange(of: viewModel.messages.last?.content) { _, _ in
                    scrollToBottom(proxy: proxy)
                }
            }

            // Quick action chips
            if let result = analysisResult, viewModel.messages.isEmpty {
                quickActionChips(for: result)
            }

            Divider()

            // Input bar
            inputBar
        }
    }

    private func resolveSelectedModel() {
        let providerStr = UserDefaults.standard.string(forKey: "Aura_LLMProvider") ?? "Ollama"
        let provider = LLMProvider(rawValue: providerStr) ?? .ollama
        
        switch provider {
        case .ollama:
            let saved = UserDefaults.standard.string(forKey: "Aura_OllamaModel") ?? ""
            let models = ollamaStatus.availableModels
            if !saved.isEmpty && models.contains(where: { $0.name == saved }) {
                selectedModel = saved
            } else if let first = models.first {
                selectedModel = first.name
                UserDefaults.standard.set(first.name, forKey: "Aura_OllamaModel")
            } else {
                selectedModel = ""
            }
        case .openAI:
            let saved = UserDefaults.standard.string(forKey: "Aura_OpenAIModel") ?? "gpt-4o-mini"
            selectedModel = saved
        case .claude:
            let saved = UserDefaults.standard.string(forKey: "Aura_ClaudeModel") ?? "claude-3-5-haiku-latest"
            selectedModel = saved
        }
    }

    // MARK: - Subviews

    private var panelHeader: some View {
        HStack(spacing: 8) {
            // Model status dot
            Circle()
                .fill(activeProviderIsReady ? Color.green : Color.red)
                .frame(width: 8, height: 8)

            Text("AI Analyst")
                .font(.system(size: 13, weight: .semibold))

            Spacer()

            // Model picker based on active provider
            if activeProvider == .ollama {
                if ollamaStatus.isAvailable && !ollamaStatus.availableModels.isEmpty && !selectedModel.isEmpty {
                    Picker("", selection: $selectedModel) {
                        ForEach(ollamaStatus.availableModels) { m in
                            Text(m.name).tag(m.name)
                        }
                    }
                    .pickerStyle(.menu)
                    .font(.caption)
                    .frame(maxWidth: 130)
                    .onChange(of: selectedModel) { _, v in
                        UserDefaults.standard.set(v, forKey: "Aura_OllamaModel")
                    }
                }
            } else if activeProvider == .openAI {
                Picker("", selection: $selectedModel) {
                    Text("gpt-4o-mini").tag("gpt-4o-mini")
                    Text("gpt-4o").tag("gpt-4o")
                    Text("o1-mini").tag("o1-mini")
                }
                .pickerStyle(.menu)
                .font(.caption)
                .frame(maxWidth: 130)
                .onChange(of: selectedModel) { _, v in
                    UserDefaults.standard.set(v, forKey: "Aura_OpenAIModel")
                }
            } else if activeProvider == .claude {
                Picker("", selection: $selectedModel) {
                    Text("claude-3-5-haiku-latest").tag("claude-3-5-haiku-latest")
                    Text("claude-3-5-sonnet-latest").tag("claude-3-5-sonnet-latest")
                }
                .pickerStyle(.menu)
                .font(.caption)
                .frame(maxWidth: 130)
                .onChange(of: selectedModel) { _, v in
                    UserDefaults.standard.set(v, forKey: "Aura_ClaudeModel")
                }
            }

            if !viewModel.messages.isEmpty {
                Button {
                    withAnimation(.easeInOut(duration: 0.2)) {
                        viewModel.clearConversation()
                    }
                } label: {
                    Image(systemName: "trash")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .buttonStyle(.plain)
                .help("Clear conversation")
                .keyboardShortcut("k", modifiers: .command)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
    }

    private var emptyState: some View {
        VStack(spacing: 10) {
            Image(systemName: "sparkles")
                .font(.system(size: 28))
                .foregroundStyle(
                    LinearGradient(colors: [.purple, .blue], startPoint: .topLeading, endPoint: .bottomTrailing)
                )
            Text("Ask anything about your data")
                .font(.subheadline)
                .fontWeight(.medium)
            Text("Use the quick actions below or type a custom question.")
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(.top, 40)
    }

    private func quickActionChips(for result: AnalysisResult) -> some View {
        let actions = QuickAction.actionsFor(result: result)
        return ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(actions) { action in
                    Button {
                        send(action.prompt)
                    } label: {
                        HStack(spacing: 4) {
                            Text(action.emoji)
                            Text(action.label)
                                .font(.caption)
                                .fontWeight(.medium)
                        }
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(Color.primary.opacity(0.05))
                        .cornerRadius(20)
                        .overlay(
                            RoundedRectangle(cornerRadius: 20)
                                .stroke(Color.primary.opacity(0.12), lineWidth: 1)
                        )
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
        }
    }

    private var inputBar: some View {
        HStack(spacing: 8) {
            TextField("Ask about this dataset...", text: $viewModel.inputText, axis: .vertical)
                .textFieldStyle(.plain)
                .font(.body)
                .lineLimit(1...4)
                .onSubmit { if !viewModel.isStreaming { send(viewModel.inputText) } }

            if viewModel.isStreaming {
                Button {
                    viewModel.cancelGeneration()
                } label: {
                    Image(systemName: "stop.fill")
                        .font(.system(size: 10, weight: .bold))
                        .foregroundColor(.white)
                        .padding(8)
                        .background(Color.red)
                        .clipShape(Circle())
                }
                .buttonStyle(.plain)
            } else {
                Button {
                    send(viewModel.inputText)
                } label: {
                    Image(systemName: "arrow.up")
                        .font(.system(size: 11, weight: .bold))
                        .foregroundColor(.white)
                        .padding(8)
                        .background(
                            viewModel.inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                                ? AnyShapeStyle(Color.secondary.opacity(0.3))
                                : AnyShapeStyle(LinearGradient(colors: [.purple, .blue], startPoint: .leading, endPoint: .trailing))
                        )
                        .clipShape(Circle())
                }
                .buttonStyle(.plain)
                .disabled(viewModel.inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(Color.primary.opacity(0.03))
        .cornerRadius(20)
        .padding(10)
    }


    // MARK: - Helpers

    private func send(_ text: String) {
        let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !t.isEmpty else { return }
        viewModel.sendMessage(
            t,
            model: selectedModel.isEmpty ? "llama3.2" : selectedModel,
            temperature: temperature,
            maxTokens: maxTokens
        )
    }

    private func scrollToBottom(proxy: ScrollViewProxy) {
        guard let last = viewModel.messages.last else { return }
        withAnimation(.easeOut(duration: 0.15)) {
            proxy.scrollTo(last.id, anchor: .bottom)
        }
    }
}

// MARK: - Message Bubble

private struct MessageBubble: View {
    let message: ChatMessage
    @State private var isExpanded = false

    var body: some View {
        if message.role == .tool {
            toolBubble
        } else {
            HStack(alignment: .top, spacing: 8) {
                if message.role == .user { Spacer(minLength: 40) }

                VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 4) {
                    bubbleContent
                }

                if message.role == .assistant { Spacer(minLength: 40) }
            }
        }
    }

    // MARK: - Tool (REPL) bubble

    @ViewBuilder
    private var toolBubble: some View {
        VStack(spacing: 6) {
            Button(action: { withAnimation(.spring(duration: 0.25)) { isExpanded.toggle() } }) {
                HStack(spacing: 6) {
                    if message.state == .executingCode {
                        ProgressView().scaleEffect(0.6)
                    } else {
                        Image(systemName: message.state == .error ? "exclamationmark.triangle" : "checkmark.circle")
                            .font(.system(size: 11, weight: .medium))
                            .foregroundColor(message.state == .error ? .orange : .green)
                    }
                    Text(message.state == .executingCode ? "Running Python…" : "⚙ Code executed")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundColor(.secondary)
                    Spacer()
                    Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
                        .font(.system(size: 10))
                        .foregroundColor(.secondary.opacity(0.6))
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(Color.primary.opacity(0.05))
                .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .stroke(Color.primary.opacity(0.1), lineWidth: 1)
                )
            }
            .buttonStyle(.plain)

            if isExpanded {
                VStack(alignment: .leading, spacing: 8) {
                    // Execution output text
                    Text(message.content)
                        .font(.system(.caption, design: .monospaced))
                        .foregroundColor(.secondary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(8)
                        .background(Color.primary.opacity(0.04))
                        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))

                    // Inline figures
                    ForEach(message.figures.indices, id: \.self) { idx in
                        if let data = Data(base64Encoded: message.figures[idx]),
                           let nsImg = NSImage(data: data) {
                            Image(nsImage: nsImg)
                                .resizable()
                                .scaledToFit()
                                .frame(maxHeight: 320)
                                .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                        }
                    }
                }
                .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .padding(.horizontal, 8)
    }

    @ViewBuilder
    private var bubbleContent: some View {
        if message.role == .user {
            Text(message.content)
                .font(.body)
                .foregroundColor(.white)
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(
                    LinearGradient(colors: [.purple, .blue.opacity(0.9)],
                                   startPoint: .topLeading, endPoint: .bottomTrailing)
                )
                .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
        } else {
            Group {
                if message.state == .streaming && message.content.isEmpty {
                    // Thinking indicator
                    ThinkingDotsView()
                } else {
                    ZStack(alignment: .topTrailing) {
                        VStack(alignment: .leading, spacing: 6) {
                            MarkdownMessageView(content: message.formattedContent.isEmpty ? " " : message.formattedContent)
                                .foregroundColor(message.state == .error ? .red : .primary)
                                .padding(.trailing, 24)

                            if message.state == .streaming {
                                // Streaming cursor
                                Text("▌")
                                    .font(.body)
                                    .foregroundColor(.secondary)
                                    .opacity(0.7)
                            }
                        }
                        
                        if !message.content.isEmpty && message.state != .streaming {
                            Button(action: {
                                NSPasteboard.general.clearContents()
                                NSPasteboard.general.setString(message.content, forType: .string)
                            }) {
                                Image(systemName: "doc.on.doc")
                                    .font(.system(size: 10))
                                    .foregroundColor(.secondary.opacity(0.7))
                            }
                            .buttonStyle(.plain)
                            .padding(.top, -2)
                            .padding(.trailing, -4)
                            .help("Copy message")
                        }
                    }
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(Color.primary.opacity(0.04))
            .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .stroke(message.state == .error ? Color.red.opacity(0.3) : Color.primary.opacity(0.07), lineWidth: 1)
            )
        }
    }
}

// MARK: - Thinking Dots

private struct ThinkingDotsView: View {
    @State private var phase = 0

    private let timer = Timer.publish(every: 0.45, on: .main, in: .common).autoconnect()

    var body: some View {
        HStack(spacing: 5) {
            ForEach(0..<3, id: \.self) { i in
                Circle()
                    .fill(Color.secondary.opacity(i == phase ? 1.0 : 0.3))
                    .frame(width: 7, height: 7)
                    .scaleEffect(i == phase ? 1.2 : 1.0)
                    .animation(.easeInOut(duration: 0.3), value: phase)
            }
        }
        .padding(.vertical, 4)
        .onReceive(timer) { _ in phase = (phase + 1) % 3 }
    }
}

// MARK: - Extensions

private extension Double {
    func clamped(to range: ClosedRange<Double>) -> Double {
        min(max(self, range.lowerBound), range.upperBound)
    }
    func nonZeroOr(_ fallback: Double) -> Double {
        self == 0.0 ? fallback : self
    }
}

// MARK: - Markdown Rich Text View

struct MarkdownMessageView: View {
    let content: String
    
    var body: some View {
        let blocks = parseBlocks(content)
        VStack(alignment: .leading, spacing: 8) {
            ForEach(blocks) { block in
                switch block.type {
                case .text(let text):
                    renderTextBlock(text)
                case .code(let lang, let code):
                    renderCodeBlock(lang: lang, code: code)
                case .table(let headers, let rows):
                    renderTable(headers: headers, rows: rows)
                case .formula(let formula):
                    renderFormulaBlock(formula)
                }
            }
        }
    }
    
    @ViewBuilder
    private func renderTextBlock(_ text: String) -> some View {
        let lines = text.components(separatedBy: .newlines)
        VStack(alignment: .leading, spacing: 4) {
            ForEach(0..<lines.count, id: \.self) { i in
                let line = lines[i]
                let trimmed = line.trimmingCharacters(in: .whitespaces)
                
                if trimmed.isEmpty {
                    Spacer().frame(height: 4)
                } else if trimmed.hasPrefix("#### ") {
                    let rawText = String(trimmed.dropFirst(5))
                    Text(parseInlineMathAndMarkdown(rawText))
                        .font(.system(.subheadline, design: .rounded))
                        .fontWeight(.bold)
                        .padding(.top, 2)
                        .textSelection(.enabled)
                } else if trimmed.hasPrefix("### ") {
                    let rawText = String(trimmed.dropFirst(4))
                    Text(parseInlineMathAndMarkdown(rawText))
                        .font(.system(.headline, design: .rounded))
                        .fontWeight(.bold)
                        .padding(.top, 4)
                        .textSelection(.enabled)
                } else if trimmed.hasPrefix("## ") {
                    let rawText = String(trimmed.dropFirst(3))
                    Text(parseInlineMathAndMarkdown(rawText))
                        .font(.system(.title3, design: .rounded))
                        .fontWeight(.bold)
                        .padding(.top, 6)
                        .textSelection(.enabled)
                } else if trimmed.hasPrefix("# ") {
                    let rawText = String(trimmed.dropFirst(2))
                    Text(parseInlineMathAndMarkdown(rawText))
                        .font(.system(.title2, design: .rounded))
                        .fontWeight(.bold)
                        .padding(.top, 8)
                        .textSelection(.enabled)
                } else if trimmed.hasPrefix("* ") || trimmed.hasPrefix("- ") {
                    let rawText = String(trimmed.dropFirst(2))
                    // Count leading spaces to determine nesting depth
                    let leadingSpaces = line.prefix(while: { $0 == " " }).count
                    let nestLevel = leadingSpaces / 2  // 2 spaces = 1 level
                    let bulletSymbol = nestLevel == 0 ? "•" : nestLevel == 1 ? "◦" : "▪"
                    HStack(alignment: .top, spacing: 6) {
                        Text(bulletSymbol)
                            .font(.body)
                            .foregroundColor(.purple)
                        Text(parseInlineMathAndMarkdown(rawText))
                            .font(.body)
                            .textSelection(.enabled)
                    }
                    .padding(.leading, CGFloat(8 + nestLevel * 16))
                } else if let numberMatch = parseNumberedList(trimmed) {
                    HStack(alignment: .top, spacing: 6) {
                        Text(numberMatch.prefix)
                            .font(.body)
                            .foregroundColor(.secondary)
                        Text(parseInlineMathAndMarkdown(numberMatch.text))
                            .font(.body)
                            .textSelection(.enabled)
                    }
                    .padding(.leading, 8)
                } else {
                    Text(parseInlineMathAndMarkdown(line))
                        .font(.body)
                        .textSelection(.enabled)
                }
            }
        }
    }
    
    @ViewBuilder
    private func renderCodeBlock(lang: String, code: String) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text(lang.isEmpty ? "CODE" : lang.uppercased())
                    .font(.system(size: 9, weight: .bold, design: .monospaced))
                    .foregroundColor(.secondary)
                Spacer()
                Button(action: {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(code, forType: .string)
                }) {
                    HStack(spacing: 4) {
                        Image(systemName: "doc.on.doc")
                        Text("Copy")
                    }
                    .font(.system(size: 10, weight: .medium))
                    .foregroundColor(.secondary)
                }
                .buttonStyle(.plain)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(Color.primary.opacity(0.04))
            
            Divider().background(Color.primary.opacity(0.06))
            
            ScrollView(.horizontal, showsIndicators: false) {
                Text(code)
                    .font(.system(.body, design: .monospaced))
                    .foregroundColor(.primary.opacity(0.95))
                    .textSelection(.enabled)
                    .padding(12)
            }
        }
        .background(Color(nsColor: .underPageBackgroundColor))
        .cornerRadius(8)
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(Color.primary.opacity(0.06), lineWidth: 1)
        )
    }
    
    @ViewBuilder
    private func renderTable(headers: [String], rows: [[String]]) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            ScrollView(.horizontal, showsIndicators: true) {
                Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 8) {
                    // Header Row
                    GridRow {
                        ForEach(headers, id: \.self) { header in
                            Text(header)
                                .font(.system(size: 11, weight: .bold))
                                .foregroundColor(.secondary)
                                .padding(.bottom, 4)
                                .gridColumnAlignment(.leading)
                        }
                    }
                    
                    // Divider Row
                    GridRow {
                        ForEach(0..<headers.count, id: \.self) { _ in
                            Divider()
                        }
                    }
                    
                    // Data Rows
                    ForEach(0..<rows.count, id: \.self) { rowIndex in
                        let row = rows[rowIndex]
                        GridRow {
                            ForEach(0..<headers.count, id: \.self) { colIndex in
                                let text = colIndex < row.count ? row[colIndex] : ""
                                Text(parseInlineMathAndMarkdown(text))
                                    .font(.system(size: 11))
                                    .foregroundColor(.primary)
                            }
                        }
                    }
                }
                .padding(12)
            }
            .background(Color.primary.opacity(0.02))
            .cornerRadius(8)
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .stroke(Color.primary.opacity(0.06), lineWidth: 1)
            )
        }
        .padding(.vertical, 4)
    }
    
    @ViewBuilder
    private func renderFormulaBlock(_ formula: String) -> some View {
        let processed = translateMathSymbols(formatFractions(formula))
        let lines = processed.components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        VStack(alignment: .center, spacing: 0) {
            ForEach(0..<lines.count, id: \.self) { idx in
                let ln = lines[idx]
                let isBar = ln.hasPrefix("─") || ln.hasPrefix("-")
                Group {
                    if isBar {
                        Text(String(repeating: "─", count: max(ln.count, 8)))
                            .font(.system(size: 12, weight: .medium, design: .default))
                            .foregroundColor(.purple.opacity(0.6))
                    } else {
                        Text(parseMathString(ln))
                            .font(.system(size: 14, weight: .medium, design: .serif))
                            .foregroundColor(.purple)
                    }
                }
                .multilineTextAlignment(.center)
                .frame(maxWidth: .infinity, alignment: .center)
                .padding(.vertical, isBar ? 1 : 4)
            }
        }
        .padding(.vertical, 12)
        .padding(.horizontal, 24)
        .frame(maxWidth: .infinity, alignment: .center)
        .background(Color.primary.opacity(0.02))
        .cornerRadius(8)
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(Color.purple.opacity(0.15), lineWidth: 1)
        )
        .padding(.vertical, 4)
    }
    
    private func parseNumberedList(_ line: String) -> (prefix: String, text: String)? {
        let pattern = "^(\\d+\\.)\\s+(.*)$"
        guard let regex = try? NSRegularExpression(pattern: pattern, options: []),
              let match = regex.firstMatch(in: line, options: [], range: NSRange(location: 0, length: line.utf16.count)) else {
            return nil
        }
        
        let prefixRange = match.range(at: 1)
        let textRange = match.range(at: 2)
        
        guard let prefixRangeSpec = Range(prefixRange, in: line),
              let textRangeSpec = Range(textRange, in: line) else {
            return nil
        }
        
        return (prefix: String(line[prefixRangeSpec]), text: String(line[textRangeSpec]))
    }
    
    private func parseInlineMathAndMarkdown(_ line: String) -> AttributedString {
        var result = AttributedString()
        
        // Let's also support \( and \) inline math
        var cleanedLine = line
        cleanedLine = cleanedLine.replacingOccurrences(of: "\\(", with: "$").replacingOccurrences(of: "\\)", with: "$")
        
        let parts = cleanedLine.components(separatedBy: "$")
        for (index, part) in parts.enumerated() {
            if index % 2 == 0 {
                // Normal markdown text
                if !part.isEmpty {
                    if let attr = try? AttributedString(markdown: part, options: AttributedString.MarkdownParsingOptions(interpretedSyntax: .inlineOnlyPreservingWhitespace)) {
                        result.append(attr)
                    } else {
                        result.append(AttributedString(part))
                    }
                }
            } else {
                // Inline math!
                if !part.isEmpty {
                    result.append(parseMathString(part))
                }
            }
        }
        return result
    }
    
    private func translateMathSymbols(_ text: String) -> String {
        var result = text
        let translations: [(String, String)] = [
            // Greek letters
            ("\\alpha", "α"), ("\\beta", "β"), ("\\gamma", "γ"), ("\\delta", "δ"),
            ("\\epsilon", "ε"), ("\\zeta", "ζ"), ("\\eta", "η"), ("\\theta", "θ"),
            ("\\iota", "ι"), ("\\kappa", "κ"), ("\\lambda", "λ"), ("\\mu", "μ"),
            ("\\nu", "ν"), ("\\xi", "ξ"), ("\\pi", "π"), ("\\rho", "ρ"),
            ("\\sigma", "σ"), ("\\tau", "τ"), ("\\upsilon", "υ"), ("\\phi", "φ"),
            ("\\chi", "χ"), ("\\psi", "ψ"), ("\\omega", "ω"),
            ("\\Delta", "Δ"), ("\\Theta", "Θ"), ("\\Lambda", "Λ"), ("\\Sigma", "Σ"),
            ("\\Phi", "Φ"), ("\\Omega", "Ω"),
            // Operators & Symbols
            ("\\sum", "∑"), ("\\prod", "∏"), ("\\infty", "∞"), ("\\partial", "∂"),
            ("\\nabla", "∇"), ("\\times", "×"), ("\\cdot", "·"), ("\\div", "÷"),
            ("\\approx", "≈"), ("\\neq", "≠"), ("\\le", "≤"), ("\\ge", "≥"),
            ("\\pm", "±"), ("\\mp", "∓"), ("\\propto", "∝"),
            // LaTeX specific formatting
            ("\\hat{y}", "ŷ"), ("\\hat{y}_i", "ŷ_i"), ("\\bar{x}", "x̄"),
            ("\\hat", "^"), ("\\sqrt", "√"), ("\\text", ""),
            ("{", ""), ("}", ""), // Remove remaining braces
            ("\\left", ""), ("\\right", ""), // Remove brackets modifiers
            ("\\,", " "), ("\\;", " ") // Spacing
        ]
        
        for (latex, unicode) in translations {
            result = result.replacingOccurrences(of: latex, with: unicode)
        }
        return result
    }
    
    private func formatFractions(_ text: String) -> String {
        var result = text
        // Match \frac{numerator}{denominator} where contents can include spaces and words.
        // Uses a greedy approach: find \frac then scan balanced braces manually.
        var output = ""
        var idx = result.startIndex
        while idx < result.endIndex {
            if result[idx...].hasPrefix("\\frac{") {
                // Advance past \frac{
                result.formIndex(&idx, offsetBy: 6) // skip \frac{
                // Collect numerator (balanced braces, depth 1 already entered)
                var depth = 1
                var numerator = ""
                while idx < result.endIndex && depth > 0 {
                    let c = result[idx]
                    if c == "{" { depth += 1; numerator.append(c) }
                    else if c == "}" { depth -= 1; if depth > 0 { numerator.append(c) } }
                    else { numerator.append(c) }
                    result.formIndex(after: &idx)
                }
                // Expect {
                guard idx < result.endIndex && result[idx] == "{" else {
                    output += "\\frac{" + numerator + "}"
                    continue
                }
                result.formIndex(after: &idx) // skip {
                // Collect denominator
                depth = 1
                var denominator = ""
                while idx < result.endIndex && depth > 0 {
                    let c = result[idx]
                    if c == "{" { depth += 1; denominator.append(c) }
                    else if c == "}" { depth -= 1; if depth > 0 { denominator.append(c) } }
                    else { denominator.append(c) }
                    result.formIndex(after: &idx)
                }
                // Build a compact stacked representation:
                // "(numerator) / (denominator)" — displayed in serif math font
                let numStr = numerator.trimmingCharacters(in: .whitespaces)
                let denStr = denominator.trimmingCharacters(in: .whitespaces)
                let lineWidth = max(numStr.count, denStr.count)
                let bar = String(repeating: "─", count: max(lineWidth + 2, 5))
                output += "\n" + numStr + "\n" + bar + "\n" + denStr + "\n"
            } else {
                output.append(result[idx])
                result.formIndex(after: &idx)
            }
        }
        return output
    }
    
    private func parseMathString(_ rawText: String) -> AttributedString {
        let formattedText = translateMathSymbols(formatFractions(rawText))
        var attrString = AttributedString()
        var i = 0
        let chars = Array(formattedText)
        
        while i < chars.count {
            let char = chars[i]
            
            if char == "_" {
                i += 1
                if i < chars.count {
                    if chars[i] == "{" {
                        i += 1
                        var subContent = ""
                        while i < chars.count && chars[i] != "}" {
                            subContent.append(chars[i])
                            i += 1
                        }
                        i += 1 // skip '}'
                        var subAttr = AttributedString(subContent)
                        subAttr.font = .system(size: 8, weight: .regular, design: .serif)
                        subAttr.baselineOffset = -3.0
                        attrString.append(subAttr)
                    } else {
                        var subAttr = AttributedString(String(chars[i]))
                        subAttr.font = .system(size: 8, weight: .regular, design: .serif)
                        subAttr.baselineOffset = -3.0
                        attrString.append(subAttr)
                        i += 1
                    }
                }
            } else if char == "^" {
                i += 1
                if i < chars.count {
                    if chars[i] == "{" {
                        i += 1
                        var superContent = ""
                        while i < chars.count && chars[i] != "}" {
                            superContent.append(chars[i])
                            i += 1
                        }
                        i += 1 // skip '}'
                        var superAttr = AttributedString(superContent)
                        superAttr.font = .system(size: 8, weight: .regular, design: .serif)
                        superAttr.baselineOffset = 4.0
                        attrString.append(superAttr)
                    } else {
                        var superAttr = AttributedString(String(chars[i]))
                        superAttr.font = .system(size: 8, weight: .regular, design: .serif)
                        superAttr.baselineOffset = 4.0
                        attrString.append(superAttr)
                        i += 1
                    }
                }
            } else {
                var charAttr = AttributedString(String(char))
                charAttr.font = .system(size: 13, weight: .regular, design: .serif)
                attrString.append(charAttr)
                i += 1
            }
        }
        return attrString
    }
    
    enum BlockType {
        case text(String)
        case code(lang: String, code: String)
        case table(headers: [String], rows: [[String]])
        case formula(String)
    }
    
    struct Block: Identifiable {
        let id = UUID()
        let type: BlockType
    }
    
    private func parseBlocks(_ text: String) -> [Block] {
        var blocks: [Block] = []
        let parts = text.components(separatedBy: "```")
        for (index, part) in parts.enumerated() {
            if index % 2 == 0 {
                // Parse tables and formulas in text parts
                blocks.append(contentsOf: parseTextContent(part))
            } else {
                let lines = part.components(separatedBy: .newlines)
                if let firstLine = lines.first, !firstLine.isEmpty && firstLine.count < 15 && !firstLine.contains(" ") {
                    let code = lines.dropFirst().joined(separator: "\n")
                    blocks.append(Block(type: .code(lang: firstLine, code: code)))
                } else {
                    blocks.append(Block(type: .code(lang: "", code: part)))
                }
            }
        }
        if blocks.isEmpty && !text.isEmpty {
            blocks.append(Block(type: .text(text)))
        }
        return blocks
    }
    
    private func parseTextContent(_ text: String) -> [Block] {
        var blocks: [Block] = []
        let parts = text.components(separatedBy: "$$")
        for (index, part) in parts.enumerated() {
            if index % 2 != 0 {
                if !part.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    blocks.append(Block(type: .formula(part)))
                }
            } else {
                blocks.append(contentsOf: parseBracketFormulasAndTables(part))
            }
        }
        return blocks
    }
    
    private func parseBracketFormulasAndTables(_ text: String) -> [Block] {
        var blocks: [Block] = []
        let parts = text.components(separatedBy: "\\[")
        for (index, part) in parts.enumerated() {
            if index == 0 {
                blocks.append(contentsOf: parseTablesAndText(part))
            } else {
                let subParts = part.components(separatedBy: "\\]")
                if let formula = subParts.first {
                    blocks.append(Block(type: .formula(formula)))
                }
                if subParts.count > 1 {
                    let remainingText = subParts.dropFirst().joined(separator: "\\]")
                    blocks.append(contentsOf: parseTablesAndText(remainingText))
                }
            }
        }
        return blocks
    }
    
    /// Returns true if a line looks like a standalone LaTeX formula
    /// (contains a math command but no surrounding prose).
    private func isFormulaLine(_ trimmed: String) -> Bool {
        let mathCommands = ["\\frac", "\\sum", "\\int", "\\prod", "\\lim", "\\sqrt",
                            "\\partial", "\\nabla", "\\infty", "\\begin{"]
        return mathCommands.contains(where: { trimmed.contains($0) })
    }

    private func parseTablesAndText(_ text: String) -> [Block] {
        let lines = text.components(separatedBy: .newlines)
        var resultBlocks: [Block] = []
        var currentTextLines: [String] = []
        
        var i = 0
        while i < lines.count {
            let line = lines[i]
            let trimmed = line.trimmingCharacters(in: .whitespaces)

            // Auto-detect bare formula lines (no $$ wrapper needed)
            if isFormulaLine(trimmed) {
                if !currentTextLines.isEmpty {
                    resultBlocks.append(Block(type: .text(currentTextLines.joined(separator: "\n"))))
                    currentTextLines.removeAll()
                }
                resultBlocks.append(Block(type: .formula(trimmed)))
                i += 1
                continue
            }
            
            let hasPipe = trimmed.contains("|")
            if hasPipe && i + 1 < lines.count {
                let nextLine = lines[i + 1].trimmingCharacters(in: .whitespaces)
                let isSeparator = nextLine.range(of: "^[|\\-:\\s]+$", options: .regularExpression) != nil && nextLine.contains("-") && nextLine.contains("|")
                
                if isSeparator {
                    if !currentTextLines.isEmpty {
                        resultBlocks.append(Block(type: .text(currentTextLines.joined(separator: "\n"))))
                        currentTextLines.removeAll()
                    }
                    
                    let rawHeaders = line.components(separatedBy: "|")
                    var headers = rawHeaders.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                    if line.hasPrefix("|") && !headers.isEmpty {
                        headers.removeFirst()
                    }
                    if line.hasSuffix("|") && !headers.isEmpty {
                        headers.removeLast()
                    }
                    
                    i += 2
                    
                    var rows: [[String]] = []
                    while i < lines.count {
                        let dataLine = lines[i].trimmingCharacters(in: .whitespaces)
                        if dataLine.contains("|") {
                            let rawCells = dataLine.components(separatedBy: "|")
                            var cells = rawCells.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                            if dataLine.hasPrefix("|") && !cells.isEmpty {
                                cells.removeFirst()
                            }
                            if dataLine.hasSuffix("|") && !cells.isEmpty {
                                cells.removeLast()
                            }
                            rows.append(cells)
                            i += 1
                        } else {
                            break
                        }
                    }
                    
                    resultBlocks.append(Block(type: .table(headers: headers, rows: rows)))
                    continue
                }
            }
            
            currentTextLines.append(line)
            i += 1
        }
        
        if !currentTextLines.isEmpty {
            resultBlocks.append(Block(type: .text(currentTextLines.joined(separator: "\n"))))
        }
        
        return resultBlocks
    }
}

struct CloudKeySetupView: View {
    let providerName: String
    
    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "key.fill")
                .font(.system(size: 40))
                .foregroundColor(.orange)
            
            Text("\(providerName) API Key Missing")
                .font(.headline)
                .fontWeight(.bold)
            
            Text("To use the cloud-hosted AI analyst, you must configure your API Key in the application Settings.")
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 240)
            
            Button("Open Settings") {
                NSApp.sendAction(Selector(("showSettingsWindow:")), to: nil, from: nil)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.small)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }
}

