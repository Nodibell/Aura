import Foundation
import WebKit

#if canImport(AppKit)
import AppKit
#endif

@MainActor
class HTMLToPDFConverter: NSObject, WKNavigationDelegate {
    static let shared = HTMLToPDFConverter()
    
    private var webView: WKWebView?
    private var continuation: CheckedContinuation<Data, Error>?
    
    private override init() {
        super.init()
    }
    
    func convert(html: String) async throws -> Data {
        return try await withCheckedThrowingContinuation { continuation in
            self.continuation = continuation
            
            let config = WKWebViewConfiguration()
            config.preferences.setValue(true, forKey: "developerExtrasEnabled")
            
            // Use A4 width (794px) so single-column layout renders at the correct scale
            let webView = WKWebView(frame: CGRect(x: 0, y: 0, width: 794, height: 2000), configuration: config)
            webView.navigationDelegate = self
            // Force print media so @media print rules (single-column, white background) are applied
            webView.mediaType = "print"
            self.webView = webView
            
            // Load from nil baseURL — JS is now inline (no external CDN)
            webView.loadHTMLString(html, baseURL: nil)
        }
    }
    
    nonisolated func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        Task { @MainActor in
            // Poll JS until all ECharts instances signal ready,
            // or bail out after a maximum of 8 seconds.
            await self.waitForChartsReady(webView: webView, maxWaitMs: 8000)
            
            guard let continuation = self.continuation else { return }
            self.continuation = nil
            
            let pdfConfig = WKPDFConfiguration()
            webView.createPDF(configuration: pdfConfig) { result in
                switch result {
                case .success(let data):
                    continuation.resume(returning: data)
                case .failure(let error):
                    continuation.resume(throwing: error)
                }
                self.webView = nil
            }
        }
    }
    
    /// Polls `window.__auraChartsReady` (set by ReportCompiler JS after all charts render) until true or timeout.
    @MainActor
    private func waitForChartsReady(webView: WKWebView, maxWaitMs: Int) async {
        let pollIntervalNs: UInt64 = 150_000_000  // 150ms
        let maxIterations = maxWaitMs / 150
        
        for _ in 0..<maxIterations {
            if let result = try? await webView.evaluateJavaScript("window.__auraChartsReady === true"),
               let ready = result as? Bool, ready {
                return
            }
            try? await Task.sleep(nanoseconds: pollIntervalNs)
        }
        // Timeout reached — proceed anyway (charts may still render acceptably)
    }
    
    nonisolated func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
        Task { @MainActor in
            self.continuation?.resume(throwing: error)
            self.continuation = nil
            self.webView = nil
        }
    }
    
    nonisolated func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
        Task { @MainActor in
            self.continuation?.resume(throwing: error)
            self.continuation = nil
            self.webView = nil
        }
    }
}
