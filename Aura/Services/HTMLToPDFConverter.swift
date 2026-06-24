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
            // Enable JavaScript and network access so ECharts can be fetched
            config.preferences.setValue(true, forKey: "developerExtrasEnabled")
            
            let webView = WKWebView(frame: CGRect(x: 0, y: 0, width: 800, height: 1000), configuration: config)
            webView.navigationDelegate = self
            self.webView = webView
            
            // Load HTML
            webView.loadHTMLString(html, baseURL: URL(string: "https://localhost"))
        }
    }
    
    nonisolated func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        Task { @MainActor in
            // Wait for ECharts to load and render the diagrams
            // A delay of 1.5 seconds ensures Javascript execution of ECharts finishes
            try? await Task.sleep(nanoseconds: 1_500_000_000)
            
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
