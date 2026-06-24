import SwiftUI
import AppKit

struct NativeProgressView: NSViewRepresentable {
    var controlSize: NSControl.ControlSize = .regular
    
    func makeNSView(context: Context) -> NSProgressIndicator {
        let indicator = NSProgressIndicator()
        indicator.style = .spinning
        indicator.controlSize = controlSize
        indicator.isDisplayedWhenStopped = false
        indicator.startAnimation(nil)
        return indicator
    }
    
    func updateNSView(_ nsView: NSProgressIndicator, context: Context) {
        nsView.controlSize = controlSize
    }
}
