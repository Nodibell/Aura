import Foundation

final class ProtectedData: @unchecked Sendable {
    private let lock = NSRecursiveLock()
    private var data = Data()
    
    func append(_ segment: Data) {
        lock.lock()
        defer { lock.unlock() }
        data.append(segment)
    }
    
    func get() -> Data {
        lock.lock()
        defer { lock.unlock() }
        return data
    }
}
