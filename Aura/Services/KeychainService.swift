import Foundation
import Security

final class KeychainService: KeychainServiceProtocol, Sendable {
    static let shared = KeychainService()
    
    private init() {}
    
    /// Save a secure string in the Keychain
    func save(_ string: String, forKey key: String) -> Bool {
        guard let data = string.data(using: .utf8) else { return false }
        
        // Delete any existing item first
        delete(forKey: key)
        
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key,
            kSecValueData as String: data,
            kSecAttrAccessible as String: kSecAttrAccessibleAfterFirstUnlock
        ]
        
        let status = SecItemAdd(query as CFDictionary, nil)
        return status == errSecSuccess
    }
    
    /// Load a secure string from the Keychain
    func load(forKey key: String) -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]
        
        var dataTypeRef: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &dataTypeRef)
        
        if status == errSecSuccess, let data = dataTypeRef as? Data {
            return String(data: data, encoding: .utf8)
        }
        return nil
    }
    
    /// Delete a secure string from the Keychain
    func delete(forKey key: String) {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key
        ]
        SecItemDelete(query as CFDictionary)
    }
    
    /// Retrieve a secure string, migrating from legacy UserDefaults if present
    func getSecureString(forKey key: String) -> String? {
        if let val = load(forKey: key) {
            return val
        }
        
        // Fallback and migrate from UserDefaults
        if let legacyVal = UserDefaults.standard.string(forKey: key) {
            _ = save(legacyVal, forKey: key)
            UserDefaults.standard.removeObject(forKey: key)
            return legacyVal
        }
        
        return nil
    }
}
