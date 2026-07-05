import Foundation

struct PredictionResult: Codable {
    let prediction: PredictionValue
    let probabilities: [String: Double]?
    var timeElapsed: Double? = nil
}

enum PredictionValue: Codable {
    case string(String)
    case number(Double)
    case boolean(Bool)
    case array([PredictionValue])
    
    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let s = try? container.decode(String.self) {
            self = .string(s)
        } else if let d = try? container.decode(Double.self) {
            self = .number(d)
        } else if let b = try? container.decode(Bool.self) {
            self = .boolean(b)
        } else if let arr = try? container.decode([PredictionValue].self) {
            self = .array(arr)
        } else {
            throw DecodingError.typeMismatch(PredictionValue.self, DecodingError.Context(codingPath: decoder.codingPath, debugDescription: "Not a string, number, boolean, or array"))
        }
    }
    
    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let s): try container.encode(s)
        case .number(let d): try container.encode(d)
        case .boolean(let b): try container.encode(b)
        case .array(let arr): try container.encode(arr)
        }
    }
    
    var displayString: String {
        switch self {
        case .string(let s): return s
        case .number(let d):
            if d.truncatingRemainder(dividingBy: 1) == 0 {
                return String(format: "%.0f", d)
            } else {
                return String(format: "%.4f", d)
            }
        case .boolean(let b): return b ? "True" : "False"
        case .array(let arr):
            return arr.map { $0.displayString }.joined(separator: ", ")
        }
    }
}
