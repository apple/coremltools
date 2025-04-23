//
//  Copyright (c) 2025, Apple Inc. All rights reserved.
//
//  Use of this source code is governed by a BSD-3-clause license that can be
//  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

// Represents a JSON-RPC 2.0 request.
///
/// This struct conforms to `Codable`, `Identifiable`, and `Sendable` protocols.
/// It is generic over the `Params` type, which must conform to `Encodable` and `Sendable`.
struct JSONRPCRequest<Params>: Codable, Identifiable, Sendable, NamedRequest where Params: Codable & Sendable {
    /// The JSON-RPC version.
    public let jsonrpc: String

    /// An identifier established by the client that MUST contain a String, Number, or NULL value if included.
    public let id: String

    /// The name of the method to be invoked.
    public let method: String

    /// The parameters to the method.
    public let params: Params?


   /// The request type name.
    public var name: String {
        method
    }
}

/// Represents a JSON-RPC 2.0 error object.
/// Conforms to `Error` for Swift error handling and `Codable` for JSON encoding/decoding.
struct JSONRPCError: Error, Codable {
    /// The error code.
    /// As per JSON-RPC 2.0 spec:
    /// - -32700 to -32600 are reserved for pre-defined errors
    /// - -32000 to -32099 are available for custom errors
    let code: Int

    /// A short description of the error.
    /// Should be limited to a concise single sentence.
    let message: String
}

// MARK: - Predefined Errors
extension JSONRPCError {
    /// Parse error (-32700): Invalid JSON was received by the server.
    static let parseError = JSONRPCError(code: -32700, message: "Parse error")

    /// Invalid Request (-32600): The JSON sent is not a valid Request object.
    static let invalidRequest = JSONRPCError(code: -32600, message: "Invalid Request")

    /// Method not found (-32601): The method does not exist / is not available.
    static let methodNotFound = JSONRPCError(code: -32601, message: "Method not found")

    /// Invalid params (-32602): Invalid method parameter(s).
    static let invalidParams = JSONRPCError(code: -32602, message: "Invalid params")

    /// Internal error (-32603): Internal JSON-RPC error.
    static let internalError = JSONRPCError(code: -32603, message: "Internal error")

    /// Creates a custom error with the given code and message.
    /// - Parameters:
    ///   - code: The error code. Should be between -32000 and -32099 for custom server errors.
    ///   - message: A short description of the error.
    /// - Returns: A new `JSONRPCError` instance.
    static func customError(code: Int, message: String) -> JSONRPCError {
        return JSONRPCError(code: code, message: message)
    }
}

/// Represents a JSON-RPC 2.0 response.
///
/// This struct conforms to `Codable`, `Identifiable`, and `Sendable` protocols.
/// It is generic over the `Result` type, which must conform to `Decodable` and `Sendable`.
struct JSONRPCResponse<Result>: Codable, Sendable where Result: Codable & Sendable {
    /// The JSON-RPC version.
    public let jsonrpc: String

    /// The id of the request this response corresponds to.
    public let id: String

    /// The result of the method invocation, if successful.
    public let result: Result?

    /// The resource.
    public let resource: String?

    /// The error object, if an error occurred during the method invocation.
    public let error: JSONRPCError?

}
