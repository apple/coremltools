//
//  Copyright (c) 2025, Apple Inc. All rights reserved.
//
//  Use of this source code is governed by a BSD-3-clause license that can be
//  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

protocol Socket {
    associatedtype Request: Sendable
    associatedtype Response: Sendable

    /// Listens for incoming data
    /// - Returns: An AsyncStream of Input values
    /// - Throws: ConnectionError if listening fails
    func receive() async throws -> AsyncStream<Request>

    /// Sends data through the connection
    /// - Parameter value: The value to send
    /// - Throws: ConnectionError if sending fails
    func send(value: Response) async throws

    ///  Shutdown the socket.
    func shutdown() async
}

/// A protocol that defines a named request.
protocol NamedRequest: Sendable {
    /// A string that uniquely identifies a request type.
    ///
    /// This property should return a unique name for the request type.
    var name: String { get }
}

/// A protocol that defines an entity associated with an optional resource URL.
///
/// `Resourceful` extends the concept of a basic entity by including
/// a reference to an external resource.
protocol Resourceful: Sendable {
    /// An optional URL representing the resource associated with this entity.
    var resource: URL? { get }
}

/// An actor that adapts generic Request and Response types to a Socket interface.
///
/// `AdapterSocket` provides a type-safe wrapper around socket operations,
/// allowing for strongly-typed request and response handling while conforming to the `Socket` protocol.
actor AdapterSocket<Request, Response>: Socket where Request: Sendable, Response: Sendable {
    /// Dictionary to store active continuations
    private var continuations: [Int : AsyncStream<Request>.Continuation] = [:]
    /// ID for the next continuation
    private var nextContinuationID: Int = 0
    /// The underlying connection used for sending and receiving payloads.
    private let underlyingSocket: any Socket
    /// A closure responsible for sending responses over the underlying connection.
    private let sendResponse: (Response) async throws -> Void
    /// A task responsible for processing incoming requests from the underlying connection. and informing listeners.
    private var task: Task<Void, Error>? = nil
    /// The response stream.
    nonisolated let responseStream: AsyncStream<Response>
    /// The response continuation.
    private let responseContinuation: AsyncStream<Response>.Continuation

    /// Initializes a new AdapterSocket with an underlying socket and conversion functions.
    ///
    /// This initializer sets up a AdapterSocket that wraps another Socket,
    /// providing conversion between the underlying socket's request/response types and
    /// the AdapterSocket's request/response types.
    ///
    /// - Parameters:
    ///   - underlyingConnection: The connection to be wrapped. It must conform to the `Connection` protocol.
    ///   - from: A closure that converts from the underlying connection's request type to this connection's request type.
    ///           This closure may throw an error if the conversion fails.
    ///   - to: A closure that converts from this connection's response type to the underlying connection's response type.
    ///         This closure may throw an error if the conversion fails.
    init<T: Socket>(socket: T,
                    receive: @escaping (T.Request) -> Request?,
                    send: @escaping (Response) -> T.Response) async throws {
        (responseStream, responseContinuation) = AsyncStream.makeStream(of: Response.self)
        underlyingSocket = socket
        sendResponse = { [socket] (response: Response) async throws -> Void in
            try await socket.send(value: send(response))
        }

        let sequence = try await socket.receive().compactMap { receive($0) }
        task = Task {[weak self] in
            for try await request in sequence {
                if Task.isCancelled {
                    return
                }

                guard let self else {
                    continue
                }

                await self.inform(request: request)
            }
        }
    }

    deinit {
        task?.cancel()
    }

    /// Informs listeners.
    private func inform(request: Request)  {
        for continuation in continuations.values {
            continuation.yield(request)
        }
    }

    /// Establishes an asynchronous stream of incoming requests.
    ///
    /// This method initializes and returns an `AsyncStream` that continuously monitors for and yields
    /// incoming requests. It provides a non-blocking way to process requests as they arrive.
    ///
    /// - Returns: An `AsyncStream<Request>` that yields incoming requests as they become available.
    ///
    /// - Throws: An error if the :
    ///   - The receive process fails to initialize.
    ///   - There's an issue with the underlying connection or network.
    public func receive() async throws -> AsyncStream<Request> {
        defer {
            nextContinuationID += 1
        }

        let continuationID = nextContinuationID
        // Create a new stream and store its continuation
        let stream = AsyncStream<Request> { continuation in
            continuations[continuationID] = continuation
        }

        guard let continuation = continuations[nextContinuationID] else {
            fatalError("Inconsistent state.")
        }

        // Set up termination handler for the continuation
        continuation.onTermination = { [weak self] _ in
            guard let self else { return }
            Task {
                await self.removeContinuation(continuationID)
            }
        }

        return stream
    }

    /// Remove a continuation when it's terminated.
    private func removeContinuation(_ continuationID: Int) async {
        continuations.removeValue(forKey: continuationID)
    }

    /// Sends a response.
    ///
    /// This method transforms the given response and sends it using the underlying socket.
    ///
    /// - Parameter value: The value to be sent.
    /// - Throws: An error if the sending process fails.
    func send(value: Response) async throws {
        try await sendResponse(value)
        responseContinuation.yield(value)
    }

    func shutdown() async {
        await underlyingSocket.shutdown()
    }
}

/// A generic actor that implements the `Socket` protocol, designed to pass through requests and responses.
actor PassThroughSocket<Request, Response>: Socket where Request: Sendable, Response:Sendable {
    /// Dictionary to store active continuations
    private var continuations: [Int : AsyncStream<Request>.Continuation] = [:]
    /// ID for the next continuation
    private var nextContinuationID: Int = 0
    /// The response stream.
    nonisolated let responseStream: AsyncStream<Response>
    /// The response continuation.
    private let responseContinuation: AsyncStream<Response>.Continuation

    init() {
        (responseStream, responseContinuation) = AsyncStream.makeStream(of: Response.self)
    }

    func receive() async throws -> AsyncStream<Request> {
        defer {
            nextContinuationID += 1
        }

        let continuationID = nextContinuationID
        // Create a new stream and store its continuation
        let (stream, continuation) = AsyncStream.makeStream(of: Request.self)
        continuations[continuationID] = continuation

        // Set up termination handler for the continuation
        continuation.onTermination = { [weak self] _ in
            guard let self else { return }
            Task {
                await self.removeContinuation(continuationID)
            }
        }

        return stream
    }

    // Remove a continuation when it's terminated
    private func removeContinuation(_ continuationID: Int) async {
        continuations.removeValue(forKey: continuationID)
    }

    func send(value: Response) async throws {
        responseContinuation.yield(value)
    }

    /// Pushes a request value to all active continuations.
    ///
    /// This method broadcasts the request to all current listeners.
    ///
    /// - Parameter value: The `Request` object to push.
    func push(value: Request) async throws {
        for continuation in continuations.values {
            continuation.yield(value)
        }
    }

    func shutdown() async {
        responseContinuation.finish()
        for continuation in continuations.values {
            continuation.finish()
        }

        continuations.removeAll()
    }
}
