//
//  Copyright (c) 2025, Apple Inc. All rights reserved.
//
//  Use of this source code is governed by a BSD-3-clause license that can be
//  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

extension URL {
    fileprivate mutating func excludeItemFromBackup() throws {
        var resourceValues = URLResourceValues()
        resourceValues.isExcludedFromBackup = true
        try setResourceValues(resourceValues)
    }
}

/// Actor responsible for managing communication via a DeviceCtl connection.
actor DeviceCtlSocket: Socket {
    /// Represents a request received via the DeviceCtl connection.
    public struct Request: Sendable, Identifiable, NamedRequest {
        /// Unique identifier for the request.
        let id: String
        /// Name of the request.
        let name: String
        /// The payload.
        let payload: Data
        /// Optional URL to a resource associated with the request.
        let resource: URL?
    }

    /// Represents a response to be sent via the DeviceCtl connection.
    public struct Response: Sendable, Identifiable {
        /// Unique identifier for the response, matching the request ID.
        let id: String
        /// The payload.
        let original: Data
        /// Optional URL to a resource associated with the response
        let resource: URL?
    }

    /// Represents a decoded message in the DeviceCtl communication protocol.
    /// This struct is used for parsing incoming messages from the connection.
    struct Message: Decodable, Sendable, Identifiable {
        /// A unique identifier.
        public let id: String
        /// Name of the request.
        public let method: String
        /// The resource name.
        public let resource: String?

        var name:String {
            method
        }
    }

    public enum Error: Swift.Error  {
        /// Indicates that the required message file does not exist.
        case unavailableMessageFile(String)
        /// The message data cannot be read.
        case invalidMessageData(Swift.Error)
        /// An error occurred during JSON decoding.
        case decodingError(Swift.Error)
        /// Max retry attempts reached.
        case maxAttemptsReached(Swift.Error)
        /// Represents missing resource file.
        case resourceMissing(String)
        /// Represents task cancellation.
        case taskCancelled

        fileprivate var isRecoverable: Bool {
            switch self {
            case .unavailableMessageFile:
                return true
            case .invalidMessageData:
                return true
            case .decodingError:
                return true
            case .resourceMissing:
                return true
            default:
                return false
            }
        }
    }

    /// The messages directory URL.
    let messagesDirectoryURL: URL

    /// The resources directory URL.
    let resourcesDirectoryURL: URL

    /// The responses directory URL.
    let responsesDirectoryURL: URL

    /// Monitors the messages directory for changes.
    let messagesDirectoryMonitor: DirectoryMonitor

    /// This is used for reading, writing, and managing files related to device communication.
    private let fm = FileManager()

    /// The maximum number of retry attempts when parsing a message from the device.
    ///
    /// This value is used to handle scenarios where a message may arrive incompletely or in fragments.
    /// The parsing process will retry up to this many times to account for:
    /// - Partial writes by the device
    /// - Network or I/O delays in message transmission
    /// - Temporary file system inconsistencies
    let maxRetryAttempts: Int = 10

    /// Tasks responsible for processing directory changes.
    private var tasks: [Task<Void, Never>] = []

    /// Dictionary to store active continuations
    private var continuations: [Int : AsyncStream<Request>.Continuation] = [:]
    /// ID for the next continuation
    private var nextContinuationID: Int = 0

    private static func retryWithBackoff<T>(
        operation: @escaping () async throws -> T,
        maxAttempts: Int,
        baseDelay: TimeInterval = 1.0,
        maxDelay: TimeInterval = 5.0
    ) async throws -> T {
        var attempts = 0
        while true {
            do {
                if Task.isCancelled {
                    throw Error.taskCancelled
                }
                return try await operation()
            } catch let error as DeviceCtlSocket.Error {
                if error.isRecoverable {
                    attempts += 1
                    if attempts >= maxAttempts {
                        throw Error.maxAttemptsReached(error)
                    }
                    let delay = min(maxDelay, baseDelay * pow(2.0, Double(attempts - 1)))
                    let jitteredDelay = Double.random(in: 0.5...1.5) * delay
                    try await Task.sleep(nanoseconds: UInt64(jitteredDelay * 1_000_000_000))
                } else {
                    throw error
                }
            }
        }
    }

    private static func createManagedDirectory(at url: URL, name: String, fileManager: FileManager) throws -> URL {
        var directoryURL = url.appendingPathComponent(name)
        if !fileManager.directoryExists(at: directoryURL.path) {
            try fileManager.createDirectory(at: directoryURL, withIntermediateDirectories: true)
        }

        do {
            try directoryURL.excludeItemFromBackup()
        } catch {
            let message = """
            Failed to exclude item from backup:
            - Item: \(directoryURL.lastPathComponent)
            - Description: \(error.localizedDescription)
            """
            print(message)
        }

        return directoryURL
    }

    private static func createPayload(payloadData: Data,
                                      identifier: String,
                                      resourceURL: URL?,
                                      directoryURL: URL,
                                      fileManager: FileManager) throws -> URL? {
        let temporaryPayloadDirectoryURL = fileManager.temporaryDirectory.appendingPathComponent(identifier)
        try? fileManager.removeItem(at: temporaryPayloadDirectoryURL)

        try fileManager.createDirectory(at: temporaryPayloadDirectoryURL, withIntermediateDirectories: false)
        let payloadDirectoryURL = directoryURL.appendingPathComponent(identifier)

        if fileManager.fileExists(atPath: payloadDirectoryURL.path) {
            try fileManager.removeItem(at: payloadDirectoryURL)
        }

        let payloadFileURL = temporaryPayloadDirectoryURL.appendingPathComponent("message.json")
        try payloadData.write(to: payloadFileURL, options: .atomic)

        if let resourceURL {
            let newResourceURL = temporaryPayloadDirectoryURL.appendingPathComponent(resourceURL.lastPathComponent)
            try fileManager.moveItem(at: resourceURL, to: newResourceURL)
        }

        let backupItemName = "\(identifier).bck"
        // Use `replaceItemAt` to move the contents atomically.
        let result = try fileManager.replaceItemAt(payloadDirectoryURL,
                                                   withItemAt: temporaryPayloadDirectoryURL,
                                                   backupItemName: backupItemName,
                                                   options: .usingNewMetadataOnly)

        return result
    }

    init(directoryURL: URL) async throws {
        messagesDirectoryURL = try Self.createManagedDirectory(at: directoryURL, name: "requests/messages", fileManager: fm)
        resourcesDirectoryURL = try Self.createManagedDirectory(at: directoryURL, name: "requests/resources", fileManager: fm)
        responsesDirectoryURL = try Self.createManagedDirectory(at: directoryURL, name: "responses", fileManager: fm)

        messagesDirectoryMonitor = try DirectoryMonitor(url: messagesDirectoryURL)
        let sequence = try await messagesDirectoryMonitor.listen()

        let task = Task {[weak self] in
            for await change in sequence {
                if Task.isCancelled {
                    return
                }

                switch change {
                case let .initial(urls):
                    await self?.processIncomingMessages(urls: urls)

                case let .added(urls):
                    await self?.processIncomingMessages(urls: urls)

                default:
                    break
                }
            }
        }

        tasks.append(task)
    }

    private func processResponse(at url: URL) async throws -> URL {
        let fm = self.fm
        return try await Self.retryWithBackoff(operation: {
            if !fm.fileExists(atPath: url.path) {
                throw Error.resourceMissing("Required resource not found: \(url.lastPathComponent)")
            }

            return url
        }, maxAttempts: maxRetryAttempts)
    }

    private func processIncomingMessage(at url: URL) async throws {
        var isDirectory: ObjCBool = false
        if fm.fileExists(atPath: url.path, isDirectory: &isDirectory) && !isDirectory.boolValue {
            return
        }

        var messageData: Data? = nil
        let messageURL = url.appendingPathComponent("message.json")
        if !fm.fileExists(atPath: messageURL.path) {
            throw Error.unavailableMessageFile("Message file not found at expected location: \(messageURL.path)")
        }

        do {
            messageData = try Data(contentsOf: messageURL, options: .mappedIfSafe)
        } catch {
            throw Error.invalidMessageData(error)
        }

        guard let messageData else {
            fatalError("Inconsistent state.")
        }

        let decoder = JSONDecoder()
        var message: Message? = nil
        do {
            message = try decoder.decode(Message.self, from: messageData)
        } catch {
            throw Error.decodingError(error)
        }

        guard let message else {
            fatalError("Inconsistent state.")
        }

        let resource = message.resource.map {resourcesDirectoryURL.appendingPathComponent("\(message.id)/\($0)")}
        if var resource {
            resource = try await processResponse(at: resource)
        }

        let request = Request(id: message.id, name: message.name, payload: messageData, resource: resource)
        for continuation in continuations.values {
            continuation.yield(request)
        }

        try? fm.removeItem(atPath: messageURL.path)
    }

    private func processIncomingMessages(urls: Set<URL>) async {
        for url in urls {
            let task = Task {
                do {
                    try await Self.retryWithBackoff(operation: {[weak self] in
                        try await self?.processIncomingMessage(at: url)
                    }, maxAttempts: maxRetryAttempts)
                } catch {
                    let errorType = type(of: error)
                    let errorDescription = error.localizedDescription
                    print("Failed to process incoming message at \(url.lastPathComponent): type=\(errorType), description=\(errorDescription)")
                }
            }

            self.tasks.append(task)
        }
    }

    /// Listens for incoming requests.
    ///
    /// This method sets up an asynchronous stream of incoming requests. It continuously
    /// monitors for new requests and yields them as they arrive.
    ///
    /// - Returns: An `AsyncStream` of `Request` objects representing incoming requests.
    /// - Throws: An error if the listening process fails to start or encounters an error while running.
    public func receive() async throws -> AsyncStream<Request> {
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

    /// Sends a response by creating a payload and notifying the client.
    ///
    /// This method performs the following steps:
    /// 1. Creates a payload using the provided response data.
    /// 2. Stores the payload in a designated directory.
    /// 3. Sends a Darwin notification to inform the client about the available response.
    ///
    /// - Parameter value: The `Response` object containing the data to be sent.
    ///   This includes:
    ///   - `original`: The original payload data.
    ///   - `id`: A unique identifier for the response.
    ///   - `resource`: An optional URL to any associated resource.
    ///
    func send(value: Response) async throws {
        let _ = try Self.createPayload(
            payloadData: value.original,
            identifier: value.id,
            resourceURL: value.resource,
            directoryURL: responsesDirectoryURL,
            fileManager: fm
        )

        DarwinNotificationCenter.shared.postNotification(named: "com.apple.modelrunner.response.ready")
    }

    func shutdown() async {
        await messagesDirectoryMonitor.shutdown()
        for task in tasks {
            task.cancel()
        }
    }
}

fileprivate struct Empty: Codable & Sendable {}

/// A protocol that defines a type which can be initialized from a `DeviceCtlSocket.Request`.
///
/// Types conforming to this protocol can decode themselves from an incoming `DeviceCtl` request.
protocol DeviceCtlRequestDecodable {
    init(request: DeviceCtlSocket.Request) throws
}

/// A protocol that defines a type which can be converted to a DeviceCtlSocket.Response.
///
/// Types conforming to this protocol can encode themselves into a `DeviceCtl` response,
protocol DeviceCtlResponseEncodable {
    func convert() throws -> DeviceCtlSocket.Response
}

/// An actor that represents a `DeviceCtl` service, handling requests and responses asynchronously.
///
/// This service uses a specific `AsyncRequestProcessor` to process requests and generate responses.
actor DeviceCtlService<ProcessorType, SocketType>: Service where ProcessorType: AsyncRequestProcessor,
                                                                 ProcessorType.Request: DeviceCtlRequestDecodable & Identifiable,
                                                                 ProcessorType.Request.ID == String,
                                                                 ProcessorType.Response: DeviceCtlResponseEncodable & Identifiable,
                                                                 ProcessorType.Response.ID == String,
                                                                 SocketType: Socket,
                                                                 SocketType.Request == DeviceCtlSocket.Request,
                                                                 SocketType.Response ==  DeviceCtlSocket.Response {

    typealias SocketType = SocketType

    /// The socket used for sending and receiving messages.
    nonisolated let socket: SocketType

    /// The processor responsible for handling requests and generating responses.
    let processor: ProcessorType

    /// The name of the service.
    let name: String

    /// The unique error code assigned to this service.
    let serviceErrorCode: Int

    /// An optional task representing ongoing asynchronous work.
    var task: Task<Void, Never>? = nil

    /// Retrieves the current task asynchronously.
    ///
    /// - Returns: The current `Task<Void, Never>?` if one exists, otherwise `nil`.
    func getTask() async -> Task<Void, Never>? {
        return task
    }

    /// Sets the current task asynchronously.
    ///
    /// - Parameter task: The `Task<Void, Never>?` to be set.
    func setTask(_ task: Task<Void, Never>?) async {
        self.task = task
    }

    /// Processes an incoming request and generates a response.
    ///
    /// This method decodes the incoming request, processes it using the `processor`,
    /// and encodes the response. It also handles any errors that occur during processing.
    ///
    /// - Parameter request: The incoming `DeviceCtlSocket.Request` to be processed.
    /// - Returns: A `DeviceCtlSocket.Response` containing the result or error information.
    func process(request: DeviceCtlSocket.Request) async -> DeviceCtlSocket.Response {
        func convert(error: any Error) ->DeviceCtlSocket.Response {
            let encoder = JSONEncoder()
            let jsonError = JSONRPCError(code: serviceErrorCode, message: error.localizedDescription)
            let jsonResponse = JSONRPCResponse<Empty>(
                jsonrpc: "2.0",
                id: request.id,
                result: Empty(),
                resource: nil,
                error: jsonError
            )
            let jsonData = try? encoder.encode(jsonResponse)
            return DeviceCtlSocket.Response(id: jsonResponse.id, original: jsonData ?? Data(), resource: nil)
        }

        var response: DeviceCtlSocket.Response? = nil

        do {
            let processorRequest = try ProcessorType.Request.init(request: request)
            let processorResponse = try await processor.process(request: processorRequest)
            response = try processorResponse.convert()
        } catch {
            response = convert(error: error)
        }

        guard let response else {
            fatalError("Inconsistent State.")
        }

        return response
    }

    /// Initializes a new instance of DeviceCtlService.
    ///
    /// - Parameters:
    ///   - socket: The socket  to be used for communication.
    ///   - processor: The `ProcessorType` to be used for processing requests.
    ///   - name: The name of the service.
    ///   - serviceErrorCode: The unique error code assigned to this service.
    init(socket: SocketType,
         processor: ProcessorType,
         name: String,
         serviceErrorCode: Int) {
        self.socket = socket
        self.processor = processor
        self.name = name
        self.serviceErrorCode = serviceErrorCode
    }
}

extension JSONRPCError {
    /// The model service error code.
    static let modelServiceErrorCode = -32001
}

extension JSONRPCError {
    /// The passthrough service error code.
    static let passthroughServiceErrorCode = -32002
}

extension Model {
    /// Creates and returns a new service instance for handling model-related requests.
    ///
    /// This method sets up a `DeviceCtlService` with a `ModelRequestProcessor` to handle
    /// requests specific to the `Model` type. It encapsulates the creation of the processor
    /// and the service, providing a convenient way to instantiate a model-specific service.
    ///
    /// - Parameter socket: The `DeviceCtlSocket` instance to be used for communication.
    ///                     This socket will be used by the service to send and receive messages.
    ///
    /// - Returns: A new service instance conforming to the `Service` protocol. The specific
    ///            type is a `DeviceCtlService` configured with a `ModelRequestProcessor`.
    ///
    /// - Note: The returned service uses "ModelService" as its name and a predefined
    ///         error code (`JSONRPCError.modelServiceErrorCode`) for error handling.
    ///
    /// Usage Example:
    /// ```
    /// let socket = DeviceCtlSocket()
    /// let modelService = Model.makeDeviceCtlService(socket: socket)
    /// // Use modelService for handling model-related requests
    /// ```
    static func makeDeviceCtlService<SocketType>(socket: SocketType) -> some Service where SocketType: Socket,
                                                                                           SocketType.Request == DeviceCtlSocket.Request,
                                                                                           SocketType.Response == DeviceCtlSocket.Response  {
        let processor = ModelRequestProcessor()
        let service = DeviceCtlService<ModelRequestProcessor, SocketType>(
            socket: socket,
            processor: processor,
            name: "MLModelService",
            serviceErrorCode: JSONRPCError.modelServiceErrorCode
        )

        return service
    }
}

enum PassThrough {
    /// A generic message structure that can be encoded, decoded, and identified.
    ///
    /// - Note: The generic type `T` must conform to both `Codable` and `Sendable` protocols.
    struct Message<T>: Codable & Sendable & NamedRequest & Identifiable where T: Codable & Sendable {
        /// Unique identifier for the message.
        let id: String
        /// The payload of the message.
        let value: T
        /// The name of the request.
        let name: String
    }

    /// An actor that processes requests by simply passing them through unchanged.
    ///
    /// This processor doesn't modify the incoming requests, it just returns them as responses.
    actor Processor<Value>: AsyncRequestProcessor {
        typealias Request = Value
        typealias Response = Value

        func process(request: Request) async throws -> Response {
            return request
        }
    }
}

extension PassThrough.Message: DeviceCtlRequestDecodable {
    /// Initializes a `PassThrough.Message` from a `DeviceCtlSocket.Request`.
    ///
    /// - Parameter request: The incoming request to decode.
    /// - Throws: An error if decoding fails.
    init(request: DeviceCtlSocket.Request) throws {
        let decoder = JSONDecoder()
        self = try decoder.decode(PassThrough.Message<T>.self, from: request.payload)
    }
}

extension PassThrough.Message: DeviceCtlResponseEncodable {
    /// Converts the message to a `DeviceCtlSocket.Response`.
    ///
    /// - Returns: A `DeviceCtlSocket.Response` containing the encoded message.
    /// - Throws: An error if encoding fails.
    func convert() throws -> DeviceCtlSocket.Response {
        let encoder = JSONEncoder()
        let jsonResponse = JSONRPCResponse<PassThrough.Message<T>>(
            jsonrpc: "2.0",
            id: id,
            result: self,
            resource: nil,
            error: nil
        )
        let jsonData = try encoder.encode(jsonResponse)
        return DeviceCtlSocket.Response(id: jsonResponse.id, original: jsonData, resource: nil)
    }
}

extension PassThrough {
    /// Creates a pass-through service for handling generic messages.
    ///
    /// This method sets up a `DeviceCtlService` with a `PassThrough.Processor` to handle
    /// messages of a specific type without modifying them.
    ///
    /// - Parameter socket: The `DeviceCtlSocket` instance to be used for communication.
    /// - Returns: A new service instance conforming to the `Service` protocol.
    ///
    /// Usage Example:
    /// ```
    /// let socket = DeviceCtlSocket()
    /// let passthroughService = PassThrough.makeDeviceCtlService<Int>(socket: socket)
    /// ```
    static func makeDeviceCtlService<ValueType, SocketType>(_ : ValueType.Type, socket: SocketType) -> some Service where ValueType: Codable & Sendable,
                                                                                                                          SocketType: Socket,
                                                                                                                          SocketType.Request == DeviceCtlSocket.Request,
                                                                                                                          SocketType.Response == DeviceCtlSocket.Response {
        let processor = PassThrough.Processor<Message<ValueType>>()
        let service = DeviceCtlService<PassThrough.Processor<Message<ValueType>>, SocketType>(
            socket: socket,
            processor: processor,
            name: "PassthroughService",
            serviceErrorCode: JSONRPCError.modelServiceErrorCode
        )

        return service
    }
}
