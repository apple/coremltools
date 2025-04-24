//
//  Copyright (c) 2025, Apple Inc. All rights reserved.
//
//  Use of this source code is governed by a BSD-3-clause license that can be
//  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

extension URL {
    /// Returns the URL for the app's documents directory.
    ///
    /// - Returns: The URL of the documents directory, or `nil` if it cannot be accessed.
    static var documents: URL? {
        return FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
    }
}

extension FileManager {
    /// Checks if a directory exists at the specified path.
    ///
    /// - Parameter path: The path to check for a directory.
    /// - Returns: `true` if a directory exists at the specified path, `false` otherwise.
    func directoryExists(at path: String) -> Bool {
        var isDirectory: ObjCBool = false
        if !fileExists(atPath: path, isDirectory:&isDirectory) {
            return false
        }

        return isDirectory.boolValue
    }
}

struct DeviceCtlCommand: Identifiable, Equatable  {
    enum Kind {
        case request
        case response
    }

    let kind: Kind
    let name: String
    let commandID: String
    let timesstamp: TimeInterval
    let id: UUID = UUID()
}

@MainActor
class DeviceCtlCommandLogger: ObservableObject {
    @Published var commands: [DeviceCtlCommand] = []

    func logRequest(name: String, commandID: String) async {
        let command = DeviceCtlCommand(
            kind: DeviceCtlCommand.Kind.request,
            name: name,
            commandID: commandID,
            timesstamp: Date().timeIntervalSince1970
        )
        commands.append(command)
    }

    func logResponse(commandID: String) async {
        let name = commands.first(where: { $0.commandID == commandID })?.name ?? ""
        let command = DeviceCtlCommand(
            kind: DeviceCtlCommand.Kind.response,
            name: name,
            commandID: commandID,
            timesstamp: Date().timeIntervalSince1970
        )

        commands.append(command)
    }

    func clear() async {
        commands.removeAll()
    }
}

/// An actor that manages the lifecycle of a `DeviceCtl` server.
actor DeviceCtlServerManager: ObservableObject {
    /// The URL of the directory used by the DeviceCtl Socket.
    private let directoryURL: URL

    /// The managed server instance.
    private var server: Server? = nil

    /// The socket used for communication with the server.
    private var socket: DeviceCtlSocket? = nil

    /// The command logger
    nonisolated let commandLogger: DeviceCtlCommandLogger


    enum Error: Swift.Error {
        case modelDirectoryCreationFailed(String)
    }

    /// Initializes a new ServerManager instance.
    ///
    /// This initializer creates a directory for the server if it doesn't exist,
    /// or cleans up an existing directory before recreating it.
    ///
    /// - Parameter name: The name of the directory to be created in the documents folder.
    /// - Parameter commandLogger: The command logger..
    init(name: String, commandLogger:DeviceCtlCommandLogger) throws {
        let fm = FileManager.default
        guard let directoryURL = fm.urls(for: .documentDirectory, in: .userDomainMask).first?.appendingPathComponent(name) else {
            throw Error.modelDirectoryCreationFailed("Failed to construct the model directory path for \(name)")
        }

        if !fm.directoryExists(at: directoryURL.path) {
            do {
                try fm.createDirectory(at: directoryURL, withIntermediateDirectories: false, attributes: nil)
            } catch {
                throw Error.modelDirectoryCreationFailed("Failed to create the model directory at \(directoryURL.path): \(error.localizedDescription)")
            }
        }

        self.directoryURL = directoryURL
        self.commandLogger = commandLogger

    }

    /// Starts the server .
    func start() async throws {
        let socket = try await DeviceCtlSocket(directoryURL: directoryURL)
        let commandLogger = self.commandLogger
        let wrappedSocket = try await AdapterSocket<DeviceCtlSocket.Request, DeviceCtlSocket.Response>(
            socket: socket,
            receive: { request in
                Task {
                    await commandLogger.logRequest(name: request.name, commandID: request.id)
                }

                return request
            },
            send: { response in
                Task {
                    await commandLogger.logResponse(commandID: response.id)
                }

                return response
            }
        )
        let server = Server.makeDeviceCtlServer(socket: wrappedSocket)
        try await server.resume()
        self.socket = socket
        self.server = server
    }

    deinit {
        guard let socket else {
            return
        }
        // Ensure the server is properly shut down when the app terminates
        Task {
            await socket.shutdown()
            print("Socket shutdown successfully")
        }
    }
}
