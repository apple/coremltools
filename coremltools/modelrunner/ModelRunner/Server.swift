//
//  Copyright (c) 2025, Apple Inc. All rights reserved.
//
//  Use of this source code is governed by a BSD-3-clause license that can be
//  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// An actor that manages multiple services and coordinates their operations.
actor Server {
    /// The collection of services managed by this server.
    private let services: [any Service]
    /// A list of service names.
    let serviceNames: [String]

    /// Initializes a new Server instance with the given services.
    ///
    /// - Parameters:
    ///  - services: An array of objects conforming to the `Service` protocol. These services will be managed by the server.
    init(services: [any Service]) {
        self.services = services
        serviceNames = services.map { $0.name }
    }

    /// Resumes all services managed by this server.
    func resume() async throws {
        for var service in services {
            try await service.resume()
        }
    }

    /// Suspends all services managed by this server.
    func suspend() async throws {
        for service in services {
            await service.suspend()
        }
    }
}

extension Server {
    /// This method initializes a `Server` with two predefined services:
    /// - A `Model` service for handling model-related requests.
    /// - A `PassThrough` service for forwarding requests without modification.
    ///
    /// The provided socket is used for communication with the services.
    ///
    /// - Parameter socket: The `DeviceCtlSocket` instance to be used for sending and receiving messages.
    ///
    /// - Returns: A new `Server` instance configured with the default services.
    static func makeDeviceCtlServer<SocketType>(socket: SocketType) -> Server where SocketType: Socket,
                                                                                    SocketType.Request == DeviceCtlSocket.Request,
                                                                                    SocketType.Response == DeviceCtlSocket.Response {
        let modelService = Model.makeDeviceCtlService(socket: socket)
        let passThroughService = PassThrough.makeDeviceCtlService(String.self, socket: socket)
        return Server(services: [modelService, passThroughService])
    }

}
