//
//  Copyright (c) 2024, Apple Inc. All rights reserved.
//
//  Use of this source code is governed by a BSD-3-clause license that can be
//  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import SwiftUI

@main
struct ModelRunnerApp: App {
    private var serverManager: DeviceCtlServerManager

    init() {
        do {
            serverManager = try DeviceCtlServerManager(name: "modelrunnerd", commandLogger: DeviceCtlCommandLogger())
        } catch {
            fatalError("Failed to initialize server: \(error)")
        }
    }

    var body: some Scene {
        WindowGroup {
            ContentView(commandLogger: serverManager.commandLogger)
                .task {
                    do {
                        try await serverManager.start()
                    } catch {
                        print("Failed to start server: \(error)")
                    }
                }
        }
    }
}
