//
//  Copyright (c) 2024, Apple Inc. All rights reserved.
//
//  Use of this source code is governed by a BSD-3-clause license that can be
//  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import SwiftUI

extension Color {
    static var systemBackground: Color {
#if os(macOS)
        return Color(NSColor.windowBackgroundColor)
#elseif os(watchOS)
        return Color.gray
#else
        return Color(UIColor.systemBackground)
#endif
    }
}

struct DeviceCommandView: View {
    let command: DeviceCtlCommand
    let dateFormatter: DateFormatter
    @Environment(\.colorScheme) var colorScheme

    init(command: DeviceCtlCommand) {
        self.command = command
        dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "HH:mm:ss"
    }


    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            VStack(alignment: .leading) {
                Text(command.name)
                    .font(.headline)
                    .foregroundColor(.primary)
            }
            VStack(alignment: .leading) {
                Text("Type: ")
                    .font(.headline)
                    .foregroundColor(.primary)
                Text(command.kind == .request ? "Request" : "Response")
                    .font(.body)
                    .foregroundColor(.secondary)
            }
            VStack(alignment: .leading) {
                Text("Time: ")
                    .font(.headline)
                    .foregroundColor(.primary)
                Text(dateFormatter.string(from: Date(timeIntervalSince1970: command.timesstamp)))
                    .font(.body)
                    .foregroundColor(.secondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(Color.systemBackground)
        .cornerRadius(10)
        .shadow(color: colorScheme == .dark ? Color.white.opacity(0.5) : Color.black.opacity(0.5), radius: 5, x: 0, y: 2)
    }
}

struct SpinnerView: View {
    var body: some View {
        ProgressView()
            .progressViewStyle(CircularProgressViewStyle(tint: .primary))
            .scaleEffect(2.0) // Adjust size
    }
}

struct ContentView: View {
    @ObservedObject var commandLogger: DeviceCtlCommandLogger

    init(commandLogger: DeviceCtlCommandLogger) {
        self.commandLogger = commandLogger
    }

    var body: some View {
        NavigationStack {
            VStack {
                if commandLogger.commands.isEmpty {
                    SpinnerView()
                } else {
                    ScrollView {
                        LazyVStack(spacing: 20) {
                            ForEach(commandLogger.commands) { command in
                                DeviceCommandView(command: command)
                                    .animation(.easeInOut, value: command)
                            }
                        }
                        .padding()
                    }
                }

            }.navigationTitle("Commands")
                .toolbar {
                    ToolbarItem(placement: .primaryAction) {
                        Button("Clear") {
                            Task {
                                await commandLogger.clear()
                            }
                        }
                    }

                }
        }
#if !os(macOS)
        .onAppear {
            /// Prevent the screen from automatically locking while the app is active.
            UIApplication.shared.isIdleTimerDisabled = true
        }
        .onDisappear {
            UIApplication.shared.isIdleTimerDisabled = false
        }
#endif
    }
}

#Preview {
    ContentView(commandLogger: DeviceCtlCommandLogger())
}
