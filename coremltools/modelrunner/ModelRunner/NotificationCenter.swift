//
//  Copyright (c) 2025, Apple Inc. All rights reserved.
//
//  Use of this source code is governed by a BSD-3-clause license that can be
//  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// A singleton class for posting Darwin notifications.
///
/// `DarwinNotificationCenter` provides a simple interface to post system-wide notifications using
/// the Darwin notification center. This is particularly useful for inter-process communication
/// between different parts of an application or between different applications.
final class DarwinNotificationCenter {
    /// The shared singleton instance of `DarwinNotificationCenter`.
    static let shared = DarwinNotificationCenter()

    /// The underlying Darwin notification center.
    private let center = CFNotificationCenterGetDarwinNotifyCenter()

    /// Private initializer to ensure the class is used as a singleton.
    private init() {}

    /// Posts a Darwin notification with the specified name.
    ///
    /// - Parameter name: The name of the notification to post. It should be a unique string identifier
    ///                   that other processes or parts of your application can listen for.
    ///
    /// - Note: The notification is posted system-wide and can be observed by any process that registers
    ///         for this notification name.
    func postNotification(named name: String) {
        let name = CFNotificationName(name as CFString)
        CFNotificationCenterPostNotification(center, name, nil, nil, true)
    }
}
