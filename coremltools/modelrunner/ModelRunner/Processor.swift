//
//  Copyright (c) 2025, Apple Inc. All rights reserved.
//
//  Use of this source code is governed by a BSD-3-clause license that can be
//  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// A protocol that defines an asynchronous request processing capability.
protocol AsyncRequestProcessor {
    /// The type of request that this processor can handle.
    associatedtype Request

    /// The type of response that this processor will return.
    associatedtype Response

    /// Processes the given request and returns a response asynchronously.
    ///
    /// This method takes a request of type `Request`, performs any necessary
    /// processing or computation, and returns a response of type `Response`.
    ///
    /// - Parameter request: The input request to be processed.
    ///
    /// - Returns: A `Response` object representing the result of processing the request.
    func process(request: Request) async throws -> Response
}
