//
//  Copyright (c) 2025, Apple Inc. All rights reserved.
//
//  Use of this source code is governed by a BSD-3-clause license that can be
//  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause


import Foundation

// Extension to FileManager to get directory contents as a Set of URLs
extension FileManager {
    fileprivate func contentsOfDirectoryBySkippingHiddenFiles(at directoryURL: URL) throws -> Set<URL> {
        // Get directory contents, skipping hidden files
        let urls = try contentsOfDirectory(at: directoryURL, includingPropertiesForKeys: [URLResourceKey.isDirectoryKey], options: [.skipsHiddenFiles])
        return Set(urls)
    }
}

// Actor to monitor directory changes
actor DirectoryMonitor {
    // Enum to represent different types of changes in the directory
    public enum Change: Equatable, Hashable  {
        case initial(Set<URL>)  // Initial state of the directory
        case added(Set<URL>)    // URLs added to the directory
        case removed(Set<URL>)  // URLs removed from the directory
    }

    /// The URL of the directory being monitored.
    private let url: URL
    /// The file descriptor for the monitored directory. Used by the dispatch source to watch for file system events.
    private let fileDescriptor: Int32
    /// The dispatch source used to monitor file system events.
    private let source: DispatchSourceFileSystemObject
    /// The current contents of the monitored directory.
    private var contents: Set<URL> = Set()
    /// FileManager instance to perform file operations
    private let fm: FileManager = FileManager()
    /// Dictionary to store active continuations
    private var continuations: [Int : AsyncStream<Change>.Continuation] = [:]
    /// ID for the next continuation
    private var nextContinuationID: Int = 0
    /// Dispatch queue for handling file system events
    private let queue: DispatchQueue = DispatchQueue(
        label: "com.apple.directoryMonitoring.queue",
        qos: .default,
        attributes: [],
        autoreleaseFrequency: .inherit,
        target: nil
    )

    // Initializer
    init(url: URL) throws {
        // Open the directory for event-only file descriptor
        fileDescriptor = open(url.path, O_EVTONLY)
        guard fileDescriptor >= 0 else {
            throw POSIXError(POSIXError.Code(rawValue: errno) ?? POSIXErrorCode.EIO)
        }

        self.url = url
        // Create a dispatch source for file system events
        source = DispatchSource.makeFileSystemObjectSource(
            fileDescriptor: fileDescriptor,
            eventMask: .write,
            queue: queue
        )

        // Set up event handler
        source.setEventHandler { [weak self] in
            guard let self else { return }
            Task {
                await self.handleDirectoryChanges()
            }
        }
    }

    public func shutdown() async {
        queue.sync {
            source.cancel()
        }
        close(fileDescriptor)
    }

    // Start monitoring the directory
    public func listen() async throws -> AsyncStream<Change> {
        defer {
            nextContinuationID += 1
        }

        let continuationID = nextContinuationID
        // Create a new stream and store its continuation
        let (stream, continuation) = AsyncStream.makeStream(of: Change.self)
        continuations[continuationID] = continuation


        // If this is the first continuation, initialize contents and resume the source
        if continuations.count == 1 && !source.isCancelled {
            contents = try fm.contentsOfDirectoryBySkippingHiddenFiles(at: url)
            source.resume()
        }

        // Set up termination handler for the continuation
        continuation.onTermination = { [weak self] _ in
            guard let self = self else { return }
            Task {
                await self.removeContinuation(continuationID)
            }
        }

        // Yield initial state
        continuation.yield(Change.initial(contents))
        return stream
    }

    // Remove a continuation when it's terminated
    private func removeContinuation(_ continuationID: Int) async {
        continuations.removeValue(forKey: continuationID)
        // If there are no more active continuations, stop monitoring
        if continuations.isEmpty && !source.isCancelled {
            source.suspend()
        }
    }

    // Handle changes in the directory
    private func handleDirectoryChanges() async {
        // Get new contents of the directory
        guard let newContents = try? fm.contentsOfDirectoryBySkippingHiddenFiles(at: url) else {
            return
        }

        // Find removed URLs
        let removedURLs = contents.subtracting(newContents)
        if !removedURLs.isEmpty {
            for continuation in continuations.values {
                continuation.yield(.removed(removedURLs))
            }
        }

        // Find added URLs
        let addedURLs = newContents.subtracting(contents)
        if !addedURLs.isEmpty {
            for continuation in continuations.values {
                continuation.yield(.added(addedURLs))
            }
        }

        // Update current contents
        contents = newContents
    }
}
