//
//  NeuralNetworkBuffer.hpp
//  CoreML
//
//  Created by Bhushan Sonawane on 11/8/19.
//  Copyright © 2019 Apple Inc. All rights reserved.
//

#pragma once

#include <fstream>
#include <string>
#include <vector>

namespace NNBuffer {

enum class BufferMode {
    Write=0,
    Append,
    Read
};

//
// NeuralNetworkBuffer - Network parameter read-write management to file
// Current management policy
// Each parameter is written to binary file in following order.
// [Length of data (size_t), Data type of data (size_t), data (length of data * size of data type)]
//
class NeuralNetworkBuffer {
public:
    // Must be constructed with file path to store parameters
    NeuralNetworkBuffer(const std::string& bufferFilePath, BufferMode mode=BufferMode::Write);
    ~NeuralNetworkBuffer();

    NeuralNetworkBuffer(const NeuralNetworkBuffer&) = delete;
    NeuralNetworkBuffer(NeuralNetworkBuffer&&) = delete;
    NeuralNetworkBuffer& operator=(const NeuralNetworkBuffer&) = delete;
    NeuralNetworkBuffer& operator=(NeuralNetworkBuffer&&) = delete;

    // Stores given buffer and returns offset in buffer file
    template<class T>
    uint64_t AddBuffer(const std::vector<T>& buffer);

    // Reads buffer from given offset and stores in vector
    // passed by reference.
    // Note that, this routine resizes the given vector.
    template<class T>
    void GetBuffer(uint64_t offset, std::vector<T>& buffer);

private:
    std::string bufferFilePath;
    std::fstream bufferStream;
};

} // namespace NNBuffer
