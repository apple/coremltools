//
//  NeuralNetworkBuffer.cpp
//  CoreML
//
//  Created by Bhushan Sonawane on 11/8/19.
//  Copyright © 2019 Apple Inc. All rights reserved.
//

#include "NeuralNetworkBuffer.hpp"
#include <fstream>
#include <vector>
#include <iostream>

namespace NNBuffer {
    /*
     * NeuralNetworkBuffer - NeuralNetworkBuffer
     */
    NeuralNetworkBuffer::NeuralNetworkBuffer(const std::string &bufferFilePath)
        : bufferFilePath(bufferFilePath),
          bufferStream(bufferFilePath, std::fstream::in | std::fstream::out | std::ios::binary | std::ios::app)
    {
    }

    /*
     * NeuralNetworkBuffer - NeuralNetworkBuffer
     */
    NeuralNetworkBuffer::~NeuralNetworkBuffer() = default;

    /*
     * NeuralNetworkBuffer - addBuffer
     * Writes given data into buffer file
     * Writes in following order
     * [Length of data, data type, data]
     * Number of bytes written = Length_Of_Data * Size_Of_Data_Type
     */
    template<class T>
    uint64_t NeuralNetworkBuffer::addBuffer(const std::vector<T> &buffer) {
        bufferStream.seekp(0, std::ios::end);
        // Get offset
        uint64_t offset = bufferStream.tellp();
        // Write length, size of data type and buffer
        uint64_t lenOfBuffer = buffer.size();
        uint64_t sizeOfBlock = sizeof(T);
        bufferStream.write((char*)&lenOfBuffer, sizeof(lenOfBuffer));
        bufferStream.write((char*)&sizeOfBlock, sizeof(sizeOfBlock));
        bufferStream.write((char*)&buffer[0], sizeOfBlock * lenOfBuffer);
        return offset;
    }

    /*
     * NeuralNetworkBuffer - getBuffer
     * Reads data from given offset
     */
    template<class T>
    void NeuralNetworkBuffer::getBuffer(const uint64_t offset, std::vector<T> &buffer) {
        uint64_t lenOfBuffer = 0;
        uint64_t sizeOfBlock = 0;
        bufferStream.seekg(offset, std::ios::beg);
        // Read length of buffer and size of each block
        bufferStream.read((char*)&lenOfBuffer, sizeof(lenOfBuffer));
        bufferStream.read((char*)&sizeOfBlock, sizeof(sizeOfBlock));
        // TODO: assert if sizeOfBlock != sizeof(T) or resize accordingly.
        // Resize buffer to fit buffer
        buffer.resize(lenOfBuffer);
        // Read buffer
        bufferStream.read((char*)&buffer[0], sizeOfBlock * lenOfBuffer);
    }

    // Explicit include templated functions
    template uint64_t NeuralNetworkBuffer::addBuffer<int>(const std::vector<int> &);
    template uint64_t NeuralNetworkBuffer::addBuffer<float>(const std::vector<float> &);
    template uint64_t NeuralNetworkBuffer::addBuffer<double>(const std::vector<double> &);

    template void NeuralNetworkBuffer::getBuffer<int>(const uint64_t, std::vector<int> &);
    template void NeuralNetworkBuffer::getBuffer<float>(const uint64_t, std::vector<float> &);
    template void NeuralNetworkBuffer::getBuffer<double>(const uint64_t, std::vector<double> &);
}