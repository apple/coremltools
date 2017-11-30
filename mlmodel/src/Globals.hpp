#ifndef MLMODEL_GLOBALS_HPP
#define MLMODEL_GLOBALS_HPP

#include <memory>
#include <string>
#include <vector>

#include "DataType.hpp"
#include "Export.hpp"

namespace CoreML {

    typedef std::vector<std::pair<std::string, FeatureType>> SchemaType;
    // Version 1 shipped as iOS 11.0
    static const int32_t MLMODEL_SPECIFICATION_VERSION_IOS11 = 1;
    // Version 2 supports fp16 weights and custom layers in neural network models.
    static const int32_t MLMODEL_SPECIFICATION_VERSION = 2;
}

#endif
