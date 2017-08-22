#ifndef MLMODEL_GLOBALS_HPP
#define MLMODEL_GLOBALS_HPP

#include <memory>
#include <string>
#include <vector>

#include "DataType.hpp"
#include "Export.hpp"

namespace CoreML {

  typedef std::vector<std::pair<std::string, FeatureType>> SchemaType;
  static const int32_t MLMODEL_SPECIFICATION_VERSION = 1;
}

#endif
