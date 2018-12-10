#ifndef MLMODEL_ONE_HOT_ENCODER_SPEC_HPP
#define MLMODEL_ONE_HOT_ENCODER_SPEC_HPP

#include "../Result.hpp"
#include "../Model.hpp"
#include "../../build/format/OneHotEncoder_enums.h"


namespace CoreML {
namespace Model {

    Result setHandleUnknown(CoreML::Specification::OneHotEncoder* ohe, MLHandleUnknown state);
    Result setUseSparse(CoreML::Specification::OneHotEncoder* ohe, bool state);
    Result setFeatureEncoding(CoreML::Specification::OneHotEncoder* ohe, const std::vector<int64_t>& container);
    Result setFeatureEncoding(CoreML::Specification::OneHotEncoder* ohe, const std::vector<std::string>& container);
    Result getHandleUnknown(CoreML::Specification::OneHotEncoder* ohe, MLHandleUnknown *state);
    Result getUseSparse(CoreML::Specification::OneHotEncoder* ohe, bool *state);
    Result getFeatureEncoding(CoreML::Specification::OneHotEncoder* ohe, std::vector<int64_t>& container);
    Result getFeatureEncoding(CoreML::Specification::OneHotEncoder* ohe, std::vector<std::string>& container);

} // namespace Model
} // namespace CoreML

#endif
