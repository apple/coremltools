#include "OneHotEncoder.hpp"
#include "../Format.hpp"

namespace CoreML {
namespace Model {

    Result setHandleUnknown(Specification::OneHotEncoder* ohe, MLHandleUnknown state) {
        ohe->set_handleunknown(static_cast<Specification::OneHotEncoder_HandleUnknown>(state));
        return Result();
    }
    
    Result setUseSparse(Specification::OneHotEncoder* ohe, bool state) {
        ohe->set_outputsparse(state);
        return Result();
    }
    
    Result setFeatureEncoding(Specification::OneHotEncoder* ohe, const std::vector<int64_t>& container) {
        ohe->clear_int64categories();

        for (auto element : container) {
            ohe->mutable_int64categories()->add_vector(element);
        }
        return Result();
    }
    
    Result setFeatureEncoding(Specification::OneHotEncoder* ohe, const std::vector<std::string>& container) {
        ohe->clear_stringcategories();
        
        for (auto element : container) {
            std::string *value = ohe->mutable_stringcategories()->add_vector();
            *value = element;
        }
        return Result();
    }

    Result getHandleUnknown(const Specification::OneHotEncoder& ohe, MLHandleUnknown *state) {
        if (state != nullptr) {
            *state = static_cast<MLHandleUnknown>(ohe.handleunknown());
        }
        return Result();
    }

    Result getUseSparse(const Specification::OneHotEncoder& ohe, bool *state) {
        if (state != nullptr) {
            *state = ohe.outputsparse();
        }
        return Result();
    }

    Result getFeatureEncoding(const Specification::OneHotEncoder& ohe, std::vector<int64_t>& container) {
        for (int i = 0; i < ohe.int64categories().vector_size(); i++) {
            container.push_back(ohe.int64categories().vector(i));
        }
        return Result();
    }

    Result getFeatureEncoding(const Specification::OneHotEncoder& ohe, std::vector<std::string>& container) {
        for (int i = 0; i < ohe.stringcategories().vector_size(); i++) {
            container.push_back(ohe.stringcategories().vector(i));
        }
        return Result();
    }

} // namespace Model
} // namespace CoreML
