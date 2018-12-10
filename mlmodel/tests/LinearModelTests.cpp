#include "MLModelTests.hpp"

// TODO -- Fix these headers.
#include "../src/Model.hpp"
#include "../src/transforms/OneHotEncoder.hpp"
#include "../src/transforms/LinearModel.hpp"
#include "../src/transforms/TreeEnsemble.hpp"


#include "framework/TestUtils.hpp"
#include <cassert>
#include <cstdio>

using namespace CoreML;
using namespace CoreML::Model;

int testLinearModelBasic() {
    CoreML::Specification::Model model;
    initRegressor(&model, "foo", "Linear regression model to predict foo");
    ML_ASSERT_GOOD(addInput(&model, "x", FeatureType::Int64()));
    ML_ASSERT_GOOD(addInput(&model, "y", FeatureType::Double()));
    ML_ASSERT_GOOD(addInput(&model, "z", FeatureType::Int64()));
    ML_ASSERT_GOOD(addOutput(&model, "foo", FeatureType::Double()));
    return 0;
}
