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

int testBasicSaveLoad () {
    Result r;
    
    const std::string path("/tmp/a.modelasset");
    Specification::Model model;
    ML_ASSERT_GOOD(Model::addInput(&model, "foo", FeatureType::String()));
    model.mutable_onehotencoder()->mutable_stringcategories()->add_vector()->assign("foo");
    ML_ASSERT_GOOD(Model::addOutput(&model, "bar", FeatureType::Array({})));
    ML_ASSERT_GOOD(Model::save(model, path));
    Specification::Model a2;
    ML_ASSERT_GOOD(Model::load(&a2, path));
    ML_ASSERT_EQ(model, a2);
    
    return 0;
}
