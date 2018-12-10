#include "MLModelTests.hpp"

// TODO -- Fix these headers.
#include "../src/Model.hpp"
#include "../src/transforms/OneHotEncoder.hpp"
#include "../src/transforms/LinearModel.hpp"
#include "../src/transforms/TreeEnsemble.hpp"


#include "framework/TestUtils.hpp"
#include <cassert>
#include <cstdio>


typedef CoreML::TreeEnsembleBase::BranchMode BranchMode;

int testTreeEnsembleBasic() {
    CoreML::TreeEnsembleRegressor tr("z", "");
    tr.setDefaultPredictionValue(0.0);
    tr.setupBranchNode(0, 0, 1, BranchMode::BranchOnValueGreaterThan, 5, 1, 2);
    tr.setupLeafNode(0, 1, 1);
    tr.setupLeafNode(0, 2, 2);

    ML_ASSERT_GOOD(CoreML::Model::addInput(&tr.getProto(), "x", CoreML::FeatureType::Double()));
    ML_ASSERT_GOOD(CoreML::Model::addInput(&tr.getProto(), "y", CoreML::FeatureType::Double()));
    ML_ASSERT_GOOD(CoreML::Model::addOutput(&tr.getProto(), "z", CoreML::FeatureType::Double()));
    ML_ASSERT_EQ(CoreML::Model::modelType(tr.getProto()), MLModelType_treeEnsembleRegressor);

    CoreML::SchemaType expectedInputSchema {
        {"x", CoreML::FeatureType::Double()},
        {"y", CoreML::FeatureType::Double()},
    };

    CoreML::SchemaType expectedOutputSchema {
        {"z", CoreML::FeatureType::Double()},
    };

    ML_ASSERT_EQ(CoreML::Model::inputSchema(tr.getProto()), expectedInputSchema);
    ML_ASSERT_EQ(CoreML::Model::outputSchema(tr.getProto()), expectedOutputSchema);

    // TODO -- don't leave stuff in /tmp
    ML_ASSERT_GOOD(CoreML::Model::save(tr.getProto(), "/tmp/tA-tree.mlmodel"));

    CoreML::Specification::Model loadedA;
    ML_ASSERT_GOOD(CoreML::Model::load(&loadedA, "/tmp/tA-tree.mlmodel"));

    ML_ASSERT_EQ(CoreML::Model::modelType(loadedA), MLModelType_treeEnsembleRegressor);

    ML_ASSERT_EQ(CoreML::Model::inputSchema(loadedA), expectedInputSchema);
    ML_ASSERT_EQ(CoreML::Model::outputSchema(loadedA), expectedOutputSchema);
    return 0;
}
