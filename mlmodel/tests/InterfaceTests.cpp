#include "MLModelTests.hpp"
#include "../src/Model.hpp"
#include "../src/Format.hpp"

#include "framework/TestUtils.hpp"

using namespace CoreML;

static Specification::Model& addRequiredField(Specification::Model& model, const std::string& name) {
    auto *input = model.mutable_description()->add_input();
    input->set_name(name);
    input->mutable_type()->mutable_int64type();
    return model;
}

static Specification::Model& addOptionalField(Specification::Model& model, const std::string& name) {
    auto *input = model.mutable_description()->add_input();
    input->set_name(name);
    input->mutable_type()->mutable_int64type();
    input->mutable_type()->set_isoptional(true);
    return model;
}

static void setupMultiArrayFeature(Specification::FeatureDescription *feature, const std::string& name) {
    feature->set_name(name);
    feature->mutable_type()->mutable_multiarraytype()->set_datatype(::CoreML::Specification::ArrayFeatureType_ArrayDataType_FLOAT32);
    feature->mutable_type()->mutable_multiarraytype()->add_shape(1);
}

static void setupStateFeature(Specification::FeatureDescription *feature, const std::string& name) {
    feature->set_name(name);
    feature->mutable_type()->mutable_statetype()->mutable_arraytype()->set_datatype(::CoreML::Specification::ArrayFeatureType_ArrayDataType_FLOAT16);
    feature->mutable_type()->mutable_statetype()->mutable_arraytype()->add_shape(1);
}

static ValidationPolicy validationPolicyForStateTests() {
    auto validationPolicy = ValidationPolicy();
    validationPolicy.allowsEmptyInput = true;
    validationPolicy.allowsEmptyOutput = true;
    validationPolicy.allowsMultipleFunctions = true;
    validationPolicy.allowsStatefulPrediction = true;
    return validationPolicy;
}

int testOptionalInputs() {
    // Test that all fields are required on a random model (normalizer)
    Specification::Model m1;
    m1.mutable_normalizer()->set_normtype(Specification::Normalizer_NormType::Normalizer_NormType_L2);
    addRequiredField(m1, "x");
    ML_ASSERT_GOOD(validateOptional(m1));

    addOptionalField(m1, "y");
    ML_ASSERT_BAD(validateOptional(m1));

    // Test that at least one optional field is required on an imputer
    // (more than one allowed)
    Specification::Model m2;
    m2.mutable_imputer()->set_imputeddoublevalue(3.14);
    addRequiredField(m2, "x");
    addOptionalField(m2, "y");
    ML_ASSERT_GOOD(validateOptional(m2));

    addOptionalField(m2, "z");
    ML_ASSERT_GOOD(validateOptional(m2));

    // Test that any fields can be optional or required for trees
    Specification::Model m3;
    (void) m3.mutable_treeensembleregressor();
    addRequiredField(m3, "x");
    ML_ASSERT_GOOD(validateOptional(m3));

    Specification::Model m4;
    (void) m4.mutable_treeensembleregressor();
    addOptionalField(m4, "x");
    ML_ASSERT_GOOD(validateOptional(m4));

    return 0;
}


int testFeatureDescriptions() {

    Specification::Model m;

    auto *feature = m.mutable_description()->add_input();
    ML_ASSERT_BAD(validateFeatureDescription(*feature, MLMODEL_SPECIFICATION_VERSION_IOS11_2, FeatureIOType::INPUT));

    // Just with a name as invalid
    feature->set_name("test_input");
    ML_ASSERT_BAD(validateFeatureDescription(*feature, MLMODEL_SPECIFICATION_VERSION_IOS11_2, FeatureIOType::INPUT));

    // Empty type, still invalid
    feature->mutable_type();
    ML_ASSERT_BAD(validateFeatureDescription(*feature, MLMODEL_SPECIFICATION_VERSION_IOS11_2, FeatureIOType::INPUT));

    // Int64 typ, now its valid
    feature->mutable_type()->mutable_int64type();
    ML_ASSERT_GOOD(validateFeatureDescription(*feature, MLMODEL_SPECIFICATION_VERSION_IOS11_2, FeatureIOType::INPUT));

    // String type, valid
    feature->mutable_type()->mutable_stringtype();
    ML_ASSERT_GOOD(validateFeatureDescription(*feature, MLMODEL_SPECIFICATION_VERSION_IOS11_2, FeatureIOType::INPUT));

    // Double type, valid
    feature->mutable_type()->mutable_doubletype();
    ML_ASSERT_GOOD(validateFeatureDescription(*feature, MLMODEL_SPECIFICATION_VERSION_IOS11_2, FeatureIOType::INPUT));

    // Multiarray type, with no params, invalid
    feature->mutable_type()->mutable_multiarraytype();
    ML_ASSERT_BAD(validateFeatureDescription(*feature, MLMODEL_SPECIFICATION_VERSION_IOS11_2, FeatureIOType::INPUT));

    // Multiarray type, double with no shape, invalid as input, valid as output
    feature->mutable_type()->mutable_multiarraytype()->set_datatype(::CoreML::Specification::ArrayFeatureType_ArrayDataType_DOUBLE);
    ML_ASSERT_BAD(validateFeatureDescription(*feature, MLMODEL_SPECIFICATION_VERSION_IOS11_2, FeatureIOType::INPUT));
    feature->mutable_type()->mutable_multiarraytype()->set_datatype(::CoreML::Specification::ArrayFeatureType_ArrayDataType_DOUBLE);
    ML_ASSERT_GOOD(validateFeatureDescription(*feature, MLMODEL_SPECIFICATION_VERSION_IOS11_2, FeatureIOType::OUTPUT));

    feature->mutable_type()->mutable_multiarraytype()->set_datatype(::CoreML::Specification::ArrayFeatureType_ArrayDataType_FLOAT32);
    ML_ASSERT_BAD(validateFeatureDescription(*feature, MLMODEL_SPECIFICATION_VERSION_IOS11_2, FeatureIOType::INPUT));
    feature->mutable_type()->mutable_multiarraytype()->set_datatype(::CoreML::Specification::ArrayFeatureType_ArrayDataType_FLOAT32);
    ML_ASSERT_GOOD(validateFeatureDescription(*feature, MLMODEL_SPECIFICATION_VERSION_IOS11_2, FeatureIOType::OUTPUT));

    feature->mutable_type()->mutable_multiarraytype()->set_datatype(::CoreML::Specification::ArrayFeatureType_ArrayDataType_INT32);
    ML_ASSERT_BAD(validateFeatureDescription(*feature, MLMODEL_SPECIFICATION_VERSION_IOS11_2, FeatureIOType::INPUT));
    feature->mutable_type()->mutable_multiarraytype()->set_datatype(::CoreML::Specification::ArrayFeatureType_ArrayDataType_INT32);
    ML_ASSERT_GOOD(validateFeatureDescription(*feature, MLMODEL_SPECIFICATION_VERSION_IOS11_2, FeatureIOType::OUTPUT));

    // Zero length shape is invalid for inputs, but valid for outputs
    feature->mutable_type()->mutable_multiarraytype()->mutable_shape();
    ML_ASSERT_BAD(validateFeatureDescription(*feature, MLMODEL_SPECIFICATION_VERSION_IOS11_2, FeatureIOType::INPUT));
    ML_ASSERT_GOOD(validateFeatureDescription(*feature, MLMODEL_SPECIFICATION_VERSION_IOS11_2, FeatureIOType::OUTPUT));

    // Non-zero length shape, valid
    feature->mutable_type()->mutable_multiarraytype()->mutable_shape()->Add(128);
    ML_ASSERT_GOOD(validateFeatureDescription(*feature, MLMODEL_SPECIFICATION_VERSION_IOS11_2, FeatureIOType::INPUT));
    ML_ASSERT_GOOD(validateFeatureDescription(*feature, MLMODEL_SPECIFICATION_VERSION_IOS11_2, FeatureIOType::OUTPUT));

    // Dictionary, with no params, invalid
    feature->mutable_type()->mutable_dictionarytype();
    ML_ASSERT_BAD(validateFeatureDescription(*feature, MLMODEL_SPECIFICATION_VERSION_NEWEST, FeatureIOType::INPUT));

    // With key type, valid
    feature->mutable_type()->mutable_dictionarytype()->mutable_stringkeytype();
    ML_ASSERT_GOOD(validateFeatureDescription(*feature, MLMODEL_SPECIFICATION_VERSION_NEWEST, FeatureIOType::INPUT));

    feature->mutable_type()->mutable_dictionarytype()->mutable_int64keytype();
    ML_ASSERT_GOOD(validateFeatureDescription(*feature, MLMODEL_SPECIFICATION_VERSION_NEWEST, FeatureIOType::INPUT));

    // Image, with no params, invalid
    feature->mutable_type()->mutable_imagetype();
    ML_ASSERT_BAD(validateFeatureDescription(*feature, MLMODEL_SPECIFICATION_VERSION_NEWEST, FeatureIOType::INPUT));

    // With just width, invalid
    feature->mutable_type()->mutable_imagetype()->set_width(10);
    ML_ASSERT_BAD(validateFeatureDescription(*feature, MLMODEL_SPECIFICATION_VERSION_NEWEST, FeatureIOType::INPUT));

    // With both width and height, still invalid because no colorspace
    feature->mutable_type()->mutable_imagetype()->set_height(20);
    ML_ASSERT_BAD(validateFeatureDescription(*feature, MLMODEL_SPECIFICATION_VERSION_NEWEST, FeatureIOType::INPUT));

    // Now with colorspace, valid
    feature->mutable_type()->mutable_imagetype()->set_colorspace(::CoreML::Specification::ImageFeatureType_ColorSpace_BGR);
    ML_ASSERT_GOOD(validateFeatureDescription(*feature, MLMODEL_SPECIFICATION_VERSION_NEWEST, FeatureIOType::INPUT));
    feature->mutable_type()->mutable_imagetype()->set_colorspace(::CoreML::Specification::ImageFeatureType_ColorSpace_RGB);
    ML_ASSERT_GOOD(validateFeatureDescription(*feature, MLMODEL_SPECIFICATION_VERSION_NEWEST, FeatureIOType::INPUT));
    feature->mutable_type()->mutable_imagetype()->set_colorspace(::CoreML::Specification::ImageFeatureType_ColorSpace_GRAYSCALE);
    ML_ASSERT_GOOD(validateFeatureDescription(*feature, MLMODEL_SPECIFICATION_VERSION_NEWEST, FeatureIOType::INPUT));
    feature->mutable_type()->mutable_imagetype()->set_colorspace(::CoreML::Specification::ImageFeatureType_ColorSpace_GRAYSCALE_FLOAT16);
    ML_ASSERT_GOOD(validateFeatureDescription(*feature, MLMODEL_SPECIFICATION_VERSION_IOS16, FeatureIOType::INPUT));
    feature->mutable_type()->mutable_imagetype()->set_colorspace(::CoreML::Specification::ImageFeatureType_ColorSpace_INVALID_COLOR_SPACE);
    ML_ASSERT_BAD(validateFeatureDescription(*feature, MLMODEL_SPECIFICATION_VERSION_NEWEST, FeatureIOType::INPUT));

    //////////////////////////////////
    // Test more recent shape constraints
    Specification::Model m2;

    auto *feature2 = m2.mutable_description()->add_input();
    feature2->set_name("feature2");
    feature2->mutable_type()->mutable_imagetype()->set_colorspace(::CoreML::Specification::ImageFeatureType_ColorSpace_BGR);

    /// Fixed Size
    // Make fixed size  6 x 5
    feature2->mutable_type()->mutable_imagetype()->set_width(6);
    feature2->mutable_type()->mutable_imagetype()->set_height(5);
    ML_ASSERT_GOOD(validateFeatureDescription(*feature2, MLMODEL_SPECIFICATION_VERSION, FeatureIOType::INPUT));

    /// Enumerated
    // Add flexibility of a single enumerated size 6 x 5
    auto *shape = feature2->mutable_type()->mutable_imagetype()->mutable_enumeratedsizes()->add_sizes();
    shape->set_width(6);
    shape->set_height(5);
    ML_ASSERT_GOOD(validateFeatureDescription(*feature2, MLMODEL_SPECIFICATION_VERSION, FeatureIOType::INPUT));

    // Reset that to a single 10 x 5 which would make the 6 x 5 invalid!
    shape->set_width(10);
    shape->set_height(5);
    ML_ASSERT_BAD(validateFeatureDescription(*feature2, MLMODEL_SPECIFICATION_VERSION, FeatureIOType::INPUT));

    // Add 6 x 5 to the list so its now [10x5, 6 x 5] which should make it valid again
    shape = feature2->mutable_type()->mutable_imagetype()->mutable_enumeratedsizes()->add_sizes();
    shape->set_width(6);
    shape->set_height(5);
    ML_ASSERT_GOOD(validateFeatureDescription(*feature2, MLMODEL_SPECIFICATION_VERSION, FeatureIOType::INPUT));

    /// Range
    // Now make it a range that inclues 6 x 5
    auto* size_range = feature2->mutable_type()->mutable_imagetype()->mutable_imagesizerange();
    size_range->mutable_widthrange()->set_lowerbound(1);
    size_range->mutable_widthrange()->set_upperbound(-1); // unbounded

    size_range->mutable_heightrange()->set_lowerbound(2);
    size_range->mutable_heightrange()->set_upperbound(5);
    ML_ASSERT_GOOD(validateFeatureDescription(*feature2, MLMODEL_SPECIFICATION_VERSION, FeatureIOType::INPUT));

    // Now make the range not include 6 x 5
    size_range->mutable_widthrange()->set_lowerbound(7);
    ML_ASSERT_BAD(validateFeatureDescription(*feature2, MLMODEL_SPECIFICATION_VERSION, FeatureIOType::INPUT));

    // Fix it to include it again
    size_range->mutable_widthrange()->set_lowerbound(2);
    ML_ASSERT_GOOD(validateFeatureDescription(*feature2, MLMODEL_SPECIFICATION_VERSION, FeatureIOType::INPUT));

    // Fail due to upper bound can't be larger than lower
    size_range->mutable_widthrange()->set_upperbound(1);
    ML_ASSERT_BAD(validateFeatureDescription(*feature2, MLMODEL_SPECIFICATION_VERSION, FeatureIOType::INPUT));


    /////////////

    auto *array_type = feature2->mutable_type()->mutable_multiarraytype();
    array_type->set_datatype(::CoreML::Specification::ArrayFeatureType_ArrayDataType_FLOAT32);

    // 10 x 5 default size
    array_type->add_shape(10);
    array_type->add_shape(5);
    ML_ASSERT_GOOD(validateFeatureDescription(*feature2, MLMODEL_SPECIFICATION_VERSION, FeatureIOType::INPUT));

    // Range
    // Now specify ranges (>1 x [5...20])
    auto rangeForDim0 = array_type->mutable_shaperange()->add_sizeranges();
    rangeForDim0->set_lowerbound(1);
    rangeForDim0->set_upperbound(-1);

    auto rangeForDim1 = array_type->mutable_shaperange()->add_sizeranges();
    rangeForDim1->set_lowerbound(5);
    rangeForDim1->set_upperbound(20);
    ML_ASSERT_GOOD(validateFeatureDescription(*feature2, MLMODEL_SPECIFICATION_VERSION, FeatureIOType::INPUT));

    // Change to (>1 x [6..20]) which is not consistent with 10 x 5
    rangeForDim1->set_lowerbound(6);
    ML_ASSERT_BAD(validateFeatureDescription(*feature2, MLMODEL_SPECIFICATION_VERSION, FeatureIOType::INPUT));

    // Enumerated
    auto eshape1 = array_type->mutable_enumeratedshapes()->add_shapes();
    eshape1->add_shape(6);
    eshape1->add_shape(2);

    // Now allow [ 6x2 ] which is inconsistent with default 10 x 5
    ML_ASSERT_BAD(validateFeatureDescription(*feature2, MLMODEL_SPECIFICATION_VERSION, FeatureIOType::INPUT));

    // Add another one to make the set [6x2 , 10x5] which is consistent
    auto eshape2 = array_type->mutable_enumeratedshapes()->add_shapes();
    eshape2->add_shape(10);
    eshape2->add_shape(5);

    ML_ASSERT_GOOD(validateFeatureDescription(*feature2, MLMODEL_SPECIFICATION_VERSION, FeatureIOType::INPUT));

    return 0;
}

int testMultiFunctionSpecificationVersion() {
    Specification::Model m;

    auto validationPolicy = ValidationPolicy();
    validationPolicy.allowsEmptyInput = true;
    validationPolicy.allowsEmptyOutput = false;
    validationPolicy.allowsMultipleFunctions = true;

    auto *description = m.mutable_description();
    description->set_defaultfunctionname("foo");
    auto *function = description->add_functions();
    function->set_name("foo");

    setupMultiArrayFeature(function->add_input(), "x");
    setupMultiArrayFeature(function->add_output(), "y");

    // Check model specification version requirements.
    ML_ASSERT_BAD_WITH_TYPE(validateModelDescription(*description, MLMODEL_SPECIFICATION_VERSION_IOS17, validationPolicy), ResultType::INVALID_COMPATIBILITY_VERSION);
    ML_ASSERT_GOOD(validateModelDescription(*description, MLMODEL_SPECIFICATION_VERSION_IOS18, validationPolicy));

    return 0;
}

int testMultiFunctionDefaultFunctionName() {
    Specification::Model m;

    auto validationPolicy = ValidationPolicy();
    validationPolicy.allowsEmptyInput = true;
    validationPolicy.allowsEmptyOutput = true;
    validationPolicy.allowsMultipleFunctions = true;

    auto *description = m.mutable_description();
    description->set_defaultfunctionname("foo");
    ML_ASSERT_BAD_WITH_TYPE(validateModelDescription(*description, MLMODEL_SPECIFICATION_VERSION_IOS18, validationPolicy), ResultType::INVALID_DEFAULT_FUNCTION_NAME);

    auto *function = description->add_functions();
    function->set_name("bar");
    ML_ASSERT_BAD_WITH_TYPE(validateModelDescription(*description, MLMODEL_SPECIFICATION_VERSION_IOS18, validationPolicy), ResultType::INVALID_DEFAULT_FUNCTION_NAME);

    function->set_name("foo");
    ML_ASSERT_GOOD(validateModelDescription(*description, MLMODEL_SPECIFICATION_VERSION_IOS18, validationPolicy));

    return 0;
}

int testMultiFunctionTopLevelFeatureDescriptionsMustBeEmpty() {
    Specification::Model m;

    auto validationPolicy = ValidationPolicy();
    validationPolicy.allowsEmptyInput = true;
    validationPolicy.allowsEmptyOutput = true;
    validationPolicy.allowsMultipleFunctions = true;

    auto *description = m.mutable_description();
    description->set_defaultfunctionname("foo");
    auto *function = description->add_functions();
    function->set_name("foo");

    ML_ASSERT_GOOD(validateModelDescription(*description, MLMODEL_SPECIFICATION_VERSION_IOS18, validationPolicy));

    description->add_input()->set_name("x");
    ML_ASSERT_BAD_WITH_TYPE(validateModelDescription(*description, MLMODEL_SPECIFICATION_VERSION_IOS18, validationPolicy), ResultType::INVALID_MODEL_INTERFACE);
    description->clear_input();

    description->add_output()->set_name("y");
    ML_ASSERT_BAD_WITH_TYPE(validateModelDescription(*description, MLMODEL_SPECIFICATION_VERSION_IOS18, validationPolicy), ResultType::INVALID_MODEL_INTERFACE);
    description->clear_output();

    description->add_traininginput()->set_name("z");
    ML_ASSERT_BAD_WITH_TYPE(validateModelDescription(*description, MLMODEL_SPECIFICATION_VERSION_IOS18, validationPolicy), ResultType::INVALID_MODEL_INTERFACE);
    description->clear_traininginput();

    description->add_state()->set_name("s");
    ML_ASSERT_BAD_WITH_TYPE(validateModelDescription(*description, MLMODEL_SPECIFICATION_VERSION_IOS18, validationPolicy), ResultType::INVALID_MODEL_INTERFACE);
    description->clear_state();

    description->set_predictedfeaturename("f");
    ML_ASSERT_BAD_WITH_TYPE(validateModelDescription(*description, MLMODEL_SPECIFICATION_VERSION_IOS18, validationPolicy), ResultType::INVALID_MODEL_INTERFACE);
    description->clear_predictedfeaturename();

    description->set_predictedprobabilitiesname("f");
    ML_ASSERT_BAD_WITH_TYPE(validateModelDescription(*description, MLMODEL_SPECIFICATION_VERSION_IOS18, validationPolicy), ResultType::INVALID_MODEL_INTERFACE);
    description->clear_predictedprobabilitiesname();

    return 0;
}

int testMultiFunctionEmptyInput() {
    Specification::Model m;

    auto validationPolicy = ValidationPolicy();
    validationPolicy.allowsEmptyInput = true;
    validationPolicy.allowsEmptyOutput = true;
    validationPolicy.allowsMultipleFunctions = true;

    auto *description = m.mutable_description();
    description->set_defaultfunctionname("foo");
    auto *function = description->add_functions();
    function->set_name("foo");

    ML_ASSERT_GOOD(validateModelDescription(*description, MLMODEL_SPECIFICATION_VERSION_IOS18, validationPolicy));

    validationPolicy.allowsEmptyInput = false;
    ML_ASSERT_BAD_WITH_TYPE(validateModelDescription(*description, MLMODEL_SPECIFICATION_VERSION_IOS18, validationPolicy), ResultType::MODEL_TYPE_DOES_NOT_SUPPORT_EMPTY_INPUT);

    return 0;
}

int testMultiFunctionAllowed() {
    Specification::Model m;

    auto validationPolicy = ValidationPolicy();
    validationPolicy.allowsEmptyInput = true;
    validationPolicy.allowsEmptyOutput = true;
    validationPolicy.allowsMultipleFunctions = true;

    auto *description = m.mutable_description();
    description->set_defaultfunctionname("foo");
    auto *function = description->add_functions();
    function->set_name("foo");

    ML_ASSERT_GOOD(validateModelDescription(*description, MLMODEL_SPECIFICATION_VERSION_IOS18, validationPolicy));

    validationPolicy.allowsMultipleFunctions = false;
    ML_ASSERT_BAD_WITH_TYPE(validateModelDescription(*description, MLMODEL_SPECIFICATION_VERSION_IOS18, validationPolicy), ResultType::MODEL_TYPE_DOES_NOT_SUPPORT_MULTI_FUNCTION);

    return 0;
}

int testStateSpecificationVersion() {
    Specification::Model m;
    auto *description = m.mutable_description();
    setupStateFeature(description->add_state(), "x");

    // Check model specification version requirements.
    auto validationPolicy = validationPolicyForStateTests();
    ML_ASSERT_BAD_WITH_TYPE(validateModelDescription(*description, MLMODEL_SPECIFICATION_VERSION_IOS17, validationPolicy), ResultType::INVALID_COMPATIBILITY_VERSION);
    ML_ASSERT_GOOD(validateModelDescription(*description, MLMODEL_SPECIFICATION_VERSION_IOS18, validationPolicy));

    return 0;
}

/// For backward compabitility reason, it is OK to declare state features in the input descriptions.
int testStateFeatureDescriptionInInputs() {
    Specification::Model m;
    auto *description = m.mutable_description();

    setupStateFeature(description->add_input(), "x");

    // Check model specification version requirements.
    auto validationPolicy = validationPolicyForStateTests();
    ML_ASSERT_GOOD(validateModelDescription(*description, MLMODEL_SPECIFICATION_VERSION_NEWEST, validationPolicy));

    return 0;
}

int testStateFeatureIsNotFP16_shouldFail() {
    Specification::Model m;
    auto *description = m.mutable_description();
    auto *state = description->add_state();

    setupStateFeature(state, "x");
    state->mutable_type()->mutable_statetype()->mutable_arraytype()->set_datatype(::CoreML::Specification::ArrayFeatureType_ArrayDataType_FLOAT32);

    // Check model specification version requirements.
    auto validationPolicy = validationPolicyForStateTests();
    ML_ASSERT_BAD_WITH_TYPE(validateModelDescription(*description, MLMODEL_SPECIFICATION_VERSION_NEWEST, validationPolicy), ResultType::INVALID_MODEL_INTERFACE);

    return 0;
}

int testStateFeatureIsOptional_shouldFail() {
    Specification::Model m;
    auto *description = m.mutable_description();
    auto *state = description->add_state();

    setupStateFeature(state, "x");
    state->mutable_type()->set_isoptional(true);

    // Check model specification version requirements.
    auto validationPolicy = validationPolicyForStateTests();
    ML_ASSERT_BAD_WITH_TYPE(validateModelDescription(*description, MLMODEL_SPECIFICATION_VERSION_NEWEST, validationPolicy), ResultType::INVALID_MODEL_INTERFACE);

    return 0;
}

int testStateFeatureHasNoDefaultShape_shouldFail() {
    Specification::Model m;
    auto *description = m.mutable_description();
    auto *state = description->add_state();

    setupStateFeature(state, "x");
    state->mutable_type()->mutable_statetype()->mutable_arraytype()->clear_shape();

    // Check model specification version requirements.
    auto validationPolicy = validationPolicyForStateTests();
    ML_ASSERT_BAD_WITH_TYPE(validateModelDescription(*description, MLMODEL_SPECIFICATION_VERSION_NEWEST, validationPolicy), ResultType::INVALID_MODEL_INTERFACE);

    return 0;
}

int testStateFeatureHasNoArrayType_shouldFail() {
    Specification::Model m;
    auto *description = m.mutable_description();
    auto *state = description->add_state();

    setupStateFeature(state, "x");
    state->mutable_type()->mutable_statetype()->clear_arraytype();

    // Check model specification version requirements.
    auto validationPolicy = validationPolicyForStateTests();
    ML_ASSERT_BAD_WITH_TYPE(validateModelDescription(*description, MLMODEL_SPECIFICATION_VERSION_NEWEST, validationPolicy), ResultType::INVALID_MODEL_INTERFACE);

    return 0;
}

int testStateFeature_ArrayUsesRangeFlexibleShape_shouldFail() {
    Specification::Model m;
    auto *description = m.mutable_description();
    auto *state = description->add_state();

    setupStateFeature(state, "x");
    state->mutable_type()->mutable_statetype()->mutable_arraytype()->mutable_shaperange();

    // Check model specification version requirements.
    auto validationPolicy = validationPolicyForStateTests();
    ML_ASSERT_BAD_WITH_TYPE(validateModelDescription(*description, MLMODEL_SPECIFICATION_VERSION_NEWEST, validationPolicy), ResultType::INVALID_MODEL_INTERFACE);

    return 0;
}

int testStateFeature_ArrayUsesEnumeratedFlexibleShape_shouldFail() {
    Specification::Model m;
    auto *description = m.mutable_description();
    auto *state = description->add_state();

    setupStateFeature(state, "x");
    state->mutable_type()->mutable_statetype()->mutable_arraytype()->mutable_enumeratedshapes();

    // Check model specification version requirements.
    auto validationPolicy = validationPolicyForStateTests();
    ML_ASSERT_BAD_WITH_TYPE(validateModelDescription(*description, MLMODEL_SPECIFICATION_VERSION_NEWEST, validationPolicy), ResultType::INVALID_MODEL_INTERFACE);

    return 0;
}
