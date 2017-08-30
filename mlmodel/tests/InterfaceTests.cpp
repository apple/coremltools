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
    ML_ASSERT_BAD(validateFeatureDescription(*feature,true));

    // Just with a name as invalid
    feature->set_name("test_input");
    ML_ASSERT_BAD(validateFeatureDescription(*feature,true));

    // Empty type, still invalid
    feature->mutable_type();
    ML_ASSERT_BAD(validateFeatureDescription(*feature,true));

    // Int64 typ, now its valid
    feature->mutable_type()->mutable_int64type();
    ML_ASSERT_GOOD(validateFeatureDescription(*feature,true));

    // String type, valid
    feature->mutable_type()->mutable_stringtype();
    ML_ASSERT_GOOD(validateFeatureDescription(*feature,true));

    // Double type, valid
    feature->mutable_type()->mutable_doubletype();
    ML_ASSERT_GOOD(validateFeatureDescription(*feature,true));

    // Multiarray type, with no params, invalid
    feature->mutable_type()->mutable_multiarraytype();
    ML_ASSERT_BAD(validateFeatureDescription(*feature,true));

    // Multiarray type, double with no shape, invalid as input, valid as output
    feature->mutable_type()->mutable_multiarraytype()->set_datatype(::CoreML::Specification::ArrayFeatureType_ArrayDataType_DOUBLE);
    ML_ASSERT_BAD(validateFeatureDescription(*feature,true));
    feature->mutable_type()->mutable_multiarraytype()->set_datatype(::CoreML::Specification::ArrayFeatureType_ArrayDataType_DOUBLE);
    ML_ASSERT_GOOD(validateFeatureDescription(*feature,false));

    feature->mutable_type()->mutable_multiarraytype()->set_datatype(::CoreML::Specification::ArrayFeatureType_ArrayDataType_FLOAT32);
    ML_ASSERT_BAD(validateFeatureDescription(*feature,true));
    feature->mutable_type()->mutable_multiarraytype()->set_datatype(::CoreML::Specification::ArrayFeatureType_ArrayDataType_FLOAT32);
    ML_ASSERT_GOOD(validateFeatureDescription(*feature,false));

    feature->mutable_type()->mutable_multiarraytype()->set_datatype(::CoreML::Specification::ArrayFeatureType_ArrayDataType_INT32);
    ML_ASSERT_BAD(validateFeatureDescription(*feature,true));
    feature->mutable_type()->mutable_multiarraytype()->set_datatype(::CoreML::Specification::ArrayFeatureType_ArrayDataType_INT32);
    ML_ASSERT_GOOD(validateFeatureDescription(*feature,false));

    // Zero length shape is invalid for inputs, but valid for outputs
    feature->mutable_type()->mutable_multiarraytype()->mutable_shape();
    ML_ASSERT_BAD(validateFeatureDescription(*feature,true));
    ML_ASSERT_GOOD(validateFeatureDescription(*feature,false));

    // Non-zero length shape, valid
    feature->mutable_type()->mutable_multiarraytype()->mutable_shape()->Add(128);
    ML_ASSERT_GOOD(validateFeatureDescription(*feature,true));
    ML_ASSERT_GOOD(validateFeatureDescription(*feature,false));

    // Dictionary, with no params, invalid
    feature->mutable_type()->mutable_dictionarytype();
    ML_ASSERT_BAD(validateFeatureDescription(*feature,true));

    // With key type, valid
    feature->mutable_type()->mutable_dictionarytype()->mutable_stringkeytype();
    ML_ASSERT_GOOD(validateFeatureDescription(*feature,true));

    feature->mutable_type()->mutable_dictionarytype()->mutable_int64keytype();
    ML_ASSERT_GOOD(validateFeatureDescription(*feature,true));

    // Image, with no params, invalid
    feature->mutable_type()->mutable_imagetype();
    ML_ASSERT_BAD(validateFeatureDescription(*feature,true));

    // With just width, invalid
    feature->mutable_type()->mutable_imagetype()->set_width(10);
    ML_ASSERT_BAD(validateFeatureDescription(*feature,true));

    // With both width and height, still invalid because no colorspace
    feature->mutable_type()->mutable_imagetype()->set_height(20);
    ML_ASSERT_BAD(validateFeatureDescription(*feature,true));

    // Now with colorspace, valid
    feature->mutable_type()->mutable_imagetype()->set_colorspace(::CoreML::Specification::ImageFeatureType_ColorSpace_BGR);
    ML_ASSERT_GOOD(validateFeatureDescription(*feature,true));
    feature->mutable_type()->mutable_imagetype()->set_colorspace(::CoreML::Specification::ImageFeatureType_ColorSpace_RGB);
    ML_ASSERT_GOOD(validateFeatureDescription(*feature,true));
    feature->mutable_type()->mutable_imagetype()->set_colorspace(::CoreML::Specification::ImageFeatureType_ColorSpace_GRAYSCALE);
    ML_ASSERT_GOOD(validateFeatureDescription(*feature,true));
    feature->mutable_type()->mutable_imagetype()->set_colorspace(::CoreML::Specification::ImageFeatureType_ColorSpace_INVALID_COLOR_SPACE);
    ML_ASSERT_BAD(validateFeatureDescription(*feature,true));


    return 0;
}
