//
//  ProgramValidatorTests.cpp
//  CoreML_tests
//
//  Created by Jeff Kilpatrick on 12/12/19.
//  Copyright © 2019 Apple Inc. All rights reserved.
//

#include "MLModelTests.hpp"

#include "framework/TestUtils.hpp"
#include "Model.hpp"
#include "ProgramModelUtils.hpp"
#include "ResultReason.hpp"

using ::CoreML::ResultReason;
using namespace ::CoreML::Specification;
using namespace ::ProgramModelUtils;

static Model ProgramWithMain(const V5::Function& main);

static ArrayFeatureType MakeArrayFeatureType(const V5::TensorType& tensorType)
{
    ArrayFeatureType aft;

    switch (tensorType.scalartype()) {
        case CoreML::Specification::V5::FLOAT32:
            aft.set_datatype(ArrayFeatureType_ArrayDataType_FLOAT32);
            break;
        case CoreML::Specification::V5::FLOAT64:
            aft.set_datatype(ArrayFeatureType_ArrayDataType_DOUBLE);
            break;
        case CoreML::Specification::V5::INT32:
            aft.set_datatype(ArrayFeatureType_ArrayDataType_INT32);
            break;
        default:
            throw std::runtime_error("Cannot create array datatype from unsupported scalar type.");
    }

    for (const auto& dim : tensorType.dimension()) {
        switch (dim.dimension_case()) {
            case CoreML::Specification::V5::Dimension::kSize:
                aft.add_shape(dim.size());
                break;
            case CoreML::Specification::V5::Dimension::kSymbol:
            case CoreML::Specification::V5::Dimension::DIMENSION_NOT_SET:
                throw std::runtime_error("Cannot create array dimension from unsupported dimension type.");
        }
    }

    return aft;
}

static FeatureDescription MakeFeatureDescription(const std::string& name, const V5::ValueType& type)
{
    FeatureDescription fd;

    fd.set_name(name);

    switch (type.type_case()) {
        case V5::ValueType::kTensorType:
            fd.mutable_type()->mutable_multiarraytype()->CopyFrom(MakeArrayFeatureType(type.tensortype()));
            break;
        case V5::ValueType::kListType:
        case V5::ValueType::kScalarType:
        case V5::ValueType::kTupleType:
        case V5::ValueType::TYPE_NOT_SET:
            throw std::runtime_error("Cannot create feature from unsupported type.");
    }

    return fd;
}

static FeatureDescription MakeFeatureDescription(const V5::NamedValueType& namedType)
{
    return MakeFeatureDescription(namedType.name(), namedType.type());
}

static Model EmptyProgram()
{
    return ProgramWithMain(MakeFunction({}, {}, MakeBlock({}, {}, {})));
};

static Model ProgramWithMain(const V5::Function& main)
{
    Model model;
    (*model.mutable_program()->mutable_functions())["main"].CopyFrom(main);

    for (const auto& input : main.inputs()) {
        auto modelInput = MakeFeatureDescription(input);
        model.mutable_description()->add_input()->CopyFrom(modelInput);
    }

    if (main.outputs_size() != main.block().outputs_size()) {
        throw std::runtime_error("main() returns different number of values than its block.");
    }

    for (int i = 0; i < main.outputs_size(); ++i) {
        const auto& outputType = main.outputs(i);
        const auto& outputName = main.block().outputs(i);
        auto modelOutput = MakeFeatureDescription(outputName, outputType);
        model.mutable_description()->add_output()->CopyFrom(modelOutput);
    }

    return model;
}

int testValidateProgramBlock()
{
    // Block arguments must exist
    {
        // [ x <- y ] { /*empty block*/ } [/*no outputs*/]
        auto block = MakeBlock({ { "x", "y" } }, // inputs
                               { }, // outputs
                               { } // ops
                               );
        auto main = MakeFunction(/*params=*/ {}, /*outputs=*/ {}, block);

        Model model;
        (*model.mutable_program()->mutable_functions())["main"].CopyFrom(main);
        ML_ASSERT_BAD(::CoreML::validate<MLModelType_program>(model));
    }

    // Block parameters must have non-empty names
    {
        // main(a : fp32[2, 4]):
        //   [ <- a] { /*empty block*/ } [/*no outputs*/]
        auto block = MakeBlock({ { "", "a" } }, // inputs
                               { }, // outputs
                               { } // ops
                               );
        NameAndTypeVec mainParams{
            { "a", MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(2), MakeDim(4) }) }
        };

        auto model = ProgramWithMain(MakeFunction(mainParams, /*outputs=*/ {}, block));
        ML_ASSERT_BAD_WITH_REASON(::CoreML::validate<MLModelType_program>(model),
                                  ResultReason::BLOCK_PARAM_NAME_EMPTY);
    }

    // Block return values must exist
    {
        // main(b : fp[2]):
        //   [/*no inputs*/] { a = const(); z = activation(data=b, type=a); } [ bonk ]
        auto block = MakeBlock({ }, // inputs
                               { "bonk" }, // outputs
                               { // ops
            MakeOp("a", "const", {}, { {"a", MakeScalarValueType(V5::ScalarType::STRING)} }, { }),
            MakeOp("z", "activation",
                   { { "type", "a" }, { "data", "b" } }, // inputs
                   { }, // outputs
                   { } /* attributes */)
        });
        NameAndTypeVec mainParams{
            { "b", MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(2) }) }
        };

        auto model = ProgramWithMain(MakeFunction(mainParams,
                                                  {MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(2) })},
                                                  block));
        ML_ASSERT_BAD_WITH_REASON(::CoreML::validate<MLModelType_program>(model),
                                  ResultReason::BLOCK_OUTPUT_VALUE_UNDEFINED);
    }

    return 0;
}

int testValidateProgramFunction()
{
    // Parameters must have names
    {
        Model model = EmptyProgram();
        (*model.mutable_program()->mutable_functions())["f"]
            .CopyFrom(MakeFunction({ {"", MakeScalarValueType(V5::BOOL)} }, // params
                                   {}, // returns
                                   MakeBlock({}, {}, {})));
        ML_ASSERT_BAD_WITH_REASON(::CoreML::validate<MLModelType_program>(model),
                                  ResultReason::FUNCTION_PARAM_NAME_EMPTY);
    }

    // We have the same number of outputs as our block
    {
        // main() -> [/*no outputs*/]
        //  [/*no inputs*/] { a = const(); } [a]
        auto main = MakeFunction(/*params=*/ {}, /*outputs=*/ {},
                                 MakeBlock(/*bindings=*/ {}, {"a"}, {
            MakeOp("a", "const", {}, { {"a", MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(2) })} }, { })
        }));

        // Create the model manually since the helper checks that blocks and functions
        // have the same number of outputs.
        Model model;
        (*model.mutable_program()->mutable_functions())["main"].CopyFrom(main);

        ML_ASSERT_BAD_WITH_REASON(::CoreML::validate<MLModelType_program>(model),
                                  ResultReason::FUNCTION_BLOCK_MISMATCHED_RETURN_COUNT);
    }

    // The types returned by our block match those in our outputs
    {
        // main() -> [ fp32[2] ]
        //  [] { a = const(); } [a : fp64[2] ]
        auto model = ProgramWithMain(MakeFunction(/*params=*/ {},
                                                  { MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(2) })},
                                                  MakeBlock(/*bindings=*/ {}, {"a"}, {
            MakeOp("a", "const", {}, { {"a", MakeTensorValueType(V5::ScalarType::FLOAT64, { MakeDim(2) })} }, { })
        })));

        ML_ASSERT_BAD_WITH_REASON(::CoreML::validate<MLModelType_program>(model),
                                  ResultReason::FUNCTION_BLOCK_MISMATCHED_RETURN_TYPE);
    }

    return 0;
}

int testValidateProgramMain()
{
    // main is a required function
    {
        Model model;
        ML_ASSERT_BAD_WITH_REASON(::CoreML::validate<MLModelType_program>(model),
                                  ResultReason::PROGRAM_MISSING_MAIN_FUNCTION);
    }
    return 0;
}

int testValidateProgramModel()
{
    {
        // Incorrect input type -- must be MultiArrayType or ImageType
        auto model = EmptyProgram();
        auto input = model.mutable_description()->add_input();
        input->set_name("anInput");
        input->mutable_type()->mutable_doubletype();
        ML_ASSERT_BAD_WITH_REASON(::CoreML::validate<MLModelType_program>(model),
                                  ResultReason::MODEL_INVALID_INPUT_TYPE);
    }

    {
        // Incorrect output type -- must be MultiArrayType or ImageType
        auto model = EmptyProgram();
        auto output = model.mutable_description()->add_output();
        output->set_name("anOutput");
        output->mutable_type()->mutable_stringtype();
        ML_ASSERT_BAD_WITH_REASON(::CoreML::validate<MLModelType_program>(model),
                                  ResultReason::MODEL_INVALID_OUTPUT_TYPE);
    }

    // Model and main() must have same number of inputs
    {
        auto model = EmptyProgram();
        auto input = model.mutable_description()->add_input();
        auto tensorType = MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(2), MakeDim(4) });
        input->CopyFrom(MakeFeatureDescription("anInput", tensorType));
        ML_ASSERT_BAD_WITH_REASON(::CoreML::validate<MLModelType_program>(model),
                                  ResultReason::MODEL_MAIN_MISMATCHED_INPUT_COUNT);
    }

    // Model and main() must have same number of outputs
    {
        auto model = EmptyProgram();
        auto output = model.mutable_description()->add_output();
        auto tensorType = MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(2), MakeDim(4) });
        output->CopyFrom(MakeFeatureDescription("anInput", tensorType));
        ML_ASSERT_BAD_WITH_REASON(::CoreML::validate<MLModelType_program>(model),
                                  ResultReason::MODEL_MAIN_MISMATCHED_OUTPUT_COUNT);
    }

    // Model and main() must have same input names
    {
        // main(a : fp32[2, 4]):
        //   [ ] { /*empty block*/ } [/*no outputs*/]
        auto block = MakeBlock({ },  // inputs
                               { },  // outputs
                               { }); // ops
        NameAndTypeVec mainParams{
            { "a", MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(2), MakeDim(4) }) }
        };

        auto model = ProgramWithMain(MakeFunction(mainParams, /*outputs=*/ {}, block));

        model.mutable_description()->mutable_input(0)->set_name("notA");
        ML_ASSERT_BAD_WITH_REASON(::CoreML::validate<MLModelType_program>(model),
                                  ResultReason::MODEL_MAIN_MISSING_INPUT_OUTPUT);
    }

    // Model inputs must be tensors
    {
        // main(a : string)
        //   [ ] { } [ ]
        auto block = MakeBlock({}, {}, {});
        NameAndTypeVec mainParams{
            { "a", MakeScalarValueType(V5::STRING) }
        };

        auto model = EmptyProgram();
        (*model.mutable_program()->mutable_functions())["main"].CopyFrom(MakeFunction(mainParams, {}, block));

        FeatureDescription fd;
        fd.set_name("a");
        fd.mutable_type()->mutable_stringtype();
        model.mutable_description()->add_input()->CopyFrom(fd);
        ML_ASSERT_BAD_WITH_REASON(::CoreML::validate<MLModelType_program>(model),
                                  ResultReason::MODEL_INVALID_INPUT_TYPE);
    }

    // Model outputs must be tensors
    {
        // main()
        //  [] { a:string = const(); } [a]
        auto block = MakeBlock({}, { "a" }, {
            MakeOp("a", "const", {}, { {"a", MakeScalarValueType(V5::ScalarType::STRING)} }, { }),
        });

        auto model = EmptyProgram();
        (*model.mutable_program()->mutable_functions())["main"].CopyFrom(MakeFunction({}, {}, block));

        FeatureDescription fd;
        fd.set_name("a");
        fd.mutable_type()->mutable_stringtype();
        model.mutable_description()->add_output()->CopyFrom(fd);
        ML_ASSERT_BAD_WITH_REASON(::CoreML::validate<MLModelType_program>(model),
                                  ResultReason::MODEL_INVALID_OUTPUT_TYPE);
    }

    // Model and main() must have same input data types
    {
        // main(a : fp32[2, 4]):
        //   [ ] { /*empty block*/ } [/*no outputs*/]
        auto block = MakeBlock({ },  // inputs
                               { },  // outputs
                               { }); // ops
        NameAndTypeVec mainParams{
            { "a", MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(2), MakeDim(4) }) }
        };

        auto model = ProgramWithMain(MakeFunction(mainParams, /*outputs=*/ {}, block));
        model.mutable_description()->mutable_input(0)->mutable_type()->
            mutable_multiarraytype()->set_datatype(ArrayFeatureType_ArrayDataType_INT32);

        ML_ASSERT_BAD_WITH_REASON(::CoreML::validate<MLModelType_program>(model),
                                  ResultReason::MODEL_MAIN_MISMATCHED_INPUT_TYPE);
    }

    // Model and main() must have same output data types
    {
        // main(a : fp32[2, 4]):
        //   [ x <- a] { /*empty block*/ } [x]
        auto block = MakeBlock({ { "x", "a" } }, // inputs
                               { "x" }, // outputs
                               { } // ops
                               );
        NameAndTypeVec mainParams{
            { "a", MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(2), MakeDim(4) }) }
        };

        TypeVec modelOutputs{ MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(2), MakeDim(4) }) };
        auto model = ProgramWithMain(MakeFunction(mainParams, { modelOutputs }, block));
        model.mutable_description()->mutable_output(0)->mutable_type()->
            mutable_multiarraytype()->set_datatype(ArrayFeatureType_ArrayDataType_INT32);
        ML_ASSERT_BAD_WITH_REASON(::CoreML::validate<MLModelType_program>(model),
                                  ResultReason::MODEL_MAIN_MISMATCHED_OUTPUT_TYPE);
    }

    // Model and main() must have compatible input tensor dimensions -- mismatched static dimensions
    {
        // main(a : fp32[2, 4]):
        //   [ ] { /*empty block*/ } [/*no outputs*/]
        auto block = MakeBlock({ },  // inputs
                               { },  // outputs
                               { }); // ops
        NameAndTypeVec mainParams{
            { "a", MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(2), MakeDim(4) }) }
        };

        auto model = ProgramWithMain(MakeFunction(mainParams, /*outputs=*/ {}, block));
        model.mutable_description()->mutable_input(0)->mutable_type()->
            mutable_multiarraytype()->mutable_shape()->Set(0, 3);
        ML_ASSERT_BAD_WITH_REASON(::CoreML::validate<MLModelType_program>(model),
                                  ResultReason::MODEL_MAIN_MISMATCHED_INPUT_SHAPE);
    }

    // Model and main() must have compatible input tensor dimensions -- enumerated model shape not
    // covered by program symbolic shape
    {
        // main(a : fp32[2, 4]):
        //   [ ] { /*empty block*/ } [/*no outputs*/]
        auto block = MakeBlock({ },  // inputs
                               { },  // outputs
                               { }); // ops
        NameAndTypeVec mainParams{
            { "a", MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(2), MakeDim(4) }) }
        };

        auto model = ProgramWithMain(MakeFunction(mainParams, /*outputs=*/ {}, block));

        // make the first dimension symbolic
        (*model.mutable_program()->mutable_functions())["main"].mutable_inputs(0)
            ->mutable_type()->mutable_tensortype()->mutable_dimension(0)->set_symbol("s0");

        { // add some enumerated shapes to the model
            auto enumShapes = model.mutable_description()->mutable_input(0)->mutable_type()->
                mutable_multiarraytype()->mutable_enumeratedshapes();

            auto enumShape = enumShapes->add_shapes();
            enumShape->add_shape(2);
            enumShape->add_shape(4);

            enumShape = enumShapes->add_shapes();
            enumShape->add_shape(4);
            enumShape->add_shape(6);
        }

        ML_ASSERT_BAD_WITH_REASON(::CoreML::validate<MLModelType_program>(model),
                                  ResultReason::MODEL_MAIN_MISMATCHED_INPUT_SHAPE);
    }

    // Model and main() must have compatible input tensor dimensions -- model shape range not
    // covered by program symbolic shape
    {
        // main(a : fp32[2, 4]):
        //   [ ] { /*empty block*/ } [/*no outputs*/]
        auto block = MakeBlock({ },  // inputs
                               { },  // outputs
                               { }); // ops
        NameAndTypeVec mainParams{
            { "a", MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(2), MakeDim(4) }) }
        };

        auto model = ProgramWithMain(MakeFunction(mainParams, /*outputs=*/ {}, block));

        { // Change the model's input shape to [ (2,2), (2,8) ]
            auto shapeRange = model.mutable_description()->mutable_input(0)->mutable_type()->
                mutable_multiarraytype()->mutable_shaperange();

            auto sizeRange = shapeRange->add_sizeranges();
            sizeRange->set_lowerbound(2);
            sizeRange->set_upperbound(2);

            sizeRange = shapeRange->add_sizeranges();
            sizeRange->set_lowerbound(2);
            sizeRange->set_upperbound(8);
        }

        ML_ASSERT_BAD_WITH_REASON(::CoreML::validate<MLModelType_program>(model),
                                  ResultReason::MODEL_MAIN_MISMATCHED_INPUT_SHAPE);
    }

    // Model and main() must have compatible output tensor dimensions -- mismatched static dimensions
    {
        // main(a : fp32[2, 4]):
        //   [ ] { /*empty block*/ } [ a ]
        auto block = MakeBlock({ },  // inputs
                               { "a" },  // outputs
                               { }); // ops
        auto fp32_2_4 = MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(2), MakeDim(4) });
        NameAndTypeVec mainParams{
            { "a", fp32_2_4 }
        };

        auto model = ProgramWithMain(MakeFunction(mainParams, { fp32_2_4 }, block));

        { // Change the model's output to fp32[2, 5]
            model.mutable_description()->mutable_output(0)->mutable_type()
                ->mutable_multiarraytype()->mutable_shape()->Set(1, 5);
        }

        ML_ASSERT_BAD_WITH_REASON(::CoreML::validate<MLModelType_program>(model),
                                  ResultReason::MODEL_MAIN_MISMATCHED_OUTPUT_SHAPE);
    }

    return 0;
}

int testValidateProgramOp()
{
    // Ops must have non-empty names
    {
        // main(b : fp[2]):
        //   [/*no inputs*/] { a = const(); z = <unnamed op> activation(data=b, type=a); } [/*no outputs*/]
        auto block = MakeBlock({ }, // inputs
                               { }, // outputs
                               { // ops
            MakeOp("a", "const", {}, { {"a", MakeScalarValueType(V5::ScalarType::STRING)} }, { }),
            MakeOp("", "activation",
                   { { "type", "a" }, { "data", "b" } }, // inputs
                   { }, // outputs
                   { } /* attributes */)
        });
        NameAndTypeVec mainParams{
            { "b", MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(2) }) }
        };

        auto model = ProgramWithMain(MakeFunction(mainParams, /*outputs=*/ {}, block));
        ML_ASSERT_BAD_WITH_REASON(::CoreML::validate<MLModelType_program>(model),
                                  ResultReason::OP_NAME_EMPTY);
    }

    // Op outputs have non-empty names
    {
        // main(b : fp[2]):
        //   [/*no inputs*/] { a = const();  = activation(data=b, type=a); } [/*no outputs*/]
        auto block = MakeBlock({ }, // inputs
                               { }, // outputs
                               { // ops
            MakeOp("a", "const", {}, { {"a", MakeScalarValueType(V5::ScalarType::STRING)} }, { }),
            MakeOp("z", "activation",
                   { { "type", "a" }, { "data", "b" } }, // inputs
                   { { "", MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(2) }) } }, // outputs
                   { } /* attributes */)
        });
        NameAndTypeVec mainParams{
            { "b", MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(2) }) }
        };

        auto model = ProgramWithMain(MakeFunction(mainParams, /*outputs=*/ {}, block));
        ML_ASSERT_BAD_WITH_REASON(::CoreML::validate<MLModelType_program>(model),
                                  ResultReason::OP_OUTPUT_NAME_EMPTY);
    }

    // Op attribute names must be non-empty
    {
        // main():
        //  [/*no inputs*/] { z = const({val: bool(false)}); } [/*no outputs*/]
        auto block = MakeBlock({ }, // inputs
                               { }, // outputs
                               { // ops
            MakeOp("z", "const",
                   { }, // inputs
                   { { "z", MakeScalarValueType(V5::ScalarType::BOOL) } }, // outputs
                   { { "", MakeBoolValue(false)} } /* attributes */)
        });

        auto model = ProgramWithMain(MakeFunction(/*params=*/ {}, /*outputs=*/ {}, block));
        ML_ASSERT_BAD_WITH_REASON(::CoreML::validate<MLModelType_program>(model),
                                  ResultReason::OP_ATTRIBUTE_NAME_EMPTY);

    }

    // Validate that we catch missing inputs
    {
        // [/*no inputs*/] { a = const(); z = activation(data=b, activation=a); } [/*no outputs*/]
        auto block = MakeBlock({ }, // inputs
                               { }, // outputs
                               { // ops
            MakeOp("a", "const", {}, { {"a", MakeScalarValueType(V5::ScalarType::STRING)} }, { }),
            MakeOp("z", "activation",
                   { { "activation", "a" }, { "data", "b" } }, // inputs
                   { }, // outputs
                   { } /* attributes */)
        });

        auto model = ProgramWithMain(MakeFunction(/*params=*/ {}, /*outputs=*/ {}, block));
        ML_ASSERT_BAD_WITH_REASON(::CoreML::validate<MLModelType_program>(model),
                                  ResultReason::OP_ARG_VALUE_UNDEFINED);
    }

    // Validate that Block inputs are found
    {
        // main(b : fp[2]):
        //   [y <- b] { a = const(); z = activation(data=y, type=x); } [/*no outputs*/]
        auto block = MakeBlock({ { "y", "b" } }, // inputs
                               { }, // outputs
                               { // ops
            MakeOp("a", "const", {}, { {"a", MakeScalarValueType(V5::ScalarType::STRING)} }, { }),
            MakeOp("z", "activation",
                   { { "type", "a" }, { "data", "y" } }, // inputs
                   { }, // outputs
                   { } /* attributes */)
        });
        NameAndTypeVec mainParams{
            { "b", MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(2) }) }
        };

        auto model = ProgramWithMain(MakeFunction(mainParams, /*outputs=*/ {}, block));
        ML_ASSERT_GOOD(::CoreML::validate<MLModelType_program>(model));
    }

    // Validate that Function inputs are found
    {
        // main(b : fp[2]):
        //   [/*no inputs*/] { a = const(); z = activation(data=b, type=a); } [/*no outputs*/]
        auto block = MakeBlock({ }, // inputs
                               { }, // outputs
                               { // ops
            MakeOp("a", "const", {}, { {"a", MakeScalarValueType(V5::ScalarType::STRING)} }, { }),
            MakeOp("z", "activation",
                   { { "type", "a" }, { "data", "b" } }, // inputs
                   { }, // outputs
                   { } /* attributes */)
        });
        NameAndTypeVec mainParams{
            { "b", MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(2) }) }
        };

        auto model = ProgramWithMain(MakeFunction(mainParams, /*outputs=*/ {}, block));
        ML_ASSERT_GOOD(::CoreML::validate<MLModelType_program>(model));
    }

    // Validate that inputs in the Parameters are found
    {
        // params: { b : fp[2] }
        // main():
        //   [/*no inputs*/] { a = const(); z = activation(data=y, type=x); } [/*no outputs*/]
        auto block = MakeBlock({ }, // inputs
                               { }, // outputs
                               { // ops
            MakeOp("a", "const", {}, { {"a", MakeScalarValueType(V5::ScalarType::STRING)} }, { }),
            MakeOp("z", "activation",
                   { { "type", "a" }, { "data", "b" } }, // inputs
                   { }, // outputs
                   {} /* attributes */)
        });

        auto main = MakeFunction(/*params=*/ {}, /*outputs=*/ {}, block);

        V5::Value bValue;
        bValue.mutable_type()->CopyFrom(MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(2) }));
        bValue.mutable_immediatevalue()->mutable_tensor()->add_floats(7.f);
        bValue.mutable_immediatevalue()->mutable_tensor()->add_floats(8.f);

        auto model = ProgramWithMain(main);
        (*model.mutable_program()->mutable_parameters())["b"].CopyFrom(bValue);

        ML_ASSERT_GOOD(::CoreML::validate<MLModelType_program>(model));
    }

    // Validate that inputs generated by previous ops are found
    {
        // main(b : fp[2]):
        //   [/*no inputs*/] {
        //     a = const();
        //     j = activation(data=b, type=a);
        //     k = activation(data=j, type=a);
        // } [/*no outputs*/]
        auto block = MakeBlock({ }, // inputs
                               { }, // outputs
                               { // ops
            MakeOp("a", "const", {}, { {"a", MakeScalarValueType(V5::ScalarType::STRING)} }, { }),
            MakeOp("j", "activation",
                   { { "type", "a" }, { "data", "b" } }, // inputs
                   { { "j", MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(2) }) } }, // outputs
                   { } /* attributes */),
            MakeOp("k", "activation",
                   { { "type", "j" }, { "data", "b" } }, // inputs
                   { }, // outputs
                   { } /* attributes */)
        });
        NameAndTypeVec mainParams{
            { "b", MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(2) }) }
        };

        auto model = ProgramWithMain(MakeFunction(mainParams, /*outputs=*/ {}, block));
        ML_ASSERT_GOOD(::CoreML::validate<MLModelType_program>(model));
    }

    return 0;
}

int testValidateProgramProgram()
{
    // Empty program is valid
    {
        ML_ASSERT_GOOD(::CoreML::validate<MLModelType_program>(EmptyProgram()));
    }

    // Functions must have names
    {
        Model model = EmptyProgram();
        (*model.mutable_program()->mutable_functions())[""]
            .CopyFrom(MakeFunction({}, {}, MakeBlock({}, {}, {})));
        ML_ASSERT_BAD_WITH_REASON(::CoreML::validate<MLModelType_program>(model),
                                  ResultReason::FUNCTION_NAME_EMPTY);
    }

    // Parameters must have names
    {
        Model model = EmptyProgram();
        (*model.mutable_program()->mutable_parameters())[""] = MakeBoolValue(true);
        ML_ASSERT_BAD_WITH_REASON(::CoreML::validate<MLModelType_program>(model),
                                  ResultReason::PARAMETER_NAME_EMPTY);
    }

    return 0;
}
