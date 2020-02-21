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

    // Block parameter names must not shadow existing declarations
    {
        // main(a : fp[2], b : fp[2]):
        //   [ a <- b ] { } [ ]
        auto block = MakeBlock({{ "a", "b" }}, {}, {});
        NameAndTypeVec mainParams {
            { "a", MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(2) }) },
            { "b", MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(2) }) }
        };

        auto main = MakeFunction(mainParams, /*outputs=*/ {}, block);
        auto model = ProgramWithMain(main);

        ML_ASSERT_BAD_WITH_REASON(::CoreML::validate<MLModelType_program>(model),
                                  ResultReason::BLOCK_PARAM_NAME_SHADOWS);
    }

    // Block return values must exist
    {
        // main(a : fp[2]):
        //   [/*no inputs*/] { z = relu(x=a); } [ bonk ]
        auto block = MakeBlock({ }, // inputs
                               { "bonk" }, // outputs
                               { // ops
            MakeOp("z", "relu",
                   { { "x", "a" } }, // inputs
                   { }, // outputs
                   { } /* attributes */)
        });
        NameAndTypeVec mainParams{
            { "a", MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(2) }) }
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

    // Parameter names must not shadow earlier declarations
    {
        // Program params: { a : bool }
        // main(a : fp[2]) { ... }
        auto main = MakeFunction({{ "a", MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(2) }) }}, {}, {});
        auto model = ProgramWithMain(main);
        (*model.mutable_program()->mutable_parameters())["a"] = MakeBoolValue(true);
        ML_ASSERT_BAD_WITH_REASON(::CoreML::validate<MLModelType_program>(model),
                                  ResultReason::FUNCTION_PARAM_NAME_SHADOWS);
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
        auto wrongValue = MakeFloatTensorValue({MakeDim(2)}, {1.f, 2.f});

        // main() -> [ fp32[2] ]
        //  [] { a = const(); } [val : fp64[2] ]
        auto model = ProgramWithMain(MakeFunction(/*params=*/ {},
                                                  { MakeTensorValueType(V5::ScalarType::FLOAT64, { MakeDim(2) })},
                                                  MakeBlock(/*bindings=*/ {}, {"a"}, {
            MakeOp("genA", "const", {}, {{"a", wrongValue.type()}}, {{"val", wrongValue}})
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

    // Model inputs must be tensors or images
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

    // Model image inputs must be coded as float32
    {
        // main(a : int8[1, 3, 10, 10])
        //   [ ] { } [ ]
        auto block = MakeBlock({}, {}, {});
        NameAndTypeVec mainParams{
            { "a", MakeTensorValueType(V5::ScalarType::INT8, { MakeDim(1), MakeDim(3), MakeDim(10), MakeDim(10) }) }
        };

        auto model = EmptyProgram();
        (*model.mutable_program()->mutable_functions())["main"].CopyFrom(MakeFunction(mainParams, {}, block));

        auto fd = MakeImageFeatureDescription("a", ImageFeatureType_ColorSpace::ImageFeatureType_ColorSpace_RGB, 10, 10);
        model.mutable_description()->add_input()->CopyFrom(fd);
        ML_ASSERT_BAD_WITH_REASON(::CoreML::validate<MLModelType_program>(model),
                                  ResultReason::MODEL_MAIN_BAD_IMAGE_INPUT_TYPE);
    }

    // Model outputs must be tensors or images
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

    // Models with image outputs must return fp32 tensor from main
    {
        // main(a : fp32[1, 1, 4, 4]):
        //   [ x <- a] { /*empty block*/ } [x]
        auto block = MakeBlock({ { "x", "a" } }, // inputs
                               { "x" }, // outputs
                               { } // ops
                               );
        NameAndTypeVec mainParams{
            { "a", MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(1), MakeDim(3), MakeDim(10), MakeDim(10) }) }
        };

        TypeVec modelOutputs{
            MakeTensorValueType(V5::ScalarType::FLOAT64, { MakeDim(1), MakeDim(3), MakeDim(10), MakeDim(10) })
        };
        auto model = ProgramWithMain(MakeFunction(mainParams, { modelOutputs }, block));
        auto fd = MakeImageFeatureDescription("a", ImageFeatureType_ColorSpace::ImageFeatureType_ColorSpace_RGB, 10, 10);
        model.mutable_description()->mutable_output(0)->CopyFrom(fd);
        ML_ASSERT_BAD_WITH_REASON(::CoreML::validate<MLModelType_program>(model),
                                  ResultReason::MODEL_MAIN_BAD_IMAGE_OUTPUT_TYPE);

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

int testValidateProgramModelInputShapes()
{
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

    // Model and main() must have compatible input image sizes -- mismatched static dimensions
    {
        // main(a : fp32[1, 3, 10, 10])
        //   [ ] { } [ ]
        auto block = MakeBlock({}, {}, {});
        NameAndTypeVec mainParams{
            { "a", MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(1), MakeDim(3), MakeDim(10), MakeDim(10) }) }
        };

        auto model = EmptyProgram();
        (*model.mutable_program()->mutable_functions())["main"].CopyFrom(MakeFunction(mainParams, {}, block));

        auto fd = MakeImageFeatureDescription("a", ImageFeatureType_ColorSpace::ImageFeatureType_ColorSpace_RGB, 10, 11);
        model.mutable_description()->add_input()->CopyFrom(fd);
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

    // Model and main() must have compatible input image sizes -- enumerated size not
    // covered by program symbolic shape
    {
        // main(a : fp32[1, 3, s0, 10]):
        //   [ ] { /*empty block*/ } [/*no outputs*/]
        auto block = MakeBlock({ },  // inputs
                               { },  // outputs
                               { }); // ops
        NameAndTypeVec mainParams{
            { "a", MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(1), MakeDim(3), MakeDim(10), MakeDim(10) }) }
        };

        auto model = ProgramWithMain(MakeFunction(mainParams, /*outputs=*/ {}, block));

        // make the height dimension symbolic
        (*model.mutable_program()->mutable_functions())["main"].mutable_inputs(0)
            ->mutable_type()->mutable_tensortype()->mutable_dimension(2)->set_symbol("s0");

        { // add some enumerated shapes to the model
            auto enumSizes = model.mutable_description()->mutable_input(0)->mutable_type()->
                mutable_imagetype()->mutable_enumeratedsizes();

            auto enumSize = enumSizes->add_sizes();
            enumSize->set_height(10);
            enumSize->set_width(10);

            enumSize = enumSizes->add_sizes();
            enumSize->set_height(10);
            enumSize->set_width(8);
        }

        ML_ASSERT_BAD_WITH_REASON(::CoreML::validate<MLModelType_program>(model),
                                  ResultReason::MODEL_MAIN_BAD_IMAGE_INPUT_SIZE);
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

    // Model and main() must have compatible input image sizes -- image size range not
    // covered by program symbolic shape
    {
        // main(a : fp32[1, 3, 10, 10]):
        //   [ ] { /*empty block*/ } [/*no outputs*/]
        auto block = MakeBlock({ },  // inputs
                               { },  // outputs
                               { }); // ops
        NameAndTypeVec mainParams{
            { "a", MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(1), MakeDim(3), MakeDim(10), MakeDim(10) }) }
        };

        auto model = ProgramWithMain(MakeFunction(mainParams, /*outputs=*/ {}, block));

        { // Change the model's input size to [ (10,10), (8,9) ]
            auto sizeRange = model.mutable_description()->mutable_input(0)->mutable_type()->
                mutable_imagetype()->mutable_imagesizerange();

            auto heightRange = sizeRange->mutable_heightrange();
            heightRange->set_lowerbound(10);
            heightRange->set_upperbound(10);

            auto widthRange = sizeRange->mutable_widthrange();
            widthRange->set_lowerbound(10);
            widthRange->set_upperbound(10);
        }

        ML_ASSERT_BAD_WITH_REASON(::CoreML::validate<MLModelType_program>(model),
                                  ResultReason::MODEL_MAIN_BAD_IMAGE_INPUT_SIZE);
    }

    return 0;
}

int testValidateProgramOp()
{
    // Ops must have non-empty names
    {
        // main(a : fp[2]):
        //   [/*no inputs*/] { z = <unnamed op> relu(x=a); } [/*no outputs*/]
        auto block = MakeBlock({ }, // inputs
                               { }, // outputs
                               { // ops
            MakeOp("", "relu",
                   { { "x", "a" }}, // inputs
                   { }, // outputs
                   { } /* attributes */)
        });
        NameAndTypeVec mainParams{
            { "a", MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(2) }) }
        };

        auto model = ProgramWithMain(MakeFunction(mainParams, /*outputs=*/ {}, block));
        ML_ASSERT_BAD_WITH_REASON(::CoreML::validate<MLModelType_program>(model),
                                  ResultReason::OP_NAME_EMPTY);
    }

    // Op outputs have non-empty names
    {
        // main(a : fp[2]):
        //   [/*no inputs*/] { = relu(x=a); } [/*no outputs*/]
        auto block = MakeBlock({ }, // inputs
                               { }, // outputs
                               { // ops
            MakeOp("z", "relu",
                   { { "x", "a" } }, // inputs
                   { { "", MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(2) }) } }, // outputs
                   { } /* attributes */)
        });
        NameAndTypeVec mainParams{
            { "a", MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(2) }) }
        };

        auto model = ProgramWithMain(MakeFunction(mainParams, /*outputs=*/ {}, block));
        ML_ASSERT_BAD_WITH_REASON(::CoreML::validate<MLModelType_program>(model),
                                  ResultReason::OP_OUTPUT_NAME_EMPTY);
    }

    // Op outputs may not shadow previous definitions
    {
        // main(b : fp[2]):
        //   [] { b = const(); } []  # ERROR: b already defined in containing scope
        auto block = MakeBlock({}, {}, {
            MakeOp("anOp", "const", {}, { { "b", MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(2) }) } }, {})
        });

        NameAndTypeVec mainParams{
            { "b", MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(2) }) }
        };

        auto model = ProgramWithMain(MakeFunction(mainParams, /*outputs=*/ {}, block));
        ML_ASSERT_BAD_WITH_REASON(::CoreML::validate<MLModelType_program>(model),
                                  ResultReason::OP_OUTPUT_NAME_SHADOWS);
    }

    // Ops cannot use their outputs as inputs
    {
        auto block = MakeBlock({}, {}, {
            MakeOp("a", "relu",
                   { { "x", "a", }},
                   { { "a", MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(2) }) } },
                   { })
        });

        auto model = ProgramWithMain(MakeFunction({}, {}, block));
        ML_ASSERT_BAD_WITH_REASON(::CoreML::validate<MLModelType_program>(model),
                                  ResultReason::OP_ARG_OUTPUT_CIRCULAR_DEFINITION);
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
        // [/*no inputs*/] { z = relu(x=a); } [/*no outputs*/]
        auto block = MakeBlock({ }, // inputs
                               { }, // outputs
                               { // ops
            MakeOp("z", "relu",
                   { { "x", "a" } }, // inputs
                   { }, // outputs
                   { } /* attributes */)
        });

        auto model = ProgramWithMain(MakeFunction(/*params=*/ {}, /*outputs=*/ {}, block));
        ML_ASSERT_BAD_WITH_REASON(::CoreML::validate<MLModelType_program>(model),
                                  ResultReason::OP_ARG_VALUE_UNDEFINED);
    }

    // Validate that we catch mis-match in number of input
    {
        // [/*no inputs*/] { z = relu(); } [/*no outputs*/]
        auto block = MakeBlock({ }, // inputs
                               { }, // outputs
                               { // ops
            MakeOp("z", "relu",
                   { }, // inputs
                   { }, // outputs
                   { } /* attributes */)
        });

        auto model = ProgramWithMain(MakeFunction(/*params=*/ {}, /*outputs=*/ {}, block));

        CoreML::Result res = CoreML::validate<MLModelType_program>(model);
        ML_ASSERT_BAD_WITH_REASON(res, ResultReason::OP_REQUIRED_ARG_NOT_FOUND);
        ML_ASSERT(res.message().find("z") != std::string::npos);
        ML_ASSERT(res.message().find("x") != std::string::npos);
    }

    // Validates that we catch mis-match in number of output
    {
        // main(a : fp[2]):
        //   [y <- a] { z = relu(x=y); } [/*no outputs*/]
        auto block = MakeBlock({ { "y", "a" } }, // inputs
                               { }, // outputs
                               { // ops
            MakeOp("z", "relu",
                   { { "x", "y" } }, // inputs
                   { }, // outputs
                   { } /* attributes */)
        });
        NameAndTypeVec mainParams{
            { "a", MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(2) }) }
        };

        auto model = ProgramWithMain(MakeFunction(mainParams, /*outputs=*/ {}, block));

        CoreML::Result res = CoreML::validate<MLModelType_program>(model);
        ML_ASSERT_BAD_WITH_REASON(res, ResultReason::OP_MISMATCHED_OUTPUT_COUNT);
        ML_ASSERT(res.message().find("z") != std::string::npos);
        ML_ASSERT(res.message().find("relu") != std::string::npos);
    }

    // Validates that we catch if input is one of the output
    {
        // main(a : fp[2]):
        //   [y <- b] { z = relu(x=y); } [/*no outputs*/]
        auto block = MakeBlock({ { "y", "a" } }, // inputs
                               { }, // outputs
                               { // ops
            MakeOp("z", "relu",
                   { { "x", "y" } }, // inputs
                   { { "y", MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(2) }) } }, // outputs
                   { } /* attributes */)
        });
        NameAndTypeVec mainParams{
            { "a", MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(2) }) }
        };

        auto model = ProgramWithMain(MakeFunction(mainParams, /*outputs=*/ {}, block));
        ML_ASSERT_BAD_WITH_REASON(::CoreML::validate<MLModelType_program>(model),
                                  ResultReason::PROGRAM_PARSE_THREW);
    }

    // Validate that Block inputs are found
    {
        // main(b : fp[2]):
        //   [y <- b] { z = relu(x=y); } [/*no outputs*/]
        auto block = MakeBlock({ { "y", "b" } }, // inputs
                               { }, // outputs
                               { // ops
            MakeOp("z", "relu",
                   { { "x", "y" } }, // inputs
                   { { "z", MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(2) }) } }, // outputs
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
        // main(a : fp[2]):
        //   [/*no inputs*/] { z = relu(x=a); } [/*no outputs*/]
        auto block = MakeBlock({ }, // inputs
                               { }, // outputs
                               { // ops
            MakeOp("z", "relu",
                   { { "x", "a" } }, // inputs
                   { { "z", MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(2) }) } }, // outputs
                   { } /* attributes */)
        });
        NameAndTypeVec mainParams{
            { "a", MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(2) }) }
        };

        auto model = ProgramWithMain(MakeFunction(mainParams, /*outputs=*/ {}, block));
        ML_ASSERT_GOOD(::CoreML::validate<MLModelType_program>(model));
    }

    // Validate that inputs in the Parameters are found
    {
        // params: { a : fp[2] }
        // main():
        //   [/*no inputs*/] { z = relu(x=a); } [/*no outputs*/]
        auto block = MakeBlock({ }, // inputs
                               { }, // outputs
                               { // ops
            MakeOp("z", "relu",
                   { { "x", "a" } }, // inputs
                   { { "z", MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(2) }) } }, // outputs
                   {} /* attributes */)
        });

        auto main = MakeFunction(/*params=*/ {}, /*outputs=*/ {}, block);

        V5::Value aValue;
        aValue.mutable_type()->CopyFrom(MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(2) }));
        aValue.mutable_immediatevalue()->mutable_tensor()->add_floats(7.f);
        aValue.mutable_immediatevalue()->mutable_tensor()->add_floats(8.f);

        auto model = ProgramWithMain(main);
        (*model.mutable_program()->mutable_parameters())["a"].CopyFrom(aValue);

        ML_ASSERT_GOOD(::CoreML::validate<MLModelType_program>(model));
    }

    // Validate that inputs generated by previous ops are found
    {
        // main(a : fp[2]):
        //   [/*no inputs*/] {
        //     j = relu(x=a);
        //     k = relu(x=j);
        // } [/*no outputs*/]
        auto block = MakeBlock({ }, // inputs
                               { }, // outputs
                               { // ops
            MakeOp("j", "relu",
                   { { "x", "a" } }, // inputs
                   { { "j", MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(2) }) } }, // outputs
                   { } /* attributes */),
            MakeOp("k", "relu",
                   { { "x", "j" } }, // inputs
                   { { "k", MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(2) }) } }, // outputs
                   { } /* attributes */)
        });
        NameAndTypeVec mainParams{
            { "a", MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(2) }) }
        };

        auto model = ProgramWithMain(MakeFunction(mainParams, /*outputs=*/ {}, block));
        ML_ASSERT_GOOD(::CoreML::validate<MLModelType_program>(model));
    }

    return 0;
}

int testValidateProgramPreprocessing()
{
    auto imgTensorType = MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(1), MakeDim(3), MakeDim(10), MakeDim(10) });

    // model(a : img[10, 10, rgb]) {
    //   main(a : fp[1, 3, 10, 10]) {
    //     s = const(1.f)
    //     blue = const(0.f)
    //     green = const(0.f)
    //     red = const(0.f)
    //     gray = const(0.f)
    //     b = scale_image(x = a, scale = s, blueBias = blue, greenBias = green, redBias = red, grayBias = gray)
    //   } -> [b]
    // }
    auto happyModel = [&imgTensorType]() {
        auto block = MakeBlock({ },  // inputs
                               { "b" },  // outputs
                               {
            MakeOp("s",     "const", { }, { { "s",     MakeScalarValueType(V5::ScalarType::FLOAT32) } }, { { "val", MakeFloatValue(1.f) } }),
            MakeOp("blue",  "const", { }, { { "blue",  MakeScalarValueType(V5::ScalarType::FLOAT32) } }, { { "val", MakeFloatValue(0.f) } }),
            MakeOp("green", "const", { }, { { "green", MakeScalarValueType(V5::ScalarType::FLOAT32) } }, { { "val", MakeFloatValue(0.f) } }),
            MakeOp("red",   "const", { }, { { "red",   MakeScalarValueType(V5::ScalarType::FLOAT32) } }, { { "val", MakeFloatValue(0.f) } }),
            MakeOp("gray",  "const", { }, { { "gray",  MakeScalarValueType(V5::ScalarType::FLOAT32) } }, { { "val", MakeFloatValue(0.f) } }),
            MakeOp("b", "scale_image", {
                { "x", "a", }, { "scale", "s" }, { "blueBias", "blue" },
                { "greenBias", "green" },{ "redBias", "red" }, { "grayBias", "gray" } },
            { { "b", imgTensorType }}, { })
        }); // ops
        NameAndTypeVec mainParams{
            { "a", imgTensorType }
        };

        auto model = ProgramWithMain(MakeFunction(mainParams, { imgTensorType }, block));
        auto fd = MakeImageFeatureDescription("a", ImageFeatureType_ColorSpace::ImageFeatureType_ColorSpace_RGB, 10, 10);
        model.mutable_description()->mutable_input(0)->CopyFrom(fd);

        return model;
    };

    // Happy path: scale_image applied to model/main image input
    {
        auto model = happyModel();
        
        auto abc = ::CoreML::validate<MLModelType_program>(model);

        ML_ASSERT_GOOD(::CoreML::validate<MLModelType_program>(model));
    }

    // scale_image applied to model/main non-image input
    {
        auto model = happyModel();
        MakeFeatureDescription("a", imgTensorType);
        model.mutable_description()->mutable_input(0)->CopyFrom(MakeFeatureDescription("a", imgTensorType));

        ML_ASSERT_BAD_WITH_REASON(::CoreML::validate<MLModelType_program>(model),
                                  ResultReason::MODEL_INVALID_INPUT_TYPE);
    }

    // scale_image applied to non-main input
    {
        auto model = happyModel();
        auto main = model.program().functions().at("main");
        (*model.mutable_program()->mutable_functions())["notMain"].CopyFrom(main);

        ML_ASSERT_BAD_WITH_REASON(::CoreML::validate<MLModelType_program>(model),
                                  ResultReason::OP_INVALID_IN_CONTEXT);
    }

    // scale_image in a nested block
    {
        // This is a bit convoluted, but what we're going to do is stuff the scale_image
        // from our happy path model into both sides of a conditional.
        // main() {
        //    <... consts ...>
        //    p = true
        //    b = cond(p, { b.0 = scale_image(...) } -> [b.0], { b.1 = scale_image(...) } -> [b.1])
        // }
        auto model = happyModel();
        auto origScale = model.program().functions().at("main").block().operations(5);
        auto scaleWithName = [&origScale](const std::string& name) {
            V5::Operation renamedScale;
            renamedScale.CopyFrom(origScale);
            renamedScale.mutable_outputs(0)->set_name(name);
            return renamedScale;
        };
        auto condOp = MakeOp("cond", "cond", {{ "pred", "p" }}, {{ "b", imgTensorType }}, {}, {
            MakeBlock({}, { "b.0" }, { scaleWithName("b.0") }),
            MakeBlock({}, { "b.1" }, { scaleWithName("b.1") })
        });

        auto mainBlock = model.mutable_program()->mutable_functions()->at("main").mutable_block();
        mainBlock->mutable_operations(5)->CopyFrom(condOp);
        mainBlock->add_operations()->CopyFrom(MakeOp("p", "const", {}, {{ "p", MakeScalarValueType(V5::ScalarType::BOOL) }}, {{ "val", MakeBoolValue(true) }}));

        auto abs123 = ::CoreML::validate<MLModelType_program>(model);
        
        ML_ASSERT_BAD_WITH_REASON(::CoreML::validate<MLModelType_program>(model),
                                  ResultReason::OP_INVALID_IN_CONTEXT);
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

int testValidatePadOp()
{
    // main(a : fp32[1]) -> (fp32[1]) {
    //   b = const(value=[[1, 1], [2, 2]])
    //   output = pad(x=a, pad=b)
    // } -> (output)

    auto tensorType = MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(2), MakeDim(3) });
    auto outType = MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(4), MakeDim(7) });
    auto paddingType = MakeTensorValueType(V5::ScalarType::INT32, { MakeDim(2), MakeDim(2) });
    auto paddingValue = MakeIntTensorValue({ MakeDim(2), MakeDim(2) }, {1, 1, 2, 2});
    auto modeType = MakeScalarValueType(V5::ScalarType::STRING);
    auto modeValue = MakeStringValue("reflect");
    auto constantType = MakeScalarValueType(V5::ScalarType::FLOAT32);
    auto constantVal = MakeFloatValue(0.0);

    NameAndTypeVec mainParams{
        { "a", tensorType }
    };
    // pad works with mode "reflect"
    {
        // main(a : fp[2]):
        //   [/*no inputs*/] {
        //      b = const(val=[[1, 1], [2, 2]]
        //      c = const(val="reflect")
        //      output = const_pad(x=a, pad=b, mode=c)
        //  } [output]
        auto block = MakeBlock({ }, // inputs
                               { }, // outputs
                               { // ops
            MakeOp("b", "const",
                   { }, // inputs
                   { {"b", paddingType} }, // outputs
                   { {"val", paddingValue }} /* attributes */),
            MakeOp("c", "const",
                   { }, // inputs
                   { {"c", modeType} }, // outputs
                   { {"val", modeValue} } /* attributes */),
            MakeOp("d", "const",
                   { }, // inputs
                   { {"d", constantType} }, // outputs
                   { {"val", constantVal} } /* attributes */),
            MakeOp("aPad", "pad",
                   { {"x", "a"}, {"pad", "b"}, {"mode", "c"}, {"constant_val", "d"} }, // inputs
                   { {"output", outType} }, // outputs
                   { } /* attributes */)
        });

        auto model = ProgramWithMain(MakeFunction(mainParams, /*outputs=*/ {}, block));
        ML_ASSERT_GOOD(::CoreML::validate<MLModelType_program>(model));
    }
    // pad works with mode "constant"
    modeValue = MakeStringValue("constant");
    {
        // main(a : fp[2]):
        //   [/*no inputs*/] {
        //      b = const(val=[[1, 1], [2]]
        //      c = const(val="reflect")
        //      output = const_pad(x=a, pad=b, mode=c)
        //  } [output]
        paddingType = MakeTensorValueType(V5::ScalarType::INT32, { MakeDim(2), MakeDim(2) });
        paddingValue = MakeIntTensorValue({ MakeDim(2), MakeDim(2) }, {1, 1, 2, 2});
        auto block = MakeBlock({ }, // inputs
                               { }, // outputs
                               { // ops
            MakeOp("b", "const",
                   { }, // inputs
                   { {"b", paddingType} }, // outputs
                   { {"val", paddingValue }} /* attributes */),
            MakeOp("c", "const",
                   { }, // inputs
                   { {"c", modeType} }, // outputs
                   { {"val", modeValue} } /* attributes */),
            MakeOp("d", "const",
                   { }, // inputs
                   { {"d", constantType} }, // outputs
                   { {"val", constantVal} } /* attributes */),
            MakeOp("aPad", "pad",
                   { {"x", "a"}, {"pad", "b"}, {"mode", "c"}, {"constant_val", "d"} }, // inputs
                   { {"output", outType} }, // outputs
                   { } /* attributes */)
        });

        auto model = ProgramWithMain(MakeFunction(mainParams, /*outputs=*/ {}, block));
        ML_ASSERT_GOOD(::CoreML::validate<MLModelType_program>(model));
    }

    // pad works with mode "replicate"
    modeValue = MakeStringValue("replicate");
    {
        // main(a : fp[2]):
        //   [/*no inputs*/] {
        //      b = const(val=[[1, 1], [2]]
        //      c = const(val="replicate")
        //      output = const_pad(x=a, pad=b, mode=c)
        //  } [output]
        paddingType = MakeTensorValueType(V5::ScalarType::INT32, { MakeDim(2), MakeDim(2) });
        paddingValue = MakeIntTensorValue({ MakeDim(2), MakeDim(2) }, {1, 1, 2, 2});
        auto block = MakeBlock({ }, // inputs
                               { }, // outputs
                               { // ops
            MakeOp("b", "const",
                   { }, // inputs
                   { {"b", paddingType} }, // outputs
                   { {"val", paddingValue }} /* attributes */),
            MakeOp("c", "const",
                   { }, // inputs
                   { {"c", modeType} }, // outputs
                   { {"val", modeValue} } /* attributes */),
            MakeOp("d", "const",
                   { }, // inputs
                   { {"d", constantType} }, // outputs
                   { {"val", constantVal} } /* attributes */),
            MakeOp("aPad", "pad",
                   { {"x", "a"}, {"pad", "b"}, {"mode", "c"}, {"constant_val", "d"} }, // inputs
                   { {"output", outType} }, // outputs
                   { } /* attributes */)
        });

        auto model = ProgramWithMain(MakeFunction(mainParams, /*outputs=*/ {}, block));
        ML_ASSERT_GOOD(::CoreML::validate<MLModelType_program>(model));
    }
    // pad fails with mode "symmetry"
    modeValue = MakeStringValue("symmetry");
    {
        // main(a : fp[2]):
        //   [/*no inputs*/] {
        //      b = const(val=[[1, 1], [2, 2]]
        //      c = const(val="symmetry")
        //      output = const_pad(x=a, pad=b, mode=c)
        //  } [output]
        paddingType = MakeTensorValueType(V5::ScalarType::INT32, { MakeDim(2), MakeDim(2) });
        paddingValue = MakeIntTensorValue({ MakeDim(2), MakeDim(2) }, {1, 1, 2, 2});
        auto block = MakeBlock({ }, // inputs
                               { }, // outputs
                               { // ops
            MakeOp("b", "const",
                   { }, // inputs
                   { {"b", paddingType} }, // outputs
                   { {"val", paddingValue }} /* attributes */),
            MakeOp("c", "const",
                   { }, // inputs
                   { {"c", modeType} }, // outputs
                   { {"val", modeValue} } /* attributes */),
            MakeOp("d", "const",
                   { }, // inputs
                   { {"d", constantType} }, // outputs
                   { {"val", constantVal} } /* attributes */),
            MakeOp("aPad", "pad",
                   { {"x", "a"}, {"pad", "b"}, {"mode", "c"}, {"constant_val", "d"} }, // inputs
                   { {"output", outType} }, // outputs
                   { } /* attributes */)
        });

        auto model = ProgramWithMain(MakeFunction(mainParams, /*outputs=*/ {}, block));
        ML_ASSERT_BAD_WITH_REASON(::CoreML::validate<MLModelType_program>(model),
                                  ResultReason::OP_PARAM_INVALID);
    }

    // pad fails with padding more than rank 2 and non-constant mode
    modeValue = MakeStringValue("replicate");
    {
        // main(a : fp[2]):
        //   [/*no inputs*/] {
        //      b = const(val=[[1, 1], [2, 2], [1, 1]]
        //      c = const(val="replicate")
        //      output = const_pad(x=a, pad=b, mode=c)
        //  } [output]
        tensorType = MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(2), MakeDim(2), MakeDim(3) });
        paddingType = MakeTensorValueType(V5::ScalarType::INT32, { MakeDim(3), MakeDim(2) });
        paddingValue = MakeIntTensorValue({ MakeDim(3), MakeDim(2) }, {1, 1, 2, 2, 1, 1});
        auto block = MakeBlock({ }, // inputs
                               { }, // outputs
                               { // ops
            MakeOp("b", "const",
                   { }, // inputs
                   { {"b", paddingType} }, // outputs
                   { {"val", paddingValue }} /* attributes */),
            MakeOp("c", "const",
                   { }, // inputs
                   { {"c", modeType} }, // outputs
                   { {"val", modeValue} } /* attributes */),
            MakeOp("d", "const",
                   { }, // inputs
                   { {"d", constantType} }, // outputs
                   { {"val", constantVal} } /* attributes */),
            MakeOp("aPad", "pad",
                   { {"x", "a"}, {"pad", "b"}, {"mode", "c"}, {"constant_val", "d"} }, // inputs
                   { {"output", outType} }, // outputs
                   { } /* attributes */)
        });

        auto model = ProgramWithMain(MakeFunction(mainParams, /*outputs=*/ {}, block));
        ML_ASSERT_BAD_WITH_REASON(::CoreML::validate<MLModelType_program>(model),
                                  ResultReason::OP_PARAM_INVALID);
    }
    // pad requires padding to be of (nx2) shape
    mainParams = { { "a", tensorType } };
    {
        // main(a : fp[2]):
        //   [/*no inputs*/] {
        //      b = const(val=[[1, 1, 1], [2, 2, 2], [1, 1, 1]]
        //      c = const(val="constant")
        //      output = const_pad(x=a, pad=b, mode=c)
        //  } [output]
        modeValue = MakeStringValue("constant");
        tensorType = MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(2), MakeDim(2), MakeDim(3) });
        paddingType = MakeTensorValueType(V5::ScalarType::INT32, { MakeDim(3), MakeDim(3) });
        paddingValue = MakeIntTensorValue({ MakeDim(3), MakeDim(3) }, {1, 1, 1, 2, 2, 2, 1, 1, 1});
        auto block = MakeBlock({ }, // inputs
                               { }, // outputs
                               { // ops
            MakeOp("b", "const",
                   { }, // inputs
                   { {"b", paddingType} }, // outputs
                   { {"val", paddingValue }} /* attributes */),
            MakeOp("c", "const",
                   { }, // inputs
                   { {"c", modeType} }, // outputs
                   { {"val", modeValue} } /* attributes */),
            MakeOp("d", "const",
                   { }, // inputs
                   { {"d", constantType} }, // outputs
                   { {"val", constantVal} } /* attributes */),
            MakeOp("aPad", "pad",
                   { {"x", "a"}, {"pad", "b"}, {"mode", "c"}, {"constant_val", "d"} }, // inputs
                   { {"output", outType} }, // outputs
                   { } /* attributes */)
        });

        auto model = ProgramWithMain(MakeFunction(mainParams, /*outputs=*/ {}, block));
        ML_ASSERT_BAD_WITH_REASON(::CoreML::validate<MLModelType_program>(model),
                                  ResultReason::OP_PARAM_INVALID);
    }

    // pad working
    {
        // main(a : fp[2]):
        //   [/*no inputs*/] {
        //      b = const(val=[[1, 1], [2, 2], [1, 1]]
        //      c = const(val="reflect")
        //      output = const_pad(x=a, pad=b, mode=c)
        //  } [output]
        modeValue = MakeStringValue("constant");
        tensorType = MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(2), MakeDim(2), MakeDim(3) });
        paddingType = MakeTensorValueType(V5::ScalarType::INT32, { MakeDim(3), MakeDim(2) });
        paddingValue = MakeIntTensorValue({ MakeDim(3), MakeDim(2) }, {1, 1, 2, 2, 1, 1});
        auto block = MakeBlock({ }, // inputs
                               { }, // outputs
                               { // ops
            MakeOp("b", "const",
                   { }, // inputs
                   { {"b", paddingType} }, // outputs
                   { {"val", paddingValue }} /* attributes */),
            MakeOp("c", "const",
                   { }, // inputs
                   { {"c", modeType} }, // outputs
                   { {"val", modeValue} } /* attributes */),
            MakeOp("d", "const",
                   { }, // inputs
                   { {"d", constantType} }, // outputs
                   { {"val", constantVal} } /* attributes */),
            MakeOp("aPad", "pad",
                   { {"x", "a"}, {"pad", "b"}, {"mode", "c"}, {"constant_val", "d"} }, // inputs
                   { {"output", outType} }, // outputs
                   { } /* attributes */)
        });

        auto model = ProgramWithMain(MakeFunction(mainParams, /*outputs=*/ {}, block));
        ML_ASSERT_GOOD(::CoreML::validate<MLModelType_program>(model));
    }

    return 0;
}

int testValidatePoolingOp()
{
    auto iType = MakeTensorValueType(V5::ScalarType::FLOAT32, {
        MakeDim(1), MakeDim(3), MakeDim(2), MakeDim(3)});
    auto oType = MakeTensorValueType(V5::ScalarType::FLOAT32, {
        MakeDim(1), MakeDim(3), MakeDim(1), MakeDim(2)});
    auto intTensor2Type = MakeTensorValueType(V5::ScalarType::INT32, {MakeDim(2)});
    auto intTensor3Type = MakeTensorValueType(V5::ScalarType::INT32, {MakeDim(3)});
    auto intTensor4Type = MakeTensorValueType(V5::ScalarType::INT32, {MakeDim(4)});
    auto boolScalarType = MakeScalarValueType(V5::ScalarType::BOOL);
    auto stringScalarType = MakeScalarValueType(V5::ScalarType::STRING);
    NameAndTypeVec mainParams{{"x", iType}};

    { // negative case: invalid length of kernel_sizes
        auto block = MakeBlock({}, {"output"}, {
            MakeOp("kernel_sizes", "const", {},
                   {{"kernel_sizes", intTensor3Type}},
                   {{"val", MakeIntTensorValue({MakeDim(3)}, {2, 1, 1})}}),
            MakeOp("strides", "const", {},
                   {{"strides", intTensor2Type}},
                   {{"val", MakeIntTensorValue({MakeDim(2)}, {2, 2})}}),
            MakeOp("pad_type", "const", {},
                   {{"pad_type", stringScalarType}},
                   {{"val", MakeStringValue("valid")}}),
            MakeOp("exclude_padding_from_average", "const", {},
                   {{"exclude_padding_from_average", boolScalarType}},
                   {{"val", MakeBoolValue(false)}}),
            MakeOp("avg_pool",
                   "avg_pool",
                   {{"x", "x"}, {"kernel_sizes", "kernel_sizes"},
                    {"strides", "strides"}, {"pad_type", "pad_type"},
                    {"exclude_padding_from_average", "exclude_padding_from_average"}},
                   {{"output", oType}}, {})
        });

        auto model = ProgramWithMain(MakeFunction(mainParams, {oType}, block));
        ML_ASSERT_BAD_WITH_REASON(::CoreML::validate<MLModelType_program>(model),
                                  ResultReason::OP_PARAM_INVALID);
    }

    { // negative case: invalid length of strides
        auto block = MakeBlock({}, {"output"}, {
            MakeOp("kernel_sizes", "const", {},
                   {{"kernel_sizes", intTensor2Type}},
                   {{"val", MakeIntTensorValue({MakeDim(2)}, {1, 1})}}),
            MakeOp("strides", "const", {},
                   {{"strides", intTensor4Type}},
                   {{"val", MakeIntTensorValue({MakeDim(4)}, {1, 1, 2, 2})}}),
            MakeOp("pad_type", "const", {},
                   {{"pad_type", stringScalarType}},
                   {{"val", MakeStringValue("valid")}}),
            MakeOp("exclude_padding_from_average", "const", {},
                   {{"exclude_padding_from_average", boolScalarType}},
                   {{"val", MakeBoolValue(false)}}),
            MakeOp("avg_pool",
                   "avg_pool",
                   {{"x", "x"}, {"kernel_sizes", "kernel_sizes"},
                    {"strides", "strides"}, {"pad_type", "pad_type"},
                    {"exclude_padding_from_average", "exclude_padding_from_average"}},
                   {{"output", oType}}, {})
        });

        auto model = ProgramWithMain(MakeFunction(mainParams, {oType}, block));
        ML_ASSERT_BAD_WITH_REASON(::CoreML::validate<MLModelType_program>(model),
                                  ResultReason::OP_PARAM_INVALID);
    }

    { // negative case: invalid pad mode
        auto block = MakeBlock({}, {"output"}, {
            MakeOp("kernel_sizes", "const", {},
                   {{"kernel_sizes", intTensor2Type}},
                   {{"val", MakeIntTensorValue({MakeDim(2)}, {1, 1})}}),
            MakeOp("strides", "const", {},
                   {{"strides", intTensor2Type}},
                   {{"val", MakeIntTensorValue({MakeDim(2)}, {1, 1})}}),
            MakeOp("pad_type", "const", {},
                   {{"pad_type", stringScalarType}},
                   {{"val", MakeStringValue("invalid_mode")}}),
            MakeOp("exclude_padding_from_average", "const", {},
                   {{"exclude_padding_from_average", boolScalarType}},
                   {{"val", MakeBoolValue(false)}}),
            MakeOp("avg_pool",
                   "avg_pool",
                   {{"x", "x"}, {"kernel_sizes", "kernel_sizes"},
                    {"strides", "strides"}, {"pad_type", "pad_type"},
                    {"exclude_padding_from_average", "exclude_padding_from_average"}},
                   {{"output", oType}}, {})
        });

        auto model = ProgramWithMain(MakeFunction(mainParams, {oType}, block));
        ML_ASSERT_BAD_WITH_REASON(::CoreML::validate<MLModelType_program>(model),
                                  ResultReason::OP_PARAM_INVALID);
    }

    return 0;
}
