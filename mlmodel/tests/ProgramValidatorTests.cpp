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

using namespace ::CoreML::Specification;
using namespace ::ProgramModelUtils;

static Model EmptyProgram()
{
    Model model;
    (*model.mutable_program()->mutable_functions())["main"]
        .CopyFrom(MakeFunction({}, {}, MakeBlock({}, {}, {})));
    return model;
};

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
        // main(a : string):
        //   [ <- a] { /*empty block*/ } [/*no outputs*/]
        auto block = MakeBlock({ { "", "a" } }, // inputs
                               { }, // outputs
                               { } // ops
                               );
        NameAndTypeVec mainParams{
            { "a", MakeScalarValueType(V5::ScalarType::STRING) }
        };
        auto main = MakeFunction(mainParams, /*outputs=*/ {}, block);

        Model model;
        (*model.mutable_program()->mutable_functions())["main"].CopyFrom(main);
        ML_ASSERT_BAD(::CoreML::validate<MLModelType_program>(model));
    }

    // Block return values must exist
    {
        // main(a : string, b : fp[2]):
        //   [/*no inputs*/] { z = activation(data=b, type=a); } [ bonk ]
        auto block = MakeBlock({ }, // inputs
                               { "bonk" }, // outputs
                               { // ops
            MakeOp("z", "activation",
                   { { "type", "a" }, { "data", "b" } }, // inputs
                   { }, // outputs
                   { } /* attributes */)
        });
        NameAndTypeVec mainParams{
            { "a", MakeScalarValueType(V5::ScalarType::STRING) },
            { "b", MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(2) }) }
        };
        auto main = MakeFunction(mainParams, /*outputs=*/ {}, block);

        Model model;
        (*model.mutable_program()->mutable_functions())["main"].CopyFrom(main);
        ML_ASSERT_BAD(::CoreML::validate<MLModelType_program>(model));
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
        ML_ASSERT_BAD(::CoreML::validate<MLModelType_program>(model));
    }

    return 0;
}

int testValidateProgramMain()
{
    // main is a required function
    {
        Model model;
        ML_ASSERT_BAD(::CoreML::validate<MLModelType_program>(model));
        return 0;
    }
}

int testValidateProgramOp()
{
    // Ops must have non-empty names
    {
        // main(a : string, b : fp[2]):
        //   [/*no inputs*/] { z = <unnamed op> activation(data=b, type=a); } [/*no outputs*/]
        auto block = MakeBlock({ }, // inputs
                               { }, // outputs
                               { // ops
            MakeOp("", "activation",
                   { { "type", "a" }, { "data", "b" } }, // inputs
                   { }, // outputs
                   { } /* attributes */)
        });
        NameAndTypeVec mainParams{
            { "a", MakeScalarValueType(V5::ScalarType::STRING) },
            { "b", MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(2) }) }
        };
        auto main = MakeFunction(mainParams, /*outputs=*/ {}, block);

        Model model;
        (*model.mutable_program()->mutable_functions())["main"].CopyFrom(main);
        ML_ASSERT_BAD(::CoreML::validate<MLModelType_program>(model));
    }

    // Op outputs have non-empty names
    {
        // main(a : string, b : fp[2]):
        //   [/*no inputs*/] {  = activation(data=b, type=a); } [/*no outputs*/]
        auto block = MakeBlock({ }, // inputs
                               { }, // outputs
                               { // ops
            MakeOp("z", "activation",
                   { { "type", "a" }, { "data", "b" } }, // inputs
                   { { "", MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(2) }) } }, // outputs
                   { } /* attributes */)
        });
        NameAndTypeVec mainParams{
            { "a", MakeScalarValueType(V5::ScalarType::STRING) },
            { "b", MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(2) }) }
        };
        auto main = MakeFunction(mainParams, /*outputs=*/ {}, block);

        Model model;
        (*model.mutable_program()->mutable_functions())["main"].CopyFrom(main);
        ML_ASSERT_BAD(::CoreML::validate<MLModelType_program>(model));
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
        auto main = MakeFunction(/*params=*/ {}, /*outputs=*/ {}, block);

        Model model;
        (*model.mutable_program()->mutable_functions())["main"].CopyFrom(main);
        ML_ASSERT_BAD(::CoreML::validate<MLModelType_program>(model));

    }

    // Validate that we catch missing inputs
    {
        // [/*no inputs*/] { z = activation(data=y, activation=x); } [/*no outputs*/]
        auto block = MakeBlock({ }, // inputs
                               { }, // outputs
                               { // ops
            MakeOp("z", "activation",
                   { { "activation", "a" }, { "data", "b" } }, // inputs
                   { }, // outputs
                   { } /* attributes */)
        });
        auto main = MakeFunction(/*params=*/ {}, /*outputs=*/ {}, block);

        Model model;
        (*model.mutable_program()->mutable_functions())["main"].CopyFrom(main);
        ML_ASSERT_BAD(::CoreML::validate<MLModelType_program>(model));
    }

    // Validate that Block inputs are found
    {
        // main(a : string, b : fp[2]):
        //   [x <- a, y <- b] { z = activation(data=y, type=x); } [/*no outputs*/]
        auto block = MakeBlock({ { "x", "a" }, { "y", "b" } }, // inputs
                               { }, // outputs
                               { // ops
            MakeOp("z", "activation",
                   { { "type", "x" }, { "data", "y" } }, // inputs
                   { }, // outputs
                   { } /* attributes */)
        });
        NameAndTypeVec mainParams{
            { "a", MakeScalarValueType(V5::ScalarType::STRING) },
            { "b", MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(2) }) }
        };
        auto main = MakeFunction(mainParams, /*outputs=*/ {}, block);

        Model model;
        (*model.mutable_program()->mutable_functions())["main"].CopyFrom(main);
        ML_ASSERT_GOOD(::CoreML::validate<MLModelType_program>(model));
    }

    // Validate that Function inputs are found
    {
        // main(a : string, b : fp[2]):
        //   [/*no inputs*/] { z = activation(data=b, type=a); } [/*no outputs*/]
        auto block = MakeBlock({ }, // inputs
                               { }, // outputs
                               { // ops
            MakeOp("z", "activation",
                   { { "type", "a" }, { "data", "b" } }, // inputs
                   { }, // outputs
                   { } /* attributes */)
        });
        NameAndTypeVec mainParams{
            { "a", MakeScalarValueType(V5::ScalarType::STRING) },
            { "b", MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(2) }) }
        };
        auto main = MakeFunction(mainParams, /*outputs=*/ {}, block);

        Model model;
        (*model.mutable_program()->mutable_functions())["main"].CopyFrom(main);
        ML_ASSERT_GOOD(::CoreML::validate<MLModelType_program>(model));
    }

    // Validate that inputs in the Parameters are found
    {
        // params: { b : fp[2] }
        // main(a : string):
        //   [/*no inputs*/] { z = activation(data=y, type=x); } [/*no outputs*/]
        auto block = MakeBlock({ }, // inputs
                               { }, // outputs
                               { // ops
            MakeOp("z", "activation",
                   { { "type", "a" }, { "data", "b" } }, // inputs
                   { }, // outputs
                   {} /* attributes */)
        });
        NameAndTypeVec mainParams{
            { "a", MakeScalarValueType(V5::ScalarType::STRING) }
        };
        auto main = MakeFunction(mainParams, /*outputs=*/ {}, block);

        V5::Value bValue;
        bValue.mutable_type()->CopyFrom(MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(2) }));
        bValue.mutable_immediatevalue()->mutable_tensor()->add_floats(7.f);
        bValue.mutable_immediatevalue()->mutable_tensor()->add_floats(8.f);

        Model model;
        (*model.mutable_program()->mutable_functions())["main"].CopyFrom(main);
        (*model.mutable_program()->mutable_parameters())["b"].CopyFrom(bValue);

        ML_ASSERT_GOOD(::CoreML::validate<MLModelType_program>(model));
    }

    // Validate that inputs generated by previous ops are found
    {
        // main(a : string, b : fp[2]):
        //   [/*no inputs*/] {
        //     j = activation(data=b, type=a);
        //     k = activation(data=j, type=a);
        // } [/*no outputs*/]
        auto block = MakeBlock({ }, // inputs
                               { }, // outputs
                               { // ops
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
            { "a", MakeScalarValueType(V5::ScalarType::STRING) },
            { "b", MakeTensorValueType(V5::ScalarType::FLOAT32, { MakeDim(2) }) }
        };
        auto main = MakeFunction(mainParams, /*outputs=*/ {}, block);

        Model model;
        (*model.mutable_program()->mutable_functions())["main"].CopyFrom(main);
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
        ML_ASSERT_BAD(::CoreML::validate<MLModelType_program>(model));
    }

    // Parameters must have names
    {
        Model model = EmptyProgram();
        (*model.mutable_program()->mutable_parameters())[""] = MakeBoolValue(true);
        ML_ASSERT_BAD(::CoreML::validate<MLModelType_program>(model));
    }

    return 0;
}
