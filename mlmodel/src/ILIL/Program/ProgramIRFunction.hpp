//
//  ProgramIRFunction.hpp
//  CoreML
//
//  Created by Jeff Kilpatrick on 11/7/19.
//  Copyright © 2019 Apple Inc. All rights reserved.
//

#pragma once

#include "ILIL/IRFunction.hpp"

namespace CoreML {

namespace Specification {
class Model;

namespace V5 {
class Function;
}
}

namespace ILIL {
namespace Program {

class ProgramIRFunction : public ILIL::IRFunction {
public:
    using FunctionSpec = ::CoreML::Specification::V5::Function;

    ~ProgramIRFunction();

    /** Create a new instance from the specified function in the given model. */
    static std::unique_ptr<ProgramIRFunction>
    Parse(const FunctionSpec& function);

protected:
    ProgramIRFunction();
};

}
}
}
