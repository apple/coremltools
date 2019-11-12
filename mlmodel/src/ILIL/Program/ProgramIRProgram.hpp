//
//  ProgramIRProgram.hpp
//  CoreML
//
//  Created by Jeff Kilpatrick on 11/7/19.
//  Copyright © 2019 Apple Inc. All rights reserved.
//

#pragma once

#include "ILIL/IRProgram.hpp"

namespace CoreML {

namespace Specification {
class Model;
namespace V5 {
class Program;
}
}

namespace ILIL {
namespace Program {

class ProgramIRProgram : public ILIL::IRProgram {
public:
    using ProgramSpec = ::CoreML::Specification::V5::Program;

    ~ProgramIRProgram();

    /** Create a new instance with the specified program spec in the given model. */
    static std::unique_ptr<ProgramIRProgram> Parse(const ProgramSpec& program);

protected:
    ProgramIRProgram();
};

}
}
}
