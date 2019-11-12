//
//  ProgramIROperation.hpp
//  CoreML
//
//  Created by Jeff Kilpatrick on 11/7/19.
//  Copyright © 2019 Apple Inc. All rights reserved.
//

#pragma once

#include "ILIL/IROperation.hpp"

#include <memory>

namespace CoreML {

namespace Specification {
class Model;
namespace V5 {
class Operation;
}
}

namespace ILIL {
namespace Program {

class ProgramIROperation : public ILIL::IROperation {
public:
    using OperationSpec = ::CoreML::Specification::V5::Operation;
    using ScopePtr = std::shared_ptr<IRScope>;

    ~ProgramIROperation();

    /** Create a new instance with the specified operation spec in the given model. */
    static std::unique_ptr<ProgramIROperation>
    Parse(const OperationSpec& operation, ScopePtr scope);

protected:
    ProgramIROperation();
};

}
}
}
