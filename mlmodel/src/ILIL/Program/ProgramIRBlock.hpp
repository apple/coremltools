//
//  ProgramIRBlock.hpp
//  CoreML
//
//  Created by Jeff Kilpatrick on 11/7/19.
//  Copyright © 2019 Apple Inc. All rights reserved.
//

#pragma once

#include "ILIL/IRBlock.hpp"

namespace CoreML {

namespace Specification {
class Model;

namespace V5 {
class Block;
}
}

namespace ILIL {
namespace Program {

class ProgramIRBlock : public ILIL::IRBlock {
public:
    using BlockSpec = ::CoreML::Specification::V5::Block;
    using ConstScopePtr = std::shared_ptr<const IRScope>;
    using ScopePtr = std::shared_ptr<IRScope>;

    ~ProgramIRBlock();

    /** Create a new instance with the given attributes. */
    static std::unique_ptr<ProgramIRBlock> Make(ScopePtr scope,
                                                InputBindingMap&& inputs,
                                                StringVec&& outputs,
                                                ConstIROperationPtrVec&& operations);

    /** Create a new instance from the specified block in the given model. */
    static std::unique_ptr<ProgramIRBlock> Parse(const BlockSpec& block, ConstScopePtr parentScope);

protected:
    ProgramIRBlock();
};

}
}
}
