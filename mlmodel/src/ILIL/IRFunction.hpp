//
//  IRFunction.hpp
//  CoreML
//
//  Created by Jeff Kilpatrick on 11/7/19.
//  Copyright © 2019 Apple Inc. All rights reserved.
//

#pragma once

#include "ILIL/IRBlock.hpp"

namespace CoreML {
namespace ILIL {

/** A function in an IRProgram. */
class IRFunction {
public:
    using ConstIRValueTypePtr = std::unique_ptr<const IRValueType>;
    using ConstIRValueTypePtrVec = std::vector<ConstIRValueTypePtr>;

    virtual ~IRFunction() = default;

    /** Get this function's body. */
    virtual const IRBlock& GetBlock() const = 0;

    /** Get this function's parameter names and types. */
    virtual const IRValueType& GetInputType(const std::string& paramName) const = 0;

    /** Get the output types. */
    virtual const ConstIRValueTypePtrVec& GetOutputTypes() const = 0;
};

}
}
