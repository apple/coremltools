//
//  IRBlock.hpp
//  CoreML
//
//  Created by Jeff Kilpatrick on 11/7/19.
//  Copyright © 2019 Apple Inc. All rights reserved.
//

#pragma once

#include "ILIL/IROperation.hpp"
#include "ILIL/IRScope.hpp"
#include <memory>

namespace CoreML {
namespace ILIL {

/**
 A basic block with single entry and exit.
 */
class IRBlock {
public:
    using InputBindingMap = std::unordered_map<std::string, std::string>;
    using ConstIROperationPtr = std::unique_ptr<IROperation>;
    using ConstIROperationPtrVec = std::vector<ConstIROperationPtr>;
    using StringVec = std::vector<std::string>;

    virtual ~IRBlock() = default;

    /** What is the name of the variable bound to the specified parameter? */
    virtual const std::string& GetArgumentName(const std::string& parameterName) = 0;

    /** Get the mapping of parameter names to argument names. */
    virtual const InputBindingMap& GetInputs() const = 0;

    /** Get this block's ops. */
    virtual const ConstIROperationPtrVec& GetOperations() const = 0;

    /** Get this block's output names. */
    virtual const StringVec& GetOutputs() const = 0;

    /** Get the scope associated with this block. */
    virtual const IRScope& GetScope() const = 0;
};

}
}
