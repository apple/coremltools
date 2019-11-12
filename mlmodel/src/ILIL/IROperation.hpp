//
//  IROperation.hpp
//  CoreML
//
//  Created by Jeff Kilpatrick on 11/7/19.
//  Copyright © 2019 Apple Inc. All rights reserved.
//

#pragma once

#include "ILIL/IROperatorDescription.hpp"
#include "ILIL/IRScope.hpp"

#include <string>
#include <vector>

namespace CoreML {
namespace ILIL {

class IRBlock;

/**
 A single operation/layer/node.
 */
class IROperation {
public:
    using IRBlockPtr = std::shared_ptr<IRBlock>;
    using IRBlockPtrVec = std::vector<IRBlockPtr>;
    using StringVec = std::vector<std::string>;

    virtual ~IROperation();

    /** Get an attribute value. */
    virtual const IRValue& GetAttribute(const std::string& name) const = 0;

    /** Get this operation's nested blocks. */
    virtual const IRBlockPtrVec& GetBlocks() const = 0;

    /** Get a description of the operator being invoked. */
    const IROperatorDescription& GetDescription() const;

    /** Get the name of the argument specified for the given parameter. */
    virtual const std::string& GetInput(const std::string& param) const = 0;

    /** Get the names of all specified arguments. */
    virtual const StringVec& GetInputNames() const = 0;

    /** Get the name of this invocation. */
    virtual const std::string& GetName() const = 0;

    /** How many inputs does this operation have? */
    virtual uint64_t GetNumInputs() const = 0;

    /** How many outputs does this operation have? */
    virtual uint64_t GetNumOutputs() const = 0;

    /** Get the names of all outputs. */
    virtual const StringVec& GetOutputNames() const = 0;

    /** Get the scope in which this op is defined. */
    virtual const IRScope& GetScope() const = 0;

    /** Get the type of operator being invoked. */
    virtual IROperatorType GetType() const = 0;

    /**
     Convenience method to get the indicated compile-time constant from our scope.
     */
    IRScope::ConstIRValuePtr GetValue(const std::string& name) const;
};

}
}
