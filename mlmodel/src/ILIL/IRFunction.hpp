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
    using ConstIRBlockPtr = std::shared_ptr<const IRBlock>;
    using ConstIRValueTypePtr = std::shared_ptr<const IRValueType>;
    using ConstIRValueTypePtrVec = std::vector<ConstIRValueTypePtr>;
    using Rename = std::pair<std::string, std::string>;
    using RenameVec = std::vector<Rename>;
    using ValueTypeMap = std::unordered_map<std::string, std::shared_ptr<const IRValueType>>;

    virtual ~IRFunction() = default;

    /** Get this function's body. */
    virtual const IRBlock& GetBlock() const = 0;

    /** Get this function's parameter declarations. */
    virtual const ValueTypeMap& GetInputs() const = 0;

    /**
     Get this function's parameter names and types.
     @throws std::out_of_range if the requested input does not exist.
     */
    virtual const IRValueType& GetInputType(const std::string& paramName) const = 0;

    /**
     Try to get this function's parameter names and types.
     @returns a pointer or nullptr of the requested input does not exist.
     */
    virtual const IRValueType* TryGetInputType(const std::string& paramName) const = 0;

    /** Get the output types. */
    virtual const ConstIRValueTypePtrVec& GetOutputTypes() const = 0;

    /** Get the scope associated with this function. */
    virtual const IRScope& GetScope() const = 0;

    /** Create a new instance with the given block. */
    virtual std::unique_ptr<const IRFunction> WithBlock(ConstIRBlockPtr block) const = 0;

    /** Create a new instance with the specified value renames applied. */
    virtual std::unique_ptr<const IRFunction> WithRenames(const RenameVec& renames) const = 0;
};

}
}
