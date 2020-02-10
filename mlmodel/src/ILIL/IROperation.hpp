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
#include <unordered_map>
#include <vector>

namespace CoreML {
namespace ILIL {

class IRBlock;

/**
 A single operation/layer/node.
 */
class IROperation {
public:
    using AttributesMap = std::unordered_map<std::string, std::shared_ptr<const IRValue>>;
    using InputBindingMap = std::unordered_map<std::string, std::string>;
    using IRBlockPtr = std::shared_ptr<const IRBlock>;
    using IRBlockPtrVec = std::vector<IRBlockPtr>;
    using Rename = std::pair<std::string, std::string>;
    using RenameVec = std::vector<Rename>;
    using ScopePtr = std::shared_ptr<IRScope>;
    using StringVec = std::vector<std::string>;

    virtual ~IROperation();

    /**
     Get an attribute value.
     @throws std::out_of_range if an attribute by the given name does not exist.
     */
    virtual const IRValue& GetAttribute(const std::string& name) const = 0;

    /** Get all attributes. */
    virtual const AttributesMap& GetAttributes() const = 0;

    /** Get this operation's nested blocks. */
    virtual const IRBlockPtrVec& GetBlocks() const = 0;

    /**
     Get the name of the argument specified for the given parameter.
     @throws std::out_of_range if the requested input does not exist.
     */
    virtual const std::string& GetInput(const std::string& param) const = 0;

    /**
     Get all parameter/argument bindings.
     */
    virtual const InputBindingMap& GetInputs() const = 0;

    /**
     Get the name of the argument specified for the given parameter.
     @return A non-owning pointer to the name of the argument for the requested
             parameter or nullptr if no argument was provided.
     */
    virtual const std::string* TryGetInput(const std::string& param) const = 0;

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
    virtual const std::string& GetType() const = 0;

    /** Create a new instance with the given blocks. */
    virtual std::unique_ptr<IROperation> WithBlocks(IRBlockPtrVec&& blocks) const = 0;

    /**
     Create a new instance with the specified value renames applied.
     @param renames Pairs of (old, new).
     @param scope A scope with renames already applied.
     */
    virtual std::unique_ptr<IROperation> WithRenames(const RenameVec& renames, ScopePtr scope) const = 0;

    /**
     Convenience method to get the indicated compile-time constant from our scope.
     */
    const IRValue& GetValue(const std::string& name) const;

    /**
     Convenience method to get the indicated compile-time constant from our scope.
     */
    const IRValue* TryGetValue(const std::string& name) const;
};

}
}
