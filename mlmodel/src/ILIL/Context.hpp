//
//  Context.h
//  CoreML
//
//  Created by Bhushan Sonawane on 1/17/20.
//  Copyright © 2020 Apple Inc. All rights reserved.
//

#ifndef Context_h
#define Context_h
#include <memory>
#include <string>
#include <unordered_map>

#include "IROperatorDescription.hpp"

namespace CoreML {
namespace ILIL {

class Context {
public:
    using IROpDescriptionMap = std::unordered_map<std::string, IROperatorDescription>;
    
    virtual ~Context();
    Context(const Context&) = delete;
    Context(Context&&) = delete;
    Context& operator=(const Context&) = delete;
    Context& operator=(Context&&) = delete;
    Context();

    const IROperatorDescription& GetOperatorDescription(const IROperation&) const;
    const IROperatorDescription* TryGetOperatorDescription(const IROperation& op) const;
    const IROperatorDescription& GetOperatorDescription(const std::string&) const;
    const IROperatorDescription* TryGetOperatorDescription(const std::string&) const;

protected:
    Context(IROpDescriptionMap&& additionalOps);
    
private:
    const IROpDescriptionMap m_opDescriptors;

}; // class Context
} // namespace ILIL
} // namespace CoreML

#endif /* Context_h */
