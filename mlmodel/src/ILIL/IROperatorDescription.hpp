//
//  IROperatorDescription.hpp
//  mlmodel
//
//  Created by Jeff Kilpatrick on 11/13/19.
//  Copyright © 2019 Apple Inc. All rights reserved.
//

#pragma once

#include "ILIL/IROperatorType.hpp"

#include <cstdint>

namespace CoreML {
namespace ILIL {

/**
 A description of an ILIL operator.
 */
class IROperatorDescription {
public:
    IROperatorDescription(uint64_t minInputs, uint64_t maxInputs, uint64_t numOutputs);

    /** Get the minimum number of inputs required by this operator. */
    uint64_t GetMinInputs() const;

    /** Get the maximum number of inputs required by this operator. */
    uint64_t GetMaxInputs() const;

    /** Get the number of outputs this operator produces. */
    uint64_t GetNumOutputs() const;

private:
    uint64_t m_minInputs;
    uint64_t m_maxInputs;
    uint64_t m_numOutputs;
};

/** Get a description of the specified operator. */
const IROperatorDescription& GetIROperatorDescription(IROperatorType type);
}
}
