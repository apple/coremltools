//
//  IROperatorDescription.cpp
//  mlmodel
//
//  Created by Jeff Kilpatrick on 11/13/19.
//  Copyright © 2019 Apple Inc. All rights reserved.
//

#include "ILIL/IROperatorDescription.hpp"

#include <limits>
#include <mutex>
#include <unordered_map>

using namespace CoreML::ILIL;

IROperatorDescription::IROperatorDescription(uint64_t minInputs, uint64_t maxInputs, uint64_t numOutputs)
    : m_minInputs(minInputs)
    , m_maxInputs(maxInputs)
    , m_numOutputs(numOutputs)
{ }

uint64_t IROperatorDescription::GetMinInputs() const
{
    return m_minInputs;
}

uint64_t IROperatorDescription::GetMaxInputs() const
{
    return m_maxInputs;
}

uint64_t IROperatorDescription::GetNumOutputs() const
{
    return m_numOutputs;
}

using OpDescriptionMap = std::unordered_map<IROperatorType, IROperatorDescription>;
static std::unique_ptr<OpDescriptionMap> MakeOpDescriptionMap()
{
    std::unique_ptr<OpDescriptionMap> map(new OpDescriptionMap{
        { IROperatorType::Activation, IROperatorDescription(2, 4, 1) },
        { IROperatorType::Const, IROperatorDescription(0, 0, 1) },
        { IROperatorType::Convolution, IROperatorDescription(1, 2, 1) },
        { IROperatorType::InnerProduct, IROperatorDescription(5, 6, 1) },
        { IROperatorType::Pooling, IROperatorDescription(1, std::numeric_limits<uint64_t>::max(), 1) }
    });

    assert(map->size() == static_cast<size_t>(IROperatorType::COUNT));
    return map;
}

const IROperatorDescription& ::CoreML::ILIL::GetIROperatorDescription(IROperatorType type)
{
#pragma clang diagnostic push
    // This warns us that tearing down "s_opDescriptions" will happen at program exit.
    // The concern is that this happens in an effectively non-deterministic
    // order which can cause crashes when statics depend on each other.
    // It can also be slow. In this case, we have no external dependencies so
    // it's safe. The cost is freeing our data (which is useless at shutdown),
    // but not terribly costly.
    // If we're deeply concerned, there are at least two ways to get around this.
    // (1) Explicitly instantiate and teardown statics from main().
    // (2) Observe that leaking this object at shutdown is really fine and just
    //     initialize it using a raw new.
    // The first one is a lot of work and the second will probably make static
    // analysis complain so I'm just going to tell the compiler to hush.
#pragma clang diagnostic ignored "-Wexit-time-destructors"
    static std::unique_ptr<OpDescriptionMap> s_opDescriptions;
#pragma clang diagnostic pop
    static std::once_flag flag;

    std::call_once(flag, []() {
        s_opDescriptions = MakeOpDescriptionMap();
    });

    return s_opDescriptions->at(type);
}
