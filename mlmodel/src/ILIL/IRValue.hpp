
//
//  IRValueType.hpp
//  CoreML
//
//  Created by Jeff Kilpatrick on 11/8/19.
//  Copyright © 2019 Apple Inc. All rights reserved.
//

#pragma once

#include "ILIL/IRValueType.hpp"

namespace CoreML {
namespace ILIL {

class IRValue {
public:
    using ConstIRValueTypePtr = std::shared_ptr<const IRValueType>;

    virtual ~IRValue();

    /** Copy this value to the given buffer. */
    virtual void CopyTo(void* dest, uint64_t destSize) const = 0;

    /** Get this value's type. */
    const IRValueType& GetType() const;

    /** Get this value's type as a pointer. */
    const ConstIRValueTypePtr& GetTypePtr() const;

    /** Convenience method to get this value as a bool. Throws std::bad_cast if not applicable. */
    virtual bool AsBool() const;

    /** Convenience method to get this value as a float. Throws std::bad_cast if not applicable. */
    virtual float AsFloat32() const;

    /** Convenience method to get this value as an int. Throws std::bad_cast if not applicable. */
    virtual int64_t AsInt64() const;

    /** Convenience method to get this value as a string. Throws std::bad_cast if not applicable. */
    virtual std::string AsString() const;

protected:
    IRValue(ConstIRValueTypePtr type);

private:
    ConstIRValueTypePtr m_type;
};

//-----------------------------------------------------------------

class IRImmediateValue : public IRValue {
public:
    virtual ~IRImmediateValue();

protected:
    IRImmediateValue(std::shared_ptr<const IRValueType> type);
};

//-----------------------------------------------------------------

template<typename T>
class IRImmediateScalarValue : public IRImmediateValue {
public:
    using ConstIRScalarValueTypePtr = std::shared_ptr<const IRScalarValueType>;

    ~IRImmediateScalarValue();
    IRImmediateScalarValue(ConstIRScalarValueTypePtr type,
                           T value);

    T GetValue() const;

    void CopyTo(void* dest, uint64_t destSize) const override;
    bool AsBool() const override;
    float AsFloat32() const override;
    int64_t AsInt64() const override;
    std::string AsString() const override;

private:
    T m_value;
};

//-----------------------------------------------------------------

template<typename T>
class IRImmediateTensorValue : public IRImmediateValue {
public:
    using ConstIRTensorValueTypePtr = std::shared_ptr<const IRTensorValueType>;

    ~IRImmediateTensorValue();

    const std::vector<T>& GetValues() const;

    IRImmediateTensorValue(ConstIRTensorValueTypePtr type,
                           std::vector<T>&& values);

    void CopyTo(void* dest, uint64_t destSize) const override;

private:
    std::vector<T> m_values;
};

//-----------------------------------------------------------------

class IRImmediateTupleValue : public IRImmediateValue {
public:
    using ConstIRTupleValueTypePtr = std::shared_ptr<const IRTupleValueType>;
    using ConstIRValueVec = std::vector<std::shared_ptr<const IRValue>>;

    ~IRImmediateTupleValue();

    IRImmediateTupleValue(ConstIRTupleValueTypePtr type,
                          ConstIRValueVec&& values);

    const ConstIRValueVec& GetValues() const;

    void CopyTo(void* dest, uint64_t destSize) const override;

private:
    ConstIRValueVec m_values;
};

//-----------------------------------------------------------------
class IRFileValue : public IRValue {
public:
    ~IRFileValue();

    IRFileValue(ConstIRValueTypePtr type, const std::string& path, uint64_t offset);

    /** Get the path to the file holding the data.*/
    const std::string& GetPath() const;

    /** Get the offset of the start of the data within the file. */
    uint64_t GetOffset() const;

    void CopyTo(void* dest, uint64_t destSize) const override;
private:
    std::string m_path;
    uint64_t m_offset;
};
}
}
