
//
//  IRValueType.hpp
//  CoreML
//
//  Created by Jeff Kilpatrick on 11/8/19.
//  Copyright © 2019 Apple Inc. All rights reserved.
//

#pragma once

#include <memory>
#include <string>
#include <vector>

namespace CoreML {
namespace ILIL {

class IRValueType;

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

    /** Convenience method to get this value as an int32. Throws std::bad_cast if not applicable. */
    virtual int32_t AsInt32() const;

    /** Convenience method to get this value as an int64. Throws std::bad_cast if not applicable. */
    virtual int64_t AsInt64() const;

    /** Convenience method to get this value as a string. Throws std::bad_cast if not applicable. */
    virtual std::string AsString() const;

protected:
    IRValue(ConstIRValueTypePtr type);

private:
    ConstIRValueTypePtr m_type;
};

//-----------------------------------------------------------------

template<typename T>
class IRScalarValue : public IRValue {
public:
    using ConstIRScalarValueTypePtr = std::shared_ptr<const class IRScalarValueType>;

    ~IRScalarValue();
    static std::unique_ptr<IRScalarValue<T>> Make(T value);

    T GetValue() const;

    void CopyTo(void* dest, uint64_t destSize) const override;
    bool AsBool() const override;
    float AsFloat32() const override;
    int32_t AsInt32() const override;
    int64_t AsInt64() const override;
    std::string AsString() const override;

private:
    IRScalarValue(ConstIRScalarValueTypePtr type, T value);

    T m_value;

    friend class IRScalarValueType;
};

//-----------------------------------------------------------------

template<typename T>
class IRTensorValue : public IRValue {
public:
    using ConstIRTensorValueTypePtr = std::shared_ptr<const class IRTensorValueType>;
    using ValueVec = std::vector<T>;

    ~IRTensorValue();

    const std::vector<T>& GetValues() const;

    void CopyTo(void* dest, uint64_t destSize) const override;

private:
    IRTensorValue(ConstIRTensorValueTypePtr type, ValueVec&& values);

    ValueVec m_values;

    friend class IRTensorValueType;
};

//-----------------------------------------------------------------

class IRTupleValue : public IRValue {
public:
    using ConstIRTupleValueTypePtr = std::shared_ptr<const class IRTupleValueType>;
    using ConstIRValueVec = std::vector<std::shared_ptr<const IRValue>>;

    ~IRTupleValue();

    const ConstIRValueVec& GetValues() const;

    void CopyTo(void* dest, uint64_t destSize) const override;

private:
    IRTupleValue(ConstIRTupleValueTypePtr type, ConstIRValueVec&& values);

    ConstIRValueVec m_values;

    friend class IRTupleValueType;
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
    
    /** Assumes the file value is a scalar and returns that scalar.
        The value will be read from disk and is not cached. */
    bool AsBool() const override;
    float AsFloat32() const override;
    int32_t AsInt32() const override;
    int64_t AsInt64() const override;
    std::string AsString() const override;
    
    /** Reads this value from disk on each call. */
    std::unique_ptr<const IRValue> GetValue() const;

    void CopyTo(void* dest, uint64_t destSize) const override;
private:
    std::string m_path;
    uint64_t m_offset;
};
}
}
