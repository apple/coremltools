/**
 * @fileoverview Handwritten code of TestAllTypesProto3.
 */
goog.module('proto.conformance.TestAllTypesProto3');

const InternalMessage = goog.require('protobuf.binary.InternalMessage');
const Kernel = goog.require('protobuf.runtime.Kernel');

/**
 * Handwritten code of conformance.TestAllTypesProto3.
 * Check google/protobuf/test_messages_proto3.proto for more details.
 * @implements {InternalMessage}
 * @final
 */
class TestAllTypesProto3 {
  /**
   * @param {!Kernel=} accessor
   * @private
   */
  constructor(accessor = Kernel.createEmpty()) {
    /** @private @const {!Kernel} */
    this.accessor_ = accessor;
  }

  /**
   * @override
   * @package
   * @return {!Kernel}
   */
  internalGetKernel() {
    return this.accessor_;
  }

  /**
   * Create a request instance with the given bytes data.
   * If we directly use the accessor created by the binary decoding, the
   * Kernel instance will only copy the same data over for encoding.  By
   * explicitly fetching data from the previous accessor and setting all fields
   * into a new accessor, we will actually test encoding/decoding for the binary
   * format.
   * @param {!ArrayBuffer} bytes
   * @return {!TestAllTypesProto3}
   */
  static deserialize(bytes) {
    const msg = new TestAllTypesProto3();
    const requestAccessor = Kernel.fromArrayBuffer(bytes);

    if (requestAccessor.hasFieldNumber(1)) {
      const value = requestAccessor.getInt32WithDefault(1);
      msg.accessor_.setInt32(1, value);
    }

    if (requestAccessor.hasFieldNumber(2)) {
      const value = requestAccessor.getInt64WithDefault(2);
      msg.accessor_.setInt64(2, value);
    }

    if (requestAccessor.hasFieldNumber(3)) {
      const value = requestAccessor.getUint32WithDefault(3);
      msg.accessor_.setUint32(3, value);
    }

    if (requestAccessor.hasFieldNumber(4)) {
      const value = requestAccessor.getUint64WithDefault(4);
      msg.accessor_.setUint64(4, value);
    }

    if (requestAccessor.hasFieldNumber(5)) {
      const value = requestAccessor.getSint32WithDefault(5);
      msg.accessor_.setSint32(5, value);
    }

    if (requestAccessor.hasFieldNumber(6)) {
      const value = requestAccessor.getSint64WithDefault(6);
      msg.accessor_.setSint64(6, value);
    }

    if (requestAccessor.hasFieldNumber(7)) {
      const value = requestAccessor.getFixed32WithDefault(7);
      msg.accessor_.setFixed32(7, value);
    }

    if (requestAccessor.hasFieldNumber(8)) {
      const value = requestAccessor.getFixed64WithDefault(8);
      msg.accessor_.setFixed64(8, value);
    }

    if (requestAccessor.hasFieldNumber(9)) {
      const value = requestAccessor.getSfixed32WithDefault(9);
      msg.accessor_.setSfixed32(9, value);
    }

    if (requestAccessor.hasFieldNumber(10)) {
      const value = requestAccessor.getSfixed64WithDefault(10);
      msg.accessor_.setSfixed64(10, value);
    }

    if (requestAccessor.hasFieldNumber(11)) {
      const value = requestAccessor.getFloatWithDefault(11);
      msg.accessor_.setFloat(11, value);
    }

    if (requestAccessor.hasFieldNumber(12)) {
      const value = requestAccessor.getDoubleWithDefault(12);
      msg.accessor_.setDouble(12, value);
    }

    if (requestAccessor.hasFieldNumber(13)) {
      const value = requestAccessor.getBoolWithDefault(13);
      msg.accessor_.setBool(13, value);
    }

    if (requestAccessor.hasFieldNumber(14)) {
      const value = requestAccessor.getStringWithDefault(14);
      msg.accessor_.setString(14, value);
    }

    if (requestAccessor.hasFieldNumber(15)) {
      const value = requestAccessor.getBytesWithDefault(15);
      msg.accessor_.setBytes(15, value);
    }

    if (requestAccessor.hasFieldNumber(18)) {
      const value = requestAccessor.getMessage(
          18, (accessor) => new TestAllTypesProto3(accessor));
      msg.accessor_.setMessage(18, value);
    }

    if (requestAccessor.hasFieldNumber(21)) {
      // Unknown enum is not checked here, because even if an enum is unknown,
      // it should be kept during encoding. For the purpose of wire format test,
      // we can simplify the implementation by treating it as an int32 field,
      // which has the same semantic except for the unknown value checking.
      const value = requestAccessor.getInt32WithDefault(21);
      msg.accessor_.setInt32(21, value);
    }

    if (requestAccessor.hasFieldNumber(31)) {
      const value = requestAccessor.getRepeatedInt32Iterable(31);
      msg.accessor_.setPackedInt32Iterable(31, value);
    }

    if (requestAccessor.hasFieldNumber(32)) {
      const value = requestAccessor.getRepeatedInt64Iterable(32);
      msg.accessor_.setPackedInt64Iterable(32, value);
    }

    if (requestAccessor.hasFieldNumber(33)) {
      const value = requestAccessor.getRepeatedUint32Iterable(33);
      msg.accessor_.setPackedUint32Iterable(33, value);
    }

    if (requestAccessor.hasFieldNumber(34)) {
      const value = requestAccessor.getRepeatedUint64Iterable(34);
      msg.accessor_.setPackedUint64Iterable(34, value);
    }

    if (requestAccessor.hasFieldNumber(35)) {
      const value = requestAccessor.getRepeatedSint32Iterable(35);
      msg.accessor_.setPackedSint32Iterable(35, value);
    }

    if (requestAccessor.hasFieldNumber(36)) {
      const value = requestAccessor.getRepeatedSint64Iterable(36);
      msg.accessor_.setPackedSint64Iterable(36, value);
    }

    if (requestAccessor.hasFieldNumber(37)) {
      const value = requestAccessor.getRepeatedFixed32Iterable(37);
      msg.accessor_.setPackedFixed32Iterable(37, value);
    }

    if (requestAccessor.hasFieldNumber(38)) {
      const value = requestAccessor.getRepeatedFixed64Iterable(38);
      msg.accessor_.setPackedFixed64Iterable(38, value);
    }

    if (requestAccessor.hasFieldNumber(39)) {
      const value = requestAccessor.getRepeatedSfixed32Iterable(39);
      msg.accessor_.setPackedSfixed32Iterable(39, value);
    }

    if (requestAccessor.hasFieldNumber(40)) {
      const value = requestAccessor.getRepeatedSfixed64Iterable(40);
      msg.accessor_.setPackedSfixed64Iterable(40, value);
    }

    if (requestAccessor.hasFieldNumber(41)) {
      const value = requestAccessor.getRepeatedFloatIterable(41);
      msg.accessor_.setPackedFloatIterable(41, value);
    }

    if (requestAccessor.hasFieldNumber(42)) {
      const value = requestAccessor.getRepeatedDoubleIterable(42);
      msg.accessor_.setPackedDoubleIterable(42, value);
    }

    if (requestAccessor.hasFieldNumber(43)) {
      const value = requestAccessor.getRepeatedBoolIterable(43);
      msg.accessor_.setPackedBoolIterable(43, value);
    }

    if (requestAccessor.hasFieldNumber(44)) {
      const value = requestAccessor.getRepeatedStringIterable(44);
      msg.accessor_.setRepeatedStringIterable(44, value);
    }

    if (requestAccessor.hasFieldNumber(45)) {
      const value = requestAccessor.getRepeatedBytesIterable(45);
      msg.accessor_.setRepeatedBytesIterable(45, value);
    }

    if (requestAccessor.hasFieldNumber(48)) {
      const value = requestAccessor.getRepeatedMessageIterable(
          48, (accessor) => new TestAllTypesProto3(accessor));
      msg.accessor_.setRepeatedMessageIterable(48, value);
    }

    if (requestAccessor.hasFieldNumber(51)) {
      // Unknown enum is not checked here, because even if an enum is unknown,
      // it should be kept during encoding. For the purpose of wire format test,
      // we can simplify the implementation by treating it as an int32 field,
      // which has the same semantic except for the unknown value checking.
      const value = requestAccessor.getRepeatedInt32Iterable(51);
      msg.accessor_.setPackedInt32Iterable(51, value);
    }

    if (requestAccessor.hasFieldNumber(89)) {
      const value = requestAccessor.getRepeatedInt32Iterable(89);
      msg.accessor_.setUnpackedInt32Iterable(89, value);
    }

    if (requestAccessor.hasFieldNumber(90)) {
      const value = requestAccessor.getRepeatedInt64Iterable(90);
      msg.accessor_.setUnpackedInt64Iterable(90, value);
    }

    if (requestAccessor.hasFieldNumber(91)) {
      const value = requestAccessor.getRepeatedUint32Iterable(91);
      msg.accessor_.setUnpackedUint32Iterable(91, value);
    }

    if (requestAccessor.hasFieldNumber(92)) {
      const value = requestAccessor.getRepeatedUint64Iterable(92);
      msg.accessor_.setUnpackedUint64Iterable(92, value);
    }

    if (requestAccessor.hasFieldNumber(93)) {
      const value = requestAccessor.getRepeatedSint32Iterable(93);
      msg.accessor_.setUnpackedSint32Iterable(93, value);
    }

    if (requestAccessor.hasFieldNumber(94)) {
      const value = requestAccessor.getRepeatedSint64Iterable(94);
      msg.accessor_.setUnpackedSint64Iterable(94, value);
    }

    if (requestAccessor.hasFieldNumber(95)) {
      const value = requestAccessor.getRepeatedFixed32Iterable(95);
      msg.accessor_.setUnpackedFixed32Iterable(95, value);
    }

    if (requestAccessor.hasFieldNumber(96)) {
      const value = requestAccessor.getRepeatedFixed64Iterable(96);
      msg.accessor_.setUnpackedFixed64Iterable(96, value);
    }

    if (requestAccessor.hasFieldNumber(97)) {
      const value = requestAccessor.getRepeatedSfixed32Iterable(97);
      msg.accessor_.setUnpackedSfixed32Iterable(97, value);
    }

    if (requestAccessor.hasFieldNumber(98)) {
      const value = requestAccessor.getRepeatedSfixed64Iterable(98);
      msg.accessor_.setUnpackedSfixed64Iterable(98, value);
    }

    if (requestAccessor.hasFieldNumber(99)) {
      const value = requestAccessor.getRepeatedFloatIterable(99);
      msg.accessor_.setUnpackedFloatIterable(99, value);
    }

    if (requestAccessor.hasFieldNumber(100)) {
      const value = requestAccessor.getRepeatedDoubleIterable(100);
      msg.accessor_.setUnpackedDoubleIterable(100, value);
    }

    if (requestAccessor.hasFieldNumber(101)) {
      const value = requestAccessor.getRepeatedBoolIterable(101);
      msg.accessor_.setUnpackedBoolIterable(101, value);
    }

    if (requestAccessor.hasFieldNumber(102)) {
      const value = requestAccessor.getRepeatedInt32Iterable(102);
      msg.accessor_.setUnpackedInt32Iterable(102, value);
    }

    return msg;
  }

  /**
   * Serializes into binary data.
   * @return {!ArrayBuffer}
   */
  serialize() {
    return this.accessor_.serialize();
  }
}

exports = TestAllTypesProto3;
