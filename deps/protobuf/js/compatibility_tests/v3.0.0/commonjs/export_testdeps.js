/**
 * @fileoverview Export symbols needed by tests in CommonJS style.
 *
 * This file is like export.js, but for symbols that are only used by tests.
 * However we exclude assert functions here, because they are exported into
 * the global namespace, so those are handled as a special case in
 * export_asserts.js.
 */

goog.require('goog.crypt.base64');
goog.require('jspb.arith.Int64');
goog.require('jspb.arith.UInt64');
goog.require('jspb.BinaryEncoder');
goog.require('jspb.BinaryDecoder');
goog.require('jspb.utils');

exports.goog = goog;
exports.jspb = jspb;
