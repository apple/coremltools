// Protocol Buffers - Google's data interchange format
// Copyright 2014 Google Inc.  All rights reserved.
// https://developers.google.com/protocol-buffers/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "message.h"

#include <inttypes.h>
#include <php.h>
#include <stdlib.h>

// This is not self-contained: it must be after other Zend includes.
#include <Zend/zend_exceptions.h>
#include <Zend/zend_inheritance.h>

#include "arena.h"
#include "array.h"
#include "convert.h"
#include "def.h"
#include "map.h"
#include "php-upb.h"
#include "protobuf.h"

// -----------------------------------------------------------------------------
// Message
// -----------------------------------------------------------------------------

typedef struct {
  zend_object std;
  zval arena;
  const Descriptor* desc;
  upb_msg *msg;
} Message;

zend_class_entry *message_ce;
static zend_object_handlers message_object_handlers;

static void Message_SuppressDefaultProperties(zend_class_entry *class_type) {
  // We suppress all default properties, because all our properties are handled
  // by our read_property handler.
  //
  // This also allows us to put our zend_object member at the beginning of our
  // struct -- instead of putting it at the end with pointer fixups to access
  // our own data, as recommended in the docs -- because Zend won't add any of
  // its own storage directly after the zend_object if default_properties_count
  // == 0.
  //
  // This is not officially supported, but since it simplifies the code, we'll
  // do it for as long as it works in practice.
  class_type->default_properties_count = 0;
}

// PHP Object Handlers /////////////////////////////////////////////////////////

/**
 * Message_create()
 *
 * PHP class entry function to allocate and initialize a new Message object.
 */
static zend_object* Message_create(zend_class_entry *class_type) {
  Message *intern = emalloc(sizeof(Message));
  Message_SuppressDefaultProperties(class_type);
  zend_object_std_init(&intern->std, class_type);
  intern->std.handlers = &message_object_handlers;
  Arena_Init(&intern->arena);
  return &intern->std;
}

/**
 * Message_dtor()
 *
 * Object handler to destroy a Message. This releases all resources associated
 * with the message. Note that it is possible to access a destroyed object from
 * PHP in rare cases.
 */
static void Message_dtor(zend_object* obj) {
  Message* intern = (Message*)obj;
  ObjCache_Delete(intern->msg);
  zval_dtor(&intern->arena);
  zend_object_std_dtor(&intern->std);
}

/**
 * get_field()
 *
 * Helper function to look up a field given a member name (as a string).
 */
static const upb_fielddef *get_field(Message *msg, PROTO_STR *member) {
  const upb_msgdef *m = msg->desc->msgdef;
  const upb_fielddef *f =
      upb_msgdef_ntof(m, PROTO_STRVAL_P(member), PROTO_STRLEN_P(member));

  if (!f) {
    zend_throw_exception_ex(NULL, 0, "No such property %s.",
                            ZSTR_VAL(msg->desc->class_entry->name));
  }

  return f;
}

static void Message_get(Message *intern, const upb_fielddef *f, zval *rv) {
  upb_arena *arena = Arena_Get(&intern->arena);

  if (upb_fielddef_ismap(f)) {
    upb_mutmsgval msgval = upb_msg_mutable(intern->msg, f, arena);
    MapField_GetPhpWrapper(rv, msgval.map, MapType_Get(f), &intern->arena);
  } else if (upb_fielddef_isseq(f)) {
    upb_mutmsgval msgval = upb_msg_mutable(intern->msg, f, arena);
    RepeatedField_GetPhpWrapper(rv, msgval.array, TypeInfo_Get(f),
                                &intern->arena);
  } else {
    if (upb_fielddef_issubmsg(f) && !upb_msg_has(intern->msg, f)) {
      ZVAL_NULL(rv);
      return;
    }
    upb_msgval msgval = upb_msg_get(intern->msg, f);
    Convert_UpbToPhp(msgval, rv, TypeInfo_Get(f), &intern->arena);
  }
}

static bool Message_set(Message *intern, const upb_fielddef *f, zval *val) {
  upb_arena *arena = Arena_Get(&intern->arena);
  upb_msgval msgval;

  if (upb_fielddef_ismap(f)) {
    msgval.map_val = MapField_GetUpbMap(val, MapType_Get(f), arena);
    if (!msgval.map_val) return false;
  } else if (upb_fielddef_isseq(f)) {
    msgval.array_val = RepeatedField_GetUpbArray(val, TypeInfo_Get(f), arena);
    if (!msgval.array_val) return false;
  } else if (upb_fielddef_issubmsg(f) && Z_TYPE_P(val) == IS_NULL) {
    upb_msg_clearfield(intern->msg, f);
    return true;
  } else {
    if (!Convert_PhpToUpb(val, &msgval, TypeInfo_Get(f), arena)) return false;
  }

  upb_msg_set(intern->msg, f, msgval, arena);
  return true;
}

static bool MessageEq(const upb_msg *m1, const upb_msg *m2, const upb_msgdef *m);

/**
 * ValueEq()
 */
bool ValueEq(upb_msgval val1, upb_msgval val2, TypeInfo type) {
  switch (type.type) {
    case UPB_TYPE_BOOL:
      return val1.bool_val == val2.bool_val;
    case UPB_TYPE_INT32:
    case UPB_TYPE_UINT32:
    case UPB_TYPE_ENUM:
      return val1.int32_val == val2.int32_val;
    case UPB_TYPE_INT64:
    case UPB_TYPE_UINT64:
      return val1.int64_val == val2.int64_val;
    case UPB_TYPE_FLOAT:
      return val1.float_val == val2.float_val;
    case UPB_TYPE_DOUBLE:
      return val1.double_val == val2.double_val;
    case UPB_TYPE_STRING:
    case UPB_TYPE_BYTES:
      return val1.str_val.size == val2.str_val.size &&
          memcmp(val1.str_val.data, val2.str_val.data, val1.str_val.size) == 0;
    case UPB_TYPE_MESSAGE:
      return MessageEq(val1.msg_val, val2.msg_val, type.desc->msgdef);
    default:
      return false;
  }
}

/**
 * MessageEq()
 */
static bool MessageEq(const upb_msg *m1, const upb_msg *m2, const upb_msgdef *m) {
  upb_msg_field_iter i;

  for(upb_msg_field_begin(&i, m);
      !upb_msg_field_done(&i);
      upb_msg_field_next(&i)) {
    const upb_fielddef *f = upb_msg_iter_field(&i);

    if (upb_fielddef_haspresence(f)) {
      if (upb_msg_has(m1, f) != upb_msg_has(m2, f)) {
        return false;
      }
      if (!upb_msg_has(m1, f)) continue;
    }

    upb_msgval val1 = upb_msg_get(m1, f);
    upb_msgval val2 = upb_msg_get(m2, f);

    if (upb_fielddef_ismap(f)) {
      if (!MapEq(val1.map_val, val2.map_val, MapType_Get(f))) return false;
    } else if (upb_fielddef_isseq(f)) {
      if (!ArrayEq(val1.array_val, val2.array_val, TypeInfo_Get(f))) return false;
    } else {
      if (!ValueEq(val1, val2, TypeInfo_Get(f))) return false;
    }
  }

  return true;
}

/**
 * Message_compare_objects()
 *
 * Object handler for comparing two message objects. Called whenever PHP code
 * does:
 *
 *   $m1 == $m2
 */
static int Message_compare_objects(zval *m1, zval *m2) {
  Message* intern1 = (Message*)Z_OBJ_P(m1);
  Message* intern2 = (Message*)Z_OBJ_P(m2);
  const upb_msgdef *m = intern1->desc->msgdef;

  if (intern2->desc->msgdef != m) return 1;

  return MessageEq(intern1->msg, intern2->msg, m) ? 0 : 1;
}

/**
 * Message_has_property()
 *
 * Object handler for testing whether a property exists. Called when PHP code
 * does any of:
 *
 *   isset($message->foobar);
 *   property_exists($message->foobar);
 *
 * Note that all properties of generated messages are private, so this should
 * only be possible to invoke from generated code, which has accessors like this
 * (if the field has presence):
 *
 *   public function hasOptionalInt32()
 *   {
 *       return isset($this->optional_int32);
 *   }
 */
static int Message_has_property(PROTO_VAL *obj, PROTO_STR *member,
                                int has_set_exists,
                                void **cache_slot) {
  Message* intern = PROTO_VAL_P(obj);
  const upb_fielddef *f = get_field(intern, member);

  if (!f) return 0;

  if (!upb_fielddef_haspresence(f)) {
    zend_throw_exception_ex(
        NULL, 0,
        "Cannot call isset() on field %s which does not have presence.",
        upb_fielddef_name(f));
    return 0;
  }

  return upb_msg_has(intern->msg, f);
}

/**
 * Message_unset_property()
 *
 * Object handler for unsetting a property. Called when PHP code calls:
 *
 *   unset($message->foobar);
 *
 * Note that all properties of generated messages are private, so this should
 * only be possible to invoke from generated code, which has accessors like this
 * (if the field has presence):
 *
 *   public function clearOptionalInt32()
 *   {
 *       unset($this->optional_int32);
 *   }
 */
static void Message_unset_property(PROTO_VAL *obj, PROTO_STR *member,
                                   void **cache_slot) {
  Message* intern = PROTO_VAL_P(obj);
  const upb_fielddef *f = get_field(intern, member);

  if (!f) return;

  if (!upb_fielddef_haspresence(f)) {
    zend_throw_exception_ex(
        NULL, 0,
        "Cannot call unset() on field %s which does not have presence.",
        upb_fielddef_name(f));
    return;
  }

  upb_msg_clearfield(intern->msg, f);
}


/**
 * Message_read_property()
 *
 * Object handler for reading a property in PHP. Called when PHP code does:
 *
 *   $x = $message->foobar;
 *
 * Note that all properties of generated messages are private, so this should
 * only be possible to invoke from generated code, which has accessors like:
 *
 *   public function getOptionalInt32()
 *   {
 *       return $this->optional_int32;
 *   }
 *
 * We lookup the field and return the scalar, RepeatedField, or MapField for
 * this field.
 */
static zval *Message_read_property(PROTO_VAL *obj, PROTO_STR *member,
                                   int type, void **cache_slot, zval *rv) {
  Message* intern = PROTO_VAL_P(obj);
  const upb_fielddef *f = get_field(intern, member);

  if (!f) return &EG(uninitialized_zval);
  Message_get(intern, f, rv);
  return rv;
}

/**
 * Message_write_property()
 *
 * Object handler for writing a property in PHP. Called when PHP code does:
 *
 *   $message->foobar = $x;
 *
 * Note that all properties of generated messages are private, so this should
 * only be possible to invoke from generated code, which has accessors like:
 *
 *   public function setOptionalInt32($var)
 *   {
 *       GPBUtil::checkInt32($var);
 *       $this->optional_int32 = $var;
 *
 *       return $this;
 *   }
 *
 * The C extension version of checkInt32() doesn't actually check anything, so
 * we perform all checking and conversion in this function.
 */
static PROTO_RETURN_VAL Message_write_property(
    PROTO_VAL *obj, PROTO_STR *member, zval *val, void **cache_slot) {
  Message* intern = PROTO_VAL_P(obj);
  const upb_fielddef *f = get_field(intern, member);

  if (f && Message_set(intern, f, val)) {
#if PHP_VERSION_ID < 70400
    return;
#else
    return val;
#endif
  } else {
#if PHP_VERSION_ID < 70400
    return;
#else
    return &EG(error_zval);
#endif
  }
}

/**
 * Message_get_property_ptr_ptr()
 *
 * Object handler for the get_property_ptr_ptr event in PHP. This returns a
 * reference to our internal properties. We don't support this, so we return
 * NULL.
 */
static zval *Message_get_property_ptr_ptr(PROTO_VAL *object, PROTO_STR *member,
                                          int type,
                                          void **cache_slot) {
  return NULL;  // We do not have a properties table.
}

/**
 * Message_clone_obj()
 *
 * Object handler for cloning an object in PHP. Called when PHP code does:
 *
 *   $msg2 = clone $msg;
 */
static zend_object *Message_clone_obj(PROTO_VAL *object) {
  Message* intern = PROTO_VAL_P(object);
  upb_msg *clone = upb_msg_new(intern->desc->msgdef, Arena_Get(&intern->arena));

  // TODO: copy unknown fields?
  // TODO: use official upb msg copy function
  memcpy(clone, intern->msg, upb_msgdef_layout(intern->desc->msgdef)->size);
  zval ret;
  Message_GetPhpWrapper(&ret, intern->desc, clone, &intern->arena);
  return Z_OBJ_P(&ret);
}

/**
 * Message_get_properties()
 *
 * Object handler for the get_properties event in PHP. This returns a HashTable
 * of our internal properties. We don't support this, so we return NULL.
 */
static HashTable *Message_get_properties(PROTO_VAL *object) {
  return NULL;  // We don't offer direct references to our properties.
}

// C Functions from message.h. /////////////////////////////////////////////////

// These are documented in the header file.

void Message_GetPhpWrapper(zval *val, const Descriptor *desc, upb_msg *msg,
                           zval *arena) {
  if (!msg) {
    ZVAL_NULL(val);
    return;
  }

  if (!ObjCache_Get(msg, val)) {
    Message *intern = emalloc(sizeof(Message));
    Message_SuppressDefaultProperties(desc->class_entry);
    zend_object_std_init(&intern->std, desc->class_entry);
    intern->std.handlers = &message_object_handlers;
    ZVAL_COPY(&intern->arena, arena);
    intern->desc = desc;
    intern->msg = msg;
    ZVAL_OBJ(val, &intern->std);
    ObjCache_Add(intern->msg, &intern->std);
  }
}

bool Message_GetUpbMessage(zval *val, const Descriptor *desc, upb_arena *arena,
                           upb_msg **msg) {
  PBPHP_ASSERT(desc);

  if (Z_ISREF_P(val)) {
    ZVAL_DEREF(val);
  }

  if (Z_TYPE_P(val) == IS_OBJECT &&
      instanceof_function(Z_OBJCE_P(val), desc->class_entry)) {
    Message *intern = (Message*)Z_OBJ_P(val);
    upb_arena_fuse(arena, Arena_Get(&intern->arena));
    *msg = intern->msg;
    return true;
  } else {
    zend_throw_exception_ex(zend_ce_type_error, 0,
                            "Given value is not an instance of %s.",
                            ZSTR_VAL(desc->class_entry->name));
    return false;
  }
}

// Message PHP methods /////////////////////////////////////////////////////////

/**
 * Message_InitFromPhp()
 *
 * Helper method to handle the initialization of a message from a PHP value, eg.
 *
 *   $m = new TestMessage([
 *       'optional_int32' => -42,
 *       'optional_bool' => true,
 *       'optional_string' => 'a',
 *       'optional_enum' => TestEnum::ONE,
 *       'optional_message' => new Sub([
 *           'a' => 33
 *       ]),
 *       'repeated_int32' => [-42, -52],
 *       'repeated_enum' => [TestEnum::ZERO, TestEnum::ONE],
 *       'repeated_message' => [new Sub(['a' => 34]),
 *                              new Sub(['a' => 35])],
 *       'map_int32_int32' => [-62 => -62],
 *       'map_int32_enum' => [1 => TestEnum::ONE],
 *       'map_int32_message' => [1 => new Sub(['a' => 36])],
 *   ]);
 *
 * The initializer must be an array.
 */
bool Message_InitFromPhp(upb_msg *msg, const upb_msgdef *m, zval *init,
                         upb_arena *arena) {
  HashTable* table = HASH_OF(init);
  HashPosition pos;

  if (Z_ISREF_P(init)) {
    ZVAL_DEREF(init);
  }

  if (Z_TYPE_P(init) != IS_ARRAY) {
    zend_throw_exception_ex(NULL, 0,
                            "Initializer for a message %s must be an array.",
                            upb_msgdef_fullname(m));
    return false;
  }

  zend_hash_internal_pointer_reset_ex(table, &pos);

  while (true) {  // Iterate over key/value pairs.
    zval key;
    zval *val;
    const upb_fielddef *f;
    upb_msgval msgval;

    zend_hash_get_current_key_zval_ex(table, &key, &pos);
    val = zend_hash_get_current_data_ex(table, &pos);

    if (!val) return true;  // Finished iteration.

    if (Z_ISREF_P(val)) {
      ZVAL_DEREF(val);
    }

    f = upb_msgdef_ntof(m, Z_STRVAL_P(&key), Z_STRLEN_P(&key));

    if (!f) {
      zend_throw_exception_ex(NULL, 0,
                              "No such field %s", Z_STRVAL_P(&key));
      return false;
    }

    if (upb_fielddef_ismap(f)) {
      msgval.map_val = MapField_GetUpbMap(val, MapType_Get(f), arena);
      if (!msgval.map_val) return false;
    } else if (upb_fielddef_isseq(f)) {
      msgval.array_val = RepeatedField_GetUpbArray(val, TypeInfo_Get(f), arena);
      if (!msgval.array_val) return false;
    } else {
      if (!Convert_PhpToUpbAutoWrap(val, &msgval, TypeInfo_Get(f), arena)) {
        return false;
      }
    }

    upb_msg_set(msg, f, msgval, arena);
    zend_hash_move_forward_ex(table, &pos);
    zval_dtor(&key);
  }
}

static void Message_Initialize(Message *intern, const Descriptor *desc) {
  intern->desc = desc;
  intern->msg = upb_msg_new(desc->msgdef, Arena_Get(&intern->arena));
  ObjCache_Add(intern->msg, &intern->std);
}

/**
 * Message::__construct()
 *
 * Constructor for Message.
 * @param array Map of initial values ['k' = val]
 */
PHP_METHOD(Message, __construct) {
  Message* intern = (Message*)Z_OBJ_P(getThis());
  const Descriptor* desc;
  zend_class_entry *ce = Z_OBJCE_P(getThis());
  upb_arena *arena = Arena_Get(&intern->arena);
  zval *init_arr = NULL;

  // This descriptor should always be available, as the generated __construct
  // method calls initOnce() to load the descriptor prior to calling us.
  //
  // However, if the user created their own class derived from Message, this
  // will trigger an infinite construction loop and blow the stack.  We
  // temporarily clear create_object to break this loop (see check in
  // NameMap_GetMessage()).
  PBPHP_ASSERT(ce->create_object == Message_create);
  ce->create_object = NULL;
  desc = Descriptor_GetFromClassEntry(ce);
  ce->create_object = Message_create;

  if (!desc) {
    zend_throw_exception_ex(
        NULL, 0,
        "Couldn't find descriptor. Note only generated code may derive from "
        "\\Google\\Protobuf\\Internal\\Message");
    return;
  }

  Message_Initialize(intern, desc);

  if (zend_parse_parameters(ZEND_NUM_ARGS(), "|a!", &init_arr) == FAILURE) {
    return;
  }

  if (init_arr) {
    Message_InitFromPhp(intern->msg, desc->msgdef, init_arr, arena);
  }
}

/**
 * Message::discardUnknownFields()
 *
 * Discards any unknown fields for this message or any submessages.
 */
PHP_METHOD(Message, discardUnknownFields) {
  Message* intern = (Message*)Z_OBJ_P(getThis());
  upb_msg_discardunknown(intern->msg, intern->desc->msgdef, 64);
}

/**
 * Message::clear()
 *
 * Clears all fields of this message.
 */
PHP_METHOD(Message, clear) {
  Message* intern = (Message*)Z_OBJ_P(getThis());
  upb_msg_clear(intern->msg, intern->desc->msgdef);
}

/**
 * Message::mergeFrom()
 *
 * Merges from the given message, which must be of the same class as us.
 * @param object Message to merge from.
 */
PHP_METHOD(Message, mergeFrom) {
  Message* intern = (Message*)Z_OBJ_P(getThis());
  Message* from;
  upb_arena *arena = Arena_Get(&intern->arena);
  const upb_msglayout *l = upb_msgdef_layout(intern->desc->msgdef);
  zval* value;
  char *pb;
  size_t size;
  bool ok;

  if (zend_parse_parameters(ZEND_NUM_ARGS(), "O", &value,
                            intern->desc->class_entry) == FAILURE) {
    return;
  }

  from = (Message*)Z_OBJ_P(value);

  // Should be guaranteed since we passed the class type to
  // zend_parse_parameters().
  PBPHP_ASSERT(from->desc == intern->desc);

  // TODO(haberman): use a temp arena for this once we can make upb_decode()
  // copy strings.
  pb = upb_encode(from->msg, l, arena, &size);

  if (!pb) {
    zend_throw_exception_ex(NULL, 0, "Max nesting exceeded");
    return;
  }

  ok = upb_decode(pb, size, intern->msg, l, arena);
  PBPHP_ASSERT(ok);
}

/**
 * Message::mergeFromString()
 *
 * Merges from the given string.
 * @param string Binary protobuf data to merge.
 */
PHP_METHOD(Message, mergeFromString) {
  Message* intern = (Message*)Z_OBJ_P(getThis());
  char *data = NULL;
  char *data_copy = NULL;
  zend_long data_len;
  const upb_msglayout *l = upb_msgdef_layout(intern->desc->msgdef);
  upb_arena *arena = Arena_Get(&intern->arena);

  if (zend_parse_parameters(ZEND_NUM_ARGS(), "s", &data, &data_len) ==
      FAILURE) {
    return;
  }

  // TODO(haberman): avoid this copy when we can make the decoder copy.
  data_copy = upb_arena_malloc(arena, data_len);
  memcpy(data_copy, data, data_len);

  if (!upb_decode(data_copy, data_len, intern->msg, l, arena)) {
    zend_throw_exception_ex(NULL, 0, "Error occurred during parsing");
    return;
  }
}

/**
 * Message::serializeToString()
 *
 * Serializes this message instance to protobuf data.
 * @return string Serialized protobuf data.
 */
PHP_METHOD(Message, serializeToString) {
  Message* intern = (Message*)Z_OBJ_P(getThis());
  const upb_msglayout *l = upb_msgdef_layout(intern->desc->msgdef);
  upb_arena *tmp_arena = upb_arena_new();
  char *data;
  size_t size;

  data = upb_encode(intern->msg, l, tmp_arena, &size);

  if (!data) {
    zend_throw_exception_ex(NULL, 0, "Error occurred during serialization");
    upb_arena_free(tmp_arena);
    return;
  }

  RETVAL_STRINGL(data, size);
  upb_arena_free(tmp_arena);
}

/**
 * Message::mergeFromJsonString()
 *
 * Merges the JSON data parsed from the given string.
 * @param string Serialized JSON data.
 */
PHP_METHOD(Message, mergeFromJsonString) {
  Message* intern = (Message*)Z_OBJ_P(getThis());
  char *data = NULL;
  char *data_copy = NULL;
  zend_long data_len;
  upb_arena *arena = Arena_Get(&intern->arena);
  upb_status status;
  zend_bool ignore_json_unknown = false;
  int options = 0;

  if (zend_parse_parameters(ZEND_NUM_ARGS(), "s|b", &data, &data_len,
                            &ignore_json_unknown) == FAILURE) {
    return;
  }

  // TODO(haberman): avoid this copy when we can make the decoder copy.
  data_copy = upb_arena_malloc(arena, data_len + 1);
  memcpy(data_copy, data, data_len);
  data_copy[data_len] = '\0';

  if (ignore_json_unknown) {
    options |= UPB_JSONDEC_IGNOREUNKNOWN;
  }

  upb_status_clear(&status);
  if (!upb_json_decode(data_copy, data_len, intern->msg, intern->desc->msgdef,
                       DescriptorPool_GetSymbolTable(), options, arena,
                       &status)) {
    zend_throw_exception_ex(NULL, 0, "Error occurred during parsing: %s",
                            upb_status_errmsg(&status));
    return;
  }
}

/**
 * Message::serializeToJsonString()
 *
 * Serializes this object to JSON.
 * @return string Serialized JSON data.
 */
PHP_METHOD(Message, serializeToJsonString) {
  Message* intern = (Message*)Z_OBJ_P(getThis());
  size_t size;
  int options = 0;
  char buf[1024];
  zend_bool preserve_proto_fieldnames = false;
  upb_status status;

  if (zend_parse_parameters(ZEND_NUM_ARGS(), "|b",
                            &preserve_proto_fieldnames) == FAILURE) {
    return;
  }

  if (preserve_proto_fieldnames) {
    options |= UPB_JSONENC_PROTONAMES;
  }

  upb_status_clear(&status);
  size = upb_json_encode(intern->msg, intern->desc->msgdef,
                         DescriptorPool_GetSymbolTable(), options, buf,
                         sizeof(buf), &status);

  if (!upb_ok(&status)) {
    zend_throw_exception_ex(NULL, 0,
                            "Error occurred during JSON serialization: %s",
                            upb_status_errmsg(&status));
    return;
  }

  if (size >= sizeof(buf)) {
    char *buf2 = malloc(size + 1);
    upb_json_encode(intern->msg, intern->desc->msgdef,
                    DescriptorPool_GetSymbolTable(), options, buf2, size + 1,
                    &status);
    RETVAL_STRINGL(buf2, size);
    free(buf2);
  } else {
    RETVAL_STRINGL(buf, size);
  }
}

/**
 * Message::readWrapperValue()
 *
 * Returns an unboxed value for the given field. This is called from generated
 * methods for wrapper fields, eg.
 *
 *   public function getDoubleValueUnwrapped()
 *   {
 *       return $this->readWrapperValue("double_value");
 *   }
 *
 * @return Unwrapped field value or null.
 */
PHP_METHOD(Message, readWrapperValue) {
  Message* intern = (Message*)Z_OBJ_P(getThis());
  char* member;
  const upb_fielddef *f;
  zend_long size;

  if (zend_parse_parameters(ZEND_NUM_ARGS(), "s", &member, &size) == FAILURE) {
    return;
  }

  f = upb_msgdef_ntof(intern->desc->msgdef, member, size);

  if (!f || !upb_msgdef_iswrapper(upb_fielddef_msgsubdef(f))) {
    zend_throw_exception_ex(NULL, 0, "Message %s has no field %s",
                            upb_msgdef_fullname(intern->desc->msgdef), member);
    return;
  }

  if (upb_msg_has(intern->msg, f)) {
    const upb_msg *wrapper = upb_msg_get(intern->msg, f).msg_val;
    const upb_msgdef *m = upb_fielddef_msgsubdef(f);
    const upb_fielddef *val_f = upb_msgdef_itof(m, 1);
    upb_msgval msgval = upb_msg_get(wrapper, val_f);
    zval ret;
    Convert_UpbToPhp(msgval, &ret, TypeInfo_Get(val_f), &intern->arena);
    RETURN_COPY_VALUE(&ret);
  } else {
    RETURN_NULL();
  }
}

/**
 * Message::writeWrapperValue()
 *
 * Sets the given wrapper field to the given unboxed value. This is called from
 * generated methods for wrapper fields, eg.
 *
 *
 *   public function setDoubleValueUnwrapped($var)
 *   {
 *       $this->writeWrapperValue("double_value", $var);
 *       return $this;
 *   }
 *
 * @param Unwrapped field value or null.
 */
PHP_METHOD(Message, writeWrapperValue) {
  Message* intern = (Message*)Z_OBJ_P(getThis());
  upb_arena *arena = Arena_Get(&intern->arena);
  char* member;
  const upb_fielddef *f;
  upb_msgval msgval;
  zend_long size;
  zval* val;

  if (zend_parse_parameters(ZEND_NUM_ARGS(), "sz", &member, &size, &val) ==
      FAILURE) {
    return;
  }

  f = upb_msgdef_ntof(intern->desc->msgdef, member, size);

  if (!f || !upb_msgdef_iswrapper(upb_fielddef_msgsubdef(f))) {
    zend_throw_exception_ex(NULL, 0, "Message %s has no field %s",
                            upb_msgdef_fullname(intern->desc->msgdef), member);
    return;
  }

  if (Z_ISREF_P(val)) {
    ZVAL_DEREF(val);
  }

  if (Z_TYPE_P(val) == IS_NULL) {
    upb_msg_clearfield(intern->msg, f);
  } else {
    const upb_msgdef *m = upb_fielddef_msgsubdef(f);
    const upb_fielddef *val_f = upb_msgdef_itof(m, 1);
    upb_msg *wrapper;

    if (!Convert_PhpToUpb(val, &msgval, TypeInfo_Get(val_f), arena)) {
      return;  // Error is already set.
    }

    wrapper = upb_msg_mutable(intern->msg, f, arena).msg;
    upb_msg_set(wrapper, val_f, msgval, arena);
  }
}

/**
 * Message::whichOneof()
 *
 * Given a oneof name, returns the name of the field that is set for this oneof,
 * or otherwise the empty string.
 *
 * @return string The field name in this oneof that is currently set.
 */
PHP_METHOD(Message, whichOneof) {
  Message* intern = (Message*)Z_OBJ_P(getThis());
  const upb_oneofdef* oneof;
  const upb_fielddef* field;
  char* name;
  zend_long len;

  if (zend_parse_parameters(ZEND_NUM_ARGS(), "s", &name, &len) == FAILURE) {
    return;
  }

  oneof = upb_msgdef_ntoo(intern->desc->msgdef, name, len);

  if (!oneof) {
    zend_throw_exception_ex(NULL, 0, "Message %s has no oneof %s",
                            upb_msgdef_fullname(intern->desc->msgdef), name);
    return;
  }

  field = upb_msg_whichoneof(intern->msg, oneof);
  RETURN_STRING(field ? upb_fielddef_name(field) : "");
}

/**
 * Message::hasOneof()
 *
 * Returns the presence of the given oneof field, given a field number. Called
 * from generated code methods such as:
 *
 *    public function hasDoubleValueOneof()
 *    {
 *        return $this->hasOneof(10);
 *    }
 *
 * @return boolean
 */
PHP_METHOD(Message, hasOneof) {
  Message* intern = (Message*)Z_OBJ_P(getThis());
  zend_long field_num;
  const upb_fielddef* f;

  if (zend_parse_parameters(ZEND_NUM_ARGS(), "l", &field_num) == FAILURE) {
    return;
  }

  f = upb_msgdef_itof(intern->desc->msgdef, field_num);

  if (!f || !upb_fielddef_realcontainingoneof(f)) {
    php_error_docref(NULL, E_USER_ERROR,
                     "Internal error, no such oneof field %d\n",
                     (int)field_num);
  }

  RETVAL_BOOL(upb_msg_has(intern->msg, f));
}

/**
 * Message::readOneof()
 *
 * Returns the contents of the given oneof field, given a field number. Called
 * from generated code methods such as:
 *
 *    public function getDoubleValueOneof()
 *    {
 *        return $this->readOneof(10);
 *    }
 *
 * @return object The oneof's field value.
 */
PHP_METHOD(Message, readOneof) {
  Message* intern = (Message*)Z_OBJ_P(getThis());
  zend_long field_num;
  const upb_fielddef* f;
  zval ret;

  if (zend_parse_parameters(ZEND_NUM_ARGS(), "l", &field_num) == FAILURE) {
    return;
  }

  f = upb_msgdef_itof(intern->desc->msgdef, field_num);

  if (!f || !upb_fielddef_realcontainingoneof(f)) {
    php_error_docref(NULL, E_USER_ERROR,
                     "Internal error, no such oneof field %d\n",
                     (int)field_num);
  }

  if (upb_fielddef_issubmsg(f) && !upb_msg_has(intern->msg, f)) {
    RETURN_NULL();
  }

  {
    upb_msgval msgval = upb_msg_get(intern->msg, f);
    Convert_UpbToPhp(msgval, &ret, TypeInfo_Get(f), &intern->arena);
  }

  RETURN_COPY_VALUE(&ret);
}

/**
 * Message::writeOneof()
 *
 * Sets the contents of the given oneof field, given a field number. Called
 * from generated code methods such as:
 *
 *    public function setDoubleValueOneof($var)
 *   {
 *       GPBUtil::checkMessage($var, \Google\Protobuf\DoubleValue::class);
 *       $this->writeOneof(10, $var);
 *
 *       return $this;
 *   }
 *
 * The C extension version of GPBUtil::check*() does nothing, so we perform
 * all type checking and conversion here.
 *
 * @param integer The field number we are setting.
 * @param object The field value we want to set.
 */
PHP_METHOD(Message, writeOneof) {
  Message* intern = (Message*)Z_OBJ_P(getThis());
  zend_long field_num;
  const upb_fielddef* f;
  upb_arena *arena = Arena_Get(&intern->arena);
  upb_msgval msgval;
  zval* val;

  if (zend_parse_parameters(ZEND_NUM_ARGS(), "lz", &field_num, &val) ==
      FAILURE) {
    return;
  }

  f = upb_msgdef_itof(intern->desc->msgdef, field_num);

  if (upb_fielddef_issubmsg(f) && Z_TYPE_P(val) == IS_NULL) {
    upb_msg_clearfield(intern->msg, f);
    return;
  } else if (!Convert_PhpToUpb(val, &msgval, TypeInfo_Get(f), arena)) {
    return;
  }

  upb_msg_set(intern->msg, f, msgval, arena);
}

ZEND_BEGIN_ARG_INFO_EX(arginfo_construct, 0, 0, 0)
  ZEND_ARG_INFO(0, data)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_mergeFrom, 0, 0, 1)
  ZEND_ARG_INFO(0, data)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_mergeFromWithArg, 0, 0, 1)
  ZEND_ARG_INFO(0, data)
  ZEND_ARG_INFO(0, arg)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_read, 0, 0, 1)
  ZEND_ARG_INFO(0, field)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_write, 0, 0, 2)
  ZEND_ARG_INFO(0, field)
  ZEND_ARG_INFO(0, value)
ZEND_END_ARG_INFO()

static zend_function_entry Message_methods[] = {
  PHP_ME(Message, clear,                 arginfo_void,      ZEND_ACC_PUBLIC)
  PHP_ME(Message, discardUnknownFields,  arginfo_void,      ZEND_ACC_PUBLIC)
  PHP_ME(Message, serializeToString,     arginfo_void,      ZEND_ACC_PUBLIC)
  PHP_ME(Message, mergeFromString,       arginfo_mergeFrom, ZEND_ACC_PUBLIC)
  PHP_ME(Message, serializeToJsonString, arginfo_void,      ZEND_ACC_PUBLIC)
  PHP_ME(Message, mergeFromJsonString,   arginfo_mergeFromWithArg, ZEND_ACC_PUBLIC)
  PHP_ME(Message, mergeFrom,             arginfo_mergeFrom, ZEND_ACC_PUBLIC)
  PHP_ME(Message, readWrapperValue,      arginfo_read,      ZEND_ACC_PROTECTED)
  PHP_ME(Message, writeWrapperValue,     arginfo_write,     ZEND_ACC_PROTECTED)
  PHP_ME(Message, hasOneof,              arginfo_read,      ZEND_ACC_PROTECTED)
  PHP_ME(Message, readOneof,             arginfo_read,      ZEND_ACC_PROTECTED)
  PHP_ME(Message, writeOneof,            arginfo_write,     ZEND_ACC_PROTECTED)
  PHP_ME(Message, whichOneof,            arginfo_read,      ZEND_ACC_PROTECTED)
  PHP_ME(Message, __construct,           arginfo_construct, ZEND_ACC_PROTECTED)
  ZEND_FE_END
};

// Well-known types ////////////////////////////////////////////////////////////

static const char TYPE_URL_PREFIX[] = "type.googleapis.com/";

static upb_msgval Message_getval(Message *intern, const char *field_name) {
  const upb_fielddef *f = upb_msgdef_ntofz(intern->desc->msgdef, field_name);
  PBPHP_ASSERT(f);
  return upb_msg_get(intern->msg, f);
}

static void Message_setval(Message *intern, const char *field_name,
                           upb_msgval val) {
  const upb_fielddef *f = upb_msgdef_ntofz(intern->desc->msgdef, field_name);
  PBPHP_ASSERT(f);
  return upb_msg_set(intern->msg, f, val, Arena_Get(&intern->arena));
}

static upb_msgval StringVal(upb_strview view) {
  upb_msgval ret;
  ret.str_val = view;
  return ret;
}

static bool TryStripUrlPrefix(upb_strview *str) {
  size_t size = strlen(TYPE_URL_PREFIX);
  if (str->size < size || memcmp(TYPE_URL_PREFIX, str->data, size) != 0) {
    return false;
  }
  str->data += size;
  str->size -= size;
  return true;
}

static bool StrViewEq(upb_strview view, const char *str) {
  size_t size = strlen(str);
  return view.size == size && memcmp(view.data, str, size) == 0;
}

PHP_METHOD(google_protobuf_Any, unpack) {
  Message* intern = (Message*)Z_OBJ_P(getThis());
  upb_strview type_url = Message_getval(intern, "type_url").str_val;
  upb_strview value = Message_getval(intern, "value").str_val;
  upb_symtab *symtab = DescriptorPool_GetSymbolTable();
  const upb_msgdef *m;
  Descriptor *desc;
  zval ret;

  // Ensure that type_url has TYPE_URL_PREFIX as a prefix.
  if (!TryStripUrlPrefix(&type_url)) {
    zend_throw_exception(
        NULL, "Type url needs to be type.googleapis.com/fully-qualified",
        0);
    return;
  }

  m = upb_symtab_lookupmsg2(symtab, type_url.data, type_url.size);

  if (m == NULL) {
    zend_throw_exception(
        NULL, "Specified message in any hasn't been added to descriptor pool",
        0);
    return;
  }

  desc = Descriptor_GetFromMessageDef(m);
  PBPHP_ASSERT(desc->class_entry->create_object == Message_create);
  zend_object *obj = Message_create(desc->class_entry);
  Message *msg = (Message*)obj;
  Message_Initialize(msg, desc);
  ZVAL_OBJ(&ret, obj);

  // Get value.
  if (!upb_decode(value.data, value.size, msg->msg,
                  upb_msgdef_layout(desc->msgdef), Arena_Get(&msg->arena))) {
    zend_throw_exception_ex(NULL, 0, "Error occurred during parsing");
    zval_dtor(&ret);
    return;
  }

  // Fuse since the parsed message could alias "value".
  upb_arena_fuse(Arena_Get(&intern->arena), Arena_Get(&msg->arena));

  RETURN_COPY_VALUE(&ret);
}

PHP_METHOD(google_protobuf_Any, pack) {
  Message* intern = (Message*)Z_OBJ_P(getThis());
  upb_arena *arena = Arena_Get(&intern->arena);
  zval *val;
  Message *msg;
  upb_strview value;
  upb_strview type_url;
  const char *full_name;
  char *buf;

  if (zend_parse_parameters(ZEND_NUM_ARGS(), "o", &val) ==
      FAILURE) {
    return;
  }

  if (!instanceof_function(Z_OBJCE_P(val), message_ce)) {
    zend_error(E_USER_ERROR, "Given value is not an instance of Message.");
    return;
  }

  msg = (Message*)Z_OBJ_P(val);

  // Serialize and set value.
  value.data = upb_encode(msg->msg, upb_msgdef_layout(msg->desc->msgdef), arena,
                          &value.size);
  Message_setval(intern, "value", StringVal(value));

  // Set type url: type_url_prefix + fully_qualified_name
  full_name = upb_msgdef_fullname(msg->desc->msgdef);
  type_url.size = strlen(TYPE_URL_PREFIX) + strlen(full_name);
  buf = upb_arena_malloc(arena, type_url.size + 1);
  memcpy(buf, TYPE_URL_PREFIX, strlen(TYPE_URL_PREFIX));
  memcpy(buf + strlen(TYPE_URL_PREFIX), full_name, strlen(full_name));
  type_url.data = buf;
  Message_setval(intern, "type_url", StringVal(type_url));
}

PHP_METHOD(google_protobuf_Any, is) {
  Message* intern = (Message*)Z_OBJ_P(getThis());
  upb_strview type_url = Message_getval(intern, "type_url").str_val;
  zend_class_entry *klass = NULL;
  const upb_msgdef *m;

  if (zend_parse_parameters(ZEND_NUM_ARGS(), "C", &klass) ==
      FAILURE) {
    return;
  }

  m = NameMap_GetMessage(klass);

  if (m == NULL) {
    RETURN_BOOL(false);
  }

  RETURN_BOOL(TryStripUrlPrefix(&type_url) &&
              StrViewEq(type_url, upb_msgdef_fullname(m)));
}

PHP_METHOD(google_protobuf_Timestamp, fromDateTime) {
  Message* intern = (Message*)Z_OBJ_P(getThis());
  zval* datetime;
  const char *classname = "\\DatetimeInterface";
  zend_string *classname_str = zend_string_init(classname, strlen(classname), 0);
  zend_class_entry *date_interface_ce = zend_lookup_class(classname_str);
  zend_string_release(classname_str);

  if (date_interface_ce == NULL) {
    zend_error(E_ERROR, "Make sure date extension is enabled.");
    return;
  }

  if (zend_parse_parameters(ZEND_NUM_ARGS(), "O", &datetime,
                            date_interface_ce) == FAILURE) {
    zend_error(E_USER_ERROR, "Expect DatetimeInterface.");
    return;
  }

  upb_msgval timestamp_seconds;
  {
    zval retval;
    zval function_name;

    ZVAL_STRING(&function_name, "date_timestamp_get");

    if (call_user_function(EG(function_table), NULL, &function_name, &retval, 1,
                           datetime) == FAILURE ||
        !Convert_PhpToUpb(&retval, &timestamp_seconds,
                          TypeInfo_FromType(UPB_TYPE_INT64), NULL)) {
      zend_error(E_ERROR, "Cannot get timestamp from DateTime.");
      return;
    }

    zval_dtor(&retval);
    zval_dtor(&function_name);
  }

  upb_msgval timestamp_nanos;
  {
    zval retval;
    zval function_name;
    zval format_string;

    ZVAL_STRING(&function_name, "date_format");
    ZVAL_STRING(&format_string, "u");

    zval params[2] = {
        *datetime,
        format_string,
    };

    if (call_user_function(EG(function_table), NULL, &function_name, &retval, 2,
                           params) == FAILURE ||
        !Convert_PhpToUpb(&retval, &timestamp_nanos,
                          TypeInfo_FromType(UPB_TYPE_INT32), NULL)) {
      zend_error(E_ERROR, "Cannot format DateTime.");
      return;
    }

    timestamp_nanos.int32_val *= 1000;

    zval_dtor(&retval);
    zval_dtor(&function_name);
    zval_dtor(&format_string);
  }

  Message_setval(intern, "seconds", timestamp_seconds);
  Message_setval(intern, "nanos", timestamp_nanos);

  RETURN_NULL();
}

PHP_METHOD(google_protobuf_Timestamp, toDateTime) {
  Message* intern = (Message*)Z_OBJ_P(getThis());
  upb_msgval seconds = Message_getval(intern, "seconds");
  upb_msgval nanos = Message_getval(intern, "nanos");

  // Get formatted time string.
  char formatted_time[32];
  snprintf(formatted_time, sizeof(formatted_time), "%" PRId64 ".%06" PRId32,
           seconds.int64_val, nanos.int32_val / 1000);

  // Create Datetime object.
  zval datetime;
  zval function_name;
  zval format_string;
  zval formatted_time_php;

  ZVAL_STRING(&function_name, "date_create_from_format");
  ZVAL_STRING(&format_string, "U.u");
  ZVAL_STRING(&formatted_time_php, formatted_time);

  zval params[2] = {
    format_string,
    formatted_time_php,
  };

  if (call_user_function(EG(function_table), NULL, &function_name, &datetime, 2,
                         params) == FAILURE) {
    zend_error(E_ERROR, "Cannot create DateTime.");
    return;
  }

  zval_dtor(&function_name);
  zval_dtor(&format_string);
  zval_dtor(&formatted_time_php);

  ZVAL_OBJ(return_value, Z_OBJ(datetime));
}

#include "wkt.inc"

/**
 * Message_ModuleInit()
 *
 * Called when the C extension is loaded to register all types.
 */
void Message_ModuleInit() {
  zend_class_entry tmp_ce;
  zend_object_handlers *h = &message_object_handlers;

  INIT_CLASS_ENTRY(tmp_ce, "Google\\Protobuf\\Internal\\Message",
                   Message_methods);

  message_ce = zend_register_internal_class(&tmp_ce);
  message_ce->create_object = Message_create;

  memcpy(h, &std_object_handlers, sizeof(zend_object_handlers));
  h->dtor_obj = Message_dtor;
#if PHP_VERSION_ID < 80000
  h->compare_objects = Message_compare_objects;
#else
  h->compare = Message_compare_objects;
#endif
  h->read_property = Message_read_property;
  h->write_property = Message_write_property;
  h->has_property = Message_has_property;
  h->unset_property = Message_unset_property;
  h->get_properties = Message_get_properties;
  h->get_property_ptr_ptr = Message_get_property_ptr_ptr;
  h->clone_obj = Message_clone_obj;

  WellKnownTypes_ModuleInit();  /* From wkt.inc. */
}
