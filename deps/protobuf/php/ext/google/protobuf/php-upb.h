/* Amalgamated source file */
/*
 * Copyright (c) 2009-2021, Google LLC
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of Google LLC nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL Google LLC BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * This is where we define macros used across upb.
 *
 * All of these macros are undef'd in port_undef.inc to avoid leaking them to
 * users.
 *
 * The correct usage is:
 *
 *   #include "upb/foobar.h"
 *   #include "upb/baz.h"
 *
 *   // MUST be last included header.
 *   #include "upb/port_def.inc"
 *
 *   // Code for this file.
 *   // <...>
 *
 *   // Can be omitted for .c files, required for .h.
 *   #include "upb/port_undef.inc"
 *
 * This file is private and must not be included by users!
 */

#if !((defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L) || \
      (defined(__cplusplus) && __cplusplus >= 201103L) ||           \
      (defined(_MSC_VER) && _MSC_VER >= 1900))
#error upb requires C99 or C++11 or MSVC >= 2015.
#endif

#include <stdint.h>
#include <stddef.h>

#if UINTPTR_MAX == 0xffffffff
#define UPB_SIZE(size32, size64) size32
#else
#define UPB_SIZE(size32, size64) size64
#endif

/* If we always read/write as a consistent type to each address, this shouldn't
 * violate aliasing.
 */
#define UPB_PTR_AT(msg, ofs, type) ((type*)((char*)(msg) + (ofs)))

#define UPB_READ_ONEOF(msg, fieldtype, offset, case_offset, case_val, default) \
  *UPB_PTR_AT(msg, case_offset, int) == case_val                              \
      ? *UPB_PTR_AT(msg, offset, fieldtype)                                   \
      : default

#define UPB_WRITE_ONEOF(msg, fieldtype, offset, value, case_offset, case_val) \
  *UPB_PTR_AT(msg, case_offset, int) = case_val;                             \
  *UPB_PTR_AT(msg, offset, fieldtype) = value;

#define UPB_MAPTYPE_STRING 0

/* UPB_INLINE: inline if possible, emit standalone code if required. */
#ifdef __cplusplus
#define UPB_INLINE inline
#elif defined (__GNUC__) || defined(__clang__)
#define UPB_INLINE static __inline__
#else
#define UPB_INLINE static
#endif

#define UPB_ALIGN_UP(size, align) (((size) + (align) - 1) / (align) * (align))
#define UPB_ALIGN_DOWN(size, align) ((size) / (align) * (align))
#define UPB_ALIGN_MALLOC(size) UPB_ALIGN_UP(size, 16)
#define UPB_ALIGN_OF(type) offsetof (struct { char c; type member; }, member)

/* Hints to the compiler about likely/unlikely branches. */
#if defined (__GNUC__) || defined(__clang__)
#define UPB_LIKELY(x) __builtin_expect((x),1)
#define UPB_UNLIKELY(x) __builtin_expect((x),0)
#else
#define UPB_LIKELY(x) (x)
#define UPB_UNLIKELY(x) (x)
#endif

/* Macros for function attributes on compilers that support them. */
#ifdef __GNUC__
#define UPB_FORCEINLINE __inline__ __attribute__((always_inline))
#define UPB_NOINLINE __attribute__((noinline))
#define UPB_NORETURN __attribute__((__noreturn__))
#define UPB_PRINTF(str, first_vararg) __attribute__((format (printf, str, first_vararg)))
#elif defined(_MSC_VER)
#define UPB_NOINLINE
#define UPB_FORCEINLINE
#define UPB_NORETURN __declspec(noreturn)
#define UPB_PRINTF(str, first_vararg)
#else  /* !defined(__GNUC__) */
#define UPB_FORCEINLINE
#define UPB_NOINLINE
#define UPB_NORETURN
#define UPB_PRINTF(str, first_vararg)
#endif

#define UPB_MAX(x, y) ((x) > (y) ? (x) : (y))
#define UPB_MIN(x, y) ((x) < (y) ? (x) : (y))

#define UPB_UNUSED(var) (void)var

/* UPB_ASSUME(): in release mode, we tell the compiler to assume this is true.
 */
#ifdef NDEBUG
#ifdef __GNUC__
#define UPB_ASSUME(expr) if (!(expr)) __builtin_unreachable()
#elif defined _MSC_VER
#define UPB_ASSUME(expr) if (!(expr)) __assume(0)
#else
#define UPB_ASSUME(expr) do {} while (false && (expr))
#endif
#else
#define UPB_ASSUME(expr) assert(expr)
#endif

/* UPB_ASSERT(): in release mode, we use the expression without letting it be
 * evaluated.  This prevents "unused variable" warnings. */
#ifdef NDEBUG
#define UPB_ASSERT(expr) do {} while (false && (expr))
#else
#define UPB_ASSERT(expr) assert(expr)
#endif

#if defined(__GNUC__) || defined(__clang__)
#define UPB_UNREACHABLE() do { assert(0); __builtin_unreachable(); } while(0)
#else
#define UPB_UNREACHABLE() do { assert(0); } while(0)
#endif

/* UPB_SETJMP() / UPB_LONGJMP(): avoid setting/restoring signal mask. */
#ifdef __APPLE__
#define UPB_SETJMP(buf) _setjmp(buf)
#define UPB_LONGJMP(buf, val) _longjmp(buf, val)
#else
#define UPB_SETJMP(buf) setjmp(buf)
#define UPB_LONGJMP(buf, val) longjmp(buf, val)
#endif

/* UPB_PTRADD(ptr, ofs): add pointer while avoiding "NULL + 0" UB */
#define UPB_PTRADD(ptr, ofs) ((ofs) ? (ptr) + (ofs) : (ptr))

/* Configure whether fasttable is switched on or not. *************************/

#ifdef __has_attribute
#define UPB_HAS_ATTRIBUTE(x) __has_attribute(x)
#else
#define UPB_HAS_ATTRIBUTE(x) 0
#endif

#if UPB_HAS_ATTRIBUTE(musttail)
#define UPB_MUSTTAIL __attribute__((musttail))
#else
#define UPB_MUSTTAIL
#endif

#undef UPB_HAS_ATTRIBUTE

/* This check is not fully robust: it does not require that we have "musttail"
 * support available. We need tail calls to avoid consuming arbitrary amounts
 * of stack space.
 *
 * GCC/Clang can mostly be trusted to generate tail calls as long as
 * optimization is enabled, but, debug builds will not generate tail calls
 * unless "musttail" is available.
 *
 * We should probably either:
 *   1. require that the compiler supports musttail.
 *   2. add some fallback code for when musttail isn't available (ie. return
 *      instead of tail calling). This is safe and portable, but this comes at
 *      a CPU cost.
 */
#if (defined(__x86_64__) || defined(__aarch64__)) && defined(__GNUC__)
#define UPB_FASTTABLE_SUPPORTED 1
#else
#define UPB_FASTTABLE_SUPPORTED 0
#endif

/* define UPB_ENABLE_FASTTABLE to force fast table support.
 * This is useful when we want to ensure we are really getting fasttable,
 * for example for testing or benchmarking. */
#if defined(UPB_ENABLE_FASTTABLE)
#if !UPB_FASTTABLE_SUPPORTED
#error fasttable is x86-64/ARM64 only and requires GCC or Clang.
#endif
#define UPB_FASTTABLE 1
/* Define UPB_TRY_ENABLE_FASTTABLE to use fasttable if possible.
 * This is useful for releasing code that might be used on multiple platforms,
 * for example the PHP or Ruby C extensions. */
#elif defined(UPB_TRY_ENABLE_FASTTABLE)
#define UPB_FASTTABLE UPB_FASTTABLE_SUPPORTED
#else
#define UPB_FASTTABLE 0
#endif

/* UPB_FASTTABLE_INIT() allows protos compiled for fasttable to gracefully
 * degrade to non-fasttable if we are using UPB_TRY_ENABLE_FASTTABLE. */
#if !UPB_FASTTABLE && defined(UPB_TRY_ENABLE_FASTTABLE)
#define UPB_FASTTABLE_INIT(...)
#else
#define UPB_FASTTABLE_INIT(...) __VA_ARGS__
#endif

#undef UPB_FASTTABLE_SUPPORTED

/* ASAN poisoning (for arena) *************************************************/

#if defined(__SANITIZE_ADDRESS__)
#define UPB_ASAN 1
#ifdef __cplusplus
extern "C" {
#endif
void __asan_poison_memory_region(void const volatile *addr, size_t size);
void __asan_unpoison_memory_region(void const volatile *addr, size_t size);
#ifdef __cplusplus
}  /* extern "C" */
#endif
#define UPB_POISON_MEMORY_REGION(addr, size) \
  __asan_poison_memory_region((addr), (size))
#define UPB_UNPOISON_MEMORY_REGION(addr, size) \
  __asan_unpoison_memory_region((addr), (size))
#else
#define UPB_ASAN 0
#define UPB_POISON_MEMORY_REGION(addr, size) \
  ((void)(addr), (void)(size))
#define UPB_UNPOISON_MEMORY_REGION(addr, size) \
  ((void)(addr), (void)(size))
#endif

/** upb/decode.h ************************************************************/
/*
 * upb_decode: parsing into a upb_msg using a upb_msglayout.
 */

#ifndef UPB_DECODE_H_
#define UPB_DECODE_H_


/** upb/msg.h ************************************************************/
/*
 * Public APIs for message operations that do not require descriptors.
 * These functions can be used even in build that does not want to depend on
 * reflection or descriptors.
 *
 * Descriptor-based reflection functionality lives in reflection.h.
 */

#ifndef UPB_MSG_H_
#define UPB_MSG_H_

#include <stddef.h>


/** upb/upb.h ************************************************************/
/*
 * This file contains shared definitions that are widely used across upb.
 */

#ifndef UPB_H_
#define UPB_H_

#include <assert.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>


#ifdef __cplusplus
extern "C" {
#endif

/* upb_status *****************************************************************/

#define UPB_STATUS_MAX_MESSAGE 127

typedef struct {
  bool ok;
  char msg[UPB_STATUS_MAX_MESSAGE];  /* Error message; NULL-terminated. */
} upb_status;

const char *upb_status_errmsg(const upb_status *status);
bool upb_ok(const upb_status *status);

/* These are no-op if |status| is NULL. */
void upb_status_clear(upb_status *status);
void upb_status_seterrmsg(upb_status *status, const char *msg);
void upb_status_seterrf(upb_status *status, const char *fmt, ...)
    UPB_PRINTF(2, 3);
void upb_status_vseterrf(upb_status *status, const char *fmt, va_list args)
    UPB_PRINTF(2, 0);
void upb_status_vappenderrf(upb_status *status, const char *fmt, va_list args)
    UPB_PRINTF(2, 0);

/** upb_strview ************************************************************/

typedef struct {
  const char *data;
  size_t size;
} upb_strview;

UPB_INLINE upb_strview upb_strview_make(const char *data, size_t size) {
  upb_strview ret;
  ret.data = data;
  ret.size = size;
  return ret;
}

UPB_INLINE upb_strview upb_strview_makez(const char *data) {
  return upb_strview_make(data, strlen(data));
}

UPB_INLINE bool upb_strview_eql(upb_strview a, upb_strview b) {
  return a.size == b.size && memcmp(a.data, b.data, a.size) == 0;
}

#define UPB_STRVIEW_INIT(ptr, len) {ptr, len}

#define UPB_STRVIEW_FORMAT "%.*s"
#define UPB_STRVIEW_ARGS(view) (int)(view).size, (view).data

/** upb_alloc *****************************************************************/

/* A upb_alloc is a possibly-stateful allocator object.
 *
 * It could either be an arena allocator (which doesn't require individual
 * free() calls) or a regular malloc() (which does).  The client must therefore
 * free memory unless it knows that the allocator is an arena allocator. */

struct upb_alloc;
typedef struct upb_alloc upb_alloc;

/* A malloc()/free() function.
 * If "size" is 0 then the function acts like free(), otherwise it acts like
 * realloc().  Only "oldsize" bytes from a previous allocation are preserved. */
typedef void *upb_alloc_func(upb_alloc *alloc, void *ptr, size_t oldsize,
                             size_t size);

struct upb_alloc {
  upb_alloc_func *func;
};

UPB_INLINE void *upb_malloc(upb_alloc *alloc, size_t size) {
  UPB_ASSERT(alloc);
  return alloc->func(alloc, NULL, 0, size);
}

UPB_INLINE void *upb_realloc(upb_alloc *alloc, void *ptr, size_t oldsize,
                             size_t size) {
  UPB_ASSERT(alloc);
  return alloc->func(alloc, ptr, oldsize, size);
}

UPB_INLINE void upb_free(upb_alloc *alloc, void *ptr) {
  assert(alloc);
  alloc->func(alloc, ptr, 0, 0);
}

/* The global allocator used by upb.  Uses the standard malloc()/free(). */

extern upb_alloc upb_alloc_global;

/* Functions that hard-code the global malloc.
 *
 * We still get benefit because we can put custom logic into our global
 * allocator, like injecting out-of-memory faults in debug/testing builds. */

UPB_INLINE void *upb_gmalloc(size_t size) {
  return upb_malloc(&upb_alloc_global, size);
}

UPB_INLINE void *upb_grealloc(void *ptr, size_t oldsize, size_t size) {
  return upb_realloc(&upb_alloc_global, ptr, oldsize, size);
}

UPB_INLINE void upb_gfree(void *ptr) {
  upb_free(&upb_alloc_global, ptr);
}

/* upb_arena ******************************************************************/

/* upb_arena is a specific allocator implementation that uses arena allocation.
 * The user provides an allocator that will be used to allocate the underlying
 * arena blocks.  Arenas by nature do not require the individual allocations
 * to be freed.  However the Arena does allow users to register cleanup
 * functions that will run when the arena is destroyed.
 *
 * A upb_arena is *not* thread-safe.
 *
 * You could write a thread-safe arena allocator that satisfies the
 * upb_alloc interface, but it would not be as efficient for the
 * single-threaded case. */

typedef void upb_cleanup_func(void *ud);

struct upb_arena;
typedef struct upb_arena upb_arena;

typedef struct {
  /* We implement the allocator interface.
   * This must be the first member of upb_arena!
   * TODO(haberman): remove once handlers are gone. */
  upb_alloc alloc;

  char *ptr, *end;
} _upb_arena_head;

/* Creates an arena from the given initial block (if any -- n may be 0).
 * Additional blocks will be allocated from |alloc|.  If |alloc| is NULL, this
 * is a fixed-size arena and cannot grow. */
upb_arena *upb_arena_init(void *mem, size_t n, upb_alloc *alloc);
void upb_arena_free(upb_arena *a);
bool upb_arena_addcleanup(upb_arena *a, void *ud, upb_cleanup_func *func);
bool upb_arena_fuse(upb_arena *a, upb_arena *b);
void *_upb_arena_slowmalloc(upb_arena *a, size_t size);

UPB_INLINE upb_alloc *upb_arena_alloc(upb_arena *a) { return (upb_alloc*)a; }

UPB_INLINE size_t _upb_arenahas(upb_arena *a) {
  _upb_arena_head *h = (_upb_arena_head*)a;
  return (size_t)(h->end - h->ptr);
}

UPB_INLINE void *upb_arena_malloc(upb_arena *a, size_t size) {
  _upb_arena_head *h = (_upb_arena_head*)a;
  void* ret;
  size = UPB_ALIGN_MALLOC(size);

  if (UPB_UNLIKELY(_upb_arenahas(a) < size)) {
    return _upb_arena_slowmalloc(a, size);
  }

  ret = h->ptr;
  h->ptr += size;
  UPB_UNPOISON_MEMORY_REGION(ret, size);

#if UPB_ASAN
  {
    size_t guard_size = 32;
    if (_upb_arenahas(a) >= guard_size) {
      h->ptr += guard_size;
    } else {
      h->ptr = h->end;
    }
  }
#endif

  return ret;
}

UPB_INLINE void *upb_arena_realloc(upb_arena *a, void *ptr, size_t oldsize,
                                   size_t size) {
  void *ret = upb_arena_malloc(a, size);

  if (ret && oldsize > 0) {
    memcpy(ret, ptr, oldsize);
  }

  return ret;
}

UPB_INLINE upb_arena *upb_arena_new(void) {
  return upb_arena_init(NULL, 0, &upb_alloc_global);
}

/* Constants ******************************************************************/

/* Generic function type. */
typedef void upb_func(void);

/* A list of types as they are encoded on-the-wire. */
typedef enum {
  UPB_WIRE_TYPE_VARINT      = 0,
  UPB_WIRE_TYPE_64BIT       = 1,
  UPB_WIRE_TYPE_DELIMITED   = 2,
  UPB_WIRE_TYPE_START_GROUP = 3,
  UPB_WIRE_TYPE_END_GROUP   = 4,
  UPB_WIRE_TYPE_32BIT       = 5
} upb_wiretype_t;

/* The types a field can have.  Note that this list is not identical to the
 * types defined in descriptor.proto, which gives INT32 and SINT32 separate
 * types (we distinguish the two with the "integer encoding" enum below). */
typedef enum {
  UPB_TYPE_BOOL     = 1,
  UPB_TYPE_FLOAT    = 2,
  UPB_TYPE_INT32    = 3,
  UPB_TYPE_UINT32   = 4,
  UPB_TYPE_ENUM     = 5,  /* Enum values are int32. */
  UPB_TYPE_MESSAGE  = 6,
  UPB_TYPE_DOUBLE   = 7,
  UPB_TYPE_INT64    = 8,
  UPB_TYPE_UINT64   = 9,
  UPB_TYPE_STRING   = 10,
  UPB_TYPE_BYTES    = 11
} upb_fieldtype_t;

/* The repeated-ness of each field; this matches descriptor.proto. */
typedef enum {
  UPB_LABEL_OPTIONAL = 1,
  UPB_LABEL_REQUIRED = 2,
  UPB_LABEL_REPEATED = 3
} upb_label_t;

/* Descriptor types, as defined in descriptor.proto. */
typedef enum {
  /* Old (long) names.  TODO(haberman): remove */
  UPB_DESCRIPTOR_TYPE_DOUBLE   = 1,
  UPB_DESCRIPTOR_TYPE_FLOAT    = 2,
  UPB_DESCRIPTOR_TYPE_INT64    = 3,
  UPB_DESCRIPTOR_TYPE_UINT64   = 4,
  UPB_DESCRIPTOR_TYPE_INT32    = 5,
  UPB_DESCRIPTOR_TYPE_FIXED64  = 6,
  UPB_DESCRIPTOR_TYPE_FIXED32  = 7,
  UPB_DESCRIPTOR_TYPE_BOOL     = 8,
  UPB_DESCRIPTOR_TYPE_STRING   = 9,
  UPB_DESCRIPTOR_TYPE_GROUP    = 10,
  UPB_DESCRIPTOR_TYPE_MESSAGE  = 11,
  UPB_DESCRIPTOR_TYPE_BYTES    = 12,
  UPB_DESCRIPTOR_TYPE_UINT32   = 13,
  UPB_DESCRIPTOR_TYPE_ENUM     = 14,
  UPB_DESCRIPTOR_TYPE_SFIXED32 = 15,
  UPB_DESCRIPTOR_TYPE_SFIXED64 = 16,
  UPB_DESCRIPTOR_TYPE_SINT32   = 17,
  UPB_DESCRIPTOR_TYPE_SINT64   = 18,

  UPB_DTYPE_DOUBLE   = 1,
  UPB_DTYPE_FLOAT    = 2,
  UPB_DTYPE_INT64    = 3,
  UPB_DTYPE_UINT64   = 4,
  UPB_DTYPE_INT32    = 5,
  UPB_DTYPE_FIXED64  = 6,
  UPB_DTYPE_FIXED32  = 7,
  UPB_DTYPE_BOOL     = 8,
  UPB_DTYPE_STRING   = 9,
  UPB_DTYPE_GROUP    = 10,
  UPB_DTYPE_MESSAGE  = 11,
  UPB_DTYPE_BYTES    = 12,
  UPB_DTYPE_UINT32   = 13,
  UPB_DTYPE_ENUM     = 14,
  UPB_DTYPE_SFIXED32 = 15,
  UPB_DTYPE_SFIXED64 = 16,
  UPB_DTYPE_SINT32   = 17,
  UPB_DTYPE_SINT64   = 18
} upb_descriptortype_t;

#define UPB_MAP_BEGIN ((size_t)-1)

UPB_INLINE bool _upb_isle(void) {
  int x = 1;
  return *(char*)&x == 1;
}

UPB_INLINE uint32_t _upb_be_swap32(uint32_t val) {
  if (_upb_isle()) {
    return val;
  } else {
    return ((val & 0xff) << 24) | ((val & 0xff00) << 8) |
           ((val & 0xff0000) >> 8) | ((val & 0xff000000) >> 24);
  }
}

UPB_INLINE uint64_t _upb_be_swap64(uint64_t val) {
  if (_upb_isle()) {
    return val;
  } else {
    return ((uint64_t)_upb_be_swap32(val) << 32) | _upb_be_swap32(val >> 32);
  }
}

UPB_INLINE int _upb_lg2ceil(int x) {
  if (x <= 1) return 0;
#ifdef __GNUC__
  return 32 - __builtin_clz(x - 1);
#else
  int lg2 = 0;
  while (1 << lg2 < x) lg2++;
  return lg2;
#endif
}

UPB_INLINE int _upb_lg2ceilsize(int x) {
  return 1 << _upb_lg2ceil(x);
}


#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* UPB_H_ */

#ifdef __cplusplus
extern "C" {
#endif

typedef void upb_msg;

/* For users these are opaque. They can be obtained from upb_msgdef_layout()
 * but users cannot access any of the members. */
struct upb_msglayout;
typedef struct upb_msglayout upb_msglayout;

/* Adds unknown data (serialized protobuf data) to the given message.  The data
 * is copied into the message instance. */
void upb_msg_addunknown(upb_msg *msg, const char *data, size_t len,
                        upb_arena *arena);

/* Returns a reference to the message's unknown data. */
const char *upb_msg_getunknown(const upb_msg *msg, size_t *len);

/** upb_extreg *******************************************************************/

/* Extension registry: a dynamic data structure that stores a map of:
 *   (upb_msglayout, number) -> extension info
 *
 * upb_decode() uses upb_extreg to look up extensions while parsing binary
 * format.
 *
 * upb_extreg is part of the mini-table (msglayout) family of objects. Like all
 * mini-table objects, it is suitable for reflection-less builds that do not
 * want to expose names into the binary.
 *
 * Unlike most mini-table types, upb_extreg requires dynamic memory allocation
 * and dynamic initialization:
 * * If reflection is being used, then upb_symtab will construct an appropriate
 *   upb_extreg automatically.
 * * For a mini-table only build, the user must manually construct the
 *   upb_extreg and populate it with all of the extensions the user cares about.
 * * A third alternative is to manually unpack relevant extensions after the
 *   main parse is complete, similar to how Any works. This is perhaps the
 *   nicest solution from the perspective of reducing dependencies, avoiding
 *   dynamic memory allocation, and avoiding the need to parse uninteresting
 *   extensions.  The downsides are:
 *     (1) parse errors are not caught during the main parse
 *     (2) the CPU hit of parsing comes during access, which could cause an
 *         undesirable stutter in application performance.
 *
 * Users cannot directly get or put into this map. Users can only add the
 * extensions from a generated module and pass the extension registry to the
 * binary decoder.
 *
 * A upb_symtab provides a upb_extreg, so any users who use reflection do not
 * need to populate a upb_extreg directly.
 */

struct upb_extreg;
typedef struct upb_extreg upb_extreg;

/* Creates a upb_extreg in the given arena.  The arena must outlive any use of
 * the extreg. */
upb_extreg *upb_extreg_new(upb_arena *arena);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* UPB_MSG_INT_H_ */

/* Must be last. */

#ifdef __cplusplus
extern "C" {
#endif

enum {
  /* If set, strings will alias the input buffer instead of copying into the
   * arena. */
  UPB_DECODE_ALIAS = 1,
};

#define UPB_DECODE_MAXDEPTH(depth) ((depth) << 16)

bool _upb_decode(const char *buf, size_t size, upb_msg *msg,
                 const upb_msglayout *l, const upb_extreg *extreg, int options,
                 upb_arena *arena);

UPB_INLINE
bool upb_decode(const char *buf, size_t size, upb_msg *msg,
                const upb_msglayout *l, upb_arena *arena) {
  return _upb_decode(buf, size, msg, l, NULL, 0, arena);
}

#ifdef __cplusplus
}  /* extern "C" */
#endif


#endif  /* UPB_DECODE_H_ */

/** upb/decode_internal.h ************************************************************/
/*
 * Internal implementation details of the decoder that are shared between
 * decode.c and decode_fast.c.
 */

#ifndef UPB_DECODE_INT_H_
#define UPB_DECODE_INT_H_

#include <setjmp.h>


/** upb/msg_internal.h ************************************************************//*
** Our memory representation for parsing tables and messages themselves.
** Functions in this file are used by generated code and possibly reflection.
**
** The definitions in this file are internal to upb.
**/

#ifndef UPB_MSG_INT_H_
#define UPB_MSG_INT_H_

#include <stdint.h>
#include <stdlib.h>
#include <string.h>


/** upb/table_internal.h ************************************************************/
/*
 * upb_table
 *
 * This header is INTERNAL-ONLY!  Its interfaces are not public or stable!
 * This file defines very fast int->upb_value (inttable) and string->upb_value
 * (strtable) hash tables.
 *
 * The table uses chained scatter with Brent's variation (inspired by the Lua
 * implementation of hash tables).  The hash function for strings is Austin
 * Appleby's "MurmurHash."
 *
 * The inttable uses uintptr_t as its key, which guarantees it can be used to
 * store pointers or integers of at least 32 bits (upb isn't really useful on
 * systems where sizeof(void*) < 4).
 *
 * The table must be homogeneous (all values of the same type).  In debug
 * mode, we check this on insert and lookup.
 */

#ifndef UPB_TABLE_H_
#define UPB_TABLE_H_

#include <stdint.h>
#include <string.h>


#ifdef __cplusplus
extern "C" {
#endif


/* upb_value ******************************************************************/

typedef struct {
  uint64_t val;
} upb_value;

/* Variant that works with a length-delimited rather than NULL-delimited string,
 * as supported by strtable. */
char *upb_strdup2(const char *s, size_t len, upb_arena *a);

UPB_INLINE void _upb_value_setval(upb_value *v, uint64_t val) {
  v->val = val;
}

/* For each value ctype, define the following set of functions:
 *
 * // Get/set an int32 from a upb_value.
 * int32_t upb_value_getint32(upb_value val);
 * void upb_value_setint32(upb_value *val, int32_t cval);
 *
 * // Construct a new upb_value from an int32.
 * upb_value upb_value_int32(int32_t val); */
#define FUNCS(name, membername, type_t, converter, proto_type) \
  UPB_INLINE void upb_value_set ## name(upb_value *val, type_t cval) { \
    val->val = (converter)cval; \
  } \
  UPB_INLINE upb_value upb_value_ ## name(type_t val) { \
    upb_value ret; \
    upb_value_set ## name(&ret, val); \
    return ret; \
  } \
  UPB_INLINE type_t upb_value_get ## name(upb_value val) { \
    return (type_t)(converter)val.val; \
  }

FUNCS(int32,    int32,        int32_t,      int32_t,    UPB_CTYPE_INT32)
FUNCS(int64,    int64,        int64_t,      int64_t,    UPB_CTYPE_INT64)
FUNCS(uint32,   uint32,       uint32_t,     uint32_t,   UPB_CTYPE_UINT32)
FUNCS(uint64,   uint64,       uint64_t,     uint64_t,   UPB_CTYPE_UINT64)
FUNCS(bool,     _bool,        bool,         bool,       UPB_CTYPE_BOOL)
FUNCS(cstr,     cstr,         char*,        uintptr_t,  UPB_CTYPE_CSTR)
FUNCS(ptr,      ptr,          void*,        uintptr_t,  UPB_CTYPE_PTR)
FUNCS(constptr, constptr,     const void*,  uintptr_t,  UPB_CTYPE_CONSTPTR)
FUNCS(fptr,     fptr,         upb_func*,    uintptr_t,  UPB_CTYPE_FPTR)

#undef FUNCS

UPB_INLINE void upb_value_setfloat(upb_value *val, float cval) {
  memcpy(&val->val, &cval, sizeof(cval));
}

UPB_INLINE void upb_value_setdouble(upb_value *val, double cval) {
  memcpy(&val->val, &cval, sizeof(cval));
}

UPB_INLINE upb_value upb_value_float(float cval) {
  upb_value ret;
  upb_value_setfloat(&ret, cval);
  return ret;
}

UPB_INLINE upb_value upb_value_double(double cval) {
  upb_value ret;
  upb_value_setdouble(&ret, cval);
  return ret;
}

#undef SET_TYPE


/* upb_tabkey *****************************************************************/

/* Either:
 *   1. an actual integer key, or
 *   2. a pointer to a string prefixed by its uint32_t length, owned by us.
 *
 * ...depending on whether this is a string table or an int table.  We would
 * make this a union of those two types, but C89 doesn't support statically
 * initializing a non-first union member. */
typedef uintptr_t upb_tabkey;

UPB_INLINE char *upb_tabstr(upb_tabkey key, uint32_t *len) {
  char* mem = (char*)key;
  if (len) memcpy(len, mem, sizeof(*len));
  return mem + sizeof(*len);
}

UPB_INLINE upb_strview upb_tabstrview(upb_tabkey key) {
  upb_strview ret;
  uint32_t len;
  ret.data = upb_tabstr(key, &len);
  ret.size = len;
  return ret;
}

/* upb_tabval *****************************************************************/

typedef struct upb_tabval {
  uint64_t val;
} upb_tabval;

#define UPB_TABVALUE_EMPTY_INIT  {-1}

/* upb_table ******************************************************************/

typedef struct _upb_tabent {
  upb_tabkey key;
  upb_tabval val;

  /* Internal chaining.  This is const so we can create static initializers for
   * tables.  We cast away const sometimes, but *only* when the containing
   * upb_table is known to be non-const.  This requires a bit of care, but
   * the subtlety is confined to table.c. */
  const struct _upb_tabent *next;
} upb_tabent;

typedef struct {
  size_t count;          /* Number of entries in the hash part. */
  uint32_t mask;         /* Mask to turn hash value -> bucket. */
  uint32_t max_count;    /* Max count before we hit our load limit. */
  uint8_t size_lg2;      /* Size of the hashtable part is 2^size_lg2 entries. */
  upb_tabent *entries;
} upb_table;

typedef struct {
  upb_table t;
} upb_strtable;

typedef struct {
  upb_table t;              /* For entries that don't fit in the array part. */
  const upb_tabval *array;  /* Array part of the table. See const note above. */
  size_t array_size;        /* Array part size. */
  size_t array_count;       /* Array part number of elements. */
} upb_inttable;

UPB_INLINE size_t upb_table_size(const upb_table *t) {
  if (t->size_lg2 == 0)
    return 0;
  else
    return 1 << t->size_lg2;
}

/* Internal-only functions, in .h file only out of necessity. */
UPB_INLINE bool upb_tabent_isempty(const upb_tabent *e) {
  return e->key == 0;
}

/* Initialize and uninitialize a table, respectively.  If memory allocation
 * failed, false is returned that the table is uninitialized. */
bool upb_inttable_init(upb_inttable *table, upb_arena *a);
bool upb_strtable_init(upb_strtable *table, size_t expected_size, upb_arena *a);

/* Returns the number of values in the table. */
size_t upb_inttable_count(const upb_inttable *t);
UPB_INLINE size_t upb_strtable_count(const upb_strtable *t) {
  return t->t.count;
}

void upb_strtable_clear(upb_strtable *t);

/* Inserts the given key into the hashtable with the given value.  The key must
 * not already exist in the hash table.  For string tables, the key must be
 * NULL-terminated, and the table will make an internal copy of the key.
 * Inttables must not insert a value of UINTPTR_MAX.
 *
 * If a table resize was required but memory allocation failed, false is
 * returned and the table is unchanged. */
bool upb_inttable_insert(upb_inttable *t, uintptr_t key, upb_value val,
                         upb_arena *a);
bool upb_strtable_insert(upb_strtable *t, const char *key, size_t len,
                         upb_value val, upb_arena *a);

/* Looks up key in this table, returning "true" if the key was found.
 * If v is non-NULL, copies the value for this key into *v. */
bool upb_inttable_lookup(const upb_inttable *t, uintptr_t key, upb_value *v);
bool upb_strtable_lookup2(const upb_strtable *t, const char *key, size_t len,
                          upb_value *v);

/* For NULL-terminated strings. */
UPB_INLINE bool upb_strtable_lookup(const upb_strtable *t, const char *key,
                                    upb_value *v) {
  return upb_strtable_lookup2(t, key, strlen(key), v);
}

/* Removes an item from the table.  Returns true if the remove was successful,
 * and stores the removed item in *val if non-NULL. */
bool upb_inttable_remove(upb_inttable *t, uintptr_t key, upb_value *val);
bool upb_strtable_remove(upb_strtable *t, const char *key, size_t len,
                          upb_value *val);

/* Updates an existing entry in an inttable.  If the entry does not exist,
 * returns false and does nothing.  Unlike insert/remove, this does not
 * invalidate iterators. */
bool upb_inttable_replace(upb_inttable *t, uintptr_t key, upb_value val);

/* Optimizes the table for the current set of entries, for both memory use and
 * lookup time.  Client should call this after all entries have been inserted;
 * inserting more entries is legal, but will likely require a table resize. */
void upb_inttable_compact(upb_inttable *t, upb_arena *a);

/* Exposed for testing only. */
bool upb_strtable_resize(upb_strtable *t, size_t size_lg2, upb_arena *a);

/* Iterators ******************************************************************/

/* Iterators for int and string tables.  We are subject to some kind of unusual
 * design constraints:
 *
 * For high-level languages:
 *  - we must be able to guarantee that we don't crash or corrupt memory even if
 *    the program accesses an invalidated iterator.
 *
 * For C++11 range-based for:
 *  - iterators must be copyable
 *  - iterators must be comparable
 *  - it must be possible to construct an "end" value.
 *
 * Iteration order is undefined.
 *
 * Modifying the table invalidates iterators.  upb_{str,int}table_done() is
 * guaranteed to work even on an invalidated iterator, as long as the table it
 * is iterating over has not been freed.  Calling next() or accessing data from
 * an invalidated iterator yields unspecified elements from the table, but it is
 * guaranteed not to crash and to return real table elements (except when done()
 * is true). */


/* upb_strtable_iter **********************************************************/

/*   upb_strtable_iter i;
 *   upb_strtable_begin(&i, t);
 *   for(; !upb_strtable_done(&i); upb_strtable_next(&i)) {
 *     const char *key = upb_strtable_iter_key(&i);
 *     const upb_value val = upb_strtable_iter_value(&i);
 *     // ...
 *   }
 */

typedef struct {
  const upb_strtable *t;
  size_t index;
} upb_strtable_iter;

void upb_strtable_begin(upb_strtable_iter *i, const upb_strtable *t);
void upb_strtable_next(upb_strtable_iter *i);
bool upb_strtable_done(const upb_strtable_iter *i);
upb_strview upb_strtable_iter_key(const upb_strtable_iter *i);
upb_value upb_strtable_iter_value(const upb_strtable_iter *i);
void upb_strtable_iter_setdone(upb_strtable_iter *i);
bool upb_strtable_iter_isequal(const upb_strtable_iter *i1,
                               const upb_strtable_iter *i2);


/* upb_inttable_iter **********************************************************/

/*   upb_inttable_iter i;
 *   upb_inttable_begin(&i, t);
 *   for(; !upb_inttable_done(&i); upb_inttable_next(&i)) {
 *     uintptr_t key = upb_inttable_iter_key(&i);
 *     upb_value val = upb_inttable_iter_value(&i);
 *     // ...
 *   }
 */

typedef struct {
  const upb_inttable *t;
  size_t index;
  bool array_part;
} upb_inttable_iter;

UPB_INLINE const upb_tabent *str_tabent(const upb_strtable_iter *i) {
  return &i->t->t.entries[i->index];
}

void upb_inttable_begin(upb_inttable_iter *i, const upb_inttable *t);
void upb_inttable_next(upb_inttable_iter *i);
bool upb_inttable_done(const upb_inttable_iter *i);
uintptr_t upb_inttable_iter_key(const upb_inttable_iter *i);
upb_value upb_inttable_iter_value(const upb_inttable_iter *i);
void upb_inttable_iter_setdone(upb_inttable_iter *i);
bool upb_inttable_iter_isequal(const upb_inttable_iter *i1,
                               const upb_inttable_iter *i2);


#ifdef __cplusplus
}  /* extern "C" */
#endif


#endif  /* UPB_TABLE_H_ */

/* Must be last. */

#ifdef __cplusplus
extern "C" {
#endif

/** upb_msglayout *************************************************************/

/* upb_msglayout represents the memory layout of a given upb_msgdef.  The
 * members are public so generated code can initialize them, but users MUST NOT
 * read or write any of its members. */

/* These aren't real labels according to descriptor.proto, but in the table we
 * use these for map/packed fields instead of UPB_LABEL_REPEATED. */
enum {
  _UPB_LABEL_MAP = 4,
  _UPB_LABEL_PACKED = 7  /* Low 3 bits are common with UPB_LABEL_REPEATED. */
};

typedef struct {
  uint32_t number;
  uint16_t offset;
  int16_t presence;       /* If >0, hasbit_index.  If <0, ~oneof_index. */
  uint16_t submsg_index;  /* undefined if descriptortype != MESSAGE or GROUP. */
  uint8_t descriptortype;
  int8_t mode;            /* upb_fieldmode, with flags from upb_labelflags */
} upb_msglayout_field;

typedef enum {
  _UPB_MODE_MAP = 0,
  _UPB_MODE_ARRAY = 1,
  _UPB_MODE_SCALAR = 2,
} upb_fieldmode;

/* Extra flags on the mode field. */
enum upb_labelflags {
  _UPB_MODE_IS_PACKED = 4,
};

UPB_INLINE upb_fieldmode _upb_getmode(const upb_msglayout_field *field) {
  return (upb_fieldmode)(field->mode & 3);
}

UPB_INLINE bool _upb_repeated_or_map(const upb_msglayout_field *field) {
  /* This works because upb_fieldmode has no value 3. */
  return !(field->mode & _UPB_MODE_SCALAR);
}

UPB_INLINE bool _upb_issubmsg(const upb_msglayout_field *field) {
  return field->descriptortype == UPB_DTYPE_MESSAGE ||
         field->descriptortype == UPB_DTYPE_GROUP;
}

struct upb_decstate;
struct upb_msglayout;

typedef const char *_upb_field_parser(struct upb_decstate *d, const char *ptr,
                                      upb_msg *msg, intptr_t table,
                                      uint64_t hasbits, uint64_t data);

typedef struct {
  uint64_t field_data;
  _upb_field_parser *field_parser;
} _upb_fasttable_entry;

struct upb_msglayout {
  const struct upb_msglayout *const* submsgs;
  const upb_msglayout_field *fields;
  /* Must be aligned to sizeof(void*).  Doesn't include internal members like
   * unknown fields, extension dict, pointer to msglayout, etc. */
  uint16_t size;
  uint16_t field_count;
  bool extendable;
  uint8_t dense_below;
  uint8_t table_mask;
  /* To constant-initialize the tables of variable length, we need a flexible
   * array member, and we need to compile in C99 mode. */
  _upb_fasttable_entry fasttable[];
};

typedef struct {
  upb_msglayout_field field;
  const upb_msglayout *extendee;
  const upb_msglayout *submsg;   /* NULL for non-submessage fields. */
} upb_msglayout_ext;

/** upb_extreg ****************************************************************/

/* Adds the given extension info for message type |l| and field number |num|
 * into the registry. Returns false if this message type and field number were
 * already in the map, or if memory allocation fails. */
bool _upb_extreg_add(upb_extreg *r, const upb_msglayout_ext *e, size_t count);

/* Looks up the extension (if any) defined for message type |l| and field
 * number |num|.  If an extension was found, copies the field info into |*ext|
 * and returns true. Otherwise returns false. */
const upb_msglayout_field *_upb_extreg_get(const upb_extreg *r,
                                           const upb_msglayout *l,
                                           uint32_t num);

/** upb_msg *******************************************************************/

/* Internal members of a upb_msg that track unknown fields and/or extensions.
 * We can change this without breaking binary compatibility.  We put these
 * before the user's data.  The user's upb_msg* points after the
 * upb_msg_internal. */

typedef struct {
  /* Total size of this structure, including the data that follows.
   * Must be aligned to 8, which is alignof(upb_msg_ext) */
  uint32_t size;

  /* Offsets relative to the beginning of this structure.
   *
   * Unknown data grows forward from the beginning to unknown_end.
   * Extension data grows backward from size to ext_begin.
   * When the two meet, we're out of data and have to realloc.
   *
   * If we imagine that the final member of this struct is:
   *   char data[size - overhead];  // overhead = sizeof(upb_msg_internaldata)
   * 
   * Then we have:
   *   unknown data: data[0 .. (unknown_end - overhead)]
   *   extensions data: data[(ext_begin - overhead) .. (size - overhead)] */
  uint32_t unknown_end;
  uint32_t ext_begin;
  /* Data follows, as if there were an array:
   *   char data[size - sizeof(upb_msg_internaldata)]; */
} upb_msg_internaldata;

typedef struct {
  upb_msg_internaldata *internal;
} upb_msg_internal;

/* Maps upb_fieldtype_t -> memory size. */
extern char _upb_fieldtype_to_size[12];

UPB_INLINE size_t upb_msg_sizeof(const upb_msglayout *l) {
  return l->size + sizeof(upb_msg_internal);
}

UPB_INLINE upb_msg *_upb_msg_new_inl(const upb_msglayout *l, upb_arena *a) {
  size_t size = upb_msg_sizeof(l);
  void *mem = upb_arena_malloc(a, size);
  upb_msg *msg;
  if (UPB_UNLIKELY(!mem)) return NULL;
  msg = UPB_PTR_AT(mem, sizeof(upb_msg_internal), upb_msg);
  memset(mem, 0, size);
  return msg;
}

/* Creates a new messages with the given layout on the given arena. */
upb_msg *_upb_msg_new(const upb_msglayout *l, upb_arena *a);

UPB_INLINE upb_msg_internal *upb_msg_getinternal(upb_msg *msg) {
  ptrdiff_t size = sizeof(upb_msg_internal);
  return (upb_msg_internal*)((char*)msg - size);
}

/* Clears the given message. */
void _upb_msg_clear(upb_msg *msg, const upb_msglayout *l);

/* Discards the unknown fields for this message only. */
void _upb_msg_discardunknown_shallow(upb_msg *msg);

/* Adds unknown data (serialized protobuf data) to the given message.  The data
 * is copied into the message instance. */
bool _upb_msg_addunknown(upb_msg *msg, const char *data, size_t len,
                         upb_arena *arena);

/** upb_msg_ext ***************************************************************/

/* The internal representation of an extension is self-describing: it contains
 * enough information that we can serialize it to binary format without needing
 * to look it up in a registry. */
typedef struct {
  const upb_msglayout_ext *ext;
  union {
    upb_strview str;
    void *ptr;
    double dbl;
    char scalar_data[8];
  } data;
} upb_msg_ext;

/* Adds the given extension data to the given message. The returned extension will
 * have its "ext" member initialized according to |ext|. */
upb_msg_ext *_upb_msg_getorcreateext(upb_msg *msg, const upb_msglayout_ext *ext,
                                     upb_arena *arena);

/* Returns an array of extensions for this message. Note: the array is
 * ordered in reverse relative to the order of creation. */
const upb_msg_ext *_upb_msg_getexts(const upb_msg *msg, size_t *count);

/* Returns an extension for the given field number, or NULL if no extension
 * exists for this field number. */
const upb_msg_ext *_upb_msg_getext(const upb_msg *msg,
                                   const upb_msglayout_ext *ext);

/** Hasbit access *************************************************************/

UPB_INLINE bool _upb_hasbit(const upb_msg *msg, size_t idx) {
  return (*UPB_PTR_AT(msg, idx / 8, const char) & (1 << (idx % 8))) != 0;
}

UPB_INLINE void _upb_sethas(const upb_msg *msg, size_t idx) {
  (*UPB_PTR_AT(msg, idx / 8, char)) |= (char)(1 << (idx % 8));
}

UPB_INLINE void _upb_clearhas(const upb_msg *msg, size_t idx) {
  (*UPB_PTR_AT(msg, idx / 8, char)) &= (char)(~(1 << (idx % 8)));
}

UPB_INLINE size_t _upb_msg_hasidx(const upb_msglayout_field *f) {
  UPB_ASSERT(f->presence > 0);
  return f->presence;
}

UPB_INLINE bool _upb_hasbit_field(const upb_msg *msg,
                                  const upb_msglayout_field *f) {
  return _upb_hasbit(msg, _upb_msg_hasidx(f));
}

UPB_INLINE void _upb_sethas_field(const upb_msg *msg,
                                  const upb_msglayout_field *f) {
  _upb_sethas(msg, _upb_msg_hasidx(f));
}

UPB_INLINE void _upb_clearhas_field(const upb_msg *msg,
                                    const upb_msglayout_field *f) {
  _upb_clearhas(msg, _upb_msg_hasidx(f));
}

/** Oneof case access *********************************************************/

UPB_INLINE uint32_t *_upb_oneofcase(upb_msg *msg, size_t case_ofs) {
  return UPB_PTR_AT(msg, case_ofs, uint32_t);
}

UPB_INLINE uint32_t _upb_getoneofcase(const void *msg, size_t case_ofs) {
  return *UPB_PTR_AT(msg, case_ofs, uint32_t);
}

UPB_INLINE size_t _upb_oneofcase_ofs(const upb_msglayout_field *f) {
  UPB_ASSERT(f->presence < 0);
  return ~(ptrdiff_t)f->presence;
}

UPB_INLINE uint32_t *_upb_oneofcase_field(upb_msg *msg,
                                          const upb_msglayout_field *f) {
  return _upb_oneofcase(msg, _upb_oneofcase_ofs(f));
}

UPB_INLINE uint32_t _upb_getoneofcase_field(const upb_msg *msg,
                                            const upb_msglayout_field *f) {
  return _upb_getoneofcase(msg, _upb_oneofcase_ofs(f));
}

UPB_INLINE bool _upb_has_submsg_nohasbit(const upb_msg *msg, size_t ofs) {
  return *UPB_PTR_AT(msg, ofs, const upb_msg*) != NULL;
}

/** upb_array *****************************************************************/

/* Our internal representation for repeated fields.  */
typedef struct {
  uintptr_t data;   /* Tagged ptr: low 3 bits of ptr are lg2(elem size). */
  size_t len;   /* Measured in elements. */
  size_t size;  /* Measured in elements. */
  uint64_t junk;
} upb_array;

UPB_INLINE const void *_upb_array_constptr(const upb_array *arr) {
  UPB_ASSERT((arr->data & 7) <= 4);
  return (void*)(arr->data & ~(uintptr_t)7);
}

UPB_INLINE uintptr_t _upb_array_tagptr(void* ptr, int elem_size_lg2) {
  UPB_ASSERT(elem_size_lg2 <= 4);
  return (uintptr_t)ptr | elem_size_lg2;
}

UPB_INLINE void *_upb_array_ptr(upb_array *arr) {
  return (void*)_upb_array_constptr(arr);
}

UPB_INLINE uintptr_t _upb_tag_arrptr(void* ptr, int elem_size_lg2) {
  UPB_ASSERT(elem_size_lg2 <= 4);
  UPB_ASSERT(((uintptr_t)ptr & 7) == 0);
  return (uintptr_t)ptr | (unsigned)elem_size_lg2;
}

UPB_INLINE upb_array *_upb_array_new(upb_arena *a, size_t init_size,
                                     int elem_size_lg2) {
  const size_t arr_size = UPB_ALIGN_UP(sizeof(upb_array), 8);
  const size_t bytes = sizeof(upb_array) + (init_size << elem_size_lg2);
  upb_array *arr = (upb_array*)upb_arena_malloc(a, bytes);
  if (!arr) return NULL;
  arr->data = _upb_tag_arrptr(UPB_PTR_AT(arr, arr_size, void), elem_size_lg2);
  arr->len = 0;
  arr->size = init_size;
  return arr;
}

/* Resizes the capacity of the array to be at least min_size. */
bool _upb_array_realloc(upb_array *arr, size_t min_size, upb_arena *arena);

/* Fallback functions for when the accessors require a resize. */
void *_upb_array_resize_fallback(upb_array **arr_ptr, size_t size,
                                 int elem_size_lg2, upb_arena *arena);
bool _upb_array_append_fallback(upb_array **arr_ptr, const void *value,
                                int elem_size_lg2, upb_arena *arena);

UPB_INLINE bool _upb_array_reserve(upb_array *arr, size_t size,
                                   upb_arena *arena) {
  if (arr->size < size) return _upb_array_realloc(arr, size, arena);
  return true;
}

UPB_INLINE bool _upb_array_resize(upb_array *arr, size_t size,
                                  upb_arena *arena) {
  if (!_upb_array_reserve(arr, size, arena)) return false;
  arr->len = size;
  return true;
}

UPB_INLINE const void *_upb_array_accessor(const void *msg, size_t ofs,
                                           size_t *size) {
  const upb_array *arr = *UPB_PTR_AT(msg, ofs, const upb_array*);
  if (arr) {
    if (size) *size = arr->len;
    return _upb_array_constptr(arr);
  } else {
    if (size) *size = 0;
    return NULL;
  }
}

UPB_INLINE void *_upb_array_mutable_accessor(void *msg, size_t ofs,
                                             size_t *size) {
  upb_array *arr = *UPB_PTR_AT(msg, ofs, upb_array*);
  if (arr) {
    if (size) *size = arr->len;
    return _upb_array_ptr(arr);
  } else {
    if (size) *size = 0;
    return NULL;
  }
}

UPB_INLINE void *_upb_array_resize_accessor2(void *msg, size_t ofs, size_t size,
                                             int elem_size_lg2,
                                             upb_arena *arena) {
  upb_array **arr_ptr = UPB_PTR_AT(msg, ofs, upb_array *);
  upb_array *arr = *arr_ptr;
  if (!arr || arr->size < size) {
    return _upb_array_resize_fallback(arr_ptr, size, elem_size_lg2, arena);
  }
  arr->len = size;
  return _upb_array_ptr(arr);
}

UPB_INLINE bool _upb_array_append_accessor2(void *msg, size_t ofs,
                                            int elem_size_lg2,
                                            const void *value,
                                            upb_arena *arena) {
  upb_array **arr_ptr = UPB_PTR_AT(msg, ofs, upb_array *);
  size_t elem_size = 1 << elem_size_lg2;
  upb_array *arr = *arr_ptr;
  void *ptr;
  if (!arr || arr->len == arr->size) {
    return _upb_array_append_fallback(arr_ptr, value, elem_size_lg2, arena);
  }
  ptr = _upb_array_ptr(arr);
  memcpy(UPB_PTR_AT(ptr, arr->len * elem_size, char), value, elem_size);
  arr->len++;
  return true;
}

/* Used by old generated code, remove once all code has been regenerated. */
UPB_INLINE int _upb_sizelg2(upb_fieldtype_t type) {
  switch (type) {
    case UPB_TYPE_BOOL:
      return 0;
    case UPB_TYPE_FLOAT:
    case UPB_TYPE_INT32:
    case UPB_TYPE_UINT32:
    case UPB_TYPE_ENUM:
      return 2;
    case UPB_TYPE_MESSAGE:
      return UPB_SIZE(2, 3);
    case UPB_TYPE_DOUBLE:
    case UPB_TYPE_INT64:
    case UPB_TYPE_UINT64:
      return 3;
    case UPB_TYPE_STRING:
    case UPB_TYPE_BYTES:
      return UPB_SIZE(3, 4);
  }
  UPB_UNREACHABLE();
}
UPB_INLINE void *_upb_array_resize_accessor(void *msg, size_t ofs, size_t size,
                                             upb_fieldtype_t type,
                                             upb_arena *arena) {
  return _upb_array_resize_accessor2(msg, ofs, size, _upb_sizelg2(type), arena);
}
UPB_INLINE bool _upb_array_append_accessor(void *msg, size_t ofs,
                                            size_t elem_size, upb_fieldtype_t type,
                                            const void *value,
                                            upb_arena *arena) {
  (void)elem_size;
  return _upb_array_append_accessor2(msg, ofs, _upb_sizelg2(type), value,
                                     arena);
}

/** upb_map *******************************************************************/

/* Right now we use strmaps for everything.  We'll likely want to use
 * integer-specific maps for integer-keyed maps.*/
typedef struct {
  /* Size of key and val, based on the map type.  Strings are represented as '0'
   * because they must be handled specially. */
  char key_size;
  char val_size;

  upb_strtable table;
} upb_map;

/* Map entries aren't actually stored, they are only used during parsing.  For
 * parsing, it helps a lot if all map entry messages have the same layout.
 * The compiler and def.c must ensure that all map entries have this layout. */
typedef struct {
  upb_msg_internal internal;
  union {
    upb_strview str;  /* For str/bytes. */
    upb_value val;    /* For all other types. */
  } k;
  union {
    upb_strview str;  /* For str/bytes. */
    upb_value val;    /* For all other types. */
  } v;
} upb_map_entry;

/* Creates a new map on the given arena with this key/value type. */
upb_map *_upb_map_new(upb_arena *a, size_t key_size, size_t value_size);

/* Converting between internal table representation and user values.
 *
 * _upb_map_tokey() and _upb_map_fromkey() are inverses.
 * _upb_map_tovalue() and _upb_map_fromvalue() are inverses.
 *
 * These functions account for the fact that strings are treated differently
 * from other types when stored in a map.
 */

UPB_INLINE upb_strview _upb_map_tokey(const void *key, size_t size) {
  if (size == UPB_MAPTYPE_STRING) {
    return *(upb_strview*)key;
  } else {
    return upb_strview_make((const char*)key, size);
  }
}

UPB_INLINE void _upb_map_fromkey(upb_strview key, void* out, size_t size) {
  if (size == UPB_MAPTYPE_STRING) {
    memcpy(out, &key, sizeof(key));
  } else {
    memcpy(out, key.data, size);
  }
}

UPB_INLINE bool _upb_map_tovalue(const void *val, size_t size, upb_value *msgval,
                                 upb_arena *a) {
  if (size == UPB_MAPTYPE_STRING) {
    upb_strview *strp = (upb_strview*)upb_arena_malloc(a, sizeof(*strp));
    if (!strp) return false;
    *strp = *(upb_strview*)val;
    *msgval = upb_value_ptr(strp);
  } else {
    memcpy(msgval, val, size);
  }
  return true;
}

UPB_INLINE void _upb_map_fromvalue(upb_value val, void* out, size_t size) {
  if (size == UPB_MAPTYPE_STRING) {
    const upb_strview *strp = (const upb_strview*)upb_value_getptr(val);
    memcpy(out, strp, sizeof(upb_strview));
  } else {
    memcpy(out, &val, size);
  }
}

/* Map operations, shared by reflection and generated code. */

UPB_INLINE size_t _upb_map_size(const upb_map *map) {
  return map->table.t.count;
}

UPB_INLINE bool _upb_map_get(const upb_map *map, const void *key,
                             size_t key_size, void *val, size_t val_size) {
  upb_value tabval;
  upb_strview k = _upb_map_tokey(key, key_size);
  bool ret = upb_strtable_lookup2(&map->table, k.data, k.size, &tabval);
  if (ret && val) {
    _upb_map_fromvalue(tabval, val, val_size);
  }
  return ret;
}

UPB_INLINE void* _upb_map_next(const upb_map *map, size_t *iter) {
  upb_strtable_iter it;
  it.t = &map->table;
  it.index = *iter;
  upb_strtable_next(&it);
  *iter = it.index;
  if (upb_strtable_done(&it)) return NULL;
  return (void*)str_tabent(&it);
}

UPB_INLINE bool _upb_map_set(upb_map *map, const void *key, size_t key_size,
                             void *val, size_t val_size, upb_arena *a) {
  upb_strview strkey = _upb_map_tokey(key, key_size);
  upb_value tabval = {0};
  if (!_upb_map_tovalue(val, val_size, &tabval, a)) return false;

  /* TODO(haberman): add overwrite operation to minimize number of lookups. */
  upb_strtable_remove(&map->table, strkey.data, strkey.size, NULL);
  return upb_strtable_insert(&map->table, strkey.data, strkey.size, tabval, a);
}

UPB_INLINE bool _upb_map_delete(upb_map *map, const void *key, size_t key_size) {
  upb_strview k = _upb_map_tokey(key, key_size);
  return upb_strtable_remove(&map->table, k.data, k.size, NULL);
}

UPB_INLINE void _upb_map_clear(upb_map *map) {
  upb_strtable_clear(&map->table);
}

/* Message map operations, these get the map from the message first. */

UPB_INLINE size_t _upb_msg_map_size(const upb_msg *msg, size_t ofs) {
  upb_map *map = *UPB_PTR_AT(msg, ofs, upb_map *);
  return map ? _upb_map_size(map) : 0;
}

UPB_INLINE bool _upb_msg_map_get(const upb_msg *msg, size_t ofs,
                                 const void *key, size_t key_size, void *val,
                                 size_t val_size) {
  upb_map *map = *UPB_PTR_AT(msg, ofs, upb_map *);
  if (!map) return false;
  return _upb_map_get(map, key, key_size, val, val_size);
}

UPB_INLINE void *_upb_msg_map_next(const upb_msg *msg, size_t ofs,
                                   size_t *iter) {
  upb_map *map = *UPB_PTR_AT(msg, ofs, upb_map *);
  if (!map) return NULL;
  return _upb_map_next(map, iter);
}

UPB_INLINE bool _upb_msg_map_set(upb_msg *msg, size_t ofs, const void *key,
                                 size_t key_size, void *val, size_t val_size,
                                 upb_arena *arena) {
  upb_map **map = UPB_PTR_AT(msg, ofs, upb_map *);
  if (!*map) {
    *map = _upb_map_new(arena, key_size, val_size);
  }
  return _upb_map_set(*map, key, key_size, val, val_size, arena);
}

UPB_INLINE bool _upb_msg_map_delete(upb_msg *msg, size_t ofs, const void *key,
                                    size_t key_size) {
  upb_map *map = *UPB_PTR_AT(msg, ofs, upb_map *);
  if (!map) return false;
  return _upb_map_delete(map, key, key_size);
}

UPB_INLINE void _upb_msg_map_clear(upb_msg *msg, size_t ofs) {
  upb_map *map = *UPB_PTR_AT(msg, ofs, upb_map *);
  if (!map) return;
  _upb_map_clear(map);
}

/* Accessing map key/value from a pointer, used by generated code only. */

UPB_INLINE void _upb_msg_map_key(const void* msg, void* key, size_t size) {
  const upb_tabent *ent = (const upb_tabent*)msg;
  uint32_t u32len;
  upb_strview k;
  k.data = upb_tabstr(ent->key, &u32len);
  k.size = u32len;
  _upb_map_fromkey(k, key, size);
}

UPB_INLINE void _upb_msg_map_value(const void* msg, void* val, size_t size) {
  const upb_tabent *ent = (const upb_tabent*)msg;
  upb_value v = {ent->val.val};
  _upb_map_fromvalue(v, val, size);
}

UPB_INLINE void _upb_msg_map_set_value(void* msg, const void* val, size_t size) {
  upb_tabent *ent = (upb_tabent*)msg;
  /* This is like _upb_map_tovalue() except the entry already exists so we can
   * reuse the allocated upb_strview for string fields. */
  if (size == UPB_MAPTYPE_STRING) {
    upb_strview *strp = (upb_strview*)(uintptr_t)ent->val.val;
    memcpy(strp, val, sizeof(*strp));
  } else {
    memcpy(&ent->val.val, val, size);
  }
}

/** _upb_mapsorter *************************************************************/

/* _upb_mapsorter sorts maps and provides ordered iteration over the entries.
 * Since maps can be recursive (map values can be messages which contain other maps).
 * _upb_mapsorter can contain a stack of maps. */

typedef struct {
  upb_tabent const**entries;
  int size;
  int cap;
} _upb_mapsorter;

typedef struct {
  int start;
  int pos;
  int end;
} _upb_sortedmap;

UPB_INLINE void _upb_mapsorter_init(_upb_mapsorter *s) {
  s->entries = NULL;
  s->size = 0;
  s->cap = 0;
}

UPB_INLINE void _upb_mapsorter_destroy(_upb_mapsorter *s) {
  if (s->entries) free(s->entries);
}

bool _upb_mapsorter_pushmap(_upb_mapsorter *s, upb_descriptortype_t key_type,
                            const upb_map *map, _upb_sortedmap *sorted);

UPB_INLINE void _upb_mapsorter_popmap(_upb_mapsorter *s, _upb_sortedmap *sorted) {
  s->size = sorted->start;
}

UPB_INLINE bool _upb_sortedmap_next(_upb_mapsorter *s, const upb_map *map,
                                    _upb_sortedmap *sorted,
                                    upb_map_entry *ent) {
  if (sorted->pos == sorted->end) return false;
  const upb_tabent *tabent = s->entries[sorted->pos++];
  upb_strview key = upb_tabstrview(tabent->key);
  _upb_map_fromkey(key, &ent->k, map->key_size);
  upb_value val = {tabent->val.val};
  _upb_map_fromvalue(val, &ent->v, map->val_size);
  return true;
}

#ifdef __cplusplus
}  /* extern "C" */
#endif


#endif /* UPB_MSG_INT_H_ */

/** upb/upb_internal.h ************************************************************/
#ifndef UPB_INT_H_
#define UPB_INT_H_


struct mem_block;
typedef struct mem_block mem_block;

struct upb_arena {
  _upb_arena_head head;
  /* Stores cleanup metadata for this arena.
   * - a pointer to the current cleanup counter.
   * - a boolean indicating if there is an unowned initial block.  */
  uintptr_t cleanup_metadata;

  /* Allocator to allocate arena blocks.  We are responsible for freeing these
   * when we are destroyed. */
  upb_alloc *block_alloc;
  uint32_t last_size;

  /* When multiple arenas are fused together, each arena points to a parent
   * arena (root points to itself). The root tracks how many live arenas
   * reference it. */
  uint32_t refcount;  /* Only used when a->parent == a */
  struct upb_arena *parent;

  /* Linked list of blocks to free/cleanup. */
  mem_block *freelist, *freelist_tail;
};

#endif  /* UPB_INT_H_ */

/* Must be last. */

#define DECODE_NOGROUP (uint32_t)-1

typedef struct upb_decstate {
  const char *end;         /* Can read up to 16 bytes slop beyond this. */
  const char *limit_ptr;   /* = end + UPB_MIN(limit, 0) */
  upb_msg *unknown_msg;    /* If non-NULL, add unknown data at buffer flip. */
  const char *unknown;     /* Start of unknown data. */
  int limit;               /* Submessage limit relative to end. */
  int depth;
  uint32_t end_group;   /* field number of END_GROUP tag, else DECODE_NOGROUP */
  bool alias;
  char patch[32];
  upb_arena arena;
  jmp_buf err;
} upb_decstate;

/* Error function that will abort decoding with longjmp(). We can't declare this
 * UPB_NORETURN, even though it is appropriate, because if we do then compilers
 * will "helpfully" refuse to tailcall to it
 * (see: https://stackoverflow.com/a/55657013), which will defeat a major goal
 * of our optimizations. That is also why we must declare it in a separate file,
 * otherwise the compiler will see that it calls longjmp() and deduce that it is
 * noreturn. */
const char *fastdecode_err(upb_decstate *d);

extern const uint8_t upb_utf8_offsets[];

UPB_INLINE
bool decode_verifyutf8_inl(const char *buf, int len) {
  int i, j;
  uint8_t offset;

  i = 0;
  while (i < len) {
    offset = upb_utf8_offsets[(uint8_t)buf[i]];
    if (offset == 0 || i + offset > len) {
      return false;
    }
    for (j = i + 1; j < i + offset; j++) {
      if ((buf[j] & 0xc0) != 0x80) {
        return false;
      }
    }
    i += offset;
  }
  return i == len;
}

/* x86-64 pointers always have the high 16 bits matching. So we can shift
 * left 8 and right 8 without loss of information. */
UPB_INLINE intptr_t decode_totable(const upb_msglayout *tablep) {
  return ((intptr_t)tablep << 8) | tablep->table_mask;
}

UPB_INLINE const upb_msglayout *decode_totablep(intptr_t table) {
  return (const upb_msglayout*)(table >> 8);
}

UPB_INLINE
const char *decode_isdonefallback_inl(upb_decstate *d, const char *ptr,
                                      int overrun) {
  if (overrun < d->limit) {
    /* Need to copy remaining data into patch buffer. */
    UPB_ASSERT(overrun < 16);
    if (d->unknown_msg) {
      if (!_upb_msg_addunknown(d->unknown_msg, d->unknown, ptr - d->unknown,
                               &d->arena)) {
        return NULL;
      }
      d->unknown = &d->patch[0] + overrun;
    }
    memset(d->patch + 16, 0, 16);
    memcpy(d->patch, d->end, 16);
    ptr = &d->patch[0] + overrun;
    d->end = &d->patch[16];
    d->limit -= 16;
    d->limit_ptr = d->end + d->limit;
    d->alias = false;
    UPB_ASSERT(ptr < d->limit_ptr);
    return ptr;
  } else {
    return NULL;
  }
}

const char *decode_isdonefallback(upb_decstate *d, const char *ptr,
                                  int overrun);

UPB_INLINE
bool decode_isdone(upb_decstate *d, const char **ptr) {
  int overrun = *ptr - d->end;
  if (UPB_LIKELY(*ptr < d->limit_ptr)) {
    return false;
  } else if (UPB_LIKELY(overrun == d->limit)) {
    return true;
  } else {
    *ptr = decode_isdonefallback(d, *ptr, overrun);
    return false;
  }
}

#if UPB_FASTTABLE
UPB_INLINE
const char *fastdecode_tagdispatch(upb_decstate *d, const char *ptr,
                                    upb_msg *msg, intptr_t table,
                                    uint64_t hasbits, uint64_t tag) {
  const upb_msglayout *table_p = decode_totablep(table);
  uint8_t mask = table;
  uint64_t data;
  size_t idx = tag & mask;
  UPB_ASSUME((idx & 7) == 0);
  idx >>= 3;
  data = table_p->fasttable[idx].field_data ^ tag;
  UPB_MUSTTAIL return table_p->fasttable[idx].field_parser(d, ptr, msg, table,
                                                           hasbits, data);
}
#endif

UPB_INLINE uint32_t fastdecode_loadtag(const char* ptr) {
  uint16_t tag;
  memcpy(&tag, ptr, 2);
  return tag;
}

UPB_INLINE void decode_checklimit(upb_decstate *d) {
  UPB_ASSERT(d->limit_ptr == d->end + UPB_MIN(0, d->limit));
}

UPB_INLINE int decode_pushlimit(upb_decstate *d, const char *ptr, int size) {
  int limit = size + (int)(ptr - d->end);
  int delta = d->limit - limit;
  decode_checklimit(d);
  d->limit = limit;
  d->limit_ptr = d->end + UPB_MIN(0, limit);
  decode_checklimit(d);
  return delta;
}

UPB_INLINE void decode_poplimit(upb_decstate *d, const char *ptr,
                                int saved_delta) {
  UPB_ASSERT(ptr - d->end == d->limit);
  decode_checklimit(d);
  d->limit += saved_delta;
  d->limit_ptr = d->end + UPB_MIN(0, d->limit);
  decode_checklimit(d);
}


#endif  /* UPB_DECODE_INT_H_ */

/** upb/encode.h ************************************************************/
/*
 * upb_encode: parsing into a upb_msg using a upb_msglayout.
 */

#ifndef UPB_ENCODE_H_
#define UPB_ENCODE_H_


/* Must be last. */

#ifdef __cplusplus
extern "C" {
#endif

enum {
  /* If set, the results of serializing will be deterministic across all
   * instances of this binary. There are no guarantees across different
   * binary builds.
   *
   * If your proto contains maps, the encoder will need to malloc()/free()
   * memory during encode. */
  UPB_ENCODE_DETERMINISTIC = 1,

  /* When set, unknown fields are not printed. */
  UPB_ENCODE_SKIPUNKNOWN = 2,
};

#define UPB_ENCODE_MAXDEPTH(depth) ((depth) << 16)

char *upb_encode_ex(const void *msg, const upb_msglayout *l, int options,
                    upb_arena *arena, size_t *size);

UPB_INLINE char *upb_encode(const void *msg, const upb_msglayout *l,
                            upb_arena *arena, size_t *size) {
  return upb_encode_ex(msg, l, 0, arena, size);
}


#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* UPB_ENCODE_H_ */

/** upb/decode_fast.h ************************************************************/
// These are the specialized field parser functions for the fast parser.
// Generated tables will refer to these by name.
//
// The function names are encoded with names like:
//
//   //  123 4
//   upb_pss_1bt();   // Parse singular string, 1 byte tag.
//
// In position 1:
//   - 'p' for parse, most function use this
//   - 'c' for copy, for when we are copying strings instead of aliasing
//
// In position 2 (cardinality):
//   - 's' for singular, with or without hasbit
//   - 'o' for oneof
//   - 'r' for non-packed repeated
//   - 'p' for packed repeated
//
// In position 3 (type):
//   - 'b1' for bool
//   - 'v4' for 4-byte varint
//   - 'v8' for 8-byte varint
//   - 'z4' for zig-zag-encoded 4-byte varint
//   - 'z8' for zig-zag-encoded 8-byte varint
//   - 'f4' for 4-byte fixed
//   - 'f8' for 8-byte fixed
//   - 'm' for sub-message
//   - 's' for string (validate UTF-8)
//   - 'b' for bytes
//
// In position 4 (tag length):
//   - '1' for one-byte tags (field numbers 1-15)
//   - '2' for two-byte tags (field numbers 16-2048)

#ifndef UPB_DECODE_FAST_H_
#define UPB_DECODE_FAST_H_


struct upb_decstate;

// The fallback, generic parsing function that can handle any field type.
// This just uses the regular (non-fast) parser to parse a single field.
const char *fastdecode_generic(struct upb_decstate *d, const char *ptr,
                               upb_msg *msg, intptr_t table, uint64_t hasbits,
                               uint64_t data);

#define UPB_PARSE_PARAMS                                                 \
  struct upb_decstate *d, const char *ptr, upb_msg *msg, intptr_t table, \
      uint64_t hasbits, uint64_t data

/* primitive fields ***********************************************************/

#define F(card, type, valbytes, tagbytes) \
  const char *upb_p##card##type##valbytes##_##tagbytes##bt(UPB_PARSE_PARAMS);

#define TYPES(card, tagbytes) \
  F(card, b, 1, tagbytes)     \
  F(card, v, 4, tagbytes)     \
  F(card, v, 8, tagbytes)     \
  F(card, z, 4, tagbytes)     \
  F(card, z, 8, tagbytes)     \
  F(card, f, 4, tagbytes)     \
  F(card, f, 8, tagbytes)

#define TAGBYTES(card) \
  TYPES(card, 1)       \
  TYPES(card, 2)

TAGBYTES(s)
TAGBYTES(o)
TAGBYTES(r)
TAGBYTES(p)

#undef F
#undef TYPES
#undef TAGBYTES

/* string fields **************************************************************/

#define F(card, tagbytes, type)                                     \
  const char *upb_p##card##type##_##tagbytes##bt(UPB_PARSE_PARAMS); \
  const char *upb_c##card##type##_##tagbytes##bt(UPB_PARSE_PARAMS);

#define UTF8(card, tagbytes) \
  F(card, tagbytes, s)       \
  F(card, tagbytes, b)

#define TAGBYTES(card) \
  UTF8(card, 1)        \
  UTF8(card, 2)

TAGBYTES(s)
TAGBYTES(o)
TAGBYTES(r)

#undef F
#undef TAGBYTES

/* sub-message fields *********************************************************/

#define F(card, tagbytes, size_ceil, ceil_arg) \
  const char *upb_p##card##m_##tagbytes##bt_max##size_ceil##b(UPB_PARSE_PARAMS);

#define SIZES(card, tagbytes) \
  F(card, tagbytes, 64, 64) \
  F(card, tagbytes, 128, 128) \
  F(card, tagbytes, 192, 192) \
  F(card, tagbytes, 256, 256) \
  F(card, tagbytes, max, -1)

#define TAGBYTES(card) \
  SIZES(card, 1) \
  SIZES(card, 2)

TAGBYTES(s)
TAGBYTES(o)
TAGBYTES(r)

#undef TAGBYTES
#undef SIZES
#undef F

#undef UPB_PARSE_PARAMS

#endif  /* UPB_DECODE_FAST_H_ */

/** google/protobuf/descriptor.upb.h ************************************************************//* This file was generated by upbc (the upb compiler) from the input
 * file:
 *
 *     google/protobuf/descriptor.proto
 *
 * Do not edit -- your changes will be discarded when the file is
 * regenerated. */

#ifndef GOOGLE_PROTOBUF_DESCRIPTOR_PROTO_UPB_H_
#define GOOGLE_PROTOBUF_DESCRIPTOR_PROTO_UPB_H_



#ifdef __cplusplus
extern "C" {
#endif

struct google_protobuf_FileDescriptorSet;
struct google_protobuf_FileDescriptorProto;
struct google_protobuf_DescriptorProto;
struct google_protobuf_DescriptorProto_ExtensionRange;
struct google_protobuf_DescriptorProto_ReservedRange;
struct google_protobuf_ExtensionRangeOptions;
struct google_protobuf_FieldDescriptorProto;
struct google_protobuf_OneofDescriptorProto;
struct google_protobuf_EnumDescriptorProto;
struct google_protobuf_EnumDescriptorProto_EnumReservedRange;
struct google_protobuf_EnumValueDescriptorProto;
struct google_protobuf_ServiceDescriptorProto;
struct google_protobuf_MethodDescriptorProto;
struct google_protobuf_FileOptions;
struct google_protobuf_MessageOptions;
struct google_protobuf_FieldOptions;
struct google_protobuf_OneofOptions;
struct google_protobuf_EnumOptions;
struct google_protobuf_EnumValueOptions;
struct google_protobuf_ServiceOptions;
struct google_protobuf_MethodOptions;
struct google_protobuf_UninterpretedOption;
struct google_protobuf_UninterpretedOption_NamePart;
struct google_protobuf_SourceCodeInfo;
struct google_protobuf_SourceCodeInfo_Location;
struct google_protobuf_GeneratedCodeInfo;
struct google_protobuf_GeneratedCodeInfo_Annotation;
typedef struct google_protobuf_FileDescriptorSet google_protobuf_FileDescriptorSet;
typedef struct google_protobuf_FileDescriptorProto google_protobuf_FileDescriptorProto;
typedef struct google_protobuf_DescriptorProto google_protobuf_DescriptorProto;
typedef struct google_protobuf_DescriptorProto_ExtensionRange google_protobuf_DescriptorProto_ExtensionRange;
typedef struct google_protobuf_DescriptorProto_ReservedRange google_protobuf_DescriptorProto_ReservedRange;
typedef struct google_protobuf_ExtensionRangeOptions google_protobuf_ExtensionRangeOptions;
typedef struct google_protobuf_FieldDescriptorProto google_protobuf_FieldDescriptorProto;
typedef struct google_protobuf_OneofDescriptorProto google_protobuf_OneofDescriptorProto;
typedef struct google_protobuf_EnumDescriptorProto google_protobuf_EnumDescriptorProto;
typedef struct google_protobuf_EnumDescriptorProto_EnumReservedRange google_protobuf_EnumDescriptorProto_EnumReservedRange;
typedef struct google_protobuf_EnumValueDescriptorProto google_protobuf_EnumValueDescriptorProto;
typedef struct google_protobuf_ServiceDescriptorProto google_protobuf_ServiceDescriptorProto;
typedef struct google_protobuf_MethodDescriptorProto google_protobuf_MethodDescriptorProto;
typedef struct google_protobuf_FileOptions google_protobuf_FileOptions;
typedef struct google_protobuf_MessageOptions google_protobuf_MessageOptions;
typedef struct google_protobuf_FieldOptions google_protobuf_FieldOptions;
typedef struct google_protobuf_OneofOptions google_protobuf_OneofOptions;
typedef struct google_protobuf_EnumOptions google_protobuf_EnumOptions;
typedef struct google_protobuf_EnumValueOptions google_protobuf_EnumValueOptions;
typedef struct google_protobuf_ServiceOptions google_protobuf_ServiceOptions;
typedef struct google_protobuf_MethodOptions google_protobuf_MethodOptions;
typedef struct google_protobuf_UninterpretedOption google_protobuf_UninterpretedOption;
typedef struct google_protobuf_UninterpretedOption_NamePart google_protobuf_UninterpretedOption_NamePart;
typedef struct google_protobuf_SourceCodeInfo google_protobuf_SourceCodeInfo;
typedef struct google_protobuf_SourceCodeInfo_Location google_protobuf_SourceCodeInfo_Location;
typedef struct google_protobuf_GeneratedCodeInfo google_protobuf_GeneratedCodeInfo;
typedef struct google_protobuf_GeneratedCodeInfo_Annotation google_protobuf_GeneratedCodeInfo_Annotation;
extern const upb_msglayout google_protobuf_FileDescriptorSet_msginit;
extern const upb_msglayout google_protobuf_FileDescriptorProto_msginit;
extern const upb_msglayout google_protobuf_DescriptorProto_msginit;
extern const upb_msglayout google_protobuf_DescriptorProto_ExtensionRange_msginit;
extern const upb_msglayout google_protobuf_DescriptorProto_ReservedRange_msginit;
extern const upb_msglayout google_protobuf_ExtensionRangeOptions_msginit;
extern const upb_msglayout google_protobuf_FieldDescriptorProto_msginit;
extern const upb_msglayout google_protobuf_OneofDescriptorProto_msginit;
extern const upb_msglayout google_protobuf_EnumDescriptorProto_msginit;
extern const upb_msglayout google_protobuf_EnumDescriptorProto_EnumReservedRange_msginit;
extern const upb_msglayout google_protobuf_EnumValueDescriptorProto_msginit;
extern const upb_msglayout google_protobuf_ServiceDescriptorProto_msginit;
extern const upb_msglayout google_protobuf_MethodDescriptorProto_msginit;
extern const upb_msglayout google_protobuf_FileOptions_msginit;
extern const upb_msglayout google_protobuf_MessageOptions_msginit;
extern const upb_msglayout google_protobuf_FieldOptions_msginit;
extern const upb_msglayout google_protobuf_OneofOptions_msginit;
extern const upb_msglayout google_protobuf_EnumOptions_msginit;
extern const upb_msglayout google_protobuf_EnumValueOptions_msginit;
extern const upb_msglayout google_protobuf_ServiceOptions_msginit;
extern const upb_msglayout google_protobuf_MethodOptions_msginit;
extern const upb_msglayout google_protobuf_UninterpretedOption_msginit;
extern const upb_msglayout google_protobuf_UninterpretedOption_NamePart_msginit;
extern const upb_msglayout google_protobuf_SourceCodeInfo_msginit;
extern const upb_msglayout google_protobuf_SourceCodeInfo_Location_msginit;
extern const upb_msglayout google_protobuf_GeneratedCodeInfo_msginit;
extern const upb_msglayout google_protobuf_GeneratedCodeInfo_Annotation_msginit;

typedef enum {
  google_protobuf_FieldDescriptorProto_LABEL_OPTIONAL = 1,
  google_protobuf_FieldDescriptorProto_LABEL_REQUIRED = 2,
  google_protobuf_FieldDescriptorProto_LABEL_REPEATED = 3
} google_protobuf_FieldDescriptorProto_Label;

typedef enum {
  google_protobuf_FieldDescriptorProto_TYPE_DOUBLE = 1,
  google_protobuf_FieldDescriptorProto_TYPE_FLOAT = 2,
  google_protobuf_FieldDescriptorProto_TYPE_INT64 = 3,
  google_protobuf_FieldDescriptorProto_TYPE_UINT64 = 4,
  google_protobuf_FieldDescriptorProto_TYPE_INT32 = 5,
  google_protobuf_FieldDescriptorProto_TYPE_FIXED64 = 6,
  google_protobuf_FieldDescriptorProto_TYPE_FIXED32 = 7,
  google_protobuf_FieldDescriptorProto_TYPE_BOOL = 8,
  google_protobuf_FieldDescriptorProto_TYPE_STRING = 9,
  google_protobuf_FieldDescriptorProto_TYPE_GROUP = 10,
  google_protobuf_FieldDescriptorProto_TYPE_MESSAGE = 11,
  google_protobuf_FieldDescriptorProto_TYPE_BYTES = 12,
  google_protobuf_FieldDescriptorProto_TYPE_UINT32 = 13,
  google_protobuf_FieldDescriptorProto_TYPE_ENUM = 14,
  google_protobuf_FieldDescriptorProto_TYPE_SFIXED32 = 15,
  google_protobuf_FieldDescriptorProto_TYPE_SFIXED64 = 16,
  google_protobuf_FieldDescriptorProto_TYPE_SINT32 = 17,
  google_protobuf_FieldDescriptorProto_TYPE_SINT64 = 18
} google_protobuf_FieldDescriptorProto_Type;

typedef enum {
  google_protobuf_FieldOptions_STRING = 0,
  google_protobuf_FieldOptions_CORD = 1,
  google_protobuf_FieldOptions_STRING_PIECE = 2
} google_protobuf_FieldOptions_CType;

typedef enum {
  google_protobuf_FieldOptions_JS_NORMAL = 0,
  google_protobuf_FieldOptions_JS_STRING = 1,
  google_protobuf_FieldOptions_JS_NUMBER = 2
} google_protobuf_FieldOptions_JSType;

typedef enum {
  google_protobuf_FileOptions_SPEED = 1,
  google_protobuf_FileOptions_CODE_SIZE = 2,
  google_protobuf_FileOptions_LITE_RUNTIME = 3
} google_protobuf_FileOptions_OptimizeMode;

typedef enum {
  google_protobuf_MethodOptions_IDEMPOTENCY_UNKNOWN = 0,
  google_protobuf_MethodOptions_NO_SIDE_EFFECTS = 1,
  google_protobuf_MethodOptions_IDEMPOTENT = 2
} google_protobuf_MethodOptions_IdempotencyLevel;


/* google.protobuf.FileDescriptorSet */

UPB_INLINE google_protobuf_FileDescriptorSet *google_protobuf_FileDescriptorSet_new(upb_arena *arena) {
  return (google_protobuf_FileDescriptorSet *)_upb_msg_new(&google_protobuf_FileDescriptorSet_msginit, arena);
}
UPB_INLINE google_protobuf_FileDescriptorSet *google_protobuf_FileDescriptorSet_parse(const char *buf, size_t size,
                        upb_arena *arena) {
  google_protobuf_FileDescriptorSet *ret = google_protobuf_FileDescriptorSet_new(arena);
  if (!ret) return NULL;
  if (!upb_decode(buf, size, ret, &google_protobuf_FileDescriptorSet_msginit, arena)) return NULL;
  return ret;
}
UPB_INLINE google_protobuf_FileDescriptorSet *google_protobuf_FileDescriptorSet_parse_ex(const char *buf, size_t size,
                           const upb_extreg *extreg, int options,
                           upb_arena *arena) {
  google_protobuf_FileDescriptorSet *ret = google_protobuf_FileDescriptorSet_new(arena);
  if (!ret) return NULL;
  if (!_upb_decode(buf, size, ret, &google_protobuf_FileDescriptorSet_msginit, extreg, options, arena)) {
    return NULL;
  }
  return ret;
}
UPB_INLINE char *google_protobuf_FileDescriptorSet_serialize(const google_protobuf_FileDescriptorSet *msg, upb_arena *arena, size_t *len) {
  return upb_encode(msg, &google_protobuf_FileDescriptorSet_msginit, arena, len);
}

UPB_INLINE bool google_protobuf_FileDescriptorSet_has_file(const google_protobuf_FileDescriptorSet *msg) { return _upb_has_submsg_nohasbit(msg, UPB_SIZE(0, 0)); }
UPB_INLINE const google_protobuf_FileDescriptorProto* const* google_protobuf_FileDescriptorSet_file(const google_protobuf_FileDescriptorSet *msg, size_t *len) { return (const google_protobuf_FileDescriptorProto* const*)_upb_array_accessor(msg, UPB_SIZE(0, 0), len); }

UPB_INLINE google_protobuf_FileDescriptorProto** google_protobuf_FileDescriptorSet_mutable_file(google_protobuf_FileDescriptorSet *msg, size_t *len) {
  return (google_protobuf_FileDescriptorProto**)_upb_array_mutable_accessor(msg, UPB_SIZE(0, 0), len);
}
UPB_INLINE google_protobuf_FileDescriptorProto** google_protobuf_FileDescriptorSet_resize_file(google_protobuf_FileDescriptorSet *msg, size_t len, upb_arena *arena) {
  return (google_protobuf_FileDescriptorProto**)_upb_array_resize_accessor2(msg, UPB_SIZE(0, 0), len, UPB_SIZE(2, 3), arena);
}
UPB_INLINE struct google_protobuf_FileDescriptorProto* google_protobuf_FileDescriptorSet_add_file(google_protobuf_FileDescriptorSet *msg, upb_arena *arena) {
  struct google_protobuf_FileDescriptorProto* sub = (struct google_protobuf_FileDescriptorProto*)_upb_msg_new(&google_protobuf_FileDescriptorProto_msginit, arena);
  bool ok = _upb_array_append_accessor2(
      msg, UPB_SIZE(0, 0), UPB_SIZE(2, 3), &sub, arena);
  if (!ok) return NULL;
  return sub;
}

/* google.protobuf.FileDescriptorProto */

UPB_INLINE google_protobuf_FileDescriptorProto *google_protobuf_FileDescriptorProto_new(upb_arena *arena) {
  return (google_protobuf_FileDescriptorProto *)_upb_msg_new(&google_protobuf_FileDescriptorProto_msginit, arena);
}
UPB_INLINE google_protobuf_FileDescriptorProto *google_protobuf_FileDescriptorProto_parse(const char *buf, size_t size,
                        upb_arena *arena) {
  google_protobuf_FileDescriptorProto *ret = google_protobuf_FileDescriptorProto_new(arena);
  if (!ret) return NULL;
  if (!upb_decode(buf, size, ret, &google_protobuf_FileDescriptorProto_msginit, arena)) return NULL;
  return ret;
}
UPB_INLINE google_protobuf_FileDescriptorProto *google_protobuf_FileDescriptorProto_parse_ex(const char *buf, size_t size,
                           const upb_extreg *extreg, int options,
                           upb_arena *arena) {
  google_protobuf_FileDescriptorProto *ret = google_protobuf_FileDescriptorProto_new(arena);
  if (!ret) return NULL;
  if (!_upb_decode(buf, size, ret, &google_protobuf_FileDescriptorProto_msginit, extreg, options, arena)) {
    return NULL;
  }
  return ret;
}
UPB_INLINE char *google_protobuf_FileDescriptorProto_serialize(const google_protobuf_FileDescriptorProto *msg, upb_arena *arena, size_t *len) {
  return upb_encode(msg, &google_protobuf_FileDescriptorProto_msginit, arena, len);
}

UPB_INLINE bool google_protobuf_FileDescriptorProto_has_name(const google_protobuf_FileDescriptorProto *msg) { return _upb_hasbit(msg, 1); }
UPB_INLINE upb_strview google_protobuf_FileDescriptorProto_name(const google_protobuf_FileDescriptorProto *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(4, 8), upb_strview); }
UPB_INLINE bool google_protobuf_FileDescriptorProto_has_package(const google_protobuf_FileDescriptorProto *msg) { return _upb_hasbit(msg, 2); }
UPB_INLINE upb_strview google_protobuf_FileDescriptorProto_package(const google_protobuf_FileDescriptorProto *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(12, 24), upb_strview); }
UPB_INLINE upb_strview const* google_protobuf_FileDescriptorProto_dependency(const google_protobuf_FileDescriptorProto *msg, size_t *len) { return (upb_strview const*)_upb_array_accessor(msg, UPB_SIZE(36, 72), len); }
UPB_INLINE bool google_protobuf_FileDescriptorProto_has_message_type(const google_protobuf_FileDescriptorProto *msg) { return _upb_has_submsg_nohasbit(msg, UPB_SIZE(40, 80)); }
UPB_INLINE const google_protobuf_DescriptorProto* const* google_protobuf_FileDescriptorProto_message_type(const google_protobuf_FileDescriptorProto *msg, size_t *len) { return (const google_protobuf_DescriptorProto* const*)_upb_array_accessor(msg, UPB_SIZE(40, 80), len); }
UPB_INLINE bool google_protobuf_FileDescriptorProto_has_enum_type(const google_protobuf_FileDescriptorProto *msg) { return _upb_has_submsg_nohasbit(msg, UPB_SIZE(44, 88)); }
UPB_INLINE const google_protobuf_EnumDescriptorProto* const* google_protobuf_FileDescriptorProto_enum_type(const google_protobuf_FileDescriptorProto *msg, size_t *len) { return (const google_protobuf_EnumDescriptorProto* const*)_upb_array_accessor(msg, UPB_SIZE(44, 88), len); }
UPB_INLINE bool google_protobuf_FileDescriptorProto_has_service(const google_protobuf_FileDescriptorProto *msg) { return _upb_has_submsg_nohasbit(msg, UPB_SIZE(48, 96)); }
UPB_INLINE const google_protobuf_ServiceDescriptorProto* const* google_protobuf_FileDescriptorProto_service(const google_protobuf_FileDescriptorProto *msg, size_t *len) { return (const google_protobuf_ServiceDescriptorProto* const*)_upb_array_accessor(msg, UPB_SIZE(48, 96), len); }
UPB_INLINE bool google_protobuf_FileDescriptorProto_has_extension(const google_protobuf_FileDescriptorProto *msg) { return _upb_has_submsg_nohasbit(msg, UPB_SIZE(52, 104)); }
UPB_INLINE const google_protobuf_FieldDescriptorProto* const* google_protobuf_FileDescriptorProto_extension(const google_protobuf_FileDescriptorProto *msg, size_t *len) { return (const google_protobuf_FieldDescriptorProto* const*)_upb_array_accessor(msg, UPB_SIZE(52, 104), len); }
UPB_INLINE bool google_protobuf_FileDescriptorProto_has_options(const google_protobuf_FileDescriptorProto *msg) { return _upb_hasbit(msg, 3); }
UPB_INLINE const google_protobuf_FileOptions* google_protobuf_FileDescriptorProto_options(const google_protobuf_FileDescriptorProto *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(28, 56), const google_protobuf_FileOptions*); }
UPB_INLINE bool google_protobuf_FileDescriptorProto_has_source_code_info(const google_protobuf_FileDescriptorProto *msg) { return _upb_hasbit(msg, 4); }
UPB_INLINE const google_protobuf_SourceCodeInfo* google_protobuf_FileDescriptorProto_source_code_info(const google_protobuf_FileDescriptorProto *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(32, 64), const google_protobuf_SourceCodeInfo*); }
UPB_INLINE int32_t const* google_protobuf_FileDescriptorProto_public_dependency(const google_protobuf_FileDescriptorProto *msg, size_t *len) { return (int32_t const*)_upb_array_accessor(msg, UPB_SIZE(56, 112), len); }
UPB_INLINE int32_t const* google_protobuf_FileDescriptorProto_weak_dependency(const google_protobuf_FileDescriptorProto *msg, size_t *len) { return (int32_t const*)_upb_array_accessor(msg, UPB_SIZE(60, 120), len); }
UPB_INLINE bool google_protobuf_FileDescriptorProto_has_syntax(const google_protobuf_FileDescriptorProto *msg) { return _upb_hasbit(msg, 5); }
UPB_INLINE upb_strview google_protobuf_FileDescriptorProto_syntax(const google_protobuf_FileDescriptorProto *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(20, 40), upb_strview); }

UPB_INLINE void google_protobuf_FileDescriptorProto_set_name(google_protobuf_FileDescriptorProto *msg, upb_strview value) {
  _upb_sethas(msg, 1);
  *UPB_PTR_AT(msg, UPB_SIZE(4, 8), upb_strview) = value;
}
UPB_INLINE void google_protobuf_FileDescriptorProto_set_package(google_protobuf_FileDescriptorProto *msg, upb_strview value) {
  _upb_sethas(msg, 2);
  *UPB_PTR_AT(msg, UPB_SIZE(12, 24), upb_strview) = value;
}
UPB_INLINE upb_strview* google_protobuf_FileDescriptorProto_mutable_dependency(google_protobuf_FileDescriptorProto *msg, size_t *len) {
  return (upb_strview*)_upb_array_mutable_accessor(msg, UPB_SIZE(36, 72), len);
}
UPB_INLINE upb_strview* google_protobuf_FileDescriptorProto_resize_dependency(google_protobuf_FileDescriptorProto *msg, size_t len, upb_arena *arena) {
  return (upb_strview*)_upb_array_resize_accessor2(msg, UPB_SIZE(36, 72), len, UPB_SIZE(3, 4), arena);
}
UPB_INLINE bool google_protobuf_FileDescriptorProto_add_dependency(google_protobuf_FileDescriptorProto *msg, upb_strview val, upb_arena *arena) {
  return _upb_array_append_accessor2(msg, UPB_SIZE(36, 72), UPB_SIZE(3, 4), &val,
      arena);
}
UPB_INLINE google_protobuf_DescriptorProto** google_protobuf_FileDescriptorProto_mutable_message_type(google_protobuf_FileDescriptorProto *msg, size_t *len) {
  return (google_protobuf_DescriptorProto**)_upb_array_mutable_accessor(msg, UPB_SIZE(40, 80), len);
}
UPB_INLINE google_protobuf_DescriptorProto** google_protobuf_FileDescriptorProto_resize_message_type(google_protobuf_FileDescriptorProto *msg, size_t len, upb_arena *arena) {
  return (google_protobuf_DescriptorProto**)_upb_array_resize_accessor2(msg, UPB_SIZE(40, 80), len, UPB_SIZE(2, 3), arena);
}
UPB_INLINE struct google_protobuf_DescriptorProto* google_protobuf_FileDescriptorProto_add_message_type(google_protobuf_FileDescriptorProto *msg, upb_arena *arena) {
  struct google_protobuf_DescriptorProto* sub = (struct google_protobuf_DescriptorProto*)_upb_msg_new(&google_protobuf_DescriptorProto_msginit, arena);
  bool ok = _upb_array_append_accessor2(
      msg, UPB_SIZE(40, 80), UPB_SIZE(2, 3), &sub, arena);
  if (!ok) return NULL;
  return sub;
}
UPB_INLINE google_protobuf_EnumDescriptorProto** google_protobuf_FileDescriptorProto_mutable_enum_type(google_protobuf_FileDescriptorProto *msg, size_t *len) {
  return (google_protobuf_EnumDescriptorProto**)_upb_array_mutable_accessor(msg, UPB_SIZE(44, 88), len);
}
UPB_INLINE google_protobuf_EnumDescriptorProto** google_protobuf_FileDescriptorProto_resize_enum_type(google_protobuf_FileDescriptorProto *msg, size_t len, upb_arena *arena) {
  return (google_protobuf_EnumDescriptorProto**)_upb_array_resize_accessor2(msg, UPB_SIZE(44, 88), len, UPB_SIZE(2, 3), arena);
}
UPB_INLINE struct google_protobuf_EnumDescriptorProto* google_protobuf_FileDescriptorProto_add_enum_type(google_protobuf_FileDescriptorProto *msg, upb_arena *arena) {
  struct google_protobuf_EnumDescriptorProto* sub = (struct google_protobuf_EnumDescriptorProto*)_upb_msg_new(&google_protobuf_EnumDescriptorProto_msginit, arena);
  bool ok = _upb_array_append_accessor2(
      msg, UPB_SIZE(44, 88), UPB_SIZE(2, 3), &sub, arena);
  if (!ok) return NULL;
  return sub;
}
UPB_INLINE google_protobuf_ServiceDescriptorProto** google_protobuf_FileDescriptorProto_mutable_service(google_protobuf_FileDescriptorProto *msg, size_t *len) {
  return (google_protobuf_ServiceDescriptorProto**)_upb_array_mutable_accessor(msg, UPB_SIZE(48, 96), len);
}
UPB_INLINE google_protobuf_ServiceDescriptorProto** google_protobuf_FileDescriptorProto_resize_service(google_protobuf_FileDescriptorProto *msg, size_t len, upb_arena *arena) {
  return (google_protobuf_ServiceDescriptorProto**)_upb_array_resize_accessor2(msg, UPB_SIZE(48, 96), len, UPB_SIZE(2, 3), arena);
}
UPB_INLINE struct google_protobuf_ServiceDescriptorProto* google_protobuf_FileDescriptorProto_add_service(google_protobuf_FileDescriptorProto *msg, upb_arena *arena) {
  struct google_protobuf_ServiceDescriptorProto* sub = (struct google_protobuf_ServiceDescriptorProto*)_upb_msg_new(&google_protobuf_ServiceDescriptorProto_msginit, arena);
  bool ok = _upb_array_append_accessor2(
      msg, UPB_SIZE(48, 96), UPB_SIZE(2, 3), &sub, arena);
  if (!ok) return NULL;
  return sub;
}
UPB_INLINE google_protobuf_FieldDescriptorProto** google_protobuf_FileDescriptorProto_mutable_extension(google_protobuf_FileDescriptorProto *msg, size_t *len) {
  return (google_protobuf_FieldDescriptorProto**)_upb_array_mutable_accessor(msg, UPB_SIZE(52, 104), len);
}
UPB_INLINE google_protobuf_FieldDescriptorProto** google_protobuf_FileDescriptorProto_resize_extension(google_protobuf_FileDescriptorProto *msg, size_t len, upb_arena *arena) {
  return (google_protobuf_FieldDescriptorProto**)_upb_array_resize_accessor2(msg, UPB_SIZE(52, 104), len, UPB_SIZE(2, 3), arena);
}
UPB_INLINE struct google_protobuf_FieldDescriptorProto* google_protobuf_FileDescriptorProto_add_extension(google_protobuf_FileDescriptorProto *msg, upb_arena *arena) {
  struct google_protobuf_FieldDescriptorProto* sub = (struct google_protobuf_FieldDescriptorProto*)_upb_msg_new(&google_protobuf_FieldDescriptorProto_msginit, arena);
  bool ok = _upb_array_append_accessor2(
      msg, UPB_SIZE(52, 104), UPB_SIZE(2, 3), &sub, arena);
  if (!ok) return NULL;
  return sub;
}
UPB_INLINE void google_protobuf_FileDescriptorProto_set_options(google_protobuf_FileDescriptorProto *msg, google_protobuf_FileOptions* value) {
  _upb_sethas(msg, 3);
  *UPB_PTR_AT(msg, UPB_SIZE(28, 56), google_protobuf_FileOptions*) = value;
}
UPB_INLINE struct google_protobuf_FileOptions* google_protobuf_FileDescriptorProto_mutable_options(google_protobuf_FileDescriptorProto *msg, upb_arena *arena) {
  struct google_protobuf_FileOptions* sub = (struct google_protobuf_FileOptions*)google_protobuf_FileDescriptorProto_options(msg);
  if (sub == NULL) {
    sub = (struct google_protobuf_FileOptions*)_upb_msg_new(&google_protobuf_FileOptions_msginit, arena);
    if (!sub) return NULL;
    google_protobuf_FileDescriptorProto_set_options(msg, sub);
  }
  return sub;
}
UPB_INLINE void google_protobuf_FileDescriptorProto_set_source_code_info(google_protobuf_FileDescriptorProto *msg, google_protobuf_SourceCodeInfo* value) {
  _upb_sethas(msg, 4);
  *UPB_PTR_AT(msg, UPB_SIZE(32, 64), google_protobuf_SourceCodeInfo*) = value;
}
UPB_INLINE struct google_protobuf_SourceCodeInfo* google_protobuf_FileDescriptorProto_mutable_source_code_info(google_protobuf_FileDescriptorProto *msg, upb_arena *arena) {
  struct google_protobuf_SourceCodeInfo* sub = (struct google_protobuf_SourceCodeInfo*)google_protobuf_FileDescriptorProto_source_code_info(msg);
  if (sub == NULL) {
    sub = (struct google_protobuf_SourceCodeInfo*)_upb_msg_new(&google_protobuf_SourceCodeInfo_msginit, arena);
    if (!sub) return NULL;
    google_protobuf_FileDescriptorProto_set_source_code_info(msg, sub);
  }
  return sub;
}
UPB_INLINE int32_t* google_protobuf_FileDescriptorProto_mutable_public_dependency(google_protobuf_FileDescriptorProto *msg, size_t *len) {
  return (int32_t*)_upb_array_mutable_accessor(msg, UPB_SIZE(56, 112), len);
}
UPB_INLINE int32_t* google_protobuf_FileDescriptorProto_resize_public_dependency(google_protobuf_FileDescriptorProto *msg, size_t len, upb_arena *arena) {
  return (int32_t*)_upb_array_resize_accessor2(msg, UPB_SIZE(56, 112), len, 2, arena);
}
UPB_INLINE bool google_protobuf_FileDescriptorProto_add_public_dependency(google_protobuf_FileDescriptorProto *msg, int32_t val, upb_arena *arena) {
  return _upb_array_append_accessor2(msg, UPB_SIZE(56, 112), 2, &val,
      arena);
}
UPB_INLINE int32_t* google_protobuf_FileDescriptorProto_mutable_weak_dependency(google_protobuf_FileDescriptorProto *msg, size_t *len) {
  return (int32_t*)_upb_array_mutable_accessor(msg, UPB_SIZE(60, 120), len);
}
UPB_INLINE int32_t* google_protobuf_FileDescriptorProto_resize_weak_dependency(google_protobuf_FileDescriptorProto *msg, size_t len, upb_arena *arena) {
  return (int32_t*)_upb_array_resize_accessor2(msg, UPB_SIZE(60, 120), len, 2, arena);
}
UPB_INLINE bool google_protobuf_FileDescriptorProto_add_weak_dependency(google_protobuf_FileDescriptorProto *msg, int32_t val, upb_arena *arena) {
  return _upb_array_append_accessor2(msg, UPB_SIZE(60, 120), 2, &val,
      arena);
}
UPB_INLINE void google_protobuf_FileDescriptorProto_set_syntax(google_protobuf_FileDescriptorProto *msg, upb_strview value) {
  _upb_sethas(msg, 5);
  *UPB_PTR_AT(msg, UPB_SIZE(20, 40), upb_strview) = value;
}

/* google.protobuf.DescriptorProto */

UPB_INLINE google_protobuf_DescriptorProto *google_protobuf_DescriptorProto_new(upb_arena *arena) {
  return (google_protobuf_DescriptorProto *)_upb_msg_new(&google_protobuf_DescriptorProto_msginit, arena);
}
UPB_INLINE google_protobuf_DescriptorProto *google_protobuf_DescriptorProto_parse(const char *buf, size_t size,
                        upb_arena *arena) {
  google_protobuf_DescriptorProto *ret = google_protobuf_DescriptorProto_new(arena);
  if (!ret) return NULL;
  if (!upb_decode(buf, size, ret, &google_protobuf_DescriptorProto_msginit, arena)) return NULL;
  return ret;
}
UPB_INLINE google_protobuf_DescriptorProto *google_protobuf_DescriptorProto_parse_ex(const char *buf, size_t size,
                           const upb_extreg *extreg, int options,
                           upb_arena *arena) {
  google_protobuf_DescriptorProto *ret = google_protobuf_DescriptorProto_new(arena);
  if (!ret) return NULL;
  if (!_upb_decode(buf, size, ret, &google_protobuf_DescriptorProto_msginit, extreg, options, arena)) {
    return NULL;
  }
  return ret;
}
UPB_INLINE char *google_protobuf_DescriptorProto_serialize(const google_protobuf_DescriptorProto *msg, upb_arena *arena, size_t *len) {
  return upb_encode(msg, &google_protobuf_DescriptorProto_msginit, arena, len);
}

UPB_INLINE bool google_protobuf_DescriptorProto_has_name(const google_protobuf_DescriptorProto *msg) { return _upb_hasbit(msg, 1); }
UPB_INLINE upb_strview google_protobuf_DescriptorProto_name(const google_protobuf_DescriptorProto *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(4, 8), upb_strview); }
UPB_INLINE bool google_protobuf_DescriptorProto_has_field(const google_protobuf_DescriptorProto *msg) { return _upb_has_submsg_nohasbit(msg, UPB_SIZE(16, 32)); }
UPB_INLINE const google_protobuf_FieldDescriptorProto* const* google_protobuf_DescriptorProto_field(const google_protobuf_DescriptorProto *msg, size_t *len) { return (const google_protobuf_FieldDescriptorProto* const*)_upb_array_accessor(msg, UPB_SIZE(16, 32), len); }
UPB_INLINE bool google_protobuf_DescriptorProto_has_nested_type(const google_protobuf_DescriptorProto *msg) { return _upb_has_submsg_nohasbit(msg, UPB_SIZE(20, 40)); }
UPB_INLINE const google_protobuf_DescriptorProto* const* google_protobuf_DescriptorProto_nested_type(const google_protobuf_DescriptorProto *msg, size_t *len) { return (const google_protobuf_DescriptorProto* const*)_upb_array_accessor(msg, UPB_SIZE(20, 40), len); }
UPB_INLINE bool google_protobuf_DescriptorProto_has_enum_type(const google_protobuf_DescriptorProto *msg) { return _upb_has_submsg_nohasbit(msg, UPB_SIZE(24, 48)); }
UPB_INLINE const google_protobuf_EnumDescriptorProto* const* google_protobuf_DescriptorProto_enum_type(const google_protobuf_DescriptorProto *msg, size_t *len) { return (const google_protobuf_EnumDescriptorProto* const*)_upb_array_accessor(msg, UPB_SIZE(24, 48), len); }
UPB_INLINE bool google_protobuf_DescriptorProto_has_extension_range(const google_protobuf_DescriptorProto *msg) { return _upb_has_submsg_nohasbit(msg, UPB_SIZE(28, 56)); }
UPB_INLINE const google_protobuf_DescriptorProto_ExtensionRange* const* google_protobuf_DescriptorProto_extension_range(const google_protobuf_DescriptorProto *msg, size_t *len) { return (const google_protobuf_DescriptorProto_ExtensionRange* const*)_upb_array_accessor(msg, UPB_SIZE(28, 56), len); }
UPB_INLINE bool google_protobuf_DescriptorProto_has_extension(const google_protobuf_DescriptorProto *msg) { return _upb_has_submsg_nohasbit(msg, UPB_SIZE(32, 64)); }
UPB_INLINE const google_protobuf_FieldDescriptorProto* const* google_protobuf_DescriptorProto_extension(const google_protobuf_DescriptorProto *msg, size_t *len) { return (const google_protobuf_FieldDescriptorProto* const*)_upb_array_accessor(msg, UPB_SIZE(32, 64), len); }
UPB_INLINE bool google_protobuf_DescriptorProto_has_options(const google_protobuf_DescriptorProto *msg) { return _upb_hasbit(msg, 2); }
UPB_INLINE const google_protobuf_MessageOptions* google_protobuf_DescriptorProto_options(const google_protobuf_DescriptorProto *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(12, 24), const google_protobuf_MessageOptions*); }
UPB_INLINE bool google_protobuf_DescriptorProto_has_oneof_decl(const google_protobuf_DescriptorProto *msg) { return _upb_has_submsg_nohasbit(msg, UPB_SIZE(36, 72)); }
UPB_INLINE const google_protobuf_OneofDescriptorProto* const* google_protobuf_DescriptorProto_oneof_decl(const google_protobuf_DescriptorProto *msg, size_t *len) { return (const google_protobuf_OneofDescriptorProto* const*)_upb_array_accessor(msg, UPB_SIZE(36, 72), len); }
UPB_INLINE bool google_protobuf_DescriptorProto_has_reserved_range(const google_protobuf_DescriptorProto *msg) { return _upb_has_submsg_nohasbit(msg, UPB_SIZE(40, 80)); }
UPB_INLINE const google_protobuf_DescriptorProto_ReservedRange* const* google_protobuf_DescriptorProto_reserved_range(const google_protobuf_DescriptorProto *msg, size_t *len) { return (const google_protobuf_DescriptorProto_ReservedRange* const*)_upb_array_accessor(msg, UPB_SIZE(40, 80), len); }
UPB_INLINE upb_strview const* google_protobuf_DescriptorProto_reserved_name(const google_protobuf_DescriptorProto *msg, size_t *len) { return (upb_strview const*)_upb_array_accessor(msg, UPB_SIZE(44, 88), len); }

UPB_INLINE void google_protobuf_DescriptorProto_set_name(google_protobuf_DescriptorProto *msg, upb_strview value) {
  _upb_sethas(msg, 1);
  *UPB_PTR_AT(msg, UPB_SIZE(4, 8), upb_strview) = value;
}
UPB_INLINE google_protobuf_FieldDescriptorProto** google_protobuf_DescriptorProto_mutable_field(google_protobuf_DescriptorProto *msg, size_t *len) {
  return (google_protobuf_FieldDescriptorProto**)_upb_array_mutable_accessor(msg, UPB_SIZE(16, 32), len);
}
UPB_INLINE google_protobuf_FieldDescriptorProto** google_protobuf_DescriptorProto_resize_field(google_protobuf_DescriptorProto *msg, size_t len, upb_arena *arena) {
  return (google_protobuf_FieldDescriptorProto**)_upb_array_resize_accessor2(msg, UPB_SIZE(16, 32), len, UPB_SIZE(2, 3), arena);
}
UPB_INLINE struct google_protobuf_FieldDescriptorProto* google_protobuf_DescriptorProto_add_field(google_protobuf_DescriptorProto *msg, upb_arena *arena) {
  struct google_protobuf_FieldDescriptorProto* sub = (struct google_protobuf_FieldDescriptorProto*)_upb_msg_new(&google_protobuf_FieldDescriptorProto_msginit, arena);
  bool ok = _upb_array_append_accessor2(
      msg, UPB_SIZE(16, 32), UPB_SIZE(2, 3), &sub, arena);
  if (!ok) return NULL;
  return sub;
}
UPB_INLINE google_protobuf_DescriptorProto** google_protobuf_DescriptorProto_mutable_nested_type(google_protobuf_DescriptorProto *msg, size_t *len) {
  return (google_protobuf_DescriptorProto**)_upb_array_mutable_accessor(msg, UPB_SIZE(20, 40), len);
}
UPB_INLINE google_protobuf_DescriptorProto** google_protobuf_DescriptorProto_resize_nested_type(google_protobuf_DescriptorProto *msg, size_t len, upb_arena *arena) {
  return (google_protobuf_DescriptorProto**)_upb_array_resize_accessor2(msg, UPB_SIZE(20, 40), len, UPB_SIZE(2, 3), arena);
}
UPB_INLINE struct google_protobuf_DescriptorProto* google_protobuf_DescriptorProto_add_nested_type(google_protobuf_DescriptorProto *msg, upb_arena *arena) {
  struct google_protobuf_DescriptorProto* sub = (struct google_protobuf_DescriptorProto*)_upb_msg_new(&google_protobuf_DescriptorProto_msginit, arena);
  bool ok = _upb_array_append_accessor2(
      msg, UPB_SIZE(20, 40), UPB_SIZE(2, 3), &sub, arena);
  if (!ok) return NULL;
  return sub;
}
UPB_INLINE google_protobuf_EnumDescriptorProto** google_protobuf_DescriptorProto_mutable_enum_type(google_protobuf_DescriptorProto *msg, size_t *len) {
  return (google_protobuf_EnumDescriptorProto**)_upb_array_mutable_accessor(msg, UPB_SIZE(24, 48), len);
}
UPB_INLINE google_protobuf_EnumDescriptorProto** google_protobuf_DescriptorProto_resize_enum_type(google_protobuf_DescriptorProto *msg, size_t len, upb_arena *arena) {
  return (google_protobuf_EnumDescriptorProto**)_upb_array_resize_accessor2(msg, UPB_SIZE(24, 48), len, UPB_SIZE(2, 3), arena);
}
UPB_INLINE struct google_protobuf_EnumDescriptorProto* google_protobuf_DescriptorProto_add_enum_type(google_protobuf_DescriptorProto *msg, upb_arena *arena) {
  struct google_protobuf_EnumDescriptorProto* sub = (struct google_protobuf_EnumDescriptorProto*)_upb_msg_new(&google_protobuf_EnumDescriptorProto_msginit, arena);
  bool ok = _upb_array_append_accessor2(
      msg, UPB_SIZE(24, 48), UPB_SIZE(2, 3), &sub, arena);
  if (!ok) return NULL;
  return sub;
}
UPB_INLINE google_protobuf_DescriptorProto_ExtensionRange** google_protobuf_DescriptorProto_mutable_extension_range(google_protobuf_DescriptorProto *msg, size_t *len) {
  return (google_protobuf_DescriptorProto_ExtensionRange**)_upb_array_mutable_accessor(msg, UPB_SIZE(28, 56), len);
}
UPB_INLINE google_protobuf_DescriptorProto_ExtensionRange** google_protobuf_DescriptorProto_resize_extension_range(google_protobuf_DescriptorProto *msg, size_t len, upb_arena *arena) {
  return (google_protobuf_DescriptorProto_ExtensionRange**)_upb_array_resize_accessor2(msg, UPB_SIZE(28, 56), len, UPB_SIZE(2, 3), arena);
}
UPB_INLINE struct google_protobuf_DescriptorProto_ExtensionRange* google_protobuf_DescriptorProto_add_extension_range(google_protobuf_DescriptorProto *msg, upb_arena *arena) {
  struct google_protobuf_DescriptorProto_ExtensionRange* sub = (struct google_protobuf_DescriptorProto_ExtensionRange*)_upb_msg_new(&google_protobuf_DescriptorProto_ExtensionRange_msginit, arena);
  bool ok = _upb_array_append_accessor2(
      msg, UPB_SIZE(28, 56), UPB_SIZE(2, 3), &sub, arena);
  if (!ok) return NULL;
  return sub;
}
UPB_INLINE google_protobuf_FieldDescriptorProto** google_protobuf_DescriptorProto_mutable_extension(google_protobuf_DescriptorProto *msg, size_t *len) {
  return (google_protobuf_FieldDescriptorProto**)_upb_array_mutable_accessor(msg, UPB_SIZE(32, 64), len);
}
UPB_INLINE google_protobuf_FieldDescriptorProto** google_protobuf_DescriptorProto_resize_extension(google_protobuf_DescriptorProto *msg, size_t len, upb_arena *arena) {
  return (google_protobuf_FieldDescriptorProto**)_upb_array_resize_accessor2(msg, UPB_SIZE(32, 64), len, UPB_SIZE(2, 3), arena);
}
UPB_INLINE struct google_protobuf_FieldDescriptorProto* google_protobuf_DescriptorProto_add_extension(google_protobuf_DescriptorProto *msg, upb_arena *arena) {
  struct google_protobuf_FieldDescriptorProto* sub = (struct google_protobuf_FieldDescriptorProto*)_upb_msg_new(&google_protobuf_FieldDescriptorProto_msginit, arena);
  bool ok = _upb_array_append_accessor2(
      msg, UPB_SIZE(32, 64), UPB_SIZE(2, 3), &sub, arena);
  if (!ok) return NULL;
  return sub;
}
UPB_INLINE void google_protobuf_DescriptorProto_set_options(google_protobuf_DescriptorProto *msg, google_protobuf_MessageOptions* value) {
  _upb_sethas(msg, 2);
  *UPB_PTR_AT(msg, UPB_SIZE(12, 24), google_protobuf_MessageOptions*) = value;
}
UPB_INLINE struct google_protobuf_MessageOptions* google_protobuf_DescriptorProto_mutable_options(google_protobuf_DescriptorProto *msg, upb_arena *arena) {
  struct google_protobuf_MessageOptions* sub = (struct google_protobuf_MessageOptions*)google_protobuf_DescriptorProto_options(msg);
  if (sub == NULL) {
    sub = (struct google_protobuf_MessageOptions*)_upb_msg_new(&google_protobuf_MessageOptions_msginit, arena);
    if (!sub) return NULL;
    google_protobuf_DescriptorProto_set_options(msg, sub);
  }
  return sub;
}
UPB_INLINE google_protobuf_OneofDescriptorProto** google_protobuf_DescriptorProto_mutable_oneof_decl(google_protobuf_DescriptorProto *msg, size_t *len) {
  return (google_protobuf_OneofDescriptorProto**)_upb_array_mutable_accessor(msg, UPB_SIZE(36, 72), len);
}
UPB_INLINE google_protobuf_OneofDescriptorProto** google_protobuf_DescriptorProto_resize_oneof_decl(google_protobuf_DescriptorProto *msg, size_t len, upb_arena *arena) {
  return (google_protobuf_OneofDescriptorProto**)_upb_array_resize_accessor2(msg, UPB_SIZE(36, 72), len, UPB_SIZE(2, 3), arena);
}
UPB_INLINE struct google_protobuf_OneofDescriptorProto* google_protobuf_DescriptorProto_add_oneof_decl(google_protobuf_DescriptorProto *msg, upb_arena *arena) {
  struct google_protobuf_OneofDescriptorProto* sub = (struct google_protobuf_OneofDescriptorProto*)_upb_msg_new(&google_protobuf_OneofDescriptorProto_msginit, arena);
  bool ok = _upb_array_append_accessor2(
      msg, UPB_SIZE(36, 72), UPB_SIZE(2, 3), &sub, arena);
  if (!ok) return NULL;
  return sub;
}
UPB_INLINE google_protobuf_DescriptorProto_ReservedRange** google_protobuf_DescriptorProto_mutable_reserved_range(google_protobuf_DescriptorProto *msg, size_t *len) {
  return (google_protobuf_DescriptorProto_ReservedRange**)_upb_array_mutable_accessor(msg, UPB_SIZE(40, 80), len);
}
UPB_INLINE google_protobuf_DescriptorProto_ReservedRange** google_protobuf_DescriptorProto_resize_reserved_range(google_protobuf_DescriptorProto *msg, size_t len, upb_arena *arena) {
  return (google_protobuf_DescriptorProto_ReservedRange**)_upb_array_resize_accessor2(msg, UPB_SIZE(40, 80), len, UPB_SIZE(2, 3), arena);
}
UPB_INLINE struct google_protobuf_DescriptorProto_ReservedRange* google_protobuf_DescriptorProto_add_reserved_range(google_protobuf_DescriptorProto *msg, upb_arena *arena) {
  struct google_protobuf_DescriptorProto_ReservedRange* sub = (struct google_protobuf_DescriptorProto_ReservedRange*)_upb_msg_new(&google_protobuf_DescriptorProto_ReservedRange_msginit, arena);
  bool ok = _upb_array_append_accessor2(
      msg, UPB_SIZE(40, 80), UPB_SIZE(2, 3), &sub, arena);
  if (!ok) return NULL;
  return sub;
}
UPB_INLINE upb_strview* google_protobuf_DescriptorProto_mutable_reserved_name(google_protobuf_DescriptorProto *msg, size_t *len) {
  return (upb_strview*)_upb_array_mutable_accessor(msg, UPB_SIZE(44, 88), len);
}
UPB_INLINE upb_strview* google_protobuf_DescriptorProto_resize_reserved_name(google_protobuf_DescriptorProto *msg, size_t len, upb_arena *arena) {
  return (upb_strview*)_upb_array_resize_accessor2(msg, UPB_SIZE(44, 88), len, UPB_SIZE(3, 4), arena);
}
UPB_INLINE bool google_protobuf_DescriptorProto_add_reserved_name(google_protobuf_DescriptorProto *msg, upb_strview val, upb_arena *arena) {
  return _upb_array_append_accessor2(msg, UPB_SIZE(44, 88), UPB_SIZE(3, 4), &val,
      arena);
}

/* google.protobuf.DescriptorProto.ExtensionRange */

UPB_INLINE google_protobuf_DescriptorProto_ExtensionRange *google_protobuf_DescriptorProto_ExtensionRange_new(upb_arena *arena) {
  return (google_protobuf_DescriptorProto_ExtensionRange *)_upb_msg_new(&google_protobuf_DescriptorProto_ExtensionRange_msginit, arena);
}
UPB_INLINE google_protobuf_DescriptorProto_ExtensionRange *google_protobuf_DescriptorProto_ExtensionRange_parse(const char *buf, size_t size,
                        upb_arena *arena) {
  google_protobuf_DescriptorProto_ExtensionRange *ret = google_protobuf_DescriptorProto_ExtensionRange_new(arena);
  if (!ret) return NULL;
  if (!upb_decode(buf, size, ret, &google_protobuf_DescriptorProto_ExtensionRange_msginit, arena)) return NULL;
  return ret;
}
UPB_INLINE google_protobuf_DescriptorProto_ExtensionRange *google_protobuf_DescriptorProto_ExtensionRange_parse_ex(const char *buf, size_t size,
                           const upb_extreg *extreg, int options,
                           upb_arena *arena) {
  google_protobuf_DescriptorProto_ExtensionRange *ret = google_protobuf_DescriptorProto_ExtensionRange_new(arena);
  if (!ret) return NULL;
  if (!_upb_decode(buf, size, ret, &google_protobuf_DescriptorProto_ExtensionRange_msginit, extreg, options, arena)) {
    return NULL;
  }
  return ret;
}
UPB_INLINE char *google_protobuf_DescriptorProto_ExtensionRange_serialize(const google_protobuf_DescriptorProto_ExtensionRange *msg, upb_arena *arena, size_t *len) {
  return upb_encode(msg, &google_protobuf_DescriptorProto_ExtensionRange_msginit, arena, len);
}

UPB_INLINE bool google_protobuf_DescriptorProto_ExtensionRange_has_start(const google_protobuf_DescriptorProto_ExtensionRange *msg) { return _upb_hasbit(msg, 1); }
UPB_INLINE int32_t google_protobuf_DescriptorProto_ExtensionRange_start(const google_protobuf_DescriptorProto_ExtensionRange *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(4, 4), int32_t); }
UPB_INLINE bool google_protobuf_DescriptorProto_ExtensionRange_has_end(const google_protobuf_DescriptorProto_ExtensionRange *msg) { return _upb_hasbit(msg, 2); }
UPB_INLINE int32_t google_protobuf_DescriptorProto_ExtensionRange_end(const google_protobuf_DescriptorProto_ExtensionRange *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(8, 8), int32_t); }
UPB_INLINE bool google_protobuf_DescriptorProto_ExtensionRange_has_options(const google_protobuf_DescriptorProto_ExtensionRange *msg) { return _upb_hasbit(msg, 3); }
UPB_INLINE const google_protobuf_ExtensionRangeOptions* google_protobuf_DescriptorProto_ExtensionRange_options(const google_protobuf_DescriptorProto_ExtensionRange *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(12, 16), const google_protobuf_ExtensionRangeOptions*); }

UPB_INLINE void google_protobuf_DescriptorProto_ExtensionRange_set_start(google_protobuf_DescriptorProto_ExtensionRange *msg, int32_t value) {
  _upb_sethas(msg, 1);
  *UPB_PTR_AT(msg, UPB_SIZE(4, 4), int32_t) = value;
}
UPB_INLINE void google_protobuf_DescriptorProto_ExtensionRange_set_end(google_protobuf_DescriptorProto_ExtensionRange *msg, int32_t value) {
  _upb_sethas(msg, 2);
  *UPB_PTR_AT(msg, UPB_SIZE(8, 8), int32_t) = value;
}
UPB_INLINE void google_protobuf_DescriptorProto_ExtensionRange_set_options(google_protobuf_DescriptorProto_ExtensionRange *msg, google_protobuf_ExtensionRangeOptions* value) {
  _upb_sethas(msg, 3);
  *UPB_PTR_AT(msg, UPB_SIZE(12, 16), google_protobuf_ExtensionRangeOptions*) = value;
}
UPB_INLINE struct google_protobuf_ExtensionRangeOptions* google_protobuf_DescriptorProto_ExtensionRange_mutable_options(google_protobuf_DescriptorProto_ExtensionRange *msg, upb_arena *arena) {
  struct google_protobuf_ExtensionRangeOptions* sub = (struct google_protobuf_ExtensionRangeOptions*)google_protobuf_DescriptorProto_ExtensionRange_options(msg);
  if (sub == NULL) {
    sub = (struct google_protobuf_ExtensionRangeOptions*)_upb_msg_new(&google_protobuf_ExtensionRangeOptions_msginit, arena);
    if (!sub) return NULL;
    google_protobuf_DescriptorProto_ExtensionRange_set_options(msg, sub);
  }
  return sub;
}

/* google.protobuf.DescriptorProto.ReservedRange */

UPB_INLINE google_protobuf_DescriptorProto_ReservedRange *google_protobuf_DescriptorProto_ReservedRange_new(upb_arena *arena) {
  return (google_protobuf_DescriptorProto_ReservedRange *)_upb_msg_new(&google_protobuf_DescriptorProto_ReservedRange_msginit, arena);
}
UPB_INLINE google_protobuf_DescriptorProto_ReservedRange *google_protobuf_DescriptorProto_ReservedRange_parse(const char *buf, size_t size,
                        upb_arena *arena) {
  google_protobuf_DescriptorProto_ReservedRange *ret = google_protobuf_DescriptorProto_ReservedRange_new(arena);
  if (!ret) return NULL;
  if (!upb_decode(buf, size, ret, &google_protobuf_DescriptorProto_ReservedRange_msginit, arena)) return NULL;
  return ret;
}
UPB_INLINE google_protobuf_DescriptorProto_ReservedRange *google_protobuf_DescriptorProto_ReservedRange_parse_ex(const char *buf, size_t size,
                           const upb_extreg *extreg, int options,
                           upb_arena *arena) {
  google_protobuf_DescriptorProto_ReservedRange *ret = google_protobuf_DescriptorProto_ReservedRange_new(arena);
  if (!ret) return NULL;
  if (!_upb_decode(buf, size, ret, &google_protobuf_DescriptorProto_ReservedRange_msginit, extreg, options, arena)) {
    return NULL;
  }
  return ret;
}
UPB_INLINE char *google_protobuf_DescriptorProto_ReservedRange_serialize(const google_protobuf_DescriptorProto_ReservedRange *msg, upb_arena *arena, size_t *len) {
  return upb_encode(msg, &google_protobuf_DescriptorProto_ReservedRange_msginit, arena, len);
}

UPB_INLINE bool google_protobuf_DescriptorProto_ReservedRange_has_start(const google_protobuf_DescriptorProto_ReservedRange *msg) { return _upb_hasbit(msg, 1); }
UPB_INLINE int32_t google_protobuf_DescriptorProto_ReservedRange_start(const google_protobuf_DescriptorProto_ReservedRange *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(4, 4), int32_t); }
UPB_INLINE bool google_protobuf_DescriptorProto_ReservedRange_has_end(const google_protobuf_DescriptorProto_ReservedRange *msg) { return _upb_hasbit(msg, 2); }
UPB_INLINE int32_t google_protobuf_DescriptorProto_ReservedRange_end(const google_protobuf_DescriptorProto_ReservedRange *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(8, 8), int32_t); }

UPB_INLINE void google_protobuf_DescriptorProto_ReservedRange_set_start(google_protobuf_DescriptorProto_ReservedRange *msg, int32_t value) {
  _upb_sethas(msg, 1);
  *UPB_PTR_AT(msg, UPB_SIZE(4, 4), int32_t) = value;
}
UPB_INLINE void google_protobuf_DescriptorProto_ReservedRange_set_end(google_protobuf_DescriptorProto_ReservedRange *msg, int32_t value) {
  _upb_sethas(msg, 2);
  *UPB_PTR_AT(msg, UPB_SIZE(8, 8), int32_t) = value;
}

/* google.protobuf.ExtensionRangeOptions */

UPB_INLINE google_protobuf_ExtensionRangeOptions *google_protobuf_ExtensionRangeOptions_new(upb_arena *arena) {
  return (google_protobuf_ExtensionRangeOptions *)_upb_msg_new(&google_protobuf_ExtensionRangeOptions_msginit, arena);
}
UPB_INLINE google_protobuf_ExtensionRangeOptions *google_protobuf_ExtensionRangeOptions_parse(const char *buf, size_t size,
                        upb_arena *arena) {
  google_protobuf_ExtensionRangeOptions *ret = google_protobuf_ExtensionRangeOptions_new(arena);
  if (!ret) return NULL;
  if (!upb_decode(buf, size, ret, &google_protobuf_ExtensionRangeOptions_msginit, arena)) return NULL;
  return ret;
}
UPB_INLINE google_protobuf_ExtensionRangeOptions *google_protobuf_ExtensionRangeOptions_parse_ex(const char *buf, size_t size,
                           const upb_extreg *extreg, int options,
                           upb_arena *arena) {
  google_protobuf_ExtensionRangeOptions *ret = google_protobuf_ExtensionRangeOptions_new(arena);
  if (!ret) return NULL;
  if (!_upb_decode(buf, size, ret, &google_protobuf_ExtensionRangeOptions_msginit, extreg, options, arena)) {
    return NULL;
  }
  return ret;
}
UPB_INLINE char *google_protobuf_ExtensionRangeOptions_serialize(const google_protobuf_ExtensionRangeOptions *msg, upb_arena *arena, size_t *len) {
  return upb_encode(msg, &google_protobuf_ExtensionRangeOptions_msginit, arena, len);
}

UPB_INLINE bool google_protobuf_ExtensionRangeOptions_has_uninterpreted_option(const google_protobuf_ExtensionRangeOptions *msg) { return _upb_has_submsg_nohasbit(msg, UPB_SIZE(0, 0)); }
UPB_INLINE const google_protobuf_UninterpretedOption* const* google_protobuf_ExtensionRangeOptions_uninterpreted_option(const google_protobuf_ExtensionRangeOptions *msg, size_t *len) { return (const google_protobuf_UninterpretedOption* const*)_upb_array_accessor(msg, UPB_SIZE(0, 0), len); }

UPB_INLINE google_protobuf_UninterpretedOption** google_protobuf_ExtensionRangeOptions_mutable_uninterpreted_option(google_protobuf_ExtensionRangeOptions *msg, size_t *len) {
  return (google_protobuf_UninterpretedOption**)_upb_array_mutable_accessor(msg, UPB_SIZE(0, 0), len);
}
UPB_INLINE google_protobuf_UninterpretedOption** google_protobuf_ExtensionRangeOptions_resize_uninterpreted_option(google_protobuf_ExtensionRangeOptions *msg, size_t len, upb_arena *arena) {
  return (google_protobuf_UninterpretedOption**)_upb_array_resize_accessor2(msg, UPB_SIZE(0, 0), len, UPB_SIZE(2, 3), arena);
}
UPB_INLINE struct google_protobuf_UninterpretedOption* google_protobuf_ExtensionRangeOptions_add_uninterpreted_option(google_protobuf_ExtensionRangeOptions *msg, upb_arena *arena) {
  struct google_protobuf_UninterpretedOption* sub = (struct google_protobuf_UninterpretedOption*)_upb_msg_new(&google_protobuf_UninterpretedOption_msginit, arena);
  bool ok = _upb_array_append_accessor2(
      msg, UPB_SIZE(0, 0), UPB_SIZE(2, 3), &sub, arena);
  if (!ok) return NULL;
  return sub;
}

/* google.protobuf.FieldDescriptorProto */

UPB_INLINE google_protobuf_FieldDescriptorProto *google_protobuf_FieldDescriptorProto_new(upb_arena *arena) {
  return (google_protobuf_FieldDescriptorProto *)_upb_msg_new(&google_protobuf_FieldDescriptorProto_msginit, arena);
}
UPB_INLINE google_protobuf_FieldDescriptorProto *google_protobuf_FieldDescriptorProto_parse(const char *buf, size_t size,
                        upb_arena *arena) {
  google_protobuf_FieldDescriptorProto *ret = google_protobuf_FieldDescriptorProto_new(arena);
  if (!ret) return NULL;
  if (!upb_decode(buf, size, ret, &google_protobuf_FieldDescriptorProto_msginit, arena)) return NULL;
  return ret;
}
UPB_INLINE google_protobuf_FieldDescriptorProto *google_protobuf_FieldDescriptorProto_parse_ex(const char *buf, size_t size,
                           const upb_extreg *extreg, int options,
                           upb_arena *arena) {
  google_protobuf_FieldDescriptorProto *ret = google_protobuf_FieldDescriptorProto_new(arena);
  if (!ret) return NULL;
  if (!_upb_decode(buf, size, ret, &google_protobuf_FieldDescriptorProto_msginit, extreg, options, arena)) {
    return NULL;
  }
  return ret;
}
UPB_INLINE char *google_protobuf_FieldDescriptorProto_serialize(const google_protobuf_FieldDescriptorProto *msg, upb_arena *arena, size_t *len) {
  return upb_encode(msg, &google_protobuf_FieldDescriptorProto_msginit, arena, len);
}

UPB_INLINE bool google_protobuf_FieldDescriptorProto_has_name(const google_protobuf_FieldDescriptorProto *msg) { return _upb_hasbit(msg, 1); }
UPB_INLINE upb_strview google_protobuf_FieldDescriptorProto_name(const google_protobuf_FieldDescriptorProto *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(24, 24), upb_strview); }
UPB_INLINE bool google_protobuf_FieldDescriptorProto_has_extendee(const google_protobuf_FieldDescriptorProto *msg) { return _upb_hasbit(msg, 2); }
UPB_INLINE upb_strview google_protobuf_FieldDescriptorProto_extendee(const google_protobuf_FieldDescriptorProto *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(32, 40), upb_strview); }
UPB_INLINE bool google_protobuf_FieldDescriptorProto_has_number(const google_protobuf_FieldDescriptorProto *msg) { return _upb_hasbit(msg, 3); }
UPB_INLINE int32_t google_protobuf_FieldDescriptorProto_number(const google_protobuf_FieldDescriptorProto *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(12, 12), int32_t); }
UPB_INLINE bool google_protobuf_FieldDescriptorProto_has_label(const google_protobuf_FieldDescriptorProto *msg) { return _upb_hasbit(msg, 4); }
UPB_INLINE int32_t google_protobuf_FieldDescriptorProto_label(const google_protobuf_FieldDescriptorProto *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(4, 4), int32_t); }
UPB_INLINE bool google_protobuf_FieldDescriptorProto_has_type(const google_protobuf_FieldDescriptorProto *msg) { return _upb_hasbit(msg, 5); }
UPB_INLINE int32_t google_protobuf_FieldDescriptorProto_type(const google_protobuf_FieldDescriptorProto *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(8, 8), int32_t); }
UPB_INLINE bool google_protobuf_FieldDescriptorProto_has_type_name(const google_protobuf_FieldDescriptorProto *msg) { return _upb_hasbit(msg, 6); }
UPB_INLINE upb_strview google_protobuf_FieldDescriptorProto_type_name(const google_protobuf_FieldDescriptorProto *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(40, 56), upb_strview); }
UPB_INLINE bool google_protobuf_FieldDescriptorProto_has_default_value(const google_protobuf_FieldDescriptorProto *msg) { return _upb_hasbit(msg, 7); }
UPB_INLINE upb_strview google_protobuf_FieldDescriptorProto_default_value(const google_protobuf_FieldDescriptorProto *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(48, 72), upb_strview); }
UPB_INLINE bool google_protobuf_FieldDescriptorProto_has_options(const google_protobuf_FieldDescriptorProto *msg) { return _upb_hasbit(msg, 8); }
UPB_INLINE const google_protobuf_FieldOptions* google_protobuf_FieldDescriptorProto_options(const google_protobuf_FieldDescriptorProto *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(64, 104), const google_protobuf_FieldOptions*); }
UPB_INLINE bool google_protobuf_FieldDescriptorProto_has_oneof_index(const google_protobuf_FieldDescriptorProto *msg) { return _upb_hasbit(msg, 9); }
UPB_INLINE int32_t google_protobuf_FieldDescriptorProto_oneof_index(const google_protobuf_FieldDescriptorProto *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(16, 16), int32_t); }
UPB_INLINE bool google_protobuf_FieldDescriptorProto_has_json_name(const google_protobuf_FieldDescriptorProto *msg) { return _upb_hasbit(msg, 10); }
UPB_INLINE upb_strview google_protobuf_FieldDescriptorProto_json_name(const google_protobuf_FieldDescriptorProto *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(56, 88), upb_strview); }
UPB_INLINE bool google_protobuf_FieldDescriptorProto_has_proto3_optional(const google_protobuf_FieldDescriptorProto *msg) { return _upb_hasbit(msg, 11); }
UPB_INLINE bool google_protobuf_FieldDescriptorProto_proto3_optional(const google_protobuf_FieldDescriptorProto *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(20, 20), bool); }

UPB_INLINE void google_protobuf_FieldDescriptorProto_set_name(google_protobuf_FieldDescriptorProto *msg, upb_strview value) {
  _upb_sethas(msg, 1);
  *UPB_PTR_AT(msg, UPB_SIZE(24, 24), upb_strview) = value;
}
UPB_INLINE void google_protobuf_FieldDescriptorProto_set_extendee(google_protobuf_FieldDescriptorProto *msg, upb_strview value) {
  _upb_sethas(msg, 2);
  *UPB_PTR_AT(msg, UPB_SIZE(32, 40), upb_strview) = value;
}
UPB_INLINE void google_protobuf_FieldDescriptorProto_set_number(google_protobuf_FieldDescriptorProto *msg, int32_t value) {
  _upb_sethas(msg, 3);
  *UPB_PTR_AT(msg, UPB_SIZE(12, 12), int32_t) = value;
}
UPB_INLINE void google_protobuf_FieldDescriptorProto_set_label(google_protobuf_FieldDescriptorProto *msg, int32_t value) {
  _upb_sethas(msg, 4);
  *UPB_PTR_AT(msg, UPB_SIZE(4, 4), int32_t) = value;
}
UPB_INLINE void google_protobuf_FieldDescriptorProto_set_type(google_protobuf_FieldDescriptorProto *msg, int32_t value) {
  _upb_sethas(msg, 5);
  *UPB_PTR_AT(msg, UPB_SIZE(8, 8), int32_t) = value;
}
UPB_INLINE void google_protobuf_FieldDescriptorProto_set_type_name(google_protobuf_FieldDescriptorProto *msg, upb_strview value) {
  _upb_sethas(msg, 6);
  *UPB_PTR_AT(msg, UPB_SIZE(40, 56), upb_strview) = value;
}
UPB_INLINE void google_protobuf_FieldDescriptorProto_set_default_value(google_protobuf_FieldDescriptorProto *msg, upb_strview value) {
  _upb_sethas(msg, 7);
  *UPB_PTR_AT(msg, UPB_SIZE(48, 72), upb_strview) = value;
}
UPB_INLINE void google_protobuf_FieldDescriptorProto_set_options(google_protobuf_FieldDescriptorProto *msg, google_protobuf_FieldOptions* value) {
  _upb_sethas(msg, 8);
  *UPB_PTR_AT(msg, UPB_SIZE(64, 104), google_protobuf_FieldOptions*) = value;
}
UPB_INLINE struct google_protobuf_FieldOptions* google_protobuf_FieldDescriptorProto_mutable_options(google_protobuf_FieldDescriptorProto *msg, upb_arena *arena) {
  struct google_protobuf_FieldOptions* sub = (struct google_protobuf_FieldOptions*)google_protobuf_FieldDescriptorProto_options(msg);
  if (sub == NULL) {
    sub = (struct google_protobuf_FieldOptions*)_upb_msg_new(&google_protobuf_FieldOptions_msginit, arena);
    if (!sub) return NULL;
    google_protobuf_FieldDescriptorProto_set_options(msg, sub);
  }
  return sub;
}
UPB_INLINE void google_protobuf_FieldDescriptorProto_set_oneof_index(google_protobuf_FieldDescriptorProto *msg, int32_t value) {
  _upb_sethas(msg, 9);
  *UPB_PTR_AT(msg, UPB_SIZE(16, 16), int32_t) = value;
}
UPB_INLINE void google_protobuf_FieldDescriptorProto_set_json_name(google_protobuf_FieldDescriptorProto *msg, upb_strview value) {
  _upb_sethas(msg, 10);
  *UPB_PTR_AT(msg, UPB_SIZE(56, 88), upb_strview) = value;
}
UPB_INLINE void google_protobuf_FieldDescriptorProto_set_proto3_optional(google_protobuf_FieldDescriptorProto *msg, bool value) {
  _upb_sethas(msg, 11);
  *UPB_PTR_AT(msg, UPB_SIZE(20, 20), bool) = value;
}

/* google.protobuf.OneofDescriptorProto */

UPB_INLINE google_protobuf_OneofDescriptorProto *google_protobuf_OneofDescriptorProto_new(upb_arena *arena) {
  return (google_protobuf_OneofDescriptorProto *)_upb_msg_new(&google_protobuf_OneofDescriptorProto_msginit, arena);
}
UPB_INLINE google_protobuf_OneofDescriptorProto *google_protobuf_OneofDescriptorProto_parse(const char *buf, size_t size,
                        upb_arena *arena) {
  google_protobuf_OneofDescriptorProto *ret = google_protobuf_OneofDescriptorProto_new(arena);
  if (!ret) return NULL;
  if (!upb_decode(buf, size, ret, &google_protobuf_OneofDescriptorProto_msginit, arena)) return NULL;
  return ret;
}
UPB_INLINE google_protobuf_OneofDescriptorProto *google_protobuf_OneofDescriptorProto_parse_ex(const char *buf, size_t size,
                           const upb_extreg *extreg, int options,
                           upb_arena *arena) {
  google_protobuf_OneofDescriptorProto *ret = google_protobuf_OneofDescriptorProto_new(arena);
  if (!ret) return NULL;
  if (!_upb_decode(buf, size, ret, &google_protobuf_OneofDescriptorProto_msginit, extreg, options, arena)) {
    return NULL;
  }
  return ret;
}
UPB_INLINE char *google_protobuf_OneofDescriptorProto_serialize(const google_protobuf_OneofDescriptorProto *msg, upb_arena *arena, size_t *len) {
  return upb_encode(msg, &google_protobuf_OneofDescriptorProto_msginit, arena, len);
}

UPB_INLINE bool google_protobuf_OneofDescriptorProto_has_name(const google_protobuf_OneofDescriptorProto *msg) { return _upb_hasbit(msg, 1); }
UPB_INLINE upb_strview google_protobuf_OneofDescriptorProto_name(const google_protobuf_OneofDescriptorProto *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(4, 8), upb_strview); }
UPB_INLINE bool google_protobuf_OneofDescriptorProto_has_options(const google_protobuf_OneofDescriptorProto *msg) { return _upb_hasbit(msg, 2); }
UPB_INLINE const google_protobuf_OneofOptions* google_protobuf_OneofDescriptorProto_options(const google_protobuf_OneofDescriptorProto *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(12, 24), const google_protobuf_OneofOptions*); }

UPB_INLINE void google_protobuf_OneofDescriptorProto_set_name(google_protobuf_OneofDescriptorProto *msg, upb_strview value) {
  _upb_sethas(msg, 1);
  *UPB_PTR_AT(msg, UPB_SIZE(4, 8), upb_strview) = value;
}
UPB_INLINE void google_protobuf_OneofDescriptorProto_set_options(google_protobuf_OneofDescriptorProto *msg, google_protobuf_OneofOptions* value) {
  _upb_sethas(msg, 2);
  *UPB_PTR_AT(msg, UPB_SIZE(12, 24), google_protobuf_OneofOptions*) = value;
}
UPB_INLINE struct google_protobuf_OneofOptions* google_protobuf_OneofDescriptorProto_mutable_options(google_protobuf_OneofDescriptorProto *msg, upb_arena *arena) {
  struct google_protobuf_OneofOptions* sub = (struct google_protobuf_OneofOptions*)google_protobuf_OneofDescriptorProto_options(msg);
  if (sub == NULL) {
    sub = (struct google_protobuf_OneofOptions*)_upb_msg_new(&google_protobuf_OneofOptions_msginit, arena);
    if (!sub) return NULL;
    google_protobuf_OneofDescriptorProto_set_options(msg, sub);
  }
  return sub;
}

/* google.protobuf.EnumDescriptorProto */

UPB_INLINE google_protobuf_EnumDescriptorProto *google_protobuf_EnumDescriptorProto_new(upb_arena *arena) {
  return (google_protobuf_EnumDescriptorProto *)_upb_msg_new(&google_protobuf_EnumDescriptorProto_msginit, arena);
}
UPB_INLINE google_protobuf_EnumDescriptorProto *google_protobuf_EnumDescriptorProto_parse(const char *buf, size_t size,
                        upb_arena *arena) {
  google_protobuf_EnumDescriptorProto *ret = google_protobuf_EnumDescriptorProto_new(arena);
  if (!ret) return NULL;
  if (!upb_decode(buf, size, ret, &google_protobuf_EnumDescriptorProto_msginit, arena)) return NULL;
  return ret;
}
UPB_INLINE google_protobuf_EnumDescriptorProto *google_protobuf_EnumDescriptorProto_parse_ex(const char *buf, size_t size,
                           const upb_extreg *extreg, int options,
                           upb_arena *arena) {
  google_protobuf_EnumDescriptorProto *ret = google_protobuf_EnumDescriptorProto_new(arena);
  if (!ret) return NULL;
  if (!_upb_decode(buf, size, ret, &google_protobuf_EnumDescriptorProto_msginit, extreg, options, arena)) {
    return NULL;
  }
  return ret;
}
UPB_INLINE char *google_protobuf_EnumDescriptorProto_serialize(const google_protobuf_EnumDescriptorProto *msg, upb_arena *arena, size_t *len) {
  return upb_encode(msg, &google_protobuf_EnumDescriptorProto_msginit, arena, len);
}

UPB_INLINE bool google_protobuf_EnumDescriptorProto_has_name(const google_protobuf_EnumDescriptorProto *msg) { return _upb_hasbit(msg, 1); }
UPB_INLINE upb_strview google_protobuf_EnumDescriptorProto_name(const google_protobuf_EnumDescriptorProto *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(4, 8), upb_strview); }
UPB_INLINE bool google_protobuf_EnumDescriptorProto_has_value(const google_protobuf_EnumDescriptorProto *msg) { return _upb_has_submsg_nohasbit(msg, UPB_SIZE(16, 32)); }
UPB_INLINE const google_protobuf_EnumValueDescriptorProto* const* google_protobuf_EnumDescriptorProto_value(const google_protobuf_EnumDescriptorProto *msg, size_t *len) { return (const google_protobuf_EnumValueDescriptorProto* const*)_upb_array_accessor(msg, UPB_SIZE(16, 32), len); }
UPB_INLINE bool google_protobuf_EnumDescriptorProto_has_options(const google_protobuf_EnumDescriptorProto *msg) { return _upb_hasbit(msg, 2); }
UPB_INLINE const google_protobuf_EnumOptions* google_protobuf_EnumDescriptorProto_options(const google_protobuf_EnumDescriptorProto *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(12, 24), const google_protobuf_EnumOptions*); }
UPB_INLINE bool google_protobuf_EnumDescriptorProto_has_reserved_range(const google_protobuf_EnumDescriptorProto *msg) { return _upb_has_submsg_nohasbit(msg, UPB_SIZE(20, 40)); }
UPB_INLINE const google_protobuf_EnumDescriptorProto_EnumReservedRange* const* google_protobuf_EnumDescriptorProto_reserved_range(const google_protobuf_EnumDescriptorProto *msg, size_t *len) { return (const google_protobuf_EnumDescriptorProto_EnumReservedRange* const*)_upb_array_accessor(msg, UPB_SIZE(20, 40), len); }
UPB_INLINE upb_strview const* google_protobuf_EnumDescriptorProto_reserved_name(const google_protobuf_EnumDescriptorProto *msg, size_t *len) { return (upb_strview const*)_upb_array_accessor(msg, UPB_SIZE(24, 48), len); }

UPB_INLINE void google_protobuf_EnumDescriptorProto_set_name(google_protobuf_EnumDescriptorProto *msg, upb_strview value) {
  _upb_sethas(msg, 1);
  *UPB_PTR_AT(msg, UPB_SIZE(4, 8), upb_strview) = value;
}
UPB_INLINE google_protobuf_EnumValueDescriptorProto** google_protobuf_EnumDescriptorProto_mutable_value(google_protobuf_EnumDescriptorProto *msg, size_t *len) {
  return (google_protobuf_EnumValueDescriptorProto**)_upb_array_mutable_accessor(msg, UPB_SIZE(16, 32), len);
}
UPB_INLINE google_protobuf_EnumValueDescriptorProto** google_protobuf_EnumDescriptorProto_resize_value(google_protobuf_EnumDescriptorProto *msg, size_t len, upb_arena *arena) {
  return (google_protobuf_EnumValueDescriptorProto**)_upb_array_resize_accessor2(msg, UPB_SIZE(16, 32), len, UPB_SIZE(2, 3), arena);
}
UPB_INLINE struct google_protobuf_EnumValueDescriptorProto* google_protobuf_EnumDescriptorProto_add_value(google_protobuf_EnumDescriptorProto *msg, upb_arena *arena) {
  struct google_protobuf_EnumValueDescriptorProto* sub = (struct google_protobuf_EnumValueDescriptorProto*)_upb_msg_new(&google_protobuf_EnumValueDescriptorProto_msginit, arena);
  bool ok = _upb_array_append_accessor2(
      msg, UPB_SIZE(16, 32), UPB_SIZE(2, 3), &sub, arena);
  if (!ok) return NULL;
  return sub;
}
UPB_INLINE void google_protobuf_EnumDescriptorProto_set_options(google_protobuf_EnumDescriptorProto *msg, google_protobuf_EnumOptions* value) {
  _upb_sethas(msg, 2);
  *UPB_PTR_AT(msg, UPB_SIZE(12, 24), google_protobuf_EnumOptions*) = value;
}
UPB_INLINE struct google_protobuf_EnumOptions* google_protobuf_EnumDescriptorProto_mutable_options(google_protobuf_EnumDescriptorProto *msg, upb_arena *arena) {
  struct google_protobuf_EnumOptions* sub = (struct google_protobuf_EnumOptions*)google_protobuf_EnumDescriptorProto_options(msg);
  if (sub == NULL) {
    sub = (struct google_protobuf_EnumOptions*)_upb_msg_new(&google_protobuf_EnumOptions_msginit, arena);
    if (!sub) return NULL;
    google_protobuf_EnumDescriptorProto_set_options(msg, sub);
  }
  return sub;
}
UPB_INLINE google_protobuf_EnumDescriptorProto_EnumReservedRange** google_protobuf_EnumDescriptorProto_mutable_reserved_range(google_protobuf_EnumDescriptorProto *msg, size_t *len) {
  return (google_protobuf_EnumDescriptorProto_EnumReservedRange**)_upb_array_mutable_accessor(msg, UPB_SIZE(20, 40), len);
}
UPB_INLINE google_protobuf_EnumDescriptorProto_EnumReservedRange** google_protobuf_EnumDescriptorProto_resize_reserved_range(google_protobuf_EnumDescriptorProto *msg, size_t len, upb_arena *arena) {
  return (google_protobuf_EnumDescriptorProto_EnumReservedRange**)_upb_array_resize_accessor2(msg, UPB_SIZE(20, 40), len, UPB_SIZE(2, 3), arena);
}
UPB_INLINE struct google_protobuf_EnumDescriptorProto_EnumReservedRange* google_protobuf_EnumDescriptorProto_add_reserved_range(google_protobuf_EnumDescriptorProto *msg, upb_arena *arena) {
  struct google_protobuf_EnumDescriptorProto_EnumReservedRange* sub = (struct google_protobuf_EnumDescriptorProto_EnumReservedRange*)_upb_msg_new(&google_protobuf_EnumDescriptorProto_EnumReservedRange_msginit, arena);
  bool ok = _upb_array_append_accessor2(
      msg, UPB_SIZE(20, 40), UPB_SIZE(2, 3), &sub, arena);
  if (!ok) return NULL;
  return sub;
}
UPB_INLINE upb_strview* google_protobuf_EnumDescriptorProto_mutable_reserved_name(google_protobuf_EnumDescriptorProto *msg, size_t *len) {
  return (upb_strview*)_upb_array_mutable_accessor(msg, UPB_SIZE(24, 48), len);
}
UPB_INLINE upb_strview* google_protobuf_EnumDescriptorProto_resize_reserved_name(google_protobuf_EnumDescriptorProto *msg, size_t len, upb_arena *arena) {
  return (upb_strview*)_upb_array_resize_accessor2(msg, UPB_SIZE(24, 48), len, UPB_SIZE(3, 4), arena);
}
UPB_INLINE bool google_protobuf_EnumDescriptorProto_add_reserved_name(google_protobuf_EnumDescriptorProto *msg, upb_strview val, upb_arena *arena) {
  return _upb_array_append_accessor2(msg, UPB_SIZE(24, 48), UPB_SIZE(3, 4), &val,
      arena);
}

/* google.protobuf.EnumDescriptorProto.EnumReservedRange */

UPB_INLINE google_protobuf_EnumDescriptorProto_EnumReservedRange *google_protobuf_EnumDescriptorProto_EnumReservedRange_new(upb_arena *arena) {
  return (google_protobuf_EnumDescriptorProto_EnumReservedRange *)_upb_msg_new(&google_protobuf_EnumDescriptorProto_EnumReservedRange_msginit, arena);
}
UPB_INLINE google_protobuf_EnumDescriptorProto_EnumReservedRange *google_protobuf_EnumDescriptorProto_EnumReservedRange_parse(const char *buf, size_t size,
                        upb_arena *arena) {
  google_protobuf_EnumDescriptorProto_EnumReservedRange *ret = google_protobuf_EnumDescriptorProto_EnumReservedRange_new(arena);
  if (!ret) return NULL;
  if (!upb_decode(buf, size, ret, &google_protobuf_EnumDescriptorProto_EnumReservedRange_msginit, arena)) return NULL;
  return ret;
}
UPB_INLINE google_protobuf_EnumDescriptorProto_EnumReservedRange *google_protobuf_EnumDescriptorProto_EnumReservedRange_parse_ex(const char *buf, size_t size,
                           const upb_extreg *extreg, int options,
                           upb_arena *arena) {
  google_protobuf_EnumDescriptorProto_EnumReservedRange *ret = google_protobuf_EnumDescriptorProto_EnumReservedRange_new(arena);
  if (!ret) return NULL;
  if (!_upb_decode(buf, size, ret, &google_protobuf_EnumDescriptorProto_EnumReservedRange_msginit, extreg, options, arena)) {
    return NULL;
  }
  return ret;
}
UPB_INLINE char *google_protobuf_EnumDescriptorProto_EnumReservedRange_serialize(const google_protobuf_EnumDescriptorProto_EnumReservedRange *msg, upb_arena *arena, size_t *len) {
  return upb_encode(msg, &google_protobuf_EnumDescriptorProto_EnumReservedRange_msginit, arena, len);
}

UPB_INLINE bool google_protobuf_EnumDescriptorProto_EnumReservedRange_has_start(const google_protobuf_EnumDescriptorProto_EnumReservedRange *msg) { return _upb_hasbit(msg, 1); }
UPB_INLINE int32_t google_protobuf_EnumDescriptorProto_EnumReservedRange_start(const google_protobuf_EnumDescriptorProto_EnumReservedRange *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(4, 4), int32_t); }
UPB_INLINE bool google_protobuf_EnumDescriptorProto_EnumReservedRange_has_end(const google_protobuf_EnumDescriptorProto_EnumReservedRange *msg) { return _upb_hasbit(msg, 2); }
UPB_INLINE int32_t google_protobuf_EnumDescriptorProto_EnumReservedRange_end(const google_protobuf_EnumDescriptorProto_EnumReservedRange *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(8, 8), int32_t); }

UPB_INLINE void google_protobuf_EnumDescriptorProto_EnumReservedRange_set_start(google_protobuf_EnumDescriptorProto_EnumReservedRange *msg, int32_t value) {
  _upb_sethas(msg, 1);
  *UPB_PTR_AT(msg, UPB_SIZE(4, 4), int32_t) = value;
}
UPB_INLINE void google_protobuf_EnumDescriptorProto_EnumReservedRange_set_end(google_protobuf_EnumDescriptorProto_EnumReservedRange *msg, int32_t value) {
  _upb_sethas(msg, 2);
  *UPB_PTR_AT(msg, UPB_SIZE(8, 8), int32_t) = value;
}

/* google.protobuf.EnumValueDescriptorProto */

UPB_INLINE google_protobuf_EnumValueDescriptorProto *google_protobuf_EnumValueDescriptorProto_new(upb_arena *arena) {
  return (google_protobuf_EnumValueDescriptorProto *)_upb_msg_new(&google_protobuf_EnumValueDescriptorProto_msginit, arena);
}
UPB_INLINE google_protobuf_EnumValueDescriptorProto *google_protobuf_EnumValueDescriptorProto_parse(const char *buf, size_t size,
                        upb_arena *arena) {
  google_protobuf_EnumValueDescriptorProto *ret = google_protobuf_EnumValueDescriptorProto_new(arena);
  if (!ret) return NULL;
  if (!upb_decode(buf, size, ret, &google_protobuf_EnumValueDescriptorProto_msginit, arena)) return NULL;
  return ret;
}
UPB_INLINE google_protobuf_EnumValueDescriptorProto *google_protobuf_EnumValueDescriptorProto_parse_ex(const char *buf, size_t size,
                           const upb_extreg *extreg, int options,
                           upb_arena *arena) {
  google_protobuf_EnumValueDescriptorProto *ret = google_protobuf_EnumValueDescriptorProto_new(arena);
  if (!ret) return NULL;
  if (!_upb_decode(buf, size, ret, &google_protobuf_EnumValueDescriptorProto_msginit, extreg, options, arena)) {
    return NULL;
  }
  return ret;
}
UPB_INLINE char *google_protobuf_EnumValueDescriptorProto_serialize(const google_protobuf_EnumValueDescriptorProto *msg, upb_arena *arena, size_t *len) {
  return upb_encode(msg, &google_protobuf_EnumValueDescriptorProto_msginit, arena, len);
}

UPB_INLINE bool google_protobuf_EnumValueDescriptorProto_has_name(const google_protobuf_EnumValueDescriptorProto *msg) { return _upb_hasbit(msg, 1); }
UPB_INLINE upb_strview google_protobuf_EnumValueDescriptorProto_name(const google_protobuf_EnumValueDescriptorProto *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(8, 8), upb_strview); }
UPB_INLINE bool google_protobuf_EnumValueDescriptorProto_has_number(const google_protobuf_EnumValueDescriptorProto *msg) { return _upb_hasbit(msg, 2); }
UPB_INLINE int32_t google_protobuf_EnumValueDescriptorProto_number(const google_protobuf_EnumValueDescriptorProto *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(4, 4), int32_t); }
UPB_INLINE bool google_protobuf_EnumValueDescriptorProto_has_options(const google_protobuf_EnumValueDescriptorProto *msg) { return _upb_hasbit(msg, 3); }
UPB_INLINE const google_protobuf_EnumValueOptions* google_protobuf_EnumValueDescriptorProto_options(const google_protobuf_EnumValueDescriptorProto *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(16, 24), const google_protobuf_EnumValueOptions*); }

UPB_INLINE void google_protobuf_EnumValueDescriptorProto_set_name(google_protobuf_EnumValueDescriptorProto *msg, upb_strview value) {
  _upb_sethas(msg, 1);
  *UPB_PTR_AT(msg, UPB_SIZE(8, 8), upb_strview) = value;
}
UPB_INLINE void google_protobuf_EnumValueDescriptorProto_set_number(google_protobuf_EnumValueDescriptorProto *msg, int32_t value) {
  _upb_sethas(msg, 2);
  *UPB_PTR_AT(msg, UPB_SIZE(4, 4), int32_t) = value;
}
UPB_INLINE void google_protobuf_EnumValueDescriptorProto_set_options(google_protobuf_EnumValueDescriptorProto *msg, google_protobuf_EnumValueOptions* value) {
  _upb_sethas(msg, 3);
  *UPB_PTR_AT(msg, UPB_SIZE(16, 24), google_protobuf_EnumValueOptions*) = value;
}
UPB_INLINE struct google_protobuf_EnumValueOptions* google_protobuf_EnumValueDescriptorProto_mutable_options(google_protobuf_EnumValueDescriptorProto *msg, upb_arena *arena) {
  struct google_protobuf_EnumValueOptions* sub = (struct google_protobuf_EnumValueOptions*)google_protobuf_EnumValueDescriptorProto_options(msg);
  if (sub == NULL) {
    sub = (struct google_protobuf_EnumValueOptions*)_upb_msg_new(&google_protobuf_EnumValueOptions_msginit, arena);
    if (!sub) return NULL;
    google_protobuf_EnumValueDescriptorProto_set_options(msg, sub);
  }
  return sub;
}

/* google.protobuf.ServiceDescriptorProto */

UPB_INLINE google_protobuf_ServiceDescriptorProto *google_protobuf_ServiceDescriptorProto_new(upb_arena *arena) {
  return (google_protobuf_ServiceDescriptorProto *)_upb_msg_new(&google_protobuf_ServiceDescriptorProto_msginit, arena);
}
UPB_INLINE google_protobuf_ServiceDescriptorProto *google_protobuf_ServiceDescriptorProto_parse(const char *buf, size_t size,
                        upb_arena *arena) {
  google_protobuf_ServiceDescriptorProto *ret = google_protobuf_ServiceDescriptorProto_new(arena);
  if (!ret) return NULL;
  if (!upb_decode(buf, size, ret, &google_protobuf_ServiceDescriptorProto_msginit, arena)) return NULL;
  return ret;
}
UPB_INLINE google_protobuf_ServiceDescriptorProto *google_protobuf_ServiceDescriptorProto_parse_ex(const char *buf, size_t size,
                           const upb_extreg *extreg, int options,
                           upb_arena *arena) {
  google_protobuf_ServiceDescriptorProto *ret = google_protobuf_ServiceDescriptorProto_new(arena);
  if (!ret) return NULL;
  if (!_upb_decode(buf, size, ret, &google_protobuf_ServiceDescriptorProto_msginit, extreg, options, arena)) {
    return NULL;
  }
  return ret;
}
UPB_INLINE char *google_protobuf_ServiceDescriptorProto_serialize(const google_protobuf_ServiceDescriptorProto *msg, upb_arena *arena, size_t *len) {
  return upb_encode(msg, &google_protobuf_ServiceDescriptorProto_msginit, arena, len);
}

UPB_INLINE bool google_protobuf_ServiceDescriptorProto_has_name(const google_protobuf_ServiceDescriptorProto *msg) { return _upb_hasbit(msg, 1); }
UPB_INLINE upb_strview google_protobuf_ServiceDescriptorProto_name(const google_protobuf_ServiceDescriptorProto *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(4, 8), upb_strview); }
UPB_INLINE bool google_protobuf_ServiceDescriptorProto_has_method(const google_protobuf_ServiceDescriptorProto *msg) { return _upb_has_submsg_nohasbit(msg, UPB_SIZE(16, 32)); }
UPB_INLINE const google_protobuf_MethodDescriptorProto* const* google_protobuf_ServiceDescriptorProto_method(const google_protobuf_ServiceDescriptorProto *msg, size_t *len) { return (const google_protobuf_MethodDescriptorProto* const*)_upb_array_accessor(msg, UPB_SIZE(16, 32), len); }
UPB_INLINE bool google_protobuf_ServiceDescriptorProto_has_options(const google_protobuf_ServiceDescriptorProto *msg) { return _upb_hasbit(msg, 2); }
UPB_INLINE const google_protobuf_ServiceOptions* google_protobuf_ServiceDescriptorProto_options(const google_protobuf_ServiceDescriptorProto *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(12, 24), const google_protobuf_ServiceOptions*); }

UPB_INLINE void google_protobuf_ServiceDescriptorProto_set_name(google_protobuf_ServiceDescriptorProto *msg, upb_strview value) {
  _upb_sethas(msg, 1);
  *UPB_PTR_AT(msg, UPB_SIZE(4, 8), upb_strview) = value;
}
UPB_INLINE google_protobuf_MethodDescriptorProto** google_protobuf_ServiceDescriptorProto_mutable_method(google_protobuf_ServiceDescriptorProto *msg, size_t *len) {
  return (google_protobuf_MethodDescriptorProto**)_upb_array_mutable_accessor(msg, UPB_SIZE(16, 32), len);
}
UPB_INLINE google_protobuf_MethodDescriptorProto** google_protobuf_ServiceDescriptorProto_resize_method(google_protobuf_ServiceDescriptorProto *msg, size_t len, upb_arena *arena) {
  return (google_protobuf_MethodDescriptorProto**)_upb_array_resize_accessor2(msg, UPB_SIZE(16, 32), len, UPB_SIZE(2, 3), arena);
}
UPB_INLINE struct google_protobuf_MethodDescriptorProto* google_protobuf_ServiceDescriptorProto_add_method(google_protobuf_ServiceDescriptorProto *msg, upb_arena *arena) {
  struct google_protobuf_MethodDescriptorProto* sub = (struct google_protobuf_MethodDescriptorProto*)_upb_msg_new(&google_protobuf_MethodDescriptorProto_msginit, arena);
  bool ok = _upb_array_append_accessor2(
      msg, UPB_SIZE(16, 32), UPB_SIZE(2, 3), &sub, arena);
  if (!ok) return NULL;
  return sub;
}
UPB_INLINE void google_protobuf_ServiceDescriptorProto_set_options(google_protobuf_ServiceDescriptorProto *msg, google_protobuf_ServiceOptions* value) {
  _upb_sethas(msg, 2);
  *UPB_PTR_AT(msg, UPB_SIZE(12, 24), google_protobuf_ServiceOptions*) = value;
}
UPB_INLINE struct google_protobuf_ServiceOptions* google_protobuf_ServiceDescriptorProto_mutable_options(google_protobuf_ServiceDescriptorProto *msg, upb_arena *arena) {
  struct google_protobuf_ServiceOptions* sub = (struct google_protobuf_ServiceOptions*)google_protobuf_ServiceDescriptorProto_options(msg);
  if (sub == NULL) {
    sub = (struct google_protobuf_ServiceOptions*)_upb_msg_new(&google_protobuf_ServiceOptions_msginit, arena);
    if (!sub) return NULL;
    google_protobuf_ServiceDescriptorProto_set_options(msg, sub);
  }
  return sub;
}

/* google.protobuf.MethodDescriptorProto */

UPB_INLINE google_protobuf_MethodDescriptorProto *google_protobuf_MethodDescriptorProto_new(upb_arena *arena) {
  return (google_protobuf_MethodDescriptorProto *)_upb_msg_new(&google_protobuf_MethodDescriptorProto_msginit, arena);
}
UPB_INLINE google_protobuf_MethodDescriptorProto *google_protobuf_MethodDescriptorProto_parse(const char *buf, size_t size,
                        upb_arena *arena) {
  google_protobuf_MethodDescriptorProto *ret = google_protobuf_MethodDescriptorProto_new(arena);
  if (!ret) return NULL;
  if (!upb_decode(buf, size, ret, &google_protobuf_MethodDescriptorProto_msginit, arena)) return NULL;
  return ret;
}
UPB_INLINE google_protobuf_MethodDescriptorProto *google_protobuf_MethodDescriptorProto_parse_ex(const char *buf, size_t size,
                           const upb_extreg *extreg, int options,
                           upb_arena *arena) {
  google_protobuf_MethodDescriptorProto *ret = google_protobuf_MethodDescriptorProto_new(arena);
  if (!ret) return NULL;
  if (!_upb_decode(buf, size, ret, &google_protobuf_MethodDescriptorProto_msginit, extreg, options, arena)) {
    return NULL;
  }
  return ret;
}
UPB_INLINE char *google_protobuf_MethodDescriptorProto_serialize(const google_protobuf_MethodDescriptorProto *msg, upb_arena *arena, size_t *len) {
  return upb_encode(msg, &google_protobuf_MethodDescriptorProto_msginit, arena, len);
}

UPB_INLINE bool google_protobuf_MethodDescriptorProto_has_name(const google_protobuf_MethodDescriptorProto *msg) { return _upb_hasbit(msg, 1); }
UPB_INLINE upb_strview google_protobuf_MethodDescriptorProto_name(const google_protobuf_MethodDescriptorProto *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(4, 8), upb_strview); }
UPB_INLINE bool google_protobuf_MethodDescriptorProto_has_input_type(const google_protobuf_MethodDescriptorProto *msg) { return _upb_hasbit(msg, 2); }
UPB_INLINE upb_strview google_protobuf_MethodDescriptorProto_input_type(const google_protobuf_MethodDescriptorProto *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(12, 24), upb_strview); }
UPB_INLINE bool google_protobuf_MethodDescriptorProto_has_output_type(const google_protobuf_MethodDescriptorProto *msg) { return _upb_hasbit(msg, 3); }
UPB_INLINE upb_strview google_protobuf_MethodDescriptorProto_output_type(const google_protobuf_MethodDescriptorProto *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(20, 40), upb_strview); }
UPB_INLINE bool google_protobuf_MethodDescriptorProto_has_options(const google_protobuf_MethodDescriptorProto *msg) { return _upb_hasbit(msg, 4); }
UPB_INLINE const google_protobuf_MethodOptions* google_protobuf_MethodDescriptorProto_options(const google_protobuf_MethodDescriptorProto *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(28, 56), const google_protobuf_MethodOptions*); }
UPB_INLINE bool google_protobuf_MethodDescriptorProto_has_client_streaming(const google_protobuf_MethodDescriptorProto *msg) { return _upb_hasbit(msg, 5); }
UPB_INLINE bool google_protobuf_MethodDescriptorProto_client_streaming(const google_protobuf_MethodDescriptorProto *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(1, 1), bool); }
UPB_INLINE bool google_protobuf_MethodDescriptorProto_has_server_streaming(const google_protobuf_MethodDescriptorProto *msg) { return _upb_hasbit(msg, 6); }
UPB_INLINE bool google_protobuf_MethodDescriptorProto_server_streaming(const google_protobuf_MethodDescriptorProto *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(2, 2), bool); }

UPB_INLINE void google_protobuf_MethodDescriptorProto_set_name(google_protobuf_MethodDescriptorProto *msg, upb_strview value) {
  _upb_sethas(msg, 1);
  *UPB_PTR_AT(msg, UPB_SIZE(4, 8), upb_strview) = value;
}
UPB_INLINE void google_protobuf_MethodDescriptorProto_set_input_type(google_protobuf_MethodDescriptorProto *msg, upb_strview value) {
  _upb_sethas(msg, 2);
  *UPB_PTR_AT(msg, UPB_SIZE(12, 24), upb_strview) = value;
}
UPB_INLINE void google_protobuf_MethodDescriptorProto_set_output_type(google_protobuf_MethodDescriptorProto *msg, upb_strview value) {
  _upb_sethas(msg, 3);
  *UPB_PTR_AT(msg, UPB_SIZE(20, 40), upb_strview) = value;
}
UPB_INLINE void google_protobuf_MethodDescriptorProto_set_options(google_protobuf_MethodDescriptorProto *msg, google_protobuf_MethodOptions* value) {
  _upb_sethas(msg, 4);
  *UPB_PTR_AT(msg, UPB_SIZE(28, 56), google_protobuf_MethodOptions*) = value;
}
UPB_INLINE struct google_protobuf_MethodOptions* google_protobuf_MethodDescriptorProto_mutable_options(google_protobuf_MethodDescriptorProto *msg, upb_arena *arena) {
  struct google_protobuf_MethodOptions* sub = (struct google_protobuf_MethodOptions*)google_protobuf_MethodDescriptorProto_options(msg);
  if (sub == NULL) {
    sub = (struct google_protobuf_MethodOptions*)_upb_msg_new(&google_protobuf_MethodOptions_msginit, arena);
    if (!sub) return NULL;
    google_protobuf_MethodDescriptorProto_set_options(msg, sub);
  }
  return sub;
}
UPB_INLINE void google_protobuf_MethodDescriptorProto_set_client_streaming(google_protobuf_MethodDescriptorProto *msg, bool value) {
  _upb_sethas(msg, 5);
  *UPB_PTR_AT(msg, UPB_SIZE(1, 1), bool) = value;
}
UPB_INLINE void google_protobuf_MethodDescriptorProto_set_server_streaming(google_protobuf_MethodDescriptorProto *msg, bool value) {
  _upb_sethas(msg, 6);
  *UPB_PTR_AT(msg, UPB_SIZE(2, 2), bool) = value;
}

/* google.protobuf.FileOptions */

UPB_INLINE google_protobuf_FileOptions *google_protobuf_FileOptions_new(upb_arena *arena) {
  return (google_protobuf_FileOptions *)_upb_msg_new(&google_protobuf_FileOptions_msginit, arena);
}
UPB_INLINE google_protobuf_FileOptions *google_protobuf_FileOptions_parse(const char *buf, size_t size,
                        upb_arena *arena) {
  google_protobuf_FileOptions *ret = google_protobuf_FileOptions_new(arena);
  if (!ret) return NULL;
  if (!upb_decode(buf, size, ret, &google_protobuf_FileOptions_msginit, arena)) return NULL;
  return ret;
}
UPB_INLINE google_protobuf_FileOptions *google_protobuf_FileOptions_parse_ex(const char *buf, size_t size,
                           const upb_extreg *extreg, int options,
                           upb_arena *arena) {
  google_protobuf_FileOptions *ret = google_protobuf_FileOptions_new(arena);
  if (!ret) return NULL;
  if (!_upb_decode(buf, size, ret, &google_protobuf_FileOptions_msginit, extreg, options, arena)) {
    return NULL;
  }
  return ret;
}
UPB_INLINE char *google_protobuf_FileOptions_serialize(const google_protobuf_FileOptions *msg, upb_arena *arena, size_t *len) {
  return upb_encode(msg, &google_protobuf_FileOptions_msginit, arena, len);
}

UPB_INLINE bool google_protobuf_FileOptions_has_java_package(const google_protobuf_FileOptions *msg) { return _upb_hasbit(msg, 1); }
UPB_INLINE upb_strview google_protobuf_FileOptions_java_package(const google_protobuf_FileOptions *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(20, 24), upb_strview); }
UPB_INLINE bool google_protobuf_FileOptions_has_java_outer_classname(const google_protobuf_FileOptions *msg) { return _upb_hasbit(msg, 2); }
UPB_INLINE upb_strview google_protobuf_FileOptions_java_outer_classname(const google_protobuf_FileOptions *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(28, 40), upb_strview); }
UPB_INLINE bool google_protobuf_FileOptions_has_optimize_for(const google_protobuf_FileOptions *msg) { return _upb_hasbit(msg, 3); }
UPB_INLINE int32_t google_protobuf_FileOptions_optimize_for(const google_protobuf_FileOptions *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(4, 4), int32_t); }
UPB_INLINE bool google_protobuf_FileOptions_has_java_multiple_files(const google_protobuf_FileOptions *msg) { return _upb_hasbit(msg, 4); }
UPB_INLINE bool google_protobuf_FileOptions_java_multiple_files(const google_protobuf_FileOptions *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(8, 8), bool); }
UPB_INLINE bool google_protobuf_FileOptions_has_go_package(const google_protobuf_FileOptions *msg) { return _upb_hasbit(msg, 5); }
UPB_INLINE upb_strview google_protobuf_FileOptions_go_package(const google_protobuf_FileOptions *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(36, 56), upb_strview); }
UPB_INLINE bool google_protobuf_FileOptions_has_cc_generic_services(const google_protobuf_FileOptions *msg) { return _upb_hasbit(msg, 6); }
UPB_INLINE bool google_protobuf_FileOptions_cc_generic_services(const google_protobuf_FileOptions *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(9, 9), bool); }
UPB_INLINE bool google_protobuf_FileOptions_has_java_generic_services(const google_protobuf_FileOptions *msg) { return _upb_hasbit(msg, 7); }
UPB_INLINE bool google_protobuf_FileOptions_java_generic_services(const google_protobuf_FileOptions *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(10, 10), bool); }
UPB_INLINE bool google_protobuf_FileOptions_has_py_generic_services(const google_protobuf_FileOptions *msg) { return _upb_hasbit(msg, 8); }
UPB_INLINE bool google_protobuf_FileOptions_py_generic_services(const google_protobuf_FileOptions *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(11, 11), bool); }
UPB_INLINE bool google_protobuf_FileOptions_has_java_generate_equals_and_hash(const google_protobuf_FileOptions *msg) { return _upb_hasbit(msg, 9); }
UPB_INLINE bool google_protobuf_FileOptions_java_generate_equals_and_hash(const google_protobuf_FileOptions *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(12, 12), bool); }
UPB_INLINE bool google_protobuf_FileOptions_has_deprecated(const google_protobuf_FileOptions *msg) { return _upb_hasbit(msg, 10); }
UPB_INLINE bool google_protobuf_FileOptions_deprecated(const google_protobuf_FileOptions *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(13, 13), bool); }
UPB_INLINE bool google_protobuf_FileOptions_has_java_string_check_utf8(const google_protobuf_FileOptions *msg) { return _upb_hasbit(msg, 11); }
UPB_INLINE bool google_protobuf_FileOptions_java_string_check_utf8(const google_protobuf_FileOptions *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(14, 14), bool); }
UPB_INLINE bool google_protobuf_FileOptions_has_cc_enable_arenas(const google_protobuf_FileOptions *msg) { return _upb_hasbit(msg, 12); }
UPB_INLINE bool google_protobuf_FileOptions_cc_enable_arenas(const google_protobuf_FileOptions *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(15, 15), bool); }
UPB_INLINE bool google_protobuf_FileOptions_has_objc_class_prefix(const google_protobuf_FileOptions *msg) { return _upb_hasbit(msg, 13); }
UPB_INLINE upb_strview google_protobuf_FileOptions_objc_class_prefix(const google_protobuf_FileOptions *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(44, 72), upb_strview); }
UPB_INLINE bool google_protobuf_FileOptions_has_csharp_namespace(const google_protobuf_FileOptions *msg) { return _upb_hasbit(msg, 14); }
UPB_INLINE upb_strview google_protobuf_FileOptions_csharp_namespace(const google_protobuf_FileOptions *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(52, 88), upb_strview); }
UPB_INLINE bool google_protobuf_FileOptions_has_swift_prefix(const google_protobuf_FileOptions *msg) { return _upb_hasbit(msg, 15); }
UPB_INLINE upb_strview google_protobuf_FileOptions_swift_prefix(const google_protobuf_FileOptions *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(60, 104), upb_strview); }
UPB_INLINE bool google_protobuf_FileOptions_has_php_class_prefix(const google_protobuf_FileOptions *msg) { return _upb_hasbit(msg, 16); }
UPB_INLINE upb_strview google_protobuf_FileOptions_php_class_prefix(const google_protobuf_FileOptions *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(68, 120), upb_strview); }
UPB_INLINE bool google_protobuf_FileOptions_has_php_namespace(const google_protobuf_FileOptions *msg) { return _upb_hasbit(msg, 17); }
UPB_INLINE upb_strview google_protobuf_FileOptions_php_namespace(const google_protobuf_FileOptions *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(76, 136), upb_strview); }
UPB_INLINE bool google_protobuf_FileOptions_has_php_generic_services(const google_protobuf_FileOptions *msg) { return _upb_hasbit(msg, 18); }
UPB_INLINE bool google_protobuf_FileOptions_php_generic_services(const google_protobuf_FileOptions *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(16, 16), bool); }
UPB_INLINE bool google_protobuf_FileOptions_has_php_metadata_namespace(const google_protobuf_FileOptions *msg) { return _upb_hasbit(msg, 19); }
UPB_INLINE upb_strview google_protobuf_FileOptions_php_metadata_namespace(const google_protobuf_FileOptions *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(84, 152), upb_strview); }
UPB_INLINE bool google_protobuf_FileOptions_has_ruby_package(const google_protobuf_FileOptions *msg) { return _upb_hasbit(msg, 20); }
UPB_INLINE upb_strview google_protobuf_FileOptions_ruby_package(const google_protobuf_FileOptions *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(92, 168), upb_strview); }
UPB_INLINE bool google_protobuf_FileOptions_has_uninterpreted_option(const google_protobuf_FileOptions *msg) { return _upb_has_submsg_nohasbit(msg, UPB_SIZE(100, 184)); }
UPB_INLINE const google_protobuf_UninterpretedOption* const* google_protobuf_FileOptions_uninterpreted_option(const google_protobuf_FileOptions *msg, size_t *len) { return (const google_protobuf_UninterpretedOption* const*)_upb_array_accessor(msg, UPB_SIZE(100, 184), len); }

UPB_INLINE void google_protobuf_FileOptions_set_java_package(google_protobuf_FileOptions *msg, upb_strview value) {
  _upb_sethas(msg, 1);
  *UPB_PTR_AT(msg, UPB_SIZE(20, 24), upb_strview) = value;
}
UPB_INLINE void google_protobuf_FileOptions_set_java_outer_classname(google_protobuf_FileOptions *msg, upb_strview value) {
  _upb_sethas(msg, 2);
  *UPB_PTR_AT(msg, UPB_SIZE(28, 40), upb_strview) = value;
}
UPB_INLINE void google_protobuf_FileOptions_set_optimize_for(google_protobuf_FileOptions *msg, int32_t value) {
  _upb_sethas(msg, 3);
  *UPB_PTR_AT(msg, UPB_SIZE(4, 4), int32_t) = value;
}
UPB_INLINE void google_protobuf_FileOptions_set_java_multiple_files(google_protobuf_FileOptions *msg, bool value) {
  _upb_sethas(msg, 4);
  *UPB_PTR_AT(msg, UPB_SIZE(8, 8), bool) = value;
}
UPB_INLINE void google_protobuf_FileOptions_set_go_package(google_protobuf_FileOptions *msg, upb_strview value) {
  _upb_sethas(msg, 5);
  *UPB_PTR_AT(msg, UPB_SIZE(36, 56), upb_strview) = value;
}
UPB_INLINE void google_protobuf_FileOptions_set_cc_generic_services(google_protobuf_FileOptions *msg, bool value) {
  _upb_sethas(msg, 6);
  *UPB_PTR_AT(msg, UPB_SIZE(9, 9), bool) = value;
}
UPB_INLINE void google_protobuf_FileOptions_set_java_generic_services(google_protobuf_FileOptions *msg, bool value) {
  _upb_sethas(msg, 7);
  *UPB_PTR_AT(msg, UPB_SIZE(10, 10), bool) = value;
}
UPB_INLINE void google_protobuf_FileOptions_set_py_generic_services(google_protobuf_FileOptions *msg, bool value) {
  _upb_sethas(msg, 8);
  *UPB_PTR_AT(msg, UPB_SIZE(11, 11), bool) = value;
}
UPB_INLINE void google_protobuf_FileOptions_set_java_generate_equals_and_hash(google_protobuf_FileOptions *msg, bool value) {
  _upb_sethas(msg, 9);
  *UPB_PTR_AT(msg, UPB_SIZE(12, 12), bool) = value;
}
UPB_INLINE void google_protobuf_FileOptions_set_deprecated(google_protobuf_FileOptions *msg, bool value) {
  _upb_sethas(msg, 10);
  *UPB_PTR_AT(msg, UPB_SIZE(13, 13), bool) = value;
}
UPB_INLINE void google_protobuf_FileOptions_set_java_string_check_utf8(google_protobuf_FileOptions *msg, bool value) {
  _upb_sethas(msg, 11);
  *UPB_PTR_AT(msg, UPB_SIZE(14, 14), bool) = value;
}
UPB_INLINE void google_protobuf_FileOptions_set_cc_enable_arenas(google_protobuf_FileOptions *msg, bool value) {
  _upb_sethas(msg, 12);
  *UPB_PTR_AT(msg, UPB_SIZE(15, 15), bool) = value;
}
UPB_INLINE void google_protobuf_FileOptions_set_objc_class_prefix(google_protobuf_FileOptions *msg, upb_strview value) {
  _upb_sethas(msg, 13);
  *UPB_PTR_AT(msg, UPB_SIZE(44, 72), upb_strview) = value;
}
UPB_INLINE void google_protobuf_FileOptions_set_csharp_namespace(google_protobuf_FileOptions *msg, upb_strview value) {
  _upb_sethas(msg, 14);
  *UPB_PTR_AT(msg, UPB_SIZE(52, 88), upb_strview) = value;
}
UPB_INLINE void google_protobuf_FileOptions_set_swift_prefix(google_protobuf_FileOptions *msg, upb_strview value) {
  _upb_sethas(msg, 15);
  *UPB_PTR_AT(msg, UPB_SIZE(60, 104), upb_strview) = value;
}
UPB_INLINE void google_protobuf_FileOptions_set_php_class_prefix(google_protobuf_FileOptions *msg, upb_strview value) {
  _upb_sethas(msg, 16);
  *UPB_PTR_AT(msg, UPB_SIZE(68, 120), upb_strview) = value;
}
UPB_INLINE void google_protobuf_FileOptions_set_php_namespace(google_protobuf_FileOptions *msg, upb_strview value) {
  _upb_sethas(msg, 17);
  *UPB_PTR_AT(msg, UPB_SIZE(76, 136), upb_strview) = value;
}
UPB_INLINE void google_protobuf_FileOptions_set_php_generic_services(google_protobuf_FileOptions *msg, bool value) {
  _upb_sethas(msg, 18);
  *UPB_PTR_AT(msg, UPB_SIZE(16, 16), bool) = value;
}
UPB_INLINE void google_protobuf_FileOptions_set_php_metadata_namespace(google_protobuf_FileOptions *msg, upb_strview value) {
  _upb_sethas(msg, 19);
  *UPB_PTR_AT(msg, UPB_SIZE(84, 152), upb_strview) = value;
}
UPB_INLINE void google_protobuf_FileOptions_set_ruby_package(google_protobuf_FileOptions *msg, upb_strview value) {
  _upb_sethas(msg, 20);
  *UPB_PTR_AT(msg, UPB_SIZE(92, 168), upb_strview) = value;
}
UPB_INLINE google_protobuf_UninterpretedOption** google_protobuf_FileOptions_mutable_uninterpreted_option(google_protobuf_FileOptions *msg, size_t *len) {
  return (google_protobuf_UninterpretedOption**)_upb_array_mutable_accessor(msg, UPB_SIZE(100, 184), len);
}
UPB_INLINE google_protobuf_UninterpretedOption** google_protobuf_FileOptions_resize_uninterpreted_option(google_protobuf_FileOptions *msg, size_t len, upb_arena *arena) {
  return (google_protobuf_UninterpretedOption**)_upb_array_resize_accessor2(msg, UPB_SIZE(100, 184), len, UPB_SIZE(2, 3), arena);
}
UPB_INLINE struct google_protobuf_UninterpretedOption* google_protobuf_FileOptions_add_uninterpreted_option(google_protobuf_FileOptions *msg, upb_arena *arena) {
  struct google_protobuf_UninterpretedOption* sub = (struct google_protobuf_UninterpretedOption*)_upb_msg_new(&google_protobuf_UninterpretedOption_msginit, arena);
  bool ok = _upb_array_append_accessor2(
      msg, UPB_SIZE(100, 184), UPB_SIZE(2, 3), &sub, arena);
  if (!ok) return NULL;
  return sub;
}

/* google.protobuf.MessageOptions */

UPB_INLINE google_protobuf_MessageOptions *google_protobuf_MessageOptions_new(upb_arena *arena) {
  return (google_protobuf_MessageOptions *)_upb_msg_new(&google_protobuf_MessageOptions_msginit, arena);
}
UPB_INLINE google_protobuf_MessageOptions *google_protobuf_MessageOptions_parse(const char *buf, size_t size,
                        upb_arena *arena) {
  google_protobuf_MessageOptions *ret = google_protobuf_MessageOptions_new(arena);
  if (!ret) return NULL;
  if (!upb_decode(buf, size, ret, &google_protobuf_MessageOptions_msginit, arena)) return NULL;
  return ret;
}
UPB_INLINE google_protobuf_MessageOptions *google_protobuf_MessageOptions_parse_ex(const char *buf, size_t size,
                           const upb_extreg *extreg, int options,
                           upb_arena *arena) {
  google_protobuf_MessageOptions *ret = google_protobuf_MessageOptions_new(arena);
  if (!ret) return NULL;
  if (!_upb_decode(buf, size, ret, &google_protobuf_MessageOptions_msginit, extreg, options, arena)) {
    return NULL;
  }
  return ret;
}
UPB_INLINE char *google_protobuf_MessageOptions_serialize(const google_protobuf_MessageOptions *msg, upb_arena *arena, size_t *len) {
  return upb_encode(msg, &google_protobuf_MessageOptions_msginit, arena, len);
}

UPB_INLINE bool google_protobuf_MessageOptions_has_message_set_wire_format(const google_protobuf_MessageOptions *msg) { return _upb_hasbit(msg, 1); }
UPB_INLINE bool google_protobuf_MessageOptions_message_set_wire_format(const google_protobuf_MessageOptions *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(1, 1), bool); }
UPB_INLINE bool google_protobuf_MessageOptions_has_no_standard_descriptor_accessor(const google_protobuf_MessageOptions *msg) { return _upb_hasbit(msg, 2); }
UPB_INLINE bool google_protobuf_MessageOptions_no_standard_descriptor_accessor(const google_protobuf_MessageOptions *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(2, 2), bool); }
UPB_INLINE bool google_protobuf_MessageOptions_has_deprecated(const google_protobuf_MessageOptions *msg) { return _upb_hasbit(msg, 3); }
UPB_INLINE bool google_protobuf_MessageOptions_deprecated(const google_protobuf_MessageOptions *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(3, 3), bool); }
UPB_INLINE bool google_protobuf_MessageOptions_has_map_entry(const google_protobuf_MessageOptions *msg) { return _upb_hasbit(msg, 4); }
UPB_INLINE bool google_protobuf_MessageOptions_map_entry(const google_protobuf_MessageOptions *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(4, 4), bool); }
UPB_INLINE bool google_protobuf_MessageOptions_has_uninterpreted_option(const google_protobuf_MessageOptions *msg) { return _upb_has_submsg_nohasbit(msg, UPB_SIZE(8, 8)); }
UPB_INLINE const google_protobuf_UninterpretedOption* const* google_protobuf_MessageOptions_uninterpreted_option(const google_protobuf_MessageOptions *msg, size_t *len) { return (const google_protobuf_UninterpretedOption* const*)_upb_array_accessor(msg, UPB_SIZE(8, 8), len); }

UPB_INLINE void google_protobuf_MessageOptions_set_message_set_wire_format(google_protobuf_MessageOptions *msg, bool value) {
  _upb_sethas(msg, 1);
  *UPB_PTR_AT(msg, UPB_SIZE(1, 1), bool) = value;
}
UPB_INLINE void google_protobuf_MessageOptions_set_no_standard_descriptor_accessor(google_protobuf_MessageOptions *msg, bool value) {
  _upb_sethas(msg, 2);
  *UPB_PTR_AT(msg, UPB_SIZE(2, 2), bool) = value;
}
UPB_INLINE void google_protobuf_MessageOptions_set_deprecated(google_protobuf_MessageOptions *msg, bool value) {
  _upb_sethas(msg, 3);
  *UPB_PTR_AT(msg, UPB_SIZE(3, 3), bool) = value;
}
UPB_INLINE void google_protobuf_MessageOptions_set_map_entry(google_protobuf_MessageOptions *msg, bool value) {
  _upb_sethas(msg, 4);
  *UPB_PTR_AT(msg, UPB_SIZE(4, 4), bool) = value;
}
UPB_INLINE google_protobuf_UninterpretedOption** google_protobuf_MessageOptions_mutable_uninterpreted_option(google_protobuf_MessageOptions *msg, size_t *len) {
  return (google_protobuf_UninterpretedOption**)_upb_array_mutable_accessor(msg, UPB_SIZE(8, 8), len);
}
UPB_INLINE google_protobuf_UninterpretedOption** google_protobuf_MessageOptions_resize_uninterpreted_option(google_protobuf_MessageOptions *msg, size_t len, upb_arena *arena) {
  return (google_protobuf_UninterpretedOption**)_upb_array_resize_accessor2(msg, UPB_SIZE(8, 8), len, UPB_SIZE(2, 3), arena);
}
UPB_INLINE struct google_protobuf_UninterpretedOption* google_protobuf_MessageOptions_add_uninterpreted_option(google_protobuf_MessageOptions *msg, upb_arena *arena) {
  struct google_protobuf_UninterpretedOption* sub = (struct google_protobuf_UninterpretedOption*)_upb_msg_new(&google_protobuf_UninterpretedOption_msginit, arena);
  bool ok = _upb_array_append_accessor2(
      msg, UPB_SIZE(8, 8), UPB_SIZE(2, 3), &sub, arena);
  if (!ok) return NULL;
  return sub;
}

/* google.protobuf.FieldOptions */

UPB_INLINE google_protobuf_FieldOptions *google_protobuf_FieldOptions_new(upb_arena *arena) {
  return (google_protobuf_FieldOptions *)_upb_msg_new(&google_protobuf_FieldOptions_msginit, arena);
}
UPB_INLINE google_protobuf_FieldOptions *google_protobuf_FieldOptions_parse(const char *buf, size_t size,
                        upb_arena *arena) {
  google_protobuf_FieldOptions *ret = google_protobuf_FieldOptions_new(arena);
  if (!ret) return NULL;
  if (!upb_decode(buf, size, ret, &google_protobuf_FieldOptions_msginit, arena)) return NULL;
  return ret;
}
UPB_INLINE google_protobuf_FieldOptions *google_protobuf_FieldOptions_parse_ex(const char *buf, size_t size,
                           const upb_extreg *extreg, int options,
                           upb_arena *arena) {
  google_protobuf_FieldOptions *ret = google_protobuf_FieldOptions_new(arena);
  if (!ret) return NULL;
  if (!_upb_decode(buf, size, ret, &google_protobuf_FieldOptions_msginit, extreg, options, arena)) {
    return NULL;
  }
  return ret;
}
UPB_INLINE char *google_protobuf_FieldOptions_serialize(const google_protobuf_FieldOptions *msg, upb_arena *arena, size_t *len) {
  return upb_encode(msg, &google_protobuf_FieldOptions_msginit, arena, len);
}

UPB_INLINE bool google_protobuf_FieldOptions_has_ctype(const google_protobuf_FieldOptions *msg) { return _upb_hasbit(msg, 1); }
UPB_INLINE int32_t google_protobuf_FieldOptions_ctype(const google_protobuf_FieldOptions *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(4, 4), int32_t); }
UPB_INLINE bool google_protobuf_FieldOptions_has_packed(const google_protobuf_FieldOptions *msg) { return _upb_hasbit(msg, 2); }
UPB_INLINE bool google_protobuf_FieldOptions_packed(const google_protobuf_FieldOptions *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(12, 12), bool); }
UPB_INLINE bool google_protobuf_FieldOptions_has_deprecated(const google_protobuf_FieldOptions *msg) { return _upb_hasbit(msg, 3); }
UPB_INLINE bool google_protobuf_FieldOptions_deprecated(const google_protobuf_FieldOptions *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(13, 13), bool); }
UPB_INLINE bool google_protobuf_FieldOptions_has_lazy(const google_protobuf_FieldOptions *msg) { return _upb_hasbit(msg, 4); }
UPB_INLINE bool google_protobuf_FieldOptions_lazy(const google_protobuf_FieldOptions *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(14, 14), bool); }
UPB_INLINE bool google_protobuf_FieldOptions_has_jstype(const google_protobuf_FieldOptions *msg) { return _upb_hasbit(msg, 5); }
UPB_INLINE int32_t google_protobuf_FieldOptions_jstype(const google_protobuf_FieldOptions *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(8, 8), int32_t); }
UPB_INLINE bool google_protobuf_FieldOptions_has_weak(const google_protobuf_FieldOptions *msg) { return _upb_hasbit(msg, 6); }
UPB_INLINE bool google_protobuf_FieldOptions_weak(const google_protobuf_FieldOptions *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(15, 15), bool); }
UPB_INLINE bool google_protobuf_FieldOptions_has_uninterpreted_option(const google_protobuf_FieldOptions *msg) { return _upb_has_submsg_nohasbit(msg, UPB_SIZE(16, 16)); }
UPB_INLINE const google_protobuf_UninterpretedOption* const* google_protobuf_FieldOptions_uninterpreted_option(const google_protobuf_FieldOptions *msg, size_t *len) { return (const google_protobuf_UninterpretedOption* const*)_upb_array_accessor(msg, UPB_SIZE(16, 16), len); }

UPB_INLINE void google_protobuf_FieldOptions_set_ctype(google_protobuf_FieldOptions *msg, int32_t value) {
  _upb_sethas(msg, 1);
  *UPB_PTR_AT(msg, UPB_SIZE(4, 4), int32_t) = value;
}
UPB_INLINE void google_protobuf_FieldOptions_set_packed(google_protobuf_FieldOptions *msg, bool value) {
  _upb_sethas(msg, 2);
  *UPB_PTR_AT(msg, UPB_SIZE(12, 12), bool) = value;
}
UPB_INLINE void google_protobuf_FieldOptions_set_deprecated(google_protobuf_FieldOptions *msg, bool value) {
  _upb_sethas(msg, 3);
  *UPB_PTR_AT(msg, UPB_SIZE(13, 13), bool) = value;
}
UPB_INLINE void google_protobuf_FieldOptions_set_lazy(google_protobuf_FieldOptions *msg, bool value) {
  _upb_sethas(msg, 4);
  *UPB_PTR_AT(msg, UPB_SIZE(14, 14), bool) = value;
}
UPB_INLINE void google_protobuf_FieldOptions_set_jstype(google_protobuf_FieldOptions *msg, int32_t value) {
  _upb_sethas(msg, 5);
  *UPB_PTR_AT(msg, UPB_SIZE(8, 8), int32_t) = value;
}
UPB_INLINE void google_protobuf_FieldOptions_set_weak(google_protobuf_FieldOptions *msg, bool value) {
  _upb_sethas(msg, 6);
  *UPB_PTR_AT(msg, UPB_SIZE(15, 15), bool) = value;
}
UPB_INLINE google_protobuf_UninterpretedOption** google_protobuf_FieldOptions_mutable_uninterpreted_option(google_protobuf_FieldOptions *msg, size_t *len) {
  return (google_protobuf_UninterpretedOption**)_upb_array_mutable_accessor(msg, UPB_SIZE(16, 16), len);
}
UPB_INLINE google_protobuf_UninterpretedOption** google_protobuf_FieldOptions_resize_uninterpreted_option(google_protobuf_FieldOptions *msg, size_t len, upb_arena *arena) {
  return (google_protobuf_UninterpretedOption**)_upb_array_resize_accessor2(msg, UPB_SIZE(16, 16), len, UPB_SIZE(2, 3), arena);
}
UPB_INLINE struct google_protobuf_UninterpretedOption* google_protobuf_FieldOptions_add_uninterpreted_option(google_protobuf_FieldOptions *msg, upb_arena *arena) {
  struct google_protobuf_UninterpretedOption* sub = (struct google_protobuf_UninterpretedOption*)_upb_msg_new(&google_protobuf_UninterpretedOption_msginit, arena);
  bool ok = _upb_array_append_accessor2(
      msg, UPB_SIZE(16, 16), UPB_SIZE(2, 3), &sub, arena);
  if (!ok) return NULL;
  return sub;
}

/* google.protobuf.OneofOptions */

UPB_INLINE google_protobuf_OneofOptions *google_protobuf_OneofOptions_new(upb_arena *arena) {
  return (google_protobuf_OneofOptions *)_upb_msg_new(&google_protobuf_OneofOptions_msginit, arena);
}
UPB_INLINE google_protobuf_OneofOptions *google_protobuf_OneofOptions_parse(const char *buf, size_t size,
                        upb_arena *arena) {
  google_protobuf_OneofOptions *ret = google_protobuf_OneofOptions_new(arena);
  if (!ret) return NULL;
  if (!upb_decode(buf, size, ret, &google_protobuf_OneofOptions_msginit, arena)) return NULL;
  return ret;
}
UPB_INLINE google_protobuf_OneofOptions *google_protobuf_OneofOptions_parse_ex(const char *buf, size_t size,
                           const upb_extreg *extreg, int options,
                           upb_arena *arena) {
  google_protobuf_OneofOptions *ret = google_protobuf_OneofOptions_new(arena);
  if (!ret) return NULL;
  if (!_upb_decode(buf, size, ret, &google_protobuf_OneofOptions_msginit, extreg, options, arena)) {
    return NULL;
  }
  return ret;
}
UPB_INLINE char *google_protobuf_OneofOptions_serialize(const google_protobuf_OneofOptions *msg, upb_arena *arena, size_t *len) {
  return upb_encode(msg, &google_protobuf_OneofOptions_msginit, arena, len);
}

UPB_INLINE bool google_protobuf_OneofOptions_has_uninterpreted_option(const google_protobuf_OneofOptions *msg) { return _upb_has_submsg_nohasbit(msg, UPB_SIZE(0, 0)); }
UPB_INLINE const google_protobuf_UninterpretedOption* const* google_protobuf_OneofOptions_uninterpreted_option(const google_protobuf_OneofOptions *msg, size_t *len) { return (const google_protobuf_UninterpretedOption* const*)_upb_array_accessor(msg, UPB_SIZE(0, 0), len); }

UPB_INLINE google_protobuf_UninterpretedOption** google_protobuf_OneofOptions_mutable_uninterpreted_option(google_protobuf_OneofOptions *msg, size_t *len) {
  return (google_protobuf_UninterpretedOption**)_upb_array_mutable_accessor(msg, UPB_SIZE(0, 0), len);
}
UPB_INLINE google_protobuf_UninterpretedOption** google_protobuf_OneofOptions_resize_uninterpreted_option(google_protobuf_OneofOptions *msg, size_t len, upb_arena *arena) {
  return (google_protobuf_UninterpretedOption**)_upb_array_resize_accessor2(msg, UPB_SIZE(0, 0), len, UPB_SIZE(2, 3), arena);
}
UPB_INLINE struct google_protobuf_UninterpretedOption* google_protobuf_OneofOptions_add_uninterpreted_option(google_protobuf_OneofOptions *msg, upb_arena *arena) {
  struct google_protobuf_UninterpretedOption* sub = (struct google_protobuf_UninterpretedOption*)_upb_msg_new(&google_protobuf_UninterpretedOption_msginit, arena);
  bool ok = _upb_array_append_accessor2(
      msg, UPB_SIZE(0, 0), UPB_SIZE(2, 3), &sub, arena);
  if (!ok) return NULL;
  return sub;
}

/* google.protobuf.EnumOptions */

UPB_INLINE google_protobuf_EnumOptions *google_protobuf_EnumOptions_new(upb_arena *arena) {
  return (google_protobuf_EnumOptions *)_upb_msg_new(&google_protobuf_EnumOptions_msginit, arena);
}
UPB_INLINE google_protobuf_EnumOptions *google_protobuf_EnumOptions_parse(const char *buf, size_t size,
                        upb_arena *arena) {
  google_protobuf_EnumOptions *ret = google_protobuf_EnumOptions_new(arena);
  if (!ret) return NULL;
  if (!upb_decode(buf, size, ret, &google_protobuf_EnumOptions_msginit, arena)) return NULL;
  return ret;
}
UPB_INLINE google_protobuf_EnumOptions *google_protobuf_EnumOptions_parse_ex(const char *buf, size_t size,
                           const upb_extreg *extreg, int options,
                           upb_arena *arena) {
  google_protobuf_EnumOptions *ret = google_protobuf_EnumOptions_new(arena);
  if (!ret) return NULL;
  if (!_upb_decode(buf, size, ret, &google_protobuf_EnumOptions_msginit, extreg, options, arena)) {
    return NULL;
  }
  return ret;
}
UPB_INLINE char *google_protobuf_EnumOptions_serialize(const google_protobuf_EnumOptions *msg, upb_arena *arena, size_t *len) {
  return upb_encode(msg, &google_protobuf_EnumOptions_msginit, arena, len);
}

UPB_INLINE bool google_protobuf_EnumOptions_has_allow_alias(const google_protobuf_EnumOptions *msg) { return _upb_hasbit(msg, 1); }
UPB_INLINE bool google_protobuf_EnumOptions_allow_alias(const google_protobuf_EnumOptions *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(1, 1), bool); }
UPB_INLINE bool google_protobuf_EnumOptions_has_deprecated(const google_protobuf_EnumOptions *msg) { return _upb_hasbit(msg, 2); }
UPB_INLINE bool google_protobuf_EnumOptions_deprecated(const google_protobuf_EnumOptions *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(2, 2), bool); }
UPB_INLINE bool google_protobuf_EnumOptions_has_uninterpreted_option(const google_protobuf_EnumOptions *msg) { return _upb_has_submsg_nohasbit(msg, UPB_SIZE(4, 8)); }
UPB_INLINE const google_protobuf_UninterpretedOption* const* google_protobuf_EnumOptions_uninterpreted_option(const google_protobuf_EnumOptions *msg, size_t *len) { return (const google_protobuf_UninterpretedOption* const*)_upb_array_accessor(msg, UPB_SIZE(4, 8), len); }

UPB_INLINE void google_protobuf_EnumOptions_set_allow_alias(google_protobuf_EnumOptions *msg, bool value) {
  _upb_sethas(msg, 1);
  *UPB_PTR_AT(msg, UPB_SIZE(1, 1), bool) = value;
}
UPB_INLINE void google_protobuf_EnumOptions_set_deprecated(google_protobuf_EnumOptions *msg, bool value) {
  _upb_sethas(msg, 2);
  *UPB_PTR_AT(msg, UPB_SIZE(2, 2), bool) = value;
}
UPB_INLINE google_protobuf_UninterpretedOption** google_protobuf_EnumOptions_mutable_uninterpreted_option(google_protobuf_EnumOptions *msg, size_t *len) {
  return (google_protobuf_UninterpretedOption**)_upb_array_mutable_accessor(msg, UPB_SIZE(4, 8), len);
}
UPB_INLINE google_protobuf_UninterpretedOption** google_protobuf_EnumOptions_resize_uninterpreted_option(google_protobuf_EnumOptions *msg, size_t len, upb_arena *arena) {
  return (google_protobuf_UninterpretedOption**)_upb_array_resize_accessor2(msg, UPB_SIZE(4, 8), len, UPB_SIZE(2, 3), arena);
}
UPB_INLINE struct google_protobuf_UninterpretedOption* google_protobuf_EnumOptions_add_uninterpreted_option(google_protobuf_EnumOptions *msg, upb_arena *arena) {
  struct google_protobuf_UninterpretedOption* sub = (struct google_protobuf_UninterpretedOption*)_upb_msg_new(&google_protobuf_UninterpretedOption_msginit, arena);
  bool ok = _upb_array_append_accessor2(
      msg, UPB_SIZE(4, 8), UPB_SIZE(2, 3), &sub, arena);
  if (!ok) return NULL;
  return sub;
}

/* google.protobuf.EnumValueOptions */

UPB_INLINE google_protobuf_EnumValueOptions *google_protobuf_EnumValueOptions_new(upb_arena *arena) {
  return (google_protobuf_EnumValueOptions *)_upb_msg_new(&google_protobuf_EnumValueOptions_msginit, arena);
}
UPB_INLINE google_protobuf_EnumValueOptions *google_protobuf_EnumValueOptions_parse(const char *buf, size_t size,
                        upb_arena *arena) {
  google_protobuf_EnumValueOptions *ret = google_protobuf_EnumValueOptions_new(arena);
  if (!ret) return NULL;
  if (!upb_decode(buf, size, ret, &google_protobuf_EnumValueOptions_msginit, arena)) return NULL;
  return ret;
}
UPB_INLINE google_protobuf_EnumValueOptions *google_protobuf_EnumValueOptions_parse_ex(const char *buf, size_t size,
                           const upb_extreg *extreg, int options,
                           upb_arena *arena) {
  google_protobuf_EnumValueOptions *ret = google_protobuf_EnumValueOptions_new(arena);
  if (!ret) return NULL;
  if (!_upb_decode(buf, size, ret, &google_protobuf_EnumValueOptions_msginit, extreg, options, arena)) {
    return NULL;
  }
  return ret;
}
UPB_INLINE char *google_protobuf_EnumValueOptions_serialize(const google_protobuf_EnumValueOptions *msg, upb_arena *arena, size_t *len) {
  return upb_encode(msg, &google_protobuf_EnumValueOptions_msginit, arena, len);
}

UPB_INLINE bool google_protobuf_EnumValueOptions_has_deprecated(const google_protobuf_EnumValueOptions *msg) { return _upb_hasbit(msg, 1); }
UPB_INLINE bool google_protobuf_EnumValueOptions_deprecated(const google_protobuf_EnumValueOptions *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(1, 1), bool); }
UPB_INLINE bool google_protobuf_EnumValueOptions_has_uninterpreted_option(const google_protobuf_EnumValueOptions *msg) { return _upb_has_submsg_nohasbit(msg, UPB_SIZE(4, 8)); }
UPB_INLINE const google_protobuf_UninterpretedOption* const* google_protobuf_EnumValueOptions_uninterpreted_option(const google_protobuf_EnumValueOptions *msg, size_t *len) { return (const google_protobuf_UninterpretedOption* const*)_upb_array_accessor(msg, UPB_SIZE(4, 8), len); }

UPB_INLINE void google_protobuf_EnumValueOptions_set_deprecated(google_protobuf_EnumValueOptions *msg, bool value) {
  _upb_sethas(msg, 1);
  *UPB_PTR_AT(msg, UPB_SIZE(1, 1), bool) = value;
}
UPB_INLINE google_protobuf_UninterpretedOption** google_protobuf_EnumValueOptions_mutable_uninterpreted_option(google_protobuf_EnumValueOptions *msg, size_t *len) {
  return (google_protobuf_UninterpretedOption**)_upb_array_mutable_accessor(msg, UPB_SIZE(4, 8), len);
}
UPB_INLINE google_protobuf_UninterpretedOption** google_protobuf_EnumValueOptions_resize_uninterpreted_option(google_protobuf_EnumValueOptions *msg, size_t len, upb_arena *arena) {
  return (google_protobuf_UninterpretedOption**)_upb_array_resize_accessor2(msg, UPB_SIZE(4, 8), len, UPB_SIZE(2, 3), arena);
}
UPB_INLINE struct google_protobuf_UninterpretedOption* google_protobuf_EnumValueOptions_add_uninterpreted_option(google_protobuf_EnumValueOptions *msg, upb_arena *arena) {
  struct google_protobuf_UninterpretedOption* sub = (struct google_protobuf_UninterpretedOption*)_upb_msg_new(&google_protobuf_UninterpretedOption_msginit, arena);
  bool ok = _upb_array_append_accessor2(
      msg, UPB_SIZE(4, 8), UPB_SIZE(2, 3), &sub, arena);
  if (!ok) return NULL;
  return sub;
}

/* google.protobuf.ServiceOptions */

UPB_INLINE google_protobuf_ServiceOptions *google_protobuf_ServiceOptions_new(upb_arena *arena) {
  return (google_protobuf_ServiceOptions *)_upb_msg_new(&google_protobuf_ServiceOptions_msginit, arena);
}
UPB_INLINE google_protobuf_ServiceOptions *google_protobuf_ServiceOptions_parse(const char *buf, size_t size,
                        upb_arena *arena) {
  google_protobuf_ServiceOptions *ret = google_protobuf_ServiceOptions_new(arena);
  if (!ret) return NULL;
  if (!upb_decode(buf, size, ret, &google_protobuf_ServiceOptions_msginit, arena)) return NULL;
  return ret;
}
UPB_INLINE google_protobuf_ServiceOptions *google_protobuf_ServiceOptions_parse_ex(const char *buf, size_t size,
                           const upb_extreg *extreg, int options,
                           upb_arena *arena) {
  google_protobuf_ServiceOptions *ret = google_protobuf_ServiceOptions_new(arena);
  if (!ret) return NULL;
  if (!_upb_decode(buf, size, ret, &google_protobuf_ServiceOptions_msginit, extreg, options, arena)) {
    return NULL;
  }
  return ret;
}
UPB_INLINE char *google_protobuf_ServiceOptions_serialize(const google_protobuf_ServiceOptions *msg, upb_arena *arena, size_t *len) {
  return upb_encode(msg, &google_protobuf_ServiceOptions_msginit, arena, len);
}

UPB_INLINE bool google_protobuf_ServiceOptions_has_deprecated(const google_protobuf_ServiceOptions *msg) { return _upb_hasbit(msg, 1); }
UPB_INLINE bool google_protobuf_ServiceOptions_deprecated(const google_protobuf_ServiceOptions *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(1, 1), bool); }
UPB_INLINE bool google_protobuf_ServiceOptions_has_uninterpreted_option(const google_protobuf_ServiceOptions *msg) { return _upb_has_submsg_nohasbit(msg, UPB_SIZE(4, 8)); }
UPB_INLINE const google_protobuf_UninterpretedOption* const* google_protobuf_ServiceOptions_uninterpreted_option(const google_protobuf_ServiceOptions *msg, size_t *len) { return (const google_protobuf_UninterpretedOption* const*)_upb_array_accessor(msg, UPB_SIZE(4, 8), len); }

UPB_INLINE void google_protobuf_ServiceOptions_set_deprecated(google_protobuf_ServiceOptions *msg, bool value) {
  _upb_sethas(msg, 1);
  *UPB_PTR_AT(msg, UPB_SIZE(1, 1), bool) = value;
}
UPB_INLINE google_protobuf_UninterpretedOption** google_protobuf_ServiceOptions_mutable_uninterpreted_option(google_protobuf_ServiceOptions *msg, size_t *len) {
  return (google_protobuf_UninterpretedOption**)_upb_array_mutable_accessor(msg, UPB_SIZE(4, 8), len);
}
UPB_INLINE google_protobuf_UninterpretedOption** google_protobuf_ServiceOptions_resize_uninterpreted_option(google_protobuf_ServiceOptions *msg, size_t len, upb_arena *arena) {
  return (google_protobuf_UninterpretedOption**)_upb_array_resize_accessor2(msg, UPB_SIZE(4, 8), len, UPB_SIZE(2, 3), arena);
}
UPB_INLINE struct google_protobuf_UninterpretedOption* google_protobuf_ServiceOptions_add_uninterpreted_option(google_protobuf_ServiceOptions *msg, upb_arena *arena) {
  struct google_protobuf_UninterpretedOption* sub = (struct google_protobuf_UninterpretedOption*)_upb_msg_new(&google_protobuf_UninterpretedOption_msginit, arena);
  bool ok = _upb_array_append_accessor2(
      msg, UPB_SIZE(4, 8), UPB_SIZE(2, 3), &sub, arena);
  if (!ok) return NULL;
  return sub;
}

/* google.protobuf.MethodOptions */

UPB_INLINE google_protobuf_MethodOptions *google_protobuf_MethodOptions_new(upb_arena *arena) {
  return (google_protobuf_MethodOptions *)_upb_msg_new(&google_protobuf_MethodOptions_msginit, arena);
}
UPB_INLINE google_protobuf_MethodOptions *google_protobuf_MethodOptions_parse(const char *buf, size_t size,
                        upb_arena *arena) {
  google_protobuf_MethodOptions *ret = google_protobuf_MethodOptions_new(arena);
  if (!ret) return NULL;
  if (!upb_decode(buf, size, ret, &google_protobuf_MethodOptions_msginit, arena)) return NULL;
  return ret;
}
UPB_INLINE google_protobuf_MethodOptions *google_protobuf_MethodOptions_parse_ex(const char *buf, size_t size,
                           const upb_extreg *extreg, int options,
                           upb_arena *arena) {
  google_protobuf_MethodOptions *ret = google_protobuf_MethodOptions_new(arena);
  if (!ret) return NULL;
  if (!_upb_decode(buf, size, ret, &google_protobuf_MethodOptions_msginit, extreg, options, arena)) {
    return NULL;
  }
  return ret;
}
UPB_INLINE char *google_protobuf_MethodOptions_serialize(const google_protobuf_MethodOptions *msg, upb_arena *arena, size_t *len) {
  return upb_encode(msg, &google_protobuf_MethodOptions_msginit, arena, len);
}

UPB_INLINE bool google_protobuf_MethodOptions_has_deprecated(const google_protobuf_MethodOptions *msg) { return _upb_hasbit(msg, 1); }
UPB_INLINE bool google_protobuf_MethodOptions_deprecated(const google_protobuf_MethodOptions *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(8, 8), bool); }
UPB_INLINE bool google_protobuf_MethodOptions_has_idempotency_level(const google_protobuf_MethodOptions *msg) { return _upb_hasbit(msg, 2); }
UPB_INLINE int32_t google_protobuf_MethodOptions_idempotency_level(const google_protobuf_MethodOptions *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(4, 4), int32_t); }
UPB_INLINE bool google_protobuf_MethodOptions_has_uninterpreted_option(const google_protobuf_MethodOptions *msg) { return _upb_has_submsg_nohasbit(msg, UPB_SIZE(12, 16)); }
UPB_INLINE const google_protobuf_UninterpretedOption* const* google_protobuf_MethodOptions_uninterpreted_option(const google_protobuf_MethodOptions *msg, size_t *len) { return (const google_protobuf_UninterpretedOption* const*)_upb_array_accessor(msg, UPB_SIZE(12, 16), len); }

UPB_INLINE void google_protobuf_MethodOptions_set_deprecated(google_protobuf_MethodOptions *msg, bool value) {
  _upb_sethas(msg, 1);
  *UPB_PTR_AT(msg, UPB_SIZE(8, 8), bool) = value;
}
UPB_INLINE void google_protobuf_MethodOptions_set_idempotency_level(google_protobuf_MethodOptions *msg, int32_t value) {
  _upb_sethas(msg, 2);
  *UPB_PTR_AT(msg, UPB_SIZE(4, 4), int32_t) = value;
}
UPB_INLINE google_protobuf_UninterpretedOption** google_protobuf_MethodOptions_mutable_uninterpreted_option(google_protobuf_MethodOptions *msg, size_t *len) {
  return (google_protobuf_UninterpretedOption**)_upb_array_mutable_accessor(msg, UPB_SIZE(12, 16), len);
}
UPB_INLINE google_protobuf_UninterpretedOption** google_protobuf_MethodOptions_resize_uninterpreted_option(google_protobuf_MethodOptions *msg, size_t len, upb_arena *arena) {
  return (google_protobuf_UninterpretedOption**)_upb_array_resize_accessor2(msg, UPB_SIZE(12, 16), len, UPB_SIZE(2, 3), arena);
}
UPB_INLINE struct google_protobuf_UninterpretedOption* google_protobuf_MethodOptions_add_uninterpreted_option(google_protobuf_MethodOptions *msg, upb_arena *arena) {
  struct google_protobuf_UninterpretedOption* sub = (struct google_protobuf_UninterpretedOption*)_upb_msg_new(&google_protobuf_UninterpretedOption_msginit, arena);
  bool ok = _upb_array_append_accessor2(
      msg, UPB_SIZE(12, 16), UPB_SIZE(2, 3), &sub, arena);
  if (!ok) return NULL;
  return sub;
}

/* google.protobuf.UninterpretedOption */

UPB_INLINE google_protobuf_UninterpretedOption *google_protobuf_UninterpretedOption_new(upb_arena *arena) {
  return (google_protobuf_UninterpretedOption *)_upb_msg_new(&google_protobuf_UninterpretedOption_msginit, arena);
}
UPB_INLINE google_protobuf_UninterpretedOption *google_protobuf_UninterpretedOption_parse(const char *buf, size_t size,
                        upb_arena *arena) {
  google_protobuf_UninterpretedOption *ret = google_protobuf_UninterpretedOption_new(arena);
  if (!ret) return NULL;
  if (!upb_decode(buf, size, ret, &google_protobuf_UninterpretedOption_msginit, arena)) return NULL;
  return ret;
}
UPB_INLINE google_protobuf_UninterpretedOption *google_protobuf_UninterpretedOption_parse_ex(const char *buf, size_t size,
                           const upb_extreg *extreg, int options,
                           upb_arena *arena) {
  google_protobuf_UninterpretedOption *ret = google_protobuf_UninterpretedOption_new(arena);
  if (!ret) return NULL;
  if (!_upb_decode(buf, size, ret, &google_protobuf_UninterpretedOption_msginit, extreg, options, arena)) {
    return NULL;
  }
  return ret;
}
UPB_INLINE char *google_protobuf_UninterpretedOption_serialize(const google_protobuf_UninterpretedOption *msg, upb_arena *arena, size_t *len) {
  return upb_encode(msg, &google_protobuf_UninterpretedOption_msginit, arena, len);
}

UPB_INLINE bool google_protobuf_UninterpretedOption_has_name(const google_protobuf_UninterpretedOption *msg) { return _upb_has_submsg_nohasbit(msg, UPB_SIZE(56, 80)); }
UPB_INLINE const google_protobuf_UninterpretedOption_NamePart* const* google_protobuf_UninterpretedOption_name(const google_protobuf_UninterpretedOption *msg, size_t *len) { return (const google_protobuf_UninterpretedOption_NamePart* const*)_upb_array_accessor(msg, UPB_SIZE(56, 80), len); }
UPB_INLINE bool google_protobuf_UninterpretedOption_has_identifier_value(const google_protobuf_UninterpretedOption *msg) { return _upb_hasbit(msg, 1); }
UPB_INLINE upb_strview google_protobuf_UninterpretedOption_identifier_value(const google_protobuf_UninterpretedOption *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(32, 32), upb_strview); }
UPB_INLINE bool google_protobuf_UninterpretedOption_has_positive_int_value(const google_protobuf_UninterpretedOption *msg) { return _upb_hasbit(msg, 2); }
UPB_INLINE uint64_t google_protobuf_UninterpretedOption_positive_int_value(const google_protobuf_UninterpretedOption *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(8, 8), uint64_t); }
UPB_INLINE bool google_protobuf_UninterpretedOption_has_negative_int_value(const google_protobuf_UninterpretedOption *msg) { return _upb_hasbit(msg, 3); }
UPB_INLINE int64_t google_protobuf_UninterpretedOption_negative_int_value(const google_protobuf_UninterpretedOption *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(16, 16), int64_t); }
UPB_INLINE bool google_protobuf_UninterpretedOption_has_double_value(const google_protobuf_UninterpretedOption *msg) { return _upb_hasbit(msg, 4); }
UPB_INLINE double google_protobuf_UninterpretedOption_double_value(const google_protobuf_UninterpretedOption *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(24, 24), double); }
UPB_INLINE bool google_protobuf_UninterpretedOption_has_string_value(const google_protobuf_UninterpretedOption *msg) { return _upb_hasbit(msg, 5); }
UPB_INLINE upb_strview google_protobuf_UninterpretedOption_string_value(const google_protobuf_UninterpretedOption *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(40, 48), upb_strview); }
UPB_INLINE bool google_protobuf_UninterpretedOption_has_aggregate_value(const google_protobuf_UninterpretedOption *msg) { return _upb_hasbit(msg, 6); }
UPB_INLINE upb_strview google_protobuf_UninterpretedOption_aggregate_value(const google_protobuf_UninterpretedOption *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(48, 64), upb_strview); }

UPB_INLINE google_protobuf_UninterpretedOption_NamePart** google_protobuf_UninterpretedOption_mutable_name(google_protobuf_UninterpretedOption *msg, size_t *len) {
  return (google_protobuf_UninterpretedOption_NamePart**)_upb_array_mutable_accessor(msg, UPB_SIZE(56, 80), len);
}
UPB_INLINE google_protobuf_UninterpretedOption_NamePart** google_protobuf_UninterpretedOption_resize_name(google_protobuf_UninterpretedOption *msg, size_t len, upb_arena *arena) {
  return (google_protobuf_UninterpretedOption_NamePart**)_upb_array_resize_accessor2(msg, UPB_SIZE(56, 80), len, UPB_SIZE(2, 3), arena);
}
UPB_INLINE struct google_protobuf_UninterpretedOption_NamePart* google_protobuf_UninterpretedOption_add_name(google_protobuf_UninterpretedOption *msg, upb_arena *arena) {
  struct google_protobuf_UninterpretedOption_NamePart* sub = (struct google_protobuf_UninterpretedOption_NamePart*)_upb_msg_new(&google_protobuf_UninterpretedOption_NamePart_msginit, arena);
  bool ok = _upb_array_append_accessor2(
      msg, UPB_SIZE(56, 80), UPB_SIZE(2, 3), &sub, arena);
  if (!ok) return NULL;
  return sub;
}
UPB_INLINE void google_protobuf_UninterpretedOption_set_identifier_value(google_protobuf_UninterpretedOption *msg, upb_strview value) {
  _upb_sethas(msg, 1);
  *UPB_PTR_AT(msg, UPB_SIZE(32, 32), upb_strview) = value;
}
UPB_INLINE void google_protobuf_UninterpretedOption_set_positive_int_value(google_protobuf_UninterpretedOption *msg, uint64_t value) {
  _upb_sethas(msg, 2);
  *UPB_PTR_AT(msg, UPB_SIZE(8, 8), uint64_t) = value;
}
UPB_INLINE void google_protobuf_UninterpretedOption_set_negative_int_value(google_protobuf_UninterpretedOption *msg, int64_t value) {
  _upb_sethas(msg, 3);
  *UPB_PTR_AT(msg, UPB_SIZE(16, 16), int64_t) = value;
}
UPB_INLINE void google_protobuf_UninterpretedOption_set_double_value(google_protobuf_UninterpretedOption *msg, double value) {
  _upb_sethas(msg, 4);
  *UPB_PTR_AT(msg, UPB_SIZE(24, 24), double) = value;
}
UPB_INLINE void google_protobuf_UninterpretedOption_set_string_value(google_protobuf_UninterpretedOption *msg, upb_strview value) {
  _upb_sethas(msg, 5);
  *UPB_PTR_AT(msg, UPB_SIZE(40, 48), upb_strview) = value;
}
UPB_INLINE void google_protobuf_UninterpretedOption_set_aggregate_value(google_protobuf_UninterpretedOption *msg, upb_strview value) {
  _upb_sethas(msg, 6);
  *UPB_PTR_AT(msg, UPB_SIZE(48, 64), upb_strview) = value;
}

/* google.protobuf.UninterpretedOption.NamePart */

UPB_INLINE google_protobuf_UninterpretedOption_NamePart *google_protobuf_UninterpretedOption_NamePart_new(upb_arena *arena) {
  return (google_protobuf_UninterpretedOption_NamePart *)_upb_msg_new(&google_protobuf_UninterpretedOption_NamePart_msginit, arena);
}
UPB_INLINE google_protobuf_UninterpretedOption_NamePart *google_protobuf_UninterpretedOption_NamePart_parse(const char *buf, size_t size,
                        upb_arena *arena) {
  google_protobuf_UninterpretedOption_NamePart *ret = google_protobuf_UninterpretedOption_NamePart_new(arena);
  if (!ret) return NULL;
  if (!upb_decode(buf, size, ret, &google_protobuf_UninterpretedOption_NamePart_msginit, arena)) return NULL;
  return ret;
}
UPB_INLINE google_protobuf_UninterpretedOption_NamePart *google_protobuf_UninterpretedOption_NamePart_parse_ex(const char *buf, size_t size,
                           const upb_extreg *extreg, int options,
                           upb_arena *arena) {
  google_protobuf_UninterpretedOption_NamePart *ret = google_protobuf_UninterpretedOption_NamePart_new(arena);
  if (!ret) return NULL;
  if (!_upb_decode(buf, size, ret, &google_protobuf_UninterpretedOption_NamePart_msginit, extreg, options, arena)) {
    return NULL;
  }
  return ret;
}
UPB_INLINE char *google_protobuf_UninterpretedOption_NamePart_serialize(const google_protobuf_UninterpretedOption_NamePart *msg, upb_arena *arena, size_t *len) {
  return upb_encode(msg, &google_protobuf_UninterpretedOption_NamePart_msginit, arena, len);
}

UPB_INLINE bool google_protobuf_UninterpretedOption_NamePart_has_name_part(const google_protobuf_UninterpretedOption_NamePart *msg) { return _upb_hasbit(msg, 1); }
UPB_INLINE upb_strview google_protobuf_UninterpretedOption_NamePart_name_part(const google_protobuf_UninterpretedOption_NamePart *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(4, 8), upb_strview); }
UPB_INLINE bool google_protobuf_UninterpretedOption_NamePart_has_is_extension(const google_protobuf_UninterpretedOption_NamePart *msg) { return _upb_hasbit(msg, 2); }
UPB_INLINE bool google_protobuf_UninterpretedOption_NamePart_is_extension(const google_protobuf_UninterpretedOption_NamePart *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(1, 1), bool); }

UPB_INLINE void google_protobuf_UninterpretedOption_NamePart_set_name_part(google_protobuf_UninterpretedOption_NamePart *msg, upb_strview value) {
  _upb_sethas(msg, 1);
  *UPB_PTR_AT(msg, UPB_SIZE(4, 8), upb_strview) = value;
}
UPB_INLINE void google_protobuf_UninterpretedOption_NamePart_set_is_extension(google_protobuf_UninterpretedOption_NamePart *msg, bool value) {
  _upb_sethas(msg, 2);
  *UPB_PTR_AT(msg, UPB_SIZE(1, 1), bool) = value;
}

/* google.protobuf.SourceCodeInfo */

UPB_INLINE google_protobuf_SourceCodeInfo *google_protobuf_SourceCodeInfo_new(upb_arena *arena) {
  return (google_protobuf_SourceCodeInfo *)_upb_msg_new(&google_protobuf_SourceCodeInfo_msginit, arena);
}
UPB_INLINE google_protobuf_SourceCodeInfo *google_protobuf_SourceCodeInfo_parse(const char *buf, size_t size,
                        upb_arena *arena) {
  google_protobuf_SourceCodeInfo *ret = google_protobuf_SourceCodeInfo_new(arena);
  if (!ret) return NULL;
  if (!upb_decode(buf, size, ret, &google_protobuf_SourceCodeInfo_msginit, arena)) return NULL;
  return ret;
}
UPB_INLINE google_protobuf_SourceCodeInfo *google_protobuf_SourceCodeInfo_parse_ex(const char *buf, size_t size,
                           const upb_extreg *extreg, int options,
                           upb_arena *arena) {
  google_protobuf_SourceCodeInfo *ret = google_protobuf_SourceCodeInfo_new(arena);
  if (!ret) return NULL;
  if (!_upb_decode(buf, size, ret, &google_protobuf_SourceCodeInfo_msginit, extreg, options, arena)) {
    return NULL;
  }
  return ret;
}
UPB_INLINE char *google_protobuf_SourceCodeInfo_serialize(const google_protobuf_SourceCodeInfo *msg, upb_arena *arena, size_t *len) {
  return upb_encode(msg, &google_protobuf_SourceCodeInfo_msginit, arena, len);
}

UPB_INLINE bool google_protobuf_SourceCodeInfo_has_location(const google_protobuf_SourceCodeInfo *msg) { return _upb_has_submsg_nohasbit(msg, UPB_SIZE(0, 0)); }
UPB_INLINE const google_protobuf_SourceCodeInfo_Location* const* google_protobuf_SourceCodeInfo_location(const google_protobuf_SourceCodeInfo *msg, size_t *len) { return (const google_protobuf_SourceCodeInfo_Location* const*)_upb_array_accessor(msg, UPB_SIZE(0, 0), len); }

UPB_INLINE google_protobuf_SourceCodeInfo_Location** google_protobuf_SourceCodeInfo_mutable_location(google_protobuf_SourceCodeInfo *msg, size_t *len) {
  return (google_protobuf_SourceCodeInfo_Location**)_upb_array_mutable_accessor(msg, UPB_SIZE(0, 0), len);
}
UPB_INLINE google_protobuf_SourceCodeInfo_Location** google_protobuf_SourceCodeInfo_resize_location(google_protobuf_SourceCodeInfo *msg, size_t len, upb_arena *arena) {
  return (google_protobuf_SourceCodeInfo_Location**)_upb_array_resize_accessor2(msg, UPB_SIZE(0, 0), len, UPB_SIZE(2, 3), arena);
}
UPB_INLINE struct google_protobuf_SourceCodeInfo_Location* google_protobuf_SourceCodeInfo_add_location(google_protobuf_SourceCodeInfo *msg, upb_arena *arena) {
  struct google_protobuf_SourceCodeInfo_Location* sub = (struct google_protobuf_SourceCodeInfo_Location*)_upb_msg_new(&google_protobuf_SourceCodeInfo_Location_msginit, arena);
  bool ok = _upb_array_append_accessor2(
      msg, UPB_SIZE(0, 0), UPB_SIZE(2, 3), &sub, arena);
  if (!ok) return NULL;
  return sub;
}

/* google.protobuf.SourceCodeInfo.Location */

UPB_INLINE google_protobuf_SourceCodeInfo_Location *google_protobuf_SourceCodeInfo_Location_new(upb_arena *arena) {
  return (google_protobuf_SourceCodeInfo_Location *)_upb_msg_new(&google_protobuf_SourceCodeInfo_Location_msginit, arena);
}
UPB_INLINE google_protobuf_SourceCodeInfo_Location *google_protobuf_SourceCodeInfo_Location_parse(const char *buf, size_t size,
                        upb_arena *arena) {
  google_protobuf_SourceCodeInfo_Location *ret = google_protobuf_SourceCodeInfo_Location_new(arena);
  if (!ret) return NULL;
  if (!upb_decode(buf, size, ret, &google_protobuf_SourceCodeInfo_Location_msginit, arena)) return NULL;
  return ret;
}
UPB_INLINE google_protobuf_SourceCodeInfo_Location *google_protobuf_SourceCodeInfo_Location_parse_ex(const char *buf, size_t size,
                           const upb_extreg *extreg, int options,
                           upb_arena *arena) {
  google_protobuf_SourceCodeInfo_Location *ret = google_protobuf_SourceCodeInfo_Location_new(arena);
  if (!ret) return NULL;
  if (!_upb_decode(buf, size, ret, &google_protobuf_SourceCodeInfo_Location_msginit, extreg, options, arena)) {
    return NULL;
  }
  return ret;
}
UPB_INLINE char *google_protobuf_SourceCodeInfo_Location_serialize(const google_protobuf_SourceCodeInfo_Location *msg, upb_arena *arena, size_t *len) {
  return upb_encode(msg, &google_protobuf_SourceCodeInfo_Location_msginit, arena, len);
}

UPB_INLINE int32_t const* google_protobuf_SourceCodeInfo_Location_path(const google_protobuf_SourceCodeInfo_Location *msg, size_t *len) { return (int32_t const*)_upb_array_accessor(msg, UPB_SIZE(20, 40), len); }
UPB_INLINE int32_t const* google_protobuf_SourceCodeInfo_Location_span(const google_protobuf_SourceCodeInfo_Location *msg, size_t *len) { return (int32_t const*)_upb_array_accessor(msg, UPB_SIZE(24, 48), len); }
UPB_INLINE bool google_protobuf_SourceCodeInfo_Location_has_leading_comments(const google_protobuf_SourceCodeInfo_Location *msg) { return _upb_hasbit(msg, 1); }
UPB_INLINE upb_strview google_protobuf_SourceCodeInfo_Location_leading_comments(const google_protobuf_SourceCodeInfo_Location *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(4, 8), upb_strview); }
UPB_INLINE bool google_protobuf_SourceCodeInfo_Location_has_trailing_comments(const google_protobuf_SourceCodeInfo_Location *msg) { return _upb_hasbit(msg, 2); }
UPB_INLINE upb_strview google_protobuf_SourceCodeInfo_Location_trailing_comments(const google_protobuf_SourceCodeInfo_Location *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(12, 24), upb_strview); }
UPB_INLINE upb_strview const* google_protobuf_SourceCodeInfo_Location_leading_detached_comments(const google_protobuf_SourceCodeInfo_Location *msg, size_t *len) { return (upb_strview const*)_upb_array_accessor(msg, UPB_SIZE(28, 56), len); }

UPB_INLINE int32_t* google_protobuf_SourceCodeInfo_Location_mutable_path(google_protobuf_SourceCodeInfo_Location *msg, size_t *len) {
  return (int32_t*)_upb_array_mutable_accessor(msg, UPB_SIZE(20, 40), len);
}
UPB_INLINE int32_t* google_protobuf_SourceCodeInfo_Location_resize_path(google_protobuf_SourceCodeInfo_Location *msg, size_t len, upb_arena *arena) {
  return (int32_t*)_upb_array_resize_accessor2(msg, UPB_SIZE(20, 40), len, 2, arena);
}
UPB_INLINE bool google_protobuf_SourceCodeInfo_Location_add_path(google_protobuf_SourceCodeInfo_Location *msg, int32_t val, upb_arena *arena) {
  return _upb_array_append_accessor2(msg, UPB_SIZE(20, 40), 2, &val,
      arena);
}
UPB_INLINE int32_t* google_protobuf_SourceCodeInfo_Location_mutable_span(google_protobuf_SourceCodeInfo_Location *msg, size_t *len) {
  return (int32_t*)_upb_array_mutable_accessor(msg, UPB_SIZE(24, 48), len);
}
UPB_INLINE int32_t* google_protobuf_SourceCodeInfo_Location_resize_span(google_protobuf_SourceCodeInfo_Location *msg, size_t len, upb_arena *arena) {
  return (int32_t*)_upb_array_resize_accessor2(msg, UPB_SIZE(24, 48), len, 2, arena);
}
UPB_INLINE bool google_protobuf_SourceCodeInfo_Location_add_span(google_protobuf_SourceCodeInfo_Location *msg, int32_t val, upb_arena *arena) {
  return _upb_array_append_accessor2(msg, UPB_SIZE(24, 48), 2, &val,
      arena);
}
UPB_INLINE void google_protobuf_SourceCodeInfo_Location_set_leading_comments(google_protobuf_SourceCodeInfo_Location *msg, upb_strview value) {
  _upb_sethas(msg, 1);
  *UPB_PTR_AT(msg, UPB_SIZE(4, 8), upb_strview) = value;
}
UPB_INLINE void google_protobuf_SourceCodeInfo_Location_set_trailing_comments(google_protobuf_SourceCodeInfo_Location *msg, upb_strview value) {
  _upb_sethas(msg, 2);
  *UPB_PTR_AT(msg, UPB_SIZE(12, 24), upb_strview) = value;
}
UPB_INLINE upb_strview* google_protobuf_SourceCodeInfo_Location_mutable_leading_detached_comments(google_protobuf_SourceCodeInfo_Location *msg, size_t *len) {
  return (upb_strview*)_upb_array_mutable_accessor(msg, UPB_SIZE(28, 56), len);
}
UPB_INLINE upb_strview* google_protobuf_SourceCodeInfo_Location_resize_leading_detached_comments(google_protobuf_SourceCodeInfo_Location *msg, size_t len, upb_arena *arena) {
  return (upb_strview*)_upb_array_resize_accessor2(msg, UPB_SIZE(28, 56), len, UPB_SIZE(3, 4), arena);
}
UPB_INLINE bool google_protobuf_SourceCodeInfo_Location_add_leading_detached_comments(google_protobuf_SourceCodeInfo_Location *msg, upb_strview val, upb_arena *arena) {
  return _upb_array_append_accessor2(msg, UPB_SIZE(28, 56), UPB_SIZE(3, 4), &val,
      arena);
}

/* google.protobuf.GeneratedCodeInfo */

UPB_INLINE google_protobuf_GeneratedCodeInfo *google_protobuf_GeneratedCodeInfo_new(upb_arena *arena) {
  return (google_protobuf_GeneratedCodeInfo *)_upb_msg_new(&google_protobuf_GeneratedCodeInfo_msginit, arena);
}
UPB_INLINE google_protobuf_GeneratedCodeInfo *google_protobuf_GeneratedCodeInfo_parse(const char *buf, size_t size,
                        upb_arena *arena) {
  google_protobuf_GeneratedCodeInfo *ret = google_protobuf_GeneratedCodeInfo_new(arena);
  if (!ret) return NULL;
  if (!upb_decode(buf, size, ret, &google_protobuf_GeneratedCodeInfo_msginit, arena)) return NULL;
  return ret;
}
UPB_INLINE google_protobuf_GeneratedCodeInfo *google_protobuf_GeneratedCodeInfo_parse_ex(const char *buf, size_t size,
                           const upb_extreg *extreg, int options,
                           upb_arena *arena) {
  google_protobuf_GeneratedCodeInfo *ret = google_protobuf_GeneratedCodeInfo_new(arena);
  if (!ret) return NULL;
  if (!_upb_decode(buf, size, ret, &google_protobuf_GeneratedCodeInfo_msginit, extreg, options, arena)) {
    return NULL;
  }
  return ret;
}
UPB_INLINE char *google_protobuf_GeneratedCodeInfo_serialize(const google_protobuf_GeneratedCodeInfo *msg, upb_arena *arena, size_t *len) {
  return upb_encode(msg, &google_protobuf_GeneratedCodeInfo_msginit, arena, len);
}

UPB_INLINE bool google_protobuf_GeneratedCodeInfo_has_annotation(const google_protobuf_GeneratedCodeInfo *msg) { return _upb_has_submsg_nohasbit(msg, UPB_SIZE(0, 0)); }
UPB_INLINE const google_protobuf_GeneratedCodeInfo_Annotation* const* google_protobuf_GeneratedCodeInfo_annotation(const google_protobuf_GeneratedCodeInfo *msg, size_t *len) { return (const google_protobuf_GeneratedCodeInfo_Annotation* const*)_upb_array_accessor(msg, UPB_SIZE(0, 0), len); }

UPB_INLINE google_protobuf_GeneratedCodeInfo_Annotation** google_protobuf_GeneratedCodeInfo_mutable_annotation(google_protobuf_GeneratedCodeInfo *msg, size_t *len) {
  return (google_protobuf_GeneratedCodeInfo_Annotation**)_upb_array_mutable_accessor(msg, UPB_SIZE(0, 0), len);
}
UPB_INLINE google_protobuf_GeneratedCodeInfo_Annotation** google_protobuf_GeneratedCodeInfo_resize_annotation(google_protobuf_GeneratedCodeInfo *msg, size_t len, upb_arena *arena) {
  return (google_protobuf_GeneratedCodeInfo_Annotation**)_upb_array_resize_accessor2(msg, UPB_SIZE(0, 0), len, UPB_SIZE(2, 3), arena);
}
UPB_INLINE struct google_protobuf_GeneratedCodeInfo_Annotation* google_protobuf_GeneratedCodeInfo_add_annotation(google_protobuf_GeneratedCodeInfo *msg, upb_arena *arena) {
  struct google_protobuf_GeneratedCodeInfo_Annotation* sub = (struct google_protobuf_GeneratedCodeInfo_Annotation*)_upb_msg_new(&google_protobuf_GeneratedCodeInfo_Annotation_msginit, arena);
  bool ok = _upb_array_append_accessor2(
      msg, UPB_SIZE(0, 0), UPB_SIZE(2, 3), &sub, arena);
  if (!ok) return NULL;
  return sub;
}

/* google.protobuf.GeneratedCodeInfo.Annotation */

UPB_INLINE google_protobuf_GeneratedCodeInfo_Annotation *google_protobuf_GeneratedCodeInfo_Annotation_new(upb_arena *arena) {
  return (google_protobuf_GeneratedCodeInfo_Annotation *)_upb_msg_new(&google_protobuf_GeneratedCodeInfo_Annotation_msginit, arena);
}
UPB_INLINE google_protobuf_GeneratedCodeInfo_Annotation *google_protobuf_GeneratedCodeInfo_Annotation_parse(const char *buf, size_t size,
                        upb_arena *arena) {
  google_protobuf_GeneratedCodeInfo_Annotation *ret = google_protobuf_GeneratedCodeInfo_Annotation_new(arena);
  if (!ret) return NULL;
  if (!upb_decode(buf, size, ret, &google_protobuf_GeneratedCodeInfo_Annotation_msginit, arena)) return NULL;
  return ret;
}
UPB_INLINE google_protobuf_GeneratedCodeInfo_Annotation *google_protobuf_GeneratedCodeInfo_Annotation_parse_ex(const char *buf, size_t size,
                           const upb_extreg *extreg, int options,
                           upb_arena *arena) {
  google_protobuf_GeneratedCodeInfo_Annotation *ret = google_protobuf_GeneratedCodeInfo_Annotation_new(arena);
  if (!ret) return NULL;
  if (!_upb_decode(buf, size, ret, &google_protobuf_GeneratedCodeInfo_Annotation_msginit, extreg, options, arena)) {
    return NULL;
  }
  return ret;
}
UPB_INLINE char *google_protobuf_GeneratedCodeInfo_Annotation_serialize(const google_protobuf_GeneratedCodeInfo_Annotation *msg, upb_arena *arena, size_t *len) {
  return upb_encode(msg, &google_protobuf_GeneratedCodeInfo_Annotation_msginit, arena, len);
}

UPB_INLINE int32_t const* google_protobuf_GeneratedCodeInfo_Annotation_path(const google_protobuf_GeneratedCodeInfo_Annotation *msg, size_t *len) { return (int32_t const*)_upb_array_accessor(msg, UPB_SIZE(20, 32), len); }
UPB_INLINE bool google_protobuf_GeneratedCodeInfo_Annotation_has_source_file(const google_protobuf_GeneratedCodeInfo_Annotation *msg) { return _upb_hasbit(msg, 1); }
UPB_INLINE upb_strview google_protobuf_GeneratedCodeInfo_Annotation_source_file(const google_protobuf_GeneratedCodeInfo_Annotation *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(12, 16), upb_strview); }
UPB_INLINE bool google_protobuf_GeneratedCodeInfo_Annotation_has_begin(const google_protobuf_GeneratedCodeInfo_Annotation *msg) { return _upb_hasbit(msg, 2); }
UPB_INLINE int32_t google_protobuf_GeneratedCodeInfo_Annotation_begin(const google_protobuf_GeneratedCodeInfo_Annotation *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(4, 4), int32_t); }
UPB_INLINE bool google_protobuf_GeneratedCodeInfo_Annotation_has_end(const google_protobuf_GeneratedCodeInfo_Annotation *msg) { return _upb_hasbit(msg, 3); }
UPB_INLINE int32_t google_protobuf_GeneratedCodeInfo_Annotation_end(const google_protobuf_GeneratedCodeInfo_Annotation *msg) { return *UPB_PTR_AT(msg, UPB_SIZE(8, 8), int32_t); }

UPB_INLINE int32_t* google_protobuf_GeneratedCodeInfo_Annotation_mutable_path(google_protobuf_GeneratedCodeInfo_Annotation *msg, size_t *len) {
  return (int32_t*)_upb_array_mutable_accessor(msg, UPB_SIZE(20, 32), len);
}
UPB_INLINE int32_t* google_protobuf_GeneratedCodeInfo_Annotation_resize_path(google_protobuf_GeneratedCodeInfo_Annotation *msg, size_t len, upb_arena *arena) {
  return (int32_t*)_upb_array_resize_accessor2(msg, UPB_SIZE(20, 32), len, 2, arena);
}
UPB_INLINE bool google_protobuf_GeneratedCodeInfo_Annotation_add_path(google_protobuf_GeneratedCodeInfo_Annotation *msg, int32_t val, upb_arena *arena) {
  return _upb_array_append_accessor2(msg, UPB_SIZE(20, 32), 2, &val,
      arena);
}
UPB_INLINE void google_protobuf_GeneratedCodeInfo_Annotation_set_source_file(google_protobuf_GeneratedCodeInfo_Annotation *msg, upb_strview value) {
  _upb_sethas(msg, 1);
  *UPB_PTR_AT(msg, UPB_SIZE(12, 16), upb_strview) = value;
}
UPB_INLINE void google_protobuf_GeneratedCodeInfo_Annotation_set_begin(google_protobuf_GeneratedCodeInfo_Annotation *msg, int32_t value) {
  _upb_sethas(msg, 2);
  *UPB_PTR_AT(msg, UPB_SIZE(4, 4), int32_t) = value;
}
UPB_INLINE void google_protobuf_GeneratedCodeInfo_Annotation_set_end(google_protobuf_GeneratedCodeInfo_Annotation *msg, int32_t value) {
  _upb_sethas(msg, 3);
  *UPB_PTR_AT(msg, UPB_SIZE(8, 8), int32_t) = value;
}

#ifdef __cplusplus
}  /* extern "C" */
#endif


#endif  /* GOOGLE_PROTOBUF_DESCRIPTOR_PROTO_UPB_H_ */

/** upb/def.h ************************************************************/
/*
 * Defs are upb's internal representation of the constructs that can appear
 * in a .proto file:
 *
 * - upb_msgdef: describes a "message" construct.
 * - upb_fielddef: describes a message field.
 * - upb_filedef: describes a .proto file and its defs.
 * - upb_enumdef: describes an enum.
 * - upb_oneofdef: describes a oneof.
 *
 * TODO: definitions of services.
 */

#ifndef UPB_DEF_H_
#define UPB_DEF_H_


/* Must be last. */

#ifdef __cplusplus
extern "C" {
#endif  /* __cplusplus */

struct upb_enumdef;
typedef struct upb_enumdef upb_enumdef;
struct upb_fielddef;
typedef struct upb_fielddef upb_fielddef;
struct upb_filedef;
typedef struct upb_filedef upb_filedef;
struct upb_msgdef;
typedef struct upb_msgdef upb_msgdef;
struct upb_oneofdef;
typedef struct upb_oneofdef upb_oneofdef;
struct upb_symtab;
typedef struct upb_symtab upb_symtab;

typedef enum {
  UPB_SYNTAX_PROTO2 = 2,
  UPB_SYNTAX_PROTO3 = 3
} upb_syntax_t;

/* All the different kind of well known type messages. For simplicity of check,
 * number wrappers and string wrappers are grouped together. Make sure the
 * order and merber of these groups are not changed.
 */
typedef enum {
  UPB_WELLKNOWN_UNSPECIFIED,
  UPB_WELLKNOWN_ANY,
  UPB_WELLKNOWN_FIELDMASK,
  UPB_WELLKNOWN_DURATION,
  UPB_WELLKNOWN_TIMESTAMP,
  /* number wrappers */
  UPB_WELLKNOWN_DOUBLEVALUE,
  UPB_WELLKNOWN_FLOATVALUE,
  UPB_WELLKNOWN_INT64VALUE,
  UPB_WELLKNOWN_UINT64VALUE,
  UPB_WELLKNOWN_INT32VALUE,
  UPB_WELLKNOWN_UINT32VALUE,
  /* string wrappers */
  UPB_WELLKNOWN_STRINGVALUE,
  UPB_WELLKNOWN_BYTESVALUE,
  UPB_WELLKNOWN_BOOLVALUE,
  UPB_WELLKNOWN_VALUE,
  UPB_WELLKNOWN_LISTVALUE,
  UPB_WELLKNOWN_STRUCT
} upb_wellknowntype_t;

/* upb_fielddef ***************************************************************/

/* Maximum field number allowed for FieldDefs.  This is an inherent limit of the
 * protobuf wire format. */
#define UPB_MAX_FIELDNUMBER ((1 << 29) - 1)

const char *upb_fielddef_fullname(const upb_fielddef *f);
upb_fieldtype_t upb_fielddef_type(const upb_fielddef *f);
upb_descriptortype_t upb_fielddef_descriptortype(const upb_fielddef *f);
upb_label_t upb_fielddef_label(const upb_fielddef *f);
uint32_t upb_fielddef_number(const upb_fielddef *f);
const char *upb_fielddef_name(const upb_fielddef *f);
const char *upb_fielddef_jsonname(const upb_fielddef *f);
bool upb_fielddef_isextension(const upb_fielddef *f);
bool upb_fielddef_lazy(const upb_fielddef *f);
bool upb_fielddef_packed(const upb_fielddef *f);
const upb_filedef *upb_fielddef_file(const upb_fielddef *f);
const upb_msgdef *upb_fielddef_containingtype(const upb_fielddef *f);
const upb_oneofdef *upb_fielddef_containingoneof(const upb_fielddef *f);
const upb_oneofdef *upb_fielddef_realcontainingoneof(const upb_fielddef *f);
uint32_t upb_fielddef_index(const upb_fielddef *f);
bool upb_fielddef_issubmsg(const upb_fielddef *f);
bool upb_fielddef_isstring(const upb_fielddef *f);
bool upb_fielddef_isseq(const upb_fielddef *f);
bool upb_fielddef_isprimitive(const upb_fielddef *f);
bool upb_fielddef_ismap(const upb_fielddef *f);
int64_t upb_fielddef_defaultint64(const upb_fielddef *f);
int32_t upb_fielddef_defaultint32(const upb_fielddef *f);
uint64_t upb_fielddef_defaultuint64(const upb_fielddef *f);
uint32_t upb_fielddef_defaultuint32(const upb_fielddef *f);
bool upb_fielddef_defaultbool(const upb_fielddef *f);
float upb_fielddef_defaultfloat(const upb_fielddef *f);
double upb_fielddef_defaultdouble(const upb_fielddef *f);
const char *upb_fielddef_defaultstr(const upb_fielddef *f, size_t *len);
bool upb_fielddef_hassubdef(const upb_fielddef *f);
bool upb_fielddef_haspresence(const upb_fielddef *f);
const upb_msgdef *upb_fielddef_msgsubdef(const upb_fielddef *f);
const upb_enumdef *upb_fielddef_enumsubdef(const upb_fielddef *f);
const upb_msglayout_field *upb_fielddef_layout(const upb_fielddef *f);

/* upb_oneofdef ***************************************************************/

typedef upb_inttable_iter upb_oneof_iter;

const char *upb_oneofdef_name(const upb_oneofdef *o);
const upb_msgdef *upb_oneofdef_containingtype(const upb_oneofdef *o);
uint32_t upb_oneofdef_index(const upb_oneofdef *o);
bool upb_oneofdef_issynthetic(const upb_oneofdef *o);
int upb_oneofdef_fieldcount(const upb_oneofdef *o);
const upb_fielddef *upb_oneofdef_field(const upb_oneofdef *o, int i);

/* Oneof lookups:
 * - ntof:  look up a field by name.
 * - ntofz: look up a field by name (as a null-terminated string).
 * - itof:  look up a field by number. */
const upb_fielddef *upb_oneofdef_ntof(const upb_oneofdef *o,
                                      const char *name, size_t length);
UPB_INLINE const upb_fielddef *upb_oneofdef_ntofz(const upb_oneofdef *o,
                                                  const char *name) {
  return upb_oneofdef_ntof(o, name, strlen(name));
}
const upb_fielddef *upb_oneofdef_itof(const upb_oneofdef *o, uint32_t num);

/* DEPRECATED, slated for removal. */
int upb_oneofdef_numfields(const upb_oneofdef *o);
void upb_oneof_begin(upb_oneof_iter *iter, const upb_oneofdef *o);
void upb_oneof_next(upb_oneof_iter *iter);
bool upb_oneof_done(upb_oneof_iter *iter);
upb_fielddef *upb_oneof_iter_field(const upb_oneof_iter *iter);
void upb_oneof_iter_setdone(upb_oneof_iter *iter);
bool upb_oneof_iter_isequal(const upb_oneof_iter *iter1,
                            const upb_oneof_iter *iter2);
/* END DEPRECATED */

/* upb_msgdef *****************************************************************/

typedef upb_inttable_iter upb_msg_field_iter;
typedef upb_strtable_iter upb_msg_oneof_iter;

/* Well-known field tag numbers for map-entry messages. */
#define UPB_MAPENTRY_KEY   1
#define UPB_MAPENTRY_VALUE 2

/* Well-known field tag numbers for Any messages. */
#define UPB_ANY_TYPE 1
#define UPB_ANY_VALUE 2

/* Well-known field tag numbers for timestamp messages. */
#define UPB_DURATION_SECONDS 1
#define UPB_DURATION_NANOS 2

/* Well-known field tag numbers for duration messages. */
#define UPB_TIMESTAMP_SECONDS 1
#define UPB_TIMESTAMP_NANOS 2

const char *upb_msgdef_fullname(const upb_msgdef *m);
const upb_filedef *upb_msgdef_file(const upb_msgdef *m);
const char *upb_msgdef_name(const upb_msgdef *m);
upb_syntax_t upb_msgdef_syntax(const upb_msgdef *m);
bool upb_msgdef_mapentry(const upb_msgdef *m);
upb_wellknowntype_t upb_msgdef_wellknowntype(const upb_msgdef *m);
bool upb_msgdef_iswrapper(const upb_msgdef *m);
bool upb_msgdef_isnumberwrapper(const upb_msgdef *m);
int upb_msgdef_fieldcount(const upb_msgdef *m);
int upb_msgdef_oneofcount(const upb_msgdef *m);
const upb_fielddef *upb_msgdef_field(const upb_msgdef *m, int i);
const upb_oneofdef *upb_msgdef_oneof(const upb_msgdef *m, int i);
const upb_fielddef *upb_msgdef_itof(const upb_msgdef *m, uint32_t i);
const upb_fielddef *upb_msgdef_ntof(const upb_msgdef *m, const char *name,
                                    size_t len);
const upb_oneofdef *upb_msgdef_ntoo(const upb_msgdef *m, const char *name,
                                    size_t len);
const upb_msglayout *upb_msgdef_layout(const upb_msgdef *m);

UPB_INLINE const upb_oneofdef *upb_msgdef_ntooz(const upb_msgdef *m,
                                               const char *name) {
  return upb_msgdef_ntoo(m, name, strlen(name));
}

UPB_INLINE const upb_fielddef *upb_msgdef_ntofz(const upb_msgdef *m,
                                                const char *name) {
  return upb_msgdef_ntof(m, name, strlen(name));
}

/* Lookup of either field or oneof by name.  Returns whether either was found.
 * If the return is true, then the found def will be set, and the non-found
 * one set to NULL. */
bool upb_msgdef_lookupname(const upb_msgdef *m, const char *name, size_t len,
                           const upb_fielddef **f, const upb_oneofdef **o);

UPB_INLINE bool upb_msgdef_lookupnamez(const upb_msgdef *m, const char *name,
                                       const upb_fielddef **f,
                                       const upb_oneofdef **o) {
  return upb_msgdef_lookupname(m, name, strlen(name), f, o);
}

/* Returns a field by either JSON name or regular proto name. */
const upb_fielddef *upb_msgdef_lookupjsonname(const upb_msgdef *m,
                                              const char *name, size_t len);

/* DEPRECATED, slated for removal */
int upb_msgdef_numfields(const upb_msgdef *m);
int upb_msgdef_numoneofs(const upb_msgdef *m);
int upb_msgdef_numrealoneofs(const upb_msgdef *m);
void upb_msg_field_begin(upb_msg_field_iter *iter, const upb_msgdef *m);
void upb_msg_field_next(upb_msg_field_iter *iter);
bool upb_msg_field_done(const upb_msg_field_iter *iter);
upb_fielddef *upb_msg_iter_field(const upb_msg_field_iter *iter);
void upb_msg_field_iter_setdone(upb_msg_field_iter *iter);
bool upb_msg_field_iter_isequal(const upb_msg_field_iter * iter1,
                                const upb_msg_field_iter * iter2);
void upb_msg_oneof_begin(upb_msg_oneof_iter * iter, const upb_msgdef *m);
void upb_msg_oneof_next(upb_msg_oneof_iter * iter);
bool upb_msg_oneof_done(const upb_msg_oneof_iter *iter);
const upb_oneofdef *upb_msg_iter_oneof(const upb_msg_oneof_iter *iter);
void upb_msg_oneof_iter_setdone(upb_msg_oneof_iter * iter);
bool upb_msg_oneof_iter_isequal(const upb_msg_oneof_iter *iter1,
                                const upb_msg_oneof_iter *iter2);
/* END DEPRECATED */

/* upb_enumdef ****************************************************************/

typedef upb_strtable_iter upb_enum_iter;

const char *upb_enumdef_fullname(const upb_enumdef *e);
const char *upb_enumdef_name(const upb_enumdef *e);
const upb_filedef *upb_enumdef_file(const upb_enumdef *e);
int32_t upb_enumdef_default(const upb_enumdef *e);
int upb_enumdef_numvals(const upb_enumdef *e);

/* Enum lookups:
 * - ntoi:  look up a name with specified length.
 * - ntoiz: look up a name provided as a null-terminated string.
 * - iton:  look up an integer, returning the name as a null-terminated
 *          string. */
bool upb_enumdef_ntoi(const upb_enumdef *e, const char *name, size_t len,
                      int32_t *num);
UPB_INLINE bool upb_enumdef_ntoiz(const upb_enumdef *e,
                                  const char *name, int32_t *num) {
  return upb_enumdef_ntoi(e, name, strlen(name), num);
}
const char *upb_enumdef_iton(const upb_enumdef *e, int32_t num);

void upb_enum_begin(upb_enum_iter *iter, const upb_enumdef *e);
void upb_enum_next(upb_enum_iter *iter);
bool upb_enum_done(upb_enum_iter *iter);
const char *upb_enum_iter_name(upb_enum_iter *iter);
int32_t upb_enum_iter_number(upb_enum_iter *iter);

/* upb_filedef ****************************************************************/

const char *upb_filedef_name(const upb_filedef *f);
const char *upb_filedef_package(const upb_filedef *f);
const char *upb_filedef_phpprefix(const upb_filedef *f);
const char *upb_filedef_phpnamespace(const upb_filedef *f);
upb_syntax_t upb_filedef_syntax(const upb_filedef *f);
int upb_filedef_depcount(const upb_filedef *f);
int upb_filedef_msgcount(const upb_filedef *f);
int upb_filedef_enumcount(const upb_filedef *f);
const upb_filedef *upb_filedef_dep(const upb_filedef *f, int i);
const upb_msgdef *upb_filedef_msg(const upb_filedef *f, int i);
const upb_enumdef *upb_filedef_enum(const upb_filedef *f, int i);
const upb_symtab *upb_filedef_symtab(const upb_filedef *f);

/* upb_symtab *****************************************************************/

upb_symtab *upb_symtab_new(void);
void upb_symtab_free(upb_symtab* s);
const upb_msgdef *upb_symtab_lookupmsg(const upb_symtab *s, const char *sym);
const upb_msgdef *upb_symtab_lookupmsg2(
    const upb_symtab *s, const char *sym, size_t len);
const upb_enumdef *upb_symtab_lookupenum(const upb_symtab *s, const char *sym);
const upb_filedef *upb_symtab_lookupfile(const upb_symtab *s, const char *name);
const upb_filedef *upb_symtab_lookupfile2(
    const upb_symtab *s, const char *name, size_t len);
int upb_symtab_filecount(const upb_symtab *s);
const upb_filedef *upb_symtab_addfile(
    upb_symtab *s, const google_protobuf_FileDescriptorProto *file,
    upb_status *status);
size_t _upb_symtab_bytesloaded(const upb_symtab *s);
upb_arena *_upb_symtab_arena(const upb_symtab *s);

/* For generated code only: loads a generated descriptor. */
typedef struct upb_def_init {
  struct upb_def_init **deps;     /* Dependencies of this file. */
  const upb_msglayout **layouts;  /* Pre-order layouts of all messages. */
  const char *filename;
  upb_strview descriptor;         /* Serialized descriptor. */
} upb_def_init;

bool _upb_symtab_loaddefinit(upb_symtab *s, const upb_def_init *init);


#ifdef __cplusplus
}  /* extern "C" */
#endif  /* __cplusplus */

#endif /* UPB_DEF_H_ */

/** google/protobuf/descriptor.upbdefs.h ************************************************************//* This file was generated by upbc (the upb compiler) from the input
 * file:
 *
 *     google/protobuf/descriptor.proto
 *
 * Do not edit -- your changes will be discarded when the file is
 * regenerated. */

#ifndef GOOGLE_PROTOBUF_DESCRIPTOR_PROTO_UPBDEFS_H_
#define GOOGLE_PROTOBUF_DESCRIPTOR_PROTO_UPBDEFS_H_

#ifdef __cplusplus
extern "C" {
#endif



extern upb_def_init google_protobuf_descriptor_proto_upbdefinit;

UPB_INLINE const upb_msgdef *google_protobuf_FileDescriptorSet_getmsgdef(upb_symtab *s) {
  _upb_symtab_loaddefinit(s, &google_protobuf_descriptor_proto_upbdefinit);
  return upb_symtab_lookupmsg(s, "google.protobuf.FileDescriptorSet");
}

UPB_INLINE const upb_msgdef *google_protobuf_FileDescriptorProto_getmsgdef(upb_symtab *s) {
  _upb_symtab_loaddefinit(s, &google_protobuf_descriptor_proto_upbdefinit);
  return upb_symtab_lookupmsg(s, "google.protobuf.FileDescriptorProto");
}

UPB_INLINE const upb_msgdef *google_protobuf_DescriptorProto_getmsgdef(upb_symtab *s) {
  _upb_symtab_loaddefinit(s, &google_protobuf_descriptor_proto_upbdefinit);
  return upb_symtab_lookupmsg(s, "google.protobuf.DescriptorProto");
}

UPB_INLINE const upb_msgdef *google_protobuf_DescriptorProto_ExtensionRange_getmsgdef(upb_symtab *s) {
  _upb_symtab_loaddefinit(s, &google_protobuf_descriptor_proto_upbdefinit);
  return upb_symtab_lookupmsg(s, "google.protobuf.DescriptorProto.ExtensionRange");
}

UPB_INLINE const upb_msgdef *google_protobuf_DescriptorProto_ReservedRange_getmsgdef(upb_symtab *s) {
  _upb_symtab_loaddefinit(s, &google_protobuf_descriptor_proto_upbdefinit);
  return upb_symtab_lookupmsg(s, "google.protobuf.DescriptorProto.ReservedRange");
}

UPB_INLINE const upb_msgdef *google_protobuf_ExtensionRangeOptions_getmsgdef(upb_symtab *s) {
  _upb_symtab_loaddefinit(s, &google_protobuf_descriptor_proto_upbdefinit);
  return upb_symtab_lookupmsg(s, "google.protobuf.ExtensionRangeOptions");
}

UPB_INLINE const upb_msgdef *google_protobuf_FieldDescriptorProto_getmsgdef(upb_symtab *s) {
  _upb_symtab_loaddefinit(s, &google_protobuf_descriptor_proto_upbdefinit);
  return upb_symtab_lookupmsg(s, "google.protobuf.FieldDescriptorProto");
}

UPB_INLINE const upb_msgdef *google_protobuf_OneofDescriptorProto_getmsgdef(upb_symtab *s) {
  _upb_symtab_loaddefinit(s, &google_protobuf_descriptor_proto_upbdefinit);
  return upb_symtab_lookupmsg(s, "google.protobuf.OneofDescriptorProto");
}

UPB_INLINE const upb_msgdef *google_protobuf_EnumDescriptorProto_getmsgdef(upb_symtab *s) {
  _upb_symtab_loaddefinit(s, &google_protobuf_descriptor_proto_upbdefinit);
  return upb_symtab_lookupmsg(s, "google.protobuf.EnumDescriptorProto");
}

UPB_INLINE const upb_msgdef *google_protobuf_EnumDescriptorProto_EnumReservedRange_getmsgdef(upb_symtab *s) {
  _upb_symtab_loaddefinit(s, &google_protobuf_descriptor_proto_upbdefinit);
  return upb_symtab_lookupmsg(s, "google.protobuf.EnumDescriptorProto.EnumReservedRange");
}

UPB_INLINE const upb_msgdef *google_protobuf_EnumValueDescriptorProto_getmsgdef(upb_symtab *s) {
  _upb_symtab_loaddefinit(s, &google_protobuf_descriptor_proto_upbdefinit);
  return upb_symtab_lookupmsg(s, "google.protobuf.EnumValueDescriptorProto");
}

UPB_INLINE const upb_msgdef *google_protobuf_ServiceDescriptorProto_getmsgdef(upb_symtab *s) {
  _upb_symtab_loaddefinit(s, &google_protobuf_descriptor_proto_upbdefinit);
  return upb_symtab_lookupmsg(s, "google.protobuf.ServiceDescriptorProto");
}

UPB_INLINE const upb_msgdef *google_protobuf_MethodDescriptorProto_getmsgdef(upb_symtab *s) {
  _upb_symtab_loaddefinit(s, &google_protobuf_descriptor_proto_upbdefinit);
  return upb_symtab_lookupmsg(s, "google.protobuf.MethodDescriptorProto");
}

UPB_INLINE const upb_msgdef *google_protobuf_FileOptions_getmsgdef(upb_symtab *s) {
  _upb_symtab_loaddefinit(s, &google_protobuf_descriptor_proto_upbdefinit);
  return upb_symtab_lookupmsg(s, "google.protobuf.FileOptions");
}

UPB_INLINE const upb_msgdef *google_protobuf_MessageOptions_getmsgdef(upb_symtab *s) {
  _upb_symtab_loaddefinit(s, &google_protobuf_descriptor_proto_upbdefinit);
  return upb_symtab_lookupmsg(s, "google.protobuf.MessageOptions");
}

UPB_INLINE const upb_msgdef *google_protobuf_FieldOptions_getmsgdef(upb_symtab *s) {
  _upb_symtab_loaddefinit(s, &google_protobuf_descriptor_proto_upbdefinit);
  return upb_symtab_lookupmsg(s, "google.protobuf.FieldOptions");
}

UPB_INLINE const upb_msgdef *google_protobuf_OneofOptions_getmsgdef(upb_symtab *s) {
  _upb_symtab_loaddefinit(s, &google_protobuf_descriptor_proto_upbdefinit);
  return upb_symtab_lookupmsg(s, "google.protobuf.OneofOptions");
}

UPB_INLINE const upb_msgdef *google_protobuf_EnumOptions_getmsgdef(upb_symtab *s) {
  _upb_symtab_loaddefinit(s, &google_protobuf_descriptor_proto_upbdefinit);
  return upb_symtab_lookupmsg(s, "google.protobuf.EnumOptions");
}

UPB_INLINE const upb_msgdef *google_protobuf_EnumValueOptions_getmsgdef(upb_symtab *s) {
  _upb_symtab_loaddefinit(s, &google_protobuf_descriptor_proto_upbdefinit);
  return upb_symtab_lookupmsg(s, "google.protobuf.EnumValueOptions");
}

UPB_INLINE const upb_msgdef *google_protobuf_ServiceOptions_getmsgdef(upb_symtab *s) {
  _upb_symtab_loaddefinit(s, &google_protobuf_descriptor_proto_upbdefinit);
  return upb_symtab_lookupmsg(s, "google.protobuf.ServiceOptions");
}

UPB_INLINE const upb_msgdef *google_protobuf_MethodOptions_getmsgdef(upb_symtab *s) {
  _upb_symtab_loaddefinit(s, &google_protobuf_descriptor_proto_upbdefinit);
  return upb_symtab_lookupmsg(s, "google.protobuf.MethodOptions");
}

UPB_INLINE const upb_msgdef *google_protobuf_UninterpretedOption_getmsgdef(upb_symtab *s) {
  _upb_symtab_loaddefinit(s, &google_protobuf_descriptor_proto_upbdefinit);
  return upb_symtab_lookupmsg(s, "google.protobuf.UninterpretedOption");
}

UPB_INLINE const upb_msgdef *google_protobuf_UninterpretedOption_NamePart_getmsgdef(upb_symtab *s) {
  _upb_symtab_loaddefinit(s, &google_protobuf_descriptor_proto_upbdefinit);
  return upb_symtab_lookupmsg(s, "google.protobuf.UninterpretedOption.NamePart");
}

UPB_INLINE const upb_msgdef *google_protobuf_SourceCodeInfo_getmsgdef(upb_symtab *s) {
  _upb_symtab_loaddefinit(s, &google_protobuf_descriptor_proto_upbdefinit);
  return upb_symtab_lookupmsg(s, "google.protobuf.SourceCodeInfo");
}

UPB_INLINE const upb_msgdef *google_protobuf_SourceCodeInfo_Location_getmsgdef(upb_symtab *s) {
  _upb_symtab_loaddefinit(s, &google_protobuf_descriptor_proto_upbdefinit);
  return upb_symtab_lookupmsg(s, "google.protobuf.SourceCodeInfo.Location");
}

UPB_INLINE const upb_msgdef *google_protobuf_GeneratedCodeInfo_getmsgdef(upb_symtab *s) {
  _upb_symtab_loaddefinit(s, &google_protobuf_descriptor_proto_upbdefinit);
  return upb_symtab_lookupmsg(s, "google.protobuf.GeneratedCodeInfo");
}

UPB_INLINE const upb_msgdef *google_protobuf_GeneratedCodeInfo_Annotation_getmsgdef(upb_symtab *s) {
  _upb_symtab_loaddefinit(s, &google_protobuf_descriptor_proto_upbdefinit);
  return upb_symtab_lookupmsg(s, "google.protobuf.GeneratedCodeInfo.Annotation");
}

#ifdef __cplusplus
}  /* extern "C" */
#endif


#endif  /* GOOGLE_PROTOBUF_DESCRIPTOR_PROTO_UPBDEFS_H_ */

/** upb/reflection.h ************************************************************/
#ifndef UPB_REFLECTION_H_
#define UPB_REFLECTION_H_



#ifdef __cplusplus
extern "C" {
#endif

typedef union {
  bool bool_val;
  float float_val;
  double double_val;
  int32_t int32_val;
  int64_t int64_val;
  uint32_t uint32_val;
  uint64_t uint64_val;
  const upb_map* map_val;
  const upb_msg* msg_val;
  const upb_array* array_val;
  upb_strview str_val;
} upb_msgval;

typedef union {
  upb_map* map;
  upb_msg* msg;
  upb_array* array;
} upb_mutmsgval;

upb_msgval upb_fielddef_default(const upb_fielddef *f);

/** upb_msg *******************************************************************/

/* Creates a new message of the given type in the given arena. */
upb_msg *upb_msg_new(const upb_msgdef *m, upb_arena *a);

/* Returns the value associated with this field. */
upb_msgval upb_msg_get(const upb_msg *msg, const upb_fielddef *f);

/* Returns a mutable pointer to a map, array, or submessage value.  If the given
 * arena is non-NULL this will construct a new object if it was not previously
 * present.  May not be called for primitive fields. */
upb_mutmsgval upb_msg_mutable(upb_msg *msg, const upb_fielddef *f, upb_arena *a);

/* May only be called for fields where upb_fielddef_haspresence(f) == true. */
bool upb_msg_has(const upb_msg *msg, const upb_fielddef *f);

/* Returns the field that is set in the oneof, or NULL if none are set. */
const upb_fielddef *upb_msg_whichoneof(const upb_msg *msg,
                                       const upb_oneofdef *o);

/* Sets the given field to the given value.  For a msg/array/map/string, the
 * value must be in the same arena.  */
void upb_msg_set(upb_msg *msg, const upb_fielddef *f, upb_msgval val,
                 upb_arena *a);

/* Clears any field presence and sets the value back to its default. */
void upb_msg_clearfield(upb_msg *msg, const upb_fielddef *f);

/* Clear all data and unknown fields. */
void upb_msg_clear(upb_msg *msg, const upb_msgdef *m);

/* Iterate over present fields.
 *
 * size_t iter = UPB_MSG_BEGIN;
 * const upb_fielddef *f;
 * upb_msgval val;
 * while (upb_msg_next(msg, m, ext_pool, &f, &val, &iter)) {
 *   process_field(f, val);
 * }
 *
 * If ext_pool is NULL, no extensions will be returned.  If the given symtab
 * returns extensions that don't match what is in this message, those extensions
 * will be skipped.
 */

#define UPB_MSG_BEGIN -1
bool upb_msg_next(const upb_msg *msg, const upb_msgdef *m,
                  const upb_symtab *ext_pool, const upb_fielddef **f,
                  upb_msgval *val, size_t *iter);

/* Clears all unknown field data from this message and all submessages. */
bool upb_msg_discardunknown(upb_msg *msg, const upb_msgdef *m, int maxdepth);

/** upb_array *****************************************************************/

/* Creates a new array on the given arena that holds elements of this type. */
upb_array *upb_array_new(upb_arena *a, upb_fieldtype_t type);

/* Returns the size of the array. */
size_t upb_array_size(const upb_array *arr);

/* Returns the given element, which must be within the array's current size. */
upb_msgval upb_array_get(const upb_array *arr, size_t i);

/* Sets the given element, which must be within the array's current size. */
void upb_array_set(upb_array *arr, size_t i, upb_msgval val);

/* Appends an element to the array.  Returns false on allocation failure. */
bool upb_array_append(upb_array *array, upb_msgval val, upb_arena *arena);

/* Changes the size of a vector.  New elements are initialized to empty/0.
 * Returns false on allocation failure. */
bool upb_array_resize(upb_array *array, size_t size, upb_arena *arena);

/** upb_map *******************************************************************/

/* Creates a new map on the given arena with the given key/value size. */
upb_map *upb_map_new(upb_arena *a, upb_fieldtype_t key_type,
                     upb_fieldtype_t value_type);

/* Returns the number of entries in the map. */
size_t upb_map_size(const upb_map *map);

/* Stores a value for the given key into |*val| (or the zero value if the key is
 * not present).  Returns whether the key was present.  The |val| pointer may be
 * NULL, in which case the function tests whether the given key is present.  */
bool upb_map_get(const upb_map *map, upb_msgval key, upb_msgval *val);

/* Removes all entries in the map. */
void upb_map_clear(upb_map *map);

/* Sets the given key to the given value.  Returns true if this was a new key in
 * the map, or false if an existing key was replaced. */
bool upb_map_set(upb_map *map, upb_msgval key, upb_msgval val,
                 upb_arena *arena);

/* Deletes this key from the table.  Returns true if the key was present. */
bool upb_map_delete(upb_map *map, upb_msgval key);

/* Map iteration:
 *
 * size_t iter = UPB_MAP_BEGIN;
 * while (upb_mapiter_next(map, &iter)) {
 *   upb_msgval key = upb_mapiter_key(map, iter);
 *   upb_msgval val = upb_mapiter_value(map, iter);
 *
 *   // If mutating is desired.
 *   upb_mapiter_setvalue(map, iter, value2);
 * }
 */

/* Advances to the next entry.  Returns false if no more entries are present. */
bool upb_mapiter_next(const upb_map *map, size_t *iter);

/* Returns true if the iterator still points to a valid entry, or false if the
 * iterator is past the last element. It is an error to call this function with
 * UPB_MAP_BEGIN (you must call next() at least once first). */
bool upb_mapiter_done(const upb_map *map, size_t iter);

/* Returns the key and value for this entry of the map. */
upb_msgval upb_mapiter_key(const upb_map *map, size_t iter);
upb_msgval upb_mapiter_value(const upb_map *map, size_t iter);

/* Sets the value for this entry.  The iterator must not be done, and the
 * iterator must not have been initialized const. */
void upb_mapiter_setvalue(upb_map *map, size_t iter, upb_msgval value);

#ifdef __cplusplus
}  /* extern "C" */
#endif


#endif /* UPB_REFLECTION_H_ */

/** upb/json_decode.h ************************************************************/
#ifndef UPB_JSONDECODE_H_
#define UPB_JSONDECODE_H_


#ifdef __cplusplus
extern "C" {
#endif

enum {
  UPB_JSONDEC_IGNOREUNKNOWN = 1
};

bool upb_json_decode(const char *buf, size_t size, upb_msg *msg,
                     const upb_msgdef *m, const upb_symtab *any_pool,
                     int options, upb_arena *arena, upb_status *status);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* UPB_JSONDECODE_H_ */

/** upb/json_encode.h ************************************************************/
#ifndef UPB_JSONENCODE_H_
#define UPB_JSONENCODE_H_


#ifdef __cplusplus
extern "C" {
#endif

enum {
  /* When set, emits 0/default values.  TODO(haberman): proto3 only? */
  UPB_JSONENC_EMITDEFAULTS = 1,

  /* When set, use normal (snake_caes) field names instead of JSON (camelCase)
     names. */
  UPB_JSONENC_PROTONAMES = 2
};

/* Encodes the given |msg| to JSON format.  The message's reflection is given in
 * |m|.  The symtab in |symtab| is used to find extensions (if NULL, extensions
 * will not be printed).
 *
 * Output is placed in the given buffer, and always NULL-terminated.  The output
 * size (excluding NULL) is returned.  This means that a return value >= |size|
 * implies that the output was truncated.  (These are the same semantics as
 * snprintf()). */
size_t upb_json_encode(const upb_msg *msg, const upb_msgdef *m,
                       const upb_symtab *ext_pool, int options, char *buf,
                       size_t size, upb_status *status);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* UPB_JSONENCODE_H_ */

/** upb/port_undef.inc ************************************************************/
/* See port_def.inc.  This should #undef all macros #defined there. */

#undef UPB_SIZE
#undef UPB_PTR_AT
#undef UPB_READ_ONEOF
#undef UPB_WRITE_ONEOF
#undef UPB_MAPTYPE_STRING
#undef UPB_INLINE
#undef UPB_ALIGN_UP
#undef UPB_ALIGN_DOWN
#undef UPB_ALIGN_MALLOC
#undef UPB_ALIGN_OF
#undef UPB_LIKELY
#undef UPB_UNLIKELY
#undef UPB_FORCEINLINE
#undef UPB_NOINLINE
#undef UPB_NORETURN
#undef UPB_PRINTF
#undef UPB_MAX
#undef UPB_MIN
#undef UPB_UNUSED
#undef UPB_ASSUME
#undef UPB_ASSERT
#undef UPB_UNREACHABLE
#undef UPB_SETJMP
#undef UPB_LONGJMP
#undef UPB_PTRADD
#undef UPB_MUSTTAIL
#undef UPB_FASTTABLE_SUPPORTED
#undef UPB_FASTTABLE
#undef UPB_FASTTABLE_INIT
#undef UPB_POISON_MEMORY_REGION
#undef UPB_UNPOISON_MEMORY_REGION
#undef UPB_ASAN
