// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: VisionFeaturePrint.proto

#ifndef PROTOBUF_VisionFeaturePrint_2eproto__INCLUDED
#define PROTOBUF_VisionFeaturePrint_2eproto__INCLUDED

#include <string>

#include <google/protobuf/stubs/common.h>

#if GOOGLE_PROTOBUF_VERSION < 3003000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please update
#error your headers.
#endif
#if 3003000 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_table_driven.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/metadata_lite.h>
#include <google/protobuf/message_lite.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
#include <google/protobuf/generated_enum_util.h>
// @@protoc_insertion_point(includes)
namespace CoreML {
namespace Specification {
namespace CoreMLModels {
class VisionFeaturePrint;
class VisionFeaturePrintDefaultTypeInternal;
extern VisionFeaturePrintDefaultTypeInternal _VisionFeaturePrint_default_instance_;
class VisionFeaturePrint_Scene;
class VisionFeaturePrint_SceneDefaultTypeInternal;
extern VisionFeaturePrint_SceneDefaultTypeInternal _VisionFeaturePrint_Scene_default_instance_;
}  // namespace CoreMLModels
}  // namespace Specification
}  // namespace CoreML

namespace CoreML {
namespace Specification {
namespace CoreMLModels {

namespace protobuf_VisionFeaturePrint_2eproto {
// Internal implementation detail -- do not call these.
struct TableStruct {
  static const ::google::protobuf::internal::ParseTableField entries[];
  static const ::google::protobuf::internal::AuxillaryParseTableField aux[];
  static const ::google::protobuf::internal::ParseTable schema[];
  static const ::google::protobuf::uint32 offsets[];
  static void InitDefaultsImpl();
  static void Shutdown();
};
void AddDescriptors();
void InitDefaults();
}  // namespace protobuf_VisionFeaturePrint_2eproto

enum VisionFeaturePrint_Scene_SceneVersion {
  VisionFeaturePrint_Scene_SceneVersion_SCENE_VERSION_INVALID = 0,
  VisionFeaturePrint_Scene_SceneVersion_SCENE_VERSION_1 = 1,
  VisionFeaturePrint_Scene_SceneVersion_VisionFeaturePrint_Scene_SceneVersion_INT_MIN_SENTINEL_DO_NOT_USE_ = ::google::protobuf::kint32min,
  VisionFeaturePrint_Scene_SceneVersion_VisionFeaturePrint_Scene_SceneVersion_INT_MAX_SENTINEL_DO_NOT_USE_ = ::google::protobuf::kint32max
};
bool VisionFeaturePrint_Scene_SceneVersion_IsValid(int value);
const VisionFeaturePrint_Scene_SceneVersion VisionFeaturePrint_Scene_SceneVersion_SceneVersion_MIN = VisionFeaturePrint_Scene_SceneVersion_SCENE_VERSION_INVALID;
const VisionFeaturePrint_Scene_SceneVersion VisionFeaturePrint_Scene_SceneVersion_SceneVersion_MAX = VisionFeaturePrint_Scene_SceneVersion_SCENE_VERSION_1;
const int VisionFeaturePrint_Scene_SceneVersion_SceneVersion_ARRAYSIZE = VisionFeaturePrint_Scene_SceneVersion_SceneVersion_MAX + 1;

// ===================================================================

class VisionFeaturePrint_Scene : public ::google::protobuf::MessageLite /* @@protoc_insertion_point(class_definition:CoreML.Specification.CoreMLModels.VisionFeaturePrint.Scene) */ {
 public:
  VisionFeaturePrint_Scene();
  virtual ~VisionFeaturePrint_Scene();

  VisionFeaturePrint_Scene(const VisionFeaturePrint_Scene& from);

  inline VisionFeaturePrint_Scene& operator=(const VisionFeaturePrint_Scene& from) {
    CopyFrom(from);
    return *this;
  }

  static const VisionFeaturePrint_Scene& default_instance();

  static inline const VisionFeaturePrint_Scene* internal_default_instance() {
    return reinterpret_cast<const VisionFeaturePrint_Scene*>(
               &_VisionFeaturePrint_Scene_default_instance_);
  }
  static PROTOBUF_CONSTEXPR int const kIndexInFileMessages =
    0;

  void Swap(VisionFeaturePrint_Scene* other);

  // implements Message ----------------------------------------------

  inline VisionFeaturePrint_Scene* New() const PROTOBUF_FINAL { return New(NULL); }

  VisionFeaturePrint_Scene* New(::google::protobuf::Arena* arena) const PROTOBUF_FINAL;
  void CheckTypeAndMergeFrom(const ::google::protobuf::MessageLite& from)
    PROTOBUF_FINAL;
  void CopyFrom(const VisionFeaturePrint_Scene& from);
  void MergeFrom(const VisionFeaturePrint_Scene& from);
  void Clear() PROTOBUF_FINAL;
  bool IsInitialized() const PROTOBUF_FINAL;

  size_t ByteSizeLong() const PROTOBUF_FINAL;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input) PROTOBUF_FINAL;
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const PROTOBUF_FINAL;
  void DiscardUnknownFields();
  int GetCachedSize() const PROTOBUF_FINAL { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const;
  void InternalSwap(VisionFeaturePrint_Scene* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return NULL;
  }
  inline void* MaybeArenaPtr() const {
    return NULL;
  }
  public:

  ::std::string GetTypeName() const PROTOBUF_FINAL;

  // nested types ----------------------------------------------------

  typedef VisionFeaturePrint_Scene_SceneVersion SceneVersion;
  static const SceneVersion SCENE_VERSION_INVALID =
    VisionFeaturePrint_Scene_SceneVersion_SCENE_VERSION_INVALID;
  static const SceneVersion SCENE_VERSION_1 =
    VisionFeaturePrint_Scene_SceneVersion_SCENE_VERSION_1;
  static inline bool SceneVersion_IsValid(int value) {
    return VisionFeaturePrint_Scene_SceneVersion_IsValid(value);
  }
  static const SceneVersion SceneVersion_MIN =
    VisionFeaturePrint_Scene_SceneVersion_SceneVersion_MIN;
  static const SceneVersion SceneVersion_MAX =
    VisionFeaturePrint_Scene_SceneVersion_SceneVersion_MAX;
  static const int SceneVersion_ARRAYSIZE =
    VisionFeaturePrint_Scene_SceneVersion_SceneVersion_ARRAYSIZE;

  // accessors -------------------------------------------------------

  // .CoreML.Specification.CoreMLModels.VisionFeaturePrint.Scene.SceneVersion version = 1;
  void clear_version();
  static const int kVersionFieldNumber = 1;
  ::CoreML::Specification::CoreMLModels::VisionFeaturePrint_Scene_SceneVersion version() const;
  void set_version(::CoreML::Specification::CoreMLModels::VisionFeaturePrint_Scene_SceneVersion value);

  // @@protoc_insertion_point(class_scope:CoreML.Specification.CoreMLModels.VisionFeaturePrint.Scene)
 private:

  ::google::protobuf::internal::InternalMetadataWithArenaLite _internal_metadata_;
  int version_;
  mutable int _cached_size_;
  friend struct protobuf_VisionFeaturePrint_2eproto::TableStruct;
};
// -------------------------------------------------------------------

class VisionFeaturePrint : public ::google::protobuf::MessageLite /* @@protoc_insertion_point(class_definition:CoreML.Specification.CoreMLModels.VisionFeaturePrint) */ {
 public:
  VisionFeaturePrint();
  virtual ~VisionFeaturePrint();

  VisionFeaturePrint(const VisionFeaturePrint& from);

  inline VisionFeaturePrint& operator=(const VisionFeaturePrint& from) {
    CopyFrom(from);
    return *this;
  }

  static const VisionFeaturePrint& default_instance();

  enum VisionFeaturePrintTypeCase {
    kScene = 20,
    VISIONFEATUREPRINTTYPE_NOT_SET = 0,
  };

  static inline const VisionFeaturePrint* internal_default_instance() {
    return reinterpret_cast<const VisionFeaturePrint*>(
               &_VisionFeaturePrint_default_instance_);
  }
  static PROTOBUF_CONSTEXPR int const kIndexInFileMessages =
    1;

  void Swap(VisionFeaturePrint* other);

  // implements Message ----------------------------------------------

  inline VisionFeaturePrint* New() const PROTOBUF_FINAL { return New(NULL); }

  VisionFeaturePrint* New(::google::protobuf::Arena* arena) const PROTOBUF_FINAL;
  void CheckTypeAndMergeFrom(const ::google::protobuf::MessageLite& from)
    PROTOBUF_FINAL;
  void CopyFrom(const VisionFeaturePrint& from);
  void MergeFrom(const VisionFeaturePrint& from);
  void Clear() PROTOBUF_FINAL;
  bool IsInitialized() const PROTOBUF_FINAL;

  size_t ByteSizeLong() const PROTOBUF_FINAL;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input) PROTOBUF_FINAL;
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const PROTOBUF_FINAL;
  void DiscardUnknownFields();
  int GetCachedSize() const PROTOBUF_FINAL { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const;
  void InternalSwap(VisionFeaturePrint* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return NULL;
  }
  inline void* MaybeArenaPtr() const {
    return NULL;
  }
  public:

  ::std::string GetTypeName() const PROTOBUF_FINAL;

  // nested types ----------------------------------------------------

  typedef VisionFeaturePrint_Scene Scene;

  // accessors -------------------------------------------------------

  // .CoreML.Specification.CoreMLModels.VisionFeaturePrint.Scene scene = 20;
  bool has_scene() const;
  void clear_scene();
  static const int kSceneFieldNumber = 20;
  const ::CoreML::Specification::CoreMLModels::VisionFeaturePrint_Scene& scene() const;
  ::CoreML::Specification::CoreMLModels::VisionFeaturePrint_Scene* mutable_scene();
  ::CoreML::Specification::CoreMLModels::VisionFeaturePrint_Scene* release_scene();
  void set_allocated_scene(::CoreML::Specification::CoreMLModels::VisionFeaturePrint_Scene* scene);

  VisionFeaturePrintTypeCase VisionFeaturePrintType_case() const;
  // @@protoc_insertion_point(class_scope:CoreML.Specification.CoreMLModels.VisionFeaturePrint)
 private:
  void set_has_scene();

  inline bool has_VisionFeaturePrintType() const;
  void clear_VisionFeaturePrintType();
  inline void clear_has_VisionFeaturePrintType();

  ::google::protobuf::internal::InternalMetadataWithArenaLite _internal_metadata_;
  union VisionFeaturePrintTypeUnion {
    VisionFeaturePrintTypeUnion() {}
    ::CoreML::Specification::CoreMLModels::VisionFeaturePrint_Scene* scene_;
  } VisionFeaturePrintType_;
  mutable int _cached_size_;
  ::google::protobuf::uint32 _oneof_case_[1];

  friend struct protobuf_VisionFeaturePrint_2eproto::TableStruct;
};
// ===================================================================


// ===================================================================

#if !PROTOBUF_INLINE_NOT_IN_HEADERS
// VisionFeaturePrint_Scene

// .CoreML.Specification.CoreMLModels.VisionFeaturePrint.Scene.SceneVersion version = 1;
inline void VisionFeaturePrint_Scene::clear_version() {
  version_ = 0;
}
inline ::CoreML::Specification::CoreMLModels::VisionFeaturePrint_Scene_SceneVersion VisionFeaturePrint_Scene::version() const {
  // @@protoc_insertion_point(field_get:CoreML.Specification.CoreMLModels.VisionFeaturePrint.Scene.version)
  return static_cast< ::CoreML::Specification::CoreMLModels::VisionFeaturePrint_Scene_SceneVersion >(version_);
}
inline void VisionFeaturePrint_Scene::set_version(::CoreML::Specification::CoreMLModels::VisionFeaturePrint_Scene_SceneVersion value) {
  
  version_ = value;
  // @@protoc_insertion_point(field_set:CoreML.Specification.CoreMLModels.VisionFeaturePrint.Scene.version)
}

// -------------------------------------------------------------------

// VisionFeaturePrint

// .CoreML.Specification.CoreMLModels.VisionFeaturePrint.Scene scene = 20;
inline bool VisionFeaturePrint::has_scene() const {
  return VisionFeaturePrintType_case() == kScene;
}
inline void VisionFeaturePrint::set_has_scene() {
  _oneof_case_[0] = kScene;
}
inline void VisionFeaturePrint::clear_scene() {
  if (has_scene()) {
    delete VisionFeaturePrintType_.scene_;
    clear_has_VisionFeaturePrintType();
  }
}
inline  const ::CoreML::Specification::CoreMLModels::VisionFeaturePrint_Scene& VisionFeaturePrint::scene() const {
  // @@protoc_insertion_point(field_get:CoreML.Specification.CoreMLModels.VisionFeaturePrint.scene)
  return has_scene()
      ? *VisionFeaturePrintType_.scene_
      : ::CoreML::Specification::CoreMLModels::VisionFeaturePrint_Scene::default_instance();
}
inline ::CoreML::Specification::CoreMLModels::VisionFeaturePrint_Scene* VisionFeaturePrint::mutable_scene() {
  if (!has_scene()) {
    clear_VisionFeaturePrintType();
    set_has_scene();
    VisionFeaturePrintType_.scene_ = new ::CoreML::Specification::CoreMLModels::VisionFeaturePrint_Scene;
  }
  // @@protoc_insertion_point(field_mutable:CoreML.Specification.CoreMLModels.VisionFeaturePrint.scene)
  return VisionFeaturePrintType_.scene_;
}
inline ::CoreML::Specification::CoreMLModels::VisionFeaturePrint_Scene* VisionFeaturePrint::release_scene() {
  // @@protoc_insertion_point(field_release:CoreML.Specification.CoreMLModels.VisionFeaturePrint.scene)
  if (has_scene()) {
    clear_has_VisionFeaturePrintType();
    ::CoreML::Specification::CoreMLModels::VisionFeaturePrint_Scene* temp = VisionFeaturePrintType_.scene_;
    VisionFeaturePrintType_.scene_ = NULL;
    return temp;
  } else {
    return NULL;
  }
}
inline void VisionFeaturePrint::set_allocated_scene(::CoreML::Specification::CoreMLModels::VisionFeaturePrint_Scene* scene) {
  clear_VisionFeaturePrintType();
  if (scene) {
    set_has_scene();
    VisionFeaturePrintType_.scene_ = scene;
  }
  // @@protoc_insertion_point(field_set_allocated:CoreML.Specification.CoreMLModels.VisionFeaturePrint.scene)
}

inline bool VisionFeaturePrint::has_VisionFeaturePrintType() const {
  return VisionFeaturePrintType_case() != VISIONFEATUREPRINTTYPE_NOT_SET;
}
inline void VisionFeaturePrint::clear_has_VisionFeaturePrintType() {
  _oneof_case_[0] = VISIONFEATUREPRINTTYPE_NOT_SET;
}
inline VisionFeaturePrint::VisionFeaturePrintTypeCase VisionFeaturePrint::VisionFeaturePrintType_case() const {
  return VisionFeaturePrint::VisionFeaturePrintTypeCase(_oneof_case_[0]);
}
#endif  // !PROTOBUF_INLINE_NOT_IN_HEADERS
// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)


}  // namespace CoreMLModels
}  // namespace Specification
}  // namespace CoreML

#ifndef SWIG
namespace google {
namespace protobuf {

template <> struct is_proto_enum< ::CoreML::Specification::CoreMLModels::VisionFeaturePrint_Scene_SceneVersion> : ::google::protobuf::internal::true_type {};

}  // namespace protobuf
}  // namespace google
#endif  // SWIG

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_VisionFeaturePrint_2eproto__INCLUDED
