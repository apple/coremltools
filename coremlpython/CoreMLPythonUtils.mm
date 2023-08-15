// Copyright (c) 2021, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
#import "CoreMLPythonArray.h"
#import "CoreMLPythonUtils.h"

#include <pybind11/eval.h>
#include <pybind11/numpy.h>
#include <iomanip> // for std::setfill etc

#import <Accelerate/Accelerate.h>

#if PY_MAJOR_VERSION < 3

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmacro-redefined"
#define PyBytes_Check(name) PyString_Check(name)
#pragma clang diagnostic pop
#define PyAnyInteger_Check(name) (PyLong_Check(name) || PyInt_Check(name))

#else

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#define PyAnyInteger_Check(name) (PyLong_Check(name) || (_import_array(), PyArray_IsScalar(name, Integer)))

#endif

using namespace CoreML::Python;

NSURL * Utils::stringToNSURL(const std::string& str) {
    NSString *nsstr = [NSString stringWithUTF8String:str.c_str()];
    return [NSURL fileURLWithPath:nsstr];
}

void Utils::handleError(NSError *error) {
    if (error != nil) {
        NSString *formatted = [NSString stringWithFormat:@"%@", [error userInfo]];
        throw std::runtime_error([formatted UTF8String]);
    }
}

MLDictionaryFeatureProvider * Utils::dictToFeatures(const py::dict& dict, NSError * __autoreleasing *error) {
    NSError *localError;
    MLDictionaryFeatureProvider * feautreProvider;

    @autoreleasepool {
        NSMutableDictionary<NSString *, NSObject *> *inputDict = [[NSMutableDictionary<NSString *, NSObject *> alloc] init];
        
        for (const auto element : dict) {
            std::string key = element.first.cast<std::string>();
            NSString *nsKey = [NSString stringWithUTF8String:key.c_str()];
            id nsValue = Utils::convertValueToObjC(element.second);
            inputDict[nsKey] = nsValue;
        }
        
        feautreProvider = [[MLDictionaryFeatureProvider alloc] initWithDictionary:inputDict error:&localError];
    }

    if (error != NULL) {
        *error = localError;
    }
    return feautreProvider;
}

py::dict Utils::featuresToDict(id<MLFeatureProvider> features) {
    @autoreleasepool {
        py::dict ret;
        NSSet<NSString *> *keys = [features featureNames];
        for (NSString *key in keys) {
            MLFeatureValue *value = [features featureValueForName:key];
            py::str pyKey = py::str([key UTF8String]);
            py::object pyValue = convertValueToPython(value);
            ret[pyKey] = pyValue;
        }
        return ret;
    }
}

template<typename KEYTYPE>
static NSObject * convertDictKey(const KEYTYPE& k);

template<>
NSObject * convertDictKey(const int64_t& k) {
    return [NSNumber numberWithLongLong:k];
}

template<>
NSObject * convertDictKey(const std::string& k) {
    return [NSString stringWithUTF8String:k.c_str()];
}

template<typename VALUETYPE>
static NSNumber * convertDictValue(const VALUETYPE& v);

template<>
NSNumber * convertDictValue(const int64_t& v) {
    return [NSNumber numberWithLongLong:v];
}

template<>
NSNumber * convertDictValue(const double& v) {
    return [NSNumber numberWithDouble:v];
}

template<typename KEYTYPE, typename VALUETYPE>
static MLFeatureValue * convertToNSDictionary(const std::unordered_map<KEYTYPE, VALUETYPE>& dict) {
    NSMutableDictionary<NSObject *, NSNumber *> *nsDict = [[NSMutableDictionary<NSObject *, NSNumber *> alloc] init];
    for (const auto& pair : dict) {
        NSObject<NSCopying> *key = (NSObject<NSCopying> *)convertDictKey(pair.first);
        NSNumber *value = convertDictValue(pair.second);
        assert(key != nil);
        nsDict[key] = value;
    }
    NSError *error = nil;
    MLFeatureValue * ret = [MLFeatureValue featureValueWithDictionary:nsDict error:&error];
    if (error != nil) {
        throw std::runtime_error(error.localizedDescription.UTF8String);
    }
    return ret;
}

static MLFeatureValue * convertValueToDictionary(const py::handle& handle) {
    
    if(!PyDict_Check(handle.ptr())) {
        throw std::runtime_error("Not a dictionary.");
    }
    
    // Get the first value in the dictionary; use that as a hint.
    PyObject *key = nullptr, *value = nullptr;
    Py_ssize_t pos = 0;
    
    int has_values = PyDict_Next(handle.ptr(), &pos, &key, &value);
    
    // Is it an empty dict?  If so, just return an empty dictionary.
    if(!has_values) {
        return [MLFeatureValue featureValueWithDictionary:@{} error:nullptr];
    }

    if(PyAnyInteger_Check(key)) {
        if(PyAnyInteger_Check(value)) {
            auto dict = handle.cast<std::unordered_map<int64_t, int64_t> >();
            return convertToNSDictionary(dict);
        } else if(PyFloat_Check(value)) {
            auto dict = handle.cast<std::unordered_map<int64_t, double> >();
            return convertToNSDictionary(dict);
        } else {
            throw std::runtime_error("Unknown value type for int key in dictionary.");
        }
    } else if (PyBytes_Check(key) || PyUnicode_Check(key)) {
        if(PyAnyInteger_Check(value)) {
            auto dict = handle.cast<std::unordered_map<std::string, int64_t> >();
            return convertToNSDictionary(dict);
        } else if(PyFloat_Check(value)) {
            auto dict = handle.cast<std::unordered_map<std::string, double> >();
            return convertToNSDictionary(dict);
        } else {
            throw std::runtime_error("Invalid value type for string key in dictionary.");
        }
    } else {
        throw std::runtime_error("Invalid key type dictionary.");
    }
}

static MLFeatureValue * convertValueToArray(const py::handle& handle) {
    // if this line throws, it can't be an array (caller should catch)
    py::array buf = handle.cast<py::array>();
    if (buf.shape() == nullptr) {
        throw std::runtime_error("no shape, can't be an array");
    }
    PybindCompatibleArray *array = [[PybindCompatibleArray alloc] initWithArray:buf];
    return [MLFeatureValue featureValueWithMultiArray:array];
}

static MLFeatureValue * convertValueToSequence(const py::handle& handle) {
#pragma unused (handle)
    /*
    if(!PyList_Check(handle.ptr())) {
        throw std::runtime_error("Not a list.");
    }

    py::list buf = handle.cast<py::list>();

    if(py::len(buf) == 0) {
        return [MLSequence emptySequenceWithType:MLFeatureTypeInt64];
    }

    py::handle e = buf[0];

    if(PyAnyInteger_Check(e)) {
        NSArray<NSNumber*> * v;


    }

    MLSequence* seq =
    */
    return nil;
}

static void handleCVReturn(CVReturn status) {
    if (status != kCVReturnSuccess) {
        std::stringstream msg;
        msg << "Got unexpected return code " << status << " from CoreVideo.";
        throw std::runtime_error(msg.str());
    }
}

static MLFeatureValue * convertValueToImage(const py::handle& handle) {
    // assumes handle is a valid PIL image!
    CVPixelBufferRef pixelBuffer = nil;
    
    size_t width = handle.attr("width").cast<size_t>();
    size_t height = handle.attr("height").cast<size_t>();
    OSType format;
    std::string formatStr = handle.attr("mode").cast<std::string>();
    
    if (formatStr == "RGB") {
        format = kCVPixelFormatType_32BGRA;
    } else if (formatStr == "RGBA") {
        format = kCVPixelFormatType_32BGRA;
    } else if (formatStr == "L") {
        format = kCVPixelFormatType_OneComponent8;
    } else if (formatStr == "F") {
        format = kCVPixelFormatType_OneComponent16Half;
    } else {
        std::stringstream msg;
        msg << "Unsupported image type " << formatStr << ". ";
        msg << "Supported types are: RGB, RGBA, L.";
        throw std::runtime_error(msg.str());
    }
    
    CVReturn status = CVPixelBufferCreate(kCFAllocatorDefault, width, height, format, NULL, &pixelBuffer);
    handleCVReturn(status);
    
    // get bytes out of the PIL image
    py::object tobytes = handle.attr("tobytes");
    py::object bytesResult = tobytes();
    assert(PyBytes_Check(bytesResult.ptr()));
    Py_ssize_t bytesLength = PyBytes_Size(bytesResult.ptr());
    assert(bytesLength >= 0);
    const char *bytesPtr = PyBytes_AsString(bytesResult.ptr());
    
    // copy data into the CVPixelBuffer
    status = CVPixelBufferLockBaseAddress(pixelBuffer, 0);
    handleCVReturn(status);
    void *baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer);
    assert(baseAddress != nullptr);
    assert(!CVPixelBufferIsPlanar(pixelBuffer));
    size_t bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer);
    const char *srcPointer = bytesPtr;

    vImage_Buffer srcBuffer;
    memset(&srcBuffer, 0, sizeof(srcBuffer));
    srcBuffer.data = const_cast<char *>(srcPointer);
    srcBuffer.width = width;
    srcBuffer.height = height;
    
    vImage_Buffer dstBuffer;
    memset(&dstBuffer, 0, sizeof(dstBuffer));
    dstBuffer.data = baseAddress;
    dstBuffer.width = width;
    dstBuffer.height = height;

    if (formatStr == "RGB") {
        
        // convert RGB to BGRA
        assert(bytesLength == width * height * 3);

        srcBuffer.rowBytes = width * 3;
        dstBuffer.rowBytes = bytesPerRow;
        vImageConvert_RGB888toBGRA8888(&srcBuffer, NULL, 255, &dstBuffer, false, 0);
        
    } else if (formatStr == "RGBA") {
        
        // convert RGBA to BGRA
        assert(bytesLength == width * height * 4);
        srcBuffer.rowBytes = width * 4;
        dstBuffer.rowBytes = bytesPerRow;
        uint8_t permuteMap[4] = { 2, 1, 0, 3 };
        vImagePermuteChannels_ARGB8888(&srcBuffer, &dstBuffer, permuteMap, 0);
        
    } else if (formatStr == "L") {
        
        // 8 bit grayscale.
        assert(bytesLength == width * height);

        srcBuffer.rowBytes = width;
        dstBuffer.rowBytes = bytesPerRow;
        vImageCopyBuffer(&srcBuffer, &dstBuffer, 1, 0);

    } else if (formatStr == "F") {
        
        // convert Float32 to Float16.
        assert(bytesLength == width * height * sizeof(Float32));

        srcBuffer.rowBytes = width * sizeof(Float32);
        dstBuffer.rowBytes = bytesPerRow;
        vImageConvert_PlanarFtoPlanar16F(&srcBuffer, &dstBuffer, 0);

    } else {
        std::stringstream msg;
        msg << "Unsupported image type " << formatStr << ". ";
        msg << "Supported types are: RGB, RGBA, L.";
        throw std::runtime_error(msg.str());
    }
    
#ifdef COREML_SHOW_PIL_IMAGES
    if (formatStr == "RGB") {
        // for debugging purposes, convert back to PIL image and show it
        py::object scope = py::module::import("__main__").attr("__dict__");
        py::eval<py::eval_single_statement>("import PIL.Image", scope);
        py::object pilImage = py::eval<py::eval_expr>("PIL.Image");
        
        std::string cvPixelStr(count, 0);
        const char *basePtr = static_cast<char *>(baseAddress);
        for (size_t row = 0; row < height; row++) {
            for (size_t col = 0; col < width; col++) {
                for (size_t color = 0; color < 3; color++) {
                    cvPixelStr[(row * width * 3) + (col*3) + color] = basePtr[(row * bytesPerRow) + (col*4) + color + 1];
                }
            }
        }
        
        py::bytes cvPixelBytes = py::bytes(cvPixelStr);
        py::object frombytes = pilImage.attr("frombytes");
        py::str mode = "RGB";
        auto size = py::make_tuple(width, height);
        py::object img = frombytes(mode, size, cvPixelBytes);
        img.attr("show")();
    }
#endif
    
    status = CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
    handleCVReturn(status);

    MLFeatureValue *fv = [MLFeatureValue featureValueWithPixelBuffer:pixelBuffer];
    CVPixelBufferRelease(pixelBuffer);
    
    return fv;
}

static bool IsPILImage(const py::handle& handle) {
    // TODO put try/catch around this?
    
    try {
        py::module::import("PIL.Image");
    } catch(...) {
        return false;
    }
    
    py::object scope = py::module::import("__main__").attr("__dict__");
    py::eval<py::eval_single_statement>("import PIL.Image", scope);
    py::handle imageTypeHandle = py::eval<py::eval_expr>("PIL.Image.Image", scope);
    assert(PyType_Check(imageTypeHandle.ptr())); // should be a Python type
    
    return PyObject_TypeCheck(handle.ptr(), (PyTypeObject *)(imageTypeHandle.ptr()));
}

MLFeatureValue * Utils::convertValueToObjC(const py::handle& handle) {
    
    if (PyAnyInteger_Check(handle.ptr())) {
        try {
            int64_t val = handle.cast<int64_t>();
            return [MLFeatureValue featureValueWithInt64:val];
        } catch(...) {}
    }
    
    if (PyFloat_Check(handle.ptr())) {
        try {
            double val = handle.cast<double>();
            return [MLFeatureValue featureValueWithDouble:val];
        } catch(...) {}
    }
    
    if (PyBytes_Check(handle.ptr()) || PyUnicode_Check(handle.ptr())) {
        try {
            std::string val = handle.cast<std::string>();
            return [MLFeatureValue featureValueWithString:[NSString stringWithUTF8String:val.c_str()]];
        } catch(...) {}
    }
    
    if (PyDict_Check(handle.ptr())) {
        try {
            return convertValueToDictionary(handle);
        } catch(...) {}
    }
    
    if(PyList_Check(handle.ptr()) || PyTuple_Check(handle.ptr())) {
        try {
            return convertValueToSequence(handle);
        } catch(...) {}
    }

    if(PyObject_CheckBuffer(handle.ptr())) {
        try {
            return convertValueToArray(handle);
        } catch(...) {}
    }
    
    if (IsPILImage(handle)) {
        return convertValueToImage(handle);
    }
    
    py::print("Error: value type not convertible:");
    py::print(handle);
    throw std::runtime_error("value type not convertible");
}

std::vector<size_t> Utils::convertNSArrayToCpp(NSArray<NSNumber *> *array) {
    std::vector<size_t> ret;
    for (NSNumber *value in array) {
        ret.push_back(value.unsignedLongValue);
    }
    return ret;
}

NSArray<NSNumber *>* Utils::convertCppArrayToObjC(const std::vector<size_t>& array) {
    NSMutableArray<NSNumber *>* ret = [[NSMutableArray<NSNumber *> alloc] init];
    for (size_t element : array) {
        [ret addObject:[NSNumber numberWithUnsignedLongLong:element]];
    }
    return ret;
}

static size_t sizeOfArrayElement(MLMultiArrayDataType type) {
    switch (type) {
        case MLMultiArrayDataTypeInt32:
            return sizeof(int32_t);
        case MLMultiArrayDataTypeFloat32:
        case MLMultiArrayDataTypeFloat16:
            return sizeof(float);
        case MLMultiArrayDataTypeDouble:
            return sizeof(double);
        default:
            assert(false);
            return sizeof(double);
    }
}

py::object Utils::convertArrayValueToPython(MLMultiArray *value) {
    if (value == nil) {
        return py::none();
    }
    MLMultiArrayDataType type = value.dataType;
    if (type == MLMultiArrayDataTypeFloat16) {
        // Cast to fp32 because py:array doesn't support fp16.
        // TODO: rdar://92239209 : return np.float16 instead of np.float32 when multiarray type is Float16
        value = [MLMultiArray multiArrayByConcatenatingMultiArrays:@[value] alongAxis:0 dataType:MLMultiArrayDataTypeFloat32];
        type = value.dataType;
    }

    std::vector<size_t> shape = Utils::convertNSArrayToCpp(value.shape);
    std::vector<size_t> strides = Utils::convertNSArrayToCpp(value.strides);

    // convert strides to numpy (bytes) instead of mlkit (elements)
    for (size_t& stride : strides) {
        stride *= sizeOfArrayElement(type);
    }

    __block py::object array;
    [value getBytesWithHandler:^(const void *bytes, NSInteger size) {
        switch (type) {
            case MLMultiArrayDataTypeInt32:
                array = py::array(shape, strides, reinterpret_cast<const int32_t *>(bytes));
                break;
            case MLMultiArrayDataTypeFloat32:
                array = py::array(shape, strides, reinterpret_cast<const float *>(bytes));
                break;
            case MLMultiArrayDataTypeFloat64:
                array = py::array(shape, strides, reinterpret_cast<const double *>(bytes));
                break;
            default:
                assert(false);
                array = py::object();
        }
    }];

    return array;
}

py::object Utils::convertDictionaryValueToPython(NSDictionary<NSObject *,NSNumber *> * dict) {
    if (dict == nil) {
        return py::none();
    }
    py::dict ret;
    for (NSObject *key in dict) {
        py::object pykey;
        if ([key isKindOfClass:[NSNumber class]]) {
            // can assume int32_t -- we only allow arrays of int or string keys
            NSNumber *nskey = static_cast<NSNumber *>(key);
            pykey = py::int_([nskey integerValue]);
        } else {
            assert([key isKindOfClass:[NSString class]]);
            NSString *nskey = static_cast<NSString *>(key);
            pykey = py::str([nskey UTF8String]);
        }
        
        NSNumber *value = dict[key];
        ret[pykey] = py::float_([value doubleValue]);
    }
    return std::move(ret);
}

py::object Utils::convertImageValueToPython(CVPixelBufferRef value) {
    if (CVPixelBufferIsPlanar(value)) {
        throw std::runtime_error("Only non-planar CVPixelBuffers are currently supported by this Python binding.");
    }
    
    // supports grayscale and BGRA format types
    auto formatType = CVPixelBufferGetPixelFormatType(value);
    assert(formatType == kCVPixelFormatType_32BGRA
           || formatType == kCVPixelFormatType_OneComponent8
           || formatType == kCVPixelFormatType_OneComponent16Half);

    auto height = CVPixelBufferGetHeight(value);
    auto width = CVPixelBufferGetWidth(value);

    py::str mode;
    size_t dstBytesPerRow = 0;
    if (formatType == kCVPixelFormatType_32BGRA) {
        mode = "RGBA";
        dstBytesPerRow = width * 4;
    } else if (formatType == kCVPixelFormatType_OneComponent8) {
        mode = "L";
        dstBytesPerRow = width * sizeof(uint8_t);
    } else if (formatType == kCVPixelFormatType_OneComponent16Half) {
        mode = "F";
        dstBytesPerRow = width * sizeof(Float32);
    } else {
        std::stringstream msg;
        msg << "Unsupported pixel format type: " << std::hex << std::setfill('0') << std::setw(4) << formatType << ". ";
        throw std::runtime_error(msg.str());
    }

    PyObject *dstPyBytes = PyBytes_FromStringAndSize(NULL, height * dstBytesPerRow);
    if (!dstPyBytes) {
        throw std::bad_alloc();
    }

    auto result = CVPixelBufferLockBaseAddress(value, kCVPixelBufferLock_ReadOnly);
    assert(result == kCVReturnSuccess);
    
    uint8_t *src = reinterpret_cast<uint8_t*>(CVPixelBufferGetBaseAddress(value));
    assert(src != nullptr);
    
    size_t srcBytesPerRow = CVPixelBufferGetBytesPerRow(value);

    // Prepare for vImage blitting
    vImage_Buffer srcBuffer;
    memset(&srcBuffer, 0, sizeof(srcBuffer));
    srcBuffer.data = src;
    srcBuffer.width = width;
    srcBuffer.height = height;
    srcBuffer.rowBytes = srcBytesPerRow;

    vImage_Buffer dstBuffer;
    memset(&dstBuffer, 0, sizeof(dstBuffer));
    dstBuffer.data = PyBytes_AS_STRING(dstPyBytes);
    dstBuffer.width = width;
    dstBuffer.height = height;
    dstBuffer.rowBytes = dstBytesPerRow;

    if (formatType == kCVPixelFormatType_32BGRA) {
        // convert BGRA to RGBA
        uint8_t permuteMap[4] = { 2, 1, 0, 3 };
        vImagePermuteChannels_ARGB8888(&srcBuffer, &dstBuffer, permuteMap, 0);
    } else if (formatType == kCVPixelFormatType_OneComponent8) {
        vImageCopyBuffer(&srcBuffer, &dstBuffer, 1, 0);
    } else if (formatType == kCVPixelFormatType_OneComponent16Half) {
        vImageConvert_Planar16FtoPlanarF(&srcBuffer, &dstBuffer, 0);
    } else {
        std::stringstream msg;
        msg << "Unsupported pixel format type: " << std::hex << std::setfill('0') << std::setw(4) << formatType << ". ";
        throw std::runtime_error(msg.str());
    }
    
    result = CVPixelBufferUnlockBaseAddress(value, kCVPixelBufferLock_ReadOnly);
    assert(result == kCVReturnSuccess);
    
    py::object scope = py::module::import("__main__").attr("__dict__");
    py::eval<py::eval_single_statement>("import PIL.Image", scope);
    py::object pilImage = py::eval<py::eval_expr>("PIL.Image", scope);
    py::object frombytes = pilImage.attr("frombytes");
    py::bytes dstBytes = py::reinterpret_steal<py::bytes>(dstPyBytes); // transfer ownership of `dstPyBytes` to `dstBytes`
    py::object img = frombytes(mode, py::make_tuple(width, height), dstBytes);
    return img;
}

py::object Utils::convertSequenceValueToPython(MLSequence *seq) {

    if (seq == nil) {
        return py::none();
    }

    py::list ret;

    if(seq.type == MLFeatureTypeString) {
        for(NSString* s in seq.stringValues) {
            ret.append(py::str(s.UTF8String));
        }
    } else if(seq.type == MLFeatureTypeInt64) {
        for(NSNumber* n in seq.stringValues) {
            ret.append(py::int_(n.longLongValue) );
        }
    } else {
        throw std::runtime_error("Error: Unrecognized sequence type.");
    }

    return std::move(ret);
}

py::object Utils::convertValueToPython(MLFeatureValue *value) {
    switch ([value type]) {
        case MLFeatureTypeInt64:
            return py::int_(value.int64Value);
        case MLFeatureTypeMultiArray:
            return convertArrayValueToPython(value.multiArrayValue);
        case MLFeatureTypeImage:
            return convertImageValueToPython(value.imageBufferValue);
        case MLFeatureTypeDouble:
            return py::float_(value.doubleValue);
        case MLFeatureTypeString:
            return py::str(value.stringValue.UTF8String);
        case MLFeatureTypeDictionary:
            return convertDictionaryValueToPython(value.dictionaryValue);
        case MLFeatureTypeSequence:
            if (@available(macOS 10.14, *)) {
                return convertSequenceValueToPython(value.sequenceValue);
            } else {
                // Fallback on earlier versions
            }
        case MLFeatureTypeInvalid:
            assert(false);
            return py::none();
    }
    return py::object();
}
