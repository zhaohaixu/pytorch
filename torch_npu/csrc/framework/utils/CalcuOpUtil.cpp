#include <fcntl.h>
#include <unistd.h>
#include <stdexcept>

#include <ATen/record_function.h>

#include "third_party/acl/inc/acl/acl_base.h"
#include "third_party/acl/inc/acl/acl_rt.h"
#include "torch_npu/csrc/aten/mirror/NPUMemoryOverlap.h"
#include "torch_npu/csrc/core/NPUBridge.h"
#include "torch_npu/csrc/core/NPUStorageImpl.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/core/npu/interface/AclInterface.h"
#include "torch_npu/csrc/core/npu/interface/AsyncTaskQueueInterface.h"
#include "torch_npu/csrc/core/npu/register/OptionRegister.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/framework/InferFormat.h"
#include "torch_npu/csrc/framework/contiguous/ReshapeOpt.h"
#include "torch_npu/csrc/framework/interface/AclOpCompileInterface.h"
#include "torch_npu/csrc/framework/interface/EnvVariables.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/ForceJitCompileList.h"
#include "torch_npu/csrc/framework/utils/NpuUtils.h"

namespace {
constexpr float EPSILON = 1e-6;

// check all at::ScalarType is not negative
#define ENUM_PAIR_FUNC(_1, n) static_assert(static_cast<int64_t>(at::ScalarType::n) >= 0, #n " is negative");
AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(ENUM_PAIR_FUNC)
#undef ENUM_PAIR_FUNC

#define AT_ALL_SCALAR_TYPE_AND_ACL_DATATYPE_PAIR(_)                                                                    \
    _(at::ScalarType::Byte, ACL_UINT8)                                                                                 \
    _(at::ScalarType::Char, ACL_INT8)                                                                                  \
    _(at::ScalarType::Short, ACL_INT16)                                                                                \
    _(at::ScalarType::Int, ACL_INT32)                                                                                  \
    _(at::ScalarType::Long, ACL_INT64)                                                                                 \
    _(at::ScalarType::Half, ACL_FLOAT16)                                                                               \
    _(at::ScalarType::Float, ACL_FLOAT)                                                                                \
    _(at::ScalarType::Double, ACL_DOUBLE)                                                                              \
    _(at::ScalarType::ComplexHalf, ACL_COMPLEX32)                                                                      \
    _(at::ScalarType::ComplexFloat, ACL_COMPLEX64)                                                                     \
    _(at::ScalarType::ComplexDouble, ACL_COMPLEX128)                                                                   \
    _(at::ScalarType::Bool, ACL_BOOL)                                                                                  \
    _(at::ScalarType::QInt8, ACL_DT_UNDEFINED)                                                                         \
    _(at::ScalarType::QUInt8, ACL_DT_UNDEFINED)                                                                        \
    _(at::ScalarType::QInt32, ACL_DT_UNDEFINED)                                                                        \
    _(at::ScalarType::BFloat16, ACL_BF16)                                                                              \
    _(at::ScalarType::QUInt4x2, ACL_DT_UNDEFINED)                                                                      \
    _(at::ScalarType::QUInt2x4, ACL_DT_UNDEFINED)                                                                      \
    _(at::ScalarType::Bits1x8, ACL_DT_UNDEFINED)                                                                       \
    _(at::ScalarType::Bits2x4, ACL_DT_UNDEFINED)                                                                       \
    _(at::ScalarType::Bits4x2, ACL_DT_UNDEFINED)                                                                       \
    _(at::ScalarType::Bits8, ACL_DT_UNDEFINED)                                                                         \
    _(at::ScalarType::Bits16, ACL_DT_UNDEFINED)                                                                        \
    _(at::ScalarType::Float8_e5m2, ACL_DT_UNDEFINED)                                                                    \
    _(at::ScalarType::Float8_e4m3fn, ACL_DT_UNDEFINED)                                                                \
    _(at::ScalarType::Float8_e5m2fnuz, ACL_DT_UNDEFINED)                                                               \
    _(at::ScalarType::Float8_e4m3fnuz, ACL_DT_UNDEFINED)                                                               \
    _(at::ScalarType::UInt16, ACL_UINT16)                                                                              \
    _(at::ScalarType::UInt32, ACL_UINT32)                                                                              \
    _(at::ScalarType::UInt64, ACL_UINT64)                                                                              \
    _(at::ScalarType::UInt1, ACL_DT_UNDEFINED)                                                                         \
    _(at::ScalarType::UInt2, ACL_DT_UNDEFINED)                                                                         \
    _(at::ScalarType::UInt3, ACL_DT_UNDEFINED)                                                                         \
    _(at::ScalarType::UInt4, ACL_DT_UNDEFINED)                                                                         \
    _(at::ScalarType::UInt5, ACL_DT_UNDEFINED)                                                                         \
    _(at::ScalarType::UInt6, ACL_DT_UNDEFINED)                                                                         \
    _(at::ScalarType::UInt7, ACL_DT_UNDEFINED)                                                                         \
    _(at::ScalarType::Int1, ACL_DT_UNDEFINED)                                                                          \
    _(at::ScalarType::Int2, ACL_DT_UNDEFINED)                                                                          \
    _(at::ScalarType::Int3, ACL_DT_UNDEFINED)                                                                          \
    _(at::ScalarType::Int4, ACL_DT_UNDEFINED)                                                                          \
    _(at::ScalarType::Int5, ACL_DT_UNDEFINED)                                                                          \
    _(at::ScalarType::Int6, ACL_DT_UNDEFINED)                                                                          \
    _(at::ScalarType::Int7, ACL_DT_UNDEFINED)                                                                          \
    _(at::ScalarType::Float8_e8m0fnu, ACL_DT_UNDEFINED)                                                                \
    _(at::ScalarType::Undefined, ACL_DT_UNDEFINED)                                                                     \
    _(at::ScalarType::NumOptions, ACL_DT_UNDEFINED)

constexpr aclDataType kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(at::ScalarType::NumOptions) + 1] = {
#define DEFINE_ENUM(_1, n) n,
    AT_ALL_SCALAR_TYPE_AND_ACL_DATATYPE_PAIR(DEFINE_ENUM)
#undef DEFINE_ENUM
};

// check at::ScalarType has been changed or not
#define ENUM_PAIR_FUNC(at_dtype, acl_dtype)                                                                            \
    static_assert(kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(at_dtype)] == (acl_dtype),                    \
                  #at_dtype " and " #acl_dtype " is not match any more, please check "                                 \
                            "AT_ALL_SCALAR_TYPE_AND_ACL_DATATYPE_PAIR and modify it");
AT_ALL_SCALAR_TYPE_AND_ACL_DATATYPE_PAIR(ENUM_PAIR_FUNC)
#undef DEFINE_ENUM

static std::map<const std::string, const aclDataType> STRING_SCALAR_TYPE_TO_ACL_TYPE_MAP = {
    {"uint16", ACL_UINT16}, {"uint8", ACL_UINT8}, {"uint64", ACL_UINT64}, {"string", ACL_STRING}};

static std::unordered_map<const aclDataType, const at::ScalarType>
    ACL_TYPE_TO_SCALAR_TYPE_MAP = {{ACL_DT_UNDEFINED, at::ScalarType::Undefined},
                                   {ACL_FLOAT, at::ScalarType::Float},
                                   {ACL_FLOAT16, at::ScalarType::Half},
                                   {ACL_INT8, at::ScalarType::Char},
                                   {ACL_INT32, at::ScalarType::Int},
                                   {ACL_UINT8, at::ScalarType::Byte},
                                   {ACL_INT16, at::ScalarType::Short},
                                   {ACL_UINT16, at::ScalarType::UInt16},
                                   {ACL_UINT32, at::ScalarType::UInt32},
                                   {ACL_INT64, at::ScalarType::Long},
                                   {ACL_UINT64, at::ScalarType::UInt64},
                                   {ACL_DOUBLE, at::ScalarType::Double},
                                   {ACL_BOOL, at::ScalarType::Bool},
                                   {ACL_STRING, at::ScalarType::Undefined},
                                   {ACL_COMPLEX64, at::ScalarType::ComplexFloat},
                                   {ACL_COMPLEX128, at::ScalarType::ComplexDouble},
                                   {ACL_BF16, at::ScalarType::BFloat16},
                                   {ACL_INT4, at::ScalarType::Undefined},
                                   {ACL_UINT1, at::ScalarType::Undefined},
                                   {ACL_COMPLEX32, at::ScalarType::ComplexHalf}};

extern "C" void chacha20_encrypt_do(uint32_t blockDim, void *stream, void* state, void* input, void* output, uint32_t dataSize, uint32_t workspace);
extern "C" void chacha20_encrypt_generate_mask(uint32_t blockDim, void *stream, void* state, void* output, uint32_t dataSize);

class Singleton {
public:
    static Singleton& getInstance() {
        static Singleton instance;  // C++11保证线程安全
        return instance;
    }
   
    void parallelQuarterRoundsCPU(uint32_t* state,
                               int a1, int b1, int c1, int d1,
                               int a2, int b2, int c2, int d2,
                               int a3, int b3, int c3, int d3,
                               int a4, int b4, int c4, int d4)
    {
        // 批量读取，减少内存访问
        uint32_t va1 = state[a1], vb1 = state[b1], vc1 = state[c1], vd1 = state[d1];
        uint32_t va2 = state[a2], vb2 = state[b2], vc2 = state[c2], vd2 = state[d2];
        uint32_t va3 = state[a3], vb3 = state[b3], vc3 = state[c3], vd3 = state[d3];
        uint32_t va4 = state[a4], vb4 = state[b4], vc4 = state[c4], vd4 = state[d4];

        // 第一步
        va1 += vb1; vd1 ^= va1; vd1 = (vd1 << 16) | (vd1 >> 16);
        va2 += vb2; vd2 ^= va2; vd2 = (vd2 << 16) | (vd2 >> 16);
        va3 += vb3; vd3 ^= va3; vd3 = (vd3 << 16) | (vd3 >> 16);
        va4 += vb4; vd4 ^= va4; vd4 = (vd4 << 16) | (vd4 >> 16);
        
        // 第二步
        vc1 += vd1; vb1 ^= vc1; vb1 = (vb1 << 12) | (vb1 >> 20);
        vc2 += vd2; vb2 ^= vc2; vb2 = (vb2 << 12) | (vb2 >> 20);
        vc3 += vd3; vb3 ^= vc3; vb3 = (vb3 << 12) | (vb3 >> 20);
        vc4 += vd4; vb4 ^= vc4; vb4 = (vb4 << 12) | (vb4 >> 20);
        
        // 第三步
        va1 += vb1; vd1 ^= va1; vd1 = (vd1 << 8) | (vd1 >> 24);
        va2 += vb2; vd2 ^= va2; vd2 = (vd2 << 8) | (vd2 >> 24);
        va3 += vb3; vd3 ^= va3; vd3 = (vd3 << 8) | (vd3 >> 24);
        va4 += vb4; vd4 ^= va4; vd4 = (vd4 << 8) | (vd4 >> 24);
        
        // 第四步
        vc1 += vd1; vb1 ^= vc1; vb1 = (vb1 << 7) | (vb1 >> 25);
        vc2 += vd2; vb2 ^= vc2; vb2 = (vb2 << 7) | (vb2 >> 25);
        vc3 += vd3; vb3 ^= vc3; vb3 = (vb3 << 7) | (vb3 >> 25);
        vc4 += vd4; vb4 ^= vc4; vb4 = (vb4 << 7) | (vb4 >> 25);
        
        // 批量写入
        state[a1] = va1; state[b1] = vb1; state[c1] = vc1; state[d1] = vd1;
        state[a2] = va2; state[b2] = vb2; state[c2] = vc2; state[d2] = vd2;
        state[a3] = va3; state[b3] = vb3; state[c3] = vc3; state[d3] = vd3;
        state[a4] = va4; state[b4] = vb4; state[c4] = vc4; state[d4] = vd4;
    }

    void chacha20_encrypt_generate_mask_cpu(uint32_t* state, uint32_t* output, uint32_t dataSize){
        size_t BLOCK_SIZE = 64;
        size_t totalBlocks = (dataSize + BLOCK_SIZE - 1) / BLOCK_SIZE;  
        for (uint32_t block = 0; block < totalBlocks; block++) {
            uint32_t workingState[16];
            for (int i = 0; i < 16; i++) {
                workingState[i] = state[i];
            }
            workingState[12] += block;
            for (int i = 0; i < 16; i++) {
                output[block * 16 + i] = workingState[i];
            }
            for (int round = 0; round < 10; round++) {
                parallelQuarterRoundsCPU(workingState, 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15); 
                parallelQuarterRoundsCPU(workingState, 0, 5, 10, 15, 1, 6, 11, 12, 2, 7, 8, 13, 3, 4, 9, 14);  
            }
            for (int i = 0; i < 16; i++) {
                output[block * 16 + i] += workingState[i];
            }
        }
    }

    void chacha20_encrypt_do_cpu(uint8_t* state, uint8_t* input, uint8_t* output, uint32_t dataSize){
        for (uint32_t i = 0; i < dataSize; i++) {
            output[i] = input[i] ^ state[i];
        }
    }
    
    void enc(void *input_ptr, void *output_ptr, size_t data_size, bool is_h2d) {
        std::lock_guard<std::mutex> lock(mutex_);
        unique_values_.insert(data_size);
        if (this->enc_stream == nullptr)
            this->enc_stream = c10_npu::getCurrentNPUStream();

        size_t threadnum = this->stream_buffer_size_ / 64 / 2048;

        if (is_h2d) {
            if (this->host_current_pos + data_size > stream_buffer_size_) {
                printf("generate key stream for Memcpy......\n");
                int fd = open("/dev/urandom", O_RDONLY);
                if (fd < 0) throw std::runtime_error("Failed to open /dev/urandom");
                ssize_t read_bytes = read(fd, this->host_state_buffer_, 64);
                if (read_bytes != (ssize_t)64) throw std::runtime_error("Failed to read random data");
                close(fd);
                aclrtMemcpyAsync(this->device_state_buffer_, 64, this->host_state_buffer_, 64, ACL_MEMCPY_HOST_TO_DEVICE, this->enc_stream);
                // aclrtSynchronizeStream(this->enc_stream);
                chacha20_encrypt_generate_mask(threadnum, this->enc_stream, this->device_state_buffer_, this->device_stream_buffer_, this->stream_buffer_size_);
                chacha20_encrypt_generate_mask_cpu((uint32_t*)(this->host_state_buffer_), (uint32_t*)(this->host_stream_buffer_), this->stream_buffer_size_);
                // aclrtSynchronizeStream(this->enc_stream);

                this->device_current_pos = 0;
                this->host_current_pos = 0;
            }
        } else {
            if (this->device_current_pos + data_size > stream_buffer_size_) {
                printf("generate key stream for Memcpy......\n");
                int fd = open("/dev/urandom", O_RDONLY);
                if (fd < 0) throw std::runtime_error("Failed to open /dev/urandom");
                ssize_t read_bytes = read(fd, this->host_state_buffer_, 64);
                if (read_bytes != (ssize_t)64) throw std::runtime_error("Failed to read random data");
                close(fd);
                aclrtMemcpyAsync(this->device_state_buffer_, 64, this->host_state_buffer_, 64, ACL_MEMCPY_HOST_TO_DEVICE, this->enc_stream);
                // aclrtSynchronizeStream(this->enc_stream);
                chacha20_encrypt_generate_mask(threadnum, this->enc_stream, this->device_state_buffer_, this->device_stream_buffer_, this->stream_buffer_size_);
                chacha20_encrypt_generate_mask_cpu((uint32_t*)(this->host_state_buffer_), (uint32_t*)(this->host_stream_buffer_), this->stream_buffer_size_);
                // aclrtSynchronizeStream(this->enc_stream);

                this->device_current_pos = 0;
                this->host_current_pos = 0;
            }
        }

        uint8_t* key_stream_ptr = nullptr;
        if (is_h2d){
            key_stream_ptr = reinterpret_cast<uint8_t*>(this->host_stream_buffer_) + this->host_current_pos;
            if (this->host_current_pos + data_size > stream_buffer_size_)
                printf("data_size[%d] > 512MB!!!\n", data_size);
            chacha20_encrypt_do_cpu(key_stream_ptr, (uint8_t*)input_ptr, (uint8_t*)output_ptr, data_size);
            uint32_t localSizePadding = (data_size + 31) / 32 * 32;
            this->host_current_pos += localSizePadding;
        } else {
            key_stream_ptr = reinterpret_cast<uint8_t*>(this->device_stream_buffer_) + this->device_current_pos;
            if (this->device_current_pos + data_size > stream_buffer_size_)
                printf("data_size[%d] > 512MB!!!\n", data_size);
            chacha20_encrypt_do(32, this->enc_stream, key_stream_ptr, (uint8_t*)input_ptr, (uint8_t*)output_ptr, data_size, 4096);
            aclrtSynchronizeStream(this->enc_stream);
            uint32_t localSizePadding = (data_size + 31) / 32 * 32;
            this->device_current_pos += localSizePadding;
        }
    }

    void dec(void *input_ptr, void *output_ptr, size_t data_size, bool is_h2d) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (this->enc_stream == nullptr)
            this->enc_stream = c10_npu::getCurrentNPUStream();

        size_t threadnum = this->stream_buffer_size_ / 64 / 2048;

        uint8_t* key_stream_ptr = nullptr;
        if (!is_h2d){
            key_stream_ptr = reinterpret_cast<uint8_t*>(this->host_stream_buffer_) + this->host_current_pos;
            chacha20_encrypt_do_cpu(key_stream_ptr, (uint8_t*)input_ptr, (uint8_t*)output_ptr, data_size);
            uint32_t localSizePadding = (data_size + 31) / 32 * 32;
            this->host_current_pos += localSizePadding;
            if (this->device_current_pos != this->host_current_pos)
                printf("waring!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        } else {
            key_stream_ptr = reinterpret_cast<uint8_t*>(this->device_stream_buffer_) + this->device_current_pos;
            chacha20_encrypt_do(32, this->enc_stream, key_stream_ptr, (uint8_t*)input_ptr, (uint8_t*)output_ptr, data_size, 4096);
            // aclrtSynchronizeStream(this->enc_stream);
            uint32_t localSizePadding = (data_size + 31) / 32 * 32;
            this->device_current_pos += localSizePadding;
            if (this->device_current_pos != this->host_current_pos)
                printf("waring!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        }
    }

private:
    Singleton() {
        std::lock_guard<std::mutex> lock(mutex_);

        aclrtMalloc(&device_stream_buffer_, stream_buffer_size_, ACL_MEM_MALLOC_HUGE_FIRST);
        host_stream_buffer_ = malloc(stream_buffer_size_);
        aclrtMalloc(&device_state_buffer_, 64, ACL_MEM_MALLOC_HUGE_FIRST);
        host_state_buffer_ = malloc(64);

        host_current_pos = stream_buffer_size_;
        device_current_pos = stream_buffer_size_;
    }
    ~Singleton() {
        if (device_stream_buffer_)
            aclrtFree(device_stream_buffer_);
        if (device_state_buffer_)
            aclrtFree(device_state_buffer_);
        if (host_stream_buffer_)
            free(host_stream_buffer_);
        if (host_state_buffer_)
            free(host_state_buffer_);
    }

    // 禁止拷贝和赋值
    Singleton(const Singleton&) = delete;
    Singleton& operator=(const Singleton&) = delete;

    void* host_stream_buffer_ = nullptr;
    void* device_stream_buffer_ = nullptr;
    void* host_state_buffer_ = nullptr;
    void* device_state_buffer_ = nullptr;
    size_t stream_buffer_size_ = 1024 * 1024 * 1024;
    size_t host_current_pos = 0;
    size_t device_current_pos = 0;
    aclrtStream enc_stream = nullptr;
    mutable std::mutex mutex_;
    std::unordered_set<size_t> unique_values_;
};

aclError AclrtMemcpyAsyncParamCheck(
    void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind, aclrtStream stream)
{
    auto ret = aclrtMemcpyAsync(dst, destMax, src, count, kind, stream);
    return ret;
}

aclError AclrtMemcpyParamCheck(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind)
{
    if (ACL_MEMCPY_HOST_TO_DEVICE == kind) {
        size_t i = 0;

        void* src_tmp = nullptr;
        src_tmp = malloc(count);

        // aclrtSynchronizeDevice();
        Singleton::getInstance().enc(const_cast<void*>(src), const_cast<void*>(src_tmp), count, true);
        // aclrtSynchronizeDevice();
        auto ret = aclrtMemcpy(dst, count, src_tmp, count, kind);
        // aclrtSynchronizeDevice();
        Singleton::getInstance().dec(dst, dst, count, true);
        // aclrtSynchronizeDevice();
        // NPU_CHECK_ERROR(aclrtMemcpy(src_tmp, count, dst, count, ACL_MEMCPY_DEVICE_TO_HOST));
        // aclrtSynchronizeDevice();
        free(src_tmp);
        // printf(">>>>>>>>>>>>>> H2D: %d B. <<<<<<<<<<<<<<<<<\n", count);

        return ret;
    }
    else if (ACL_MEMCPY_DEVICE_TO_HOST == kind) {
        size_t i = 0;

        void* src_tmp = nullptr;
        NPU_CHECK_ERROR(aclrtMalloc(&src_tmp, count, ACL_MEM_MALLOC_HUGE_FIRST));

        // aclrtSynchronizeDevice();
        Singleton::getInstance().enc(const_cast<void*>(src), const_cast<void*>(src_tmp), count, false);
        // aclrtSynchronizeDevice();
        auto ret = aclrtMemcpy(dst, count, src_tmp, count, kind);
        // aclrtSynchronizeDevice();
        Singleton::getInstance().dec(dst, dst, count, false);
        // aclrtSynchronizeDevice();
        NPU_CHECK_ERROR(aclrtFree(src_tmp));
        // printf(">>>>>>>>>>>>>> D2H: %d B. <<<<<<<<<<<<<<<<<\n", count);

        return ret;
    }
    else{
        auto ret = aclrtMemcpy(dst, destMax, src, count, kind);
        return ret;
    }
}
} // namespace

namespace at_npu {
namespace native {
aclDataType CalcuOpUtil::ConvertToAclDataType(const at::ScalarType &data_type)
{
    auto acl_dtype = kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(data_type)];
    TORCH_CHECK(acl_dtype != ACL_DT_UNDEFINED,
                std::string(c10::toString(data_type)) + " has not been supported",
                OPS_ERROR(ErrCode::NOT_SUPPORT))
    return acl_dtype;
}

aclDataType CalcuOpUtil::ConvertToAclDataType(const at::ScalarType &data_type, const std::string &realDataType)
{
    auto acl_dtype = kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(data_type)];
    TORCH_CHECK(acl_dtype != ACL_DT_UNDEFINED,
                std::string(c10::toString(data_type)) + " has not been supported",
                OPS_ERROR(ErrCode::NOT_SUPPORT))
    if (!realDataType.empty()) {
        return STRING_SCALAR_TYPE_TO_ACL_TYPE_MAP[realDataType];
    }
    return acl_dtype;
}

c10::Scalar CalcuOpUtil::ConvertTensorToScalar(const at::Tensor &tensor)
{
    c10::Scalar expScalar;
    const at::Tensor *aclInput = &tensor;
    if (aclInput->scalar_type() == at::ScalarType::Double) {
        double value = *(double *)aclInput->data_ptr();
        c10::Scalar scalar(value);
        expScalar = scalar;
    } else if (aclInput->scalar_type() == at::ScalarType::Long) {
        int64_t value = *(int64_t *)aclInput->data_ptr();
        c10::Scalar scalar(value);
        expScalar = scalar;
    } else if (aclInput->scalar_type() == at::ScalarType::Float) {
        float value = *(float *)aclInput->data_ptr();
        c10::Scalar scalar(value);
        expScalar = scalar;
    } else if (aclInput->scalar_type() == at::ScalarType::Int) {
        int value = *(int *)aclInput->data_ptr();
        c10::Scalar scalar(value);
        expScalar = scalar;
    } else if (aclInput->scalar_type() == at::ScalarType::Half) {
        c10::Half value = *(c10::Half *)aclInput->data_ptr();
        c10::Scalar scalar(value);
        expScalar = scalar;
    } else {
        ASCEND_LOGE("unsupport scalar type! ");
        NPU_CHECK_ERROR(ACL_ERROR_UNSUPPORTED_DATA_TYPE);
    }

    return expScalar;
}

at::Tensor CalcuOpUtil::CopyScalarToDevice(const c10::Scalar &cpu_scalar, at::ScalarType scalar_data_type)
{
    return CalcuOpUtil::CopyTensorHostToDevice(scalar_to_tensor(cpu_scalar).to(scalar_data_type));
}

at::Tensor CalcuOpUtil::CopyTensorHostToDevice(const at::Tensor &cpu_tensor)
{
    at::Tensor cpuPinMemTensor = cpu_tensor.pin_memory();
    int deviceIndex = 0;
    NPU_CHECK_ERROR(c10_npu::GetDevice(&deviceIndex));
    return cpuPinMemTensor.to(
        c10::Device(c10::DeviceType::PrivateUse1, deviceIndex), cpuPinMemTensor.scalar_type(), true, true);
}

NPUStatus CalcuOpUtil::AclrtMemcpyAsync(const std::pair<at::Tensor, int64_t> &dst,
                                        size_t dst_size,
                                        const std::pair<at::Tensor, int64_t> &src,
                                        size_t src_size,
                                        aclrtMemcpyKind kind)
{
    void *dst_ptr = reinterpret_cast<uint8_t *>(dst.first.data_ptr()) + dst.second * dst.first.itemsize();
    void *src_ptr = reinterpret_cast<uint8_t *>(src.first.data_ptr()) + src.second * src.first.itemsize();
    NPU_CHECK_ERROR(
        c10_npu::queue::LaunchAsyncCopyTask(dst_ptr, dst_size, const_cast<void *>(src_ptr), src_size, kind));

    return NPU_STATUS_SUCCESS;
}

aclError CalcuOpUtil::AclrtMemcpyWithModeSwitch(const StorageAndOffsetMemSizePair &dst,
                                                size_t dstMax,
                                                const StorageAndOffsetMemSizePair &src,
                                                size_t count,
                                                aclrtMemcpyKind kind)
{
    void *dst_ptr = static_cast<void *>(static_cast<uint8_t *>(const_cast<void *>(dst.first->data())) + dst.second);
    void *src_ptr = static_cast<void *>(static_cast<uint8_t *>(const_cast<void *>(src.first->data())) + src.second);
    return AclrtMemcpyParamCheck(dst_ptr, dstMax, const_cast<void *>(src_ptr), count, kind);
}

aclError CalcuOpUtil::AclrtMemcpyWithModeSwitch(
    const StorageAndOffsetMemSizePair &dst, size_t dstMax, const void *src, size_t count, aclrtMemcpyKind kind)
{
    void *dst_ptr = static_cast<void *>(static_cast<uint8_t *>(const_cast<void *>(dst.first->data())) + dst.second);
    return AclrtMemcpyParamCheck(dst_ptr, dstMax, src, count, kind);
}

aclError CalcuOpUtil::AclrtMemcpyWithModeSwitch(
    void *dst, size_t dstMax, const StorageAndOffsetMemSizePair &src, size_t count, aclrtMemcpyKind kind)
{
    void *src_ptr = static_cast<void *>(static_cast<uint8_t *>(const_cast<void *>(src.first->data())) + src.second);
    return AclrtMemcpyParamCheck(dst, dstMax, const_cast<void *>(src_ptr), count, kind);
}

aclError CalcuOpUtil::LaunchAsyncCopyTaskWithModeSwitch(
    const at::Tensor &dst, size_t dstMax, const at::Tensor &src, size_t count, aclrtMemcpyKind kind)
{
    aclError ret = c10_npu::queue::LaunchAsyncCopyTask(dst.data_ptr(), dstMax, src.data_ptr(), count, kind);
    return ret;
}

aclError CalcuOpUtil::LaunchAsyncCopyTaskWithModeSwitch(
    const c10::StorageImpl &dst, size_t dstMax, void *src, size_t count, aclrtMemcpyKind kind)
{
    aclError ret = c10_npu::queue::LaunchAsyncCopyTask(const_cast<void *>(dst.data()), dstMax, src, count, kind);
    return ret;
}

int64_t CalcuOpUtil::GetTensorNpuFormat(const at::Tensor &tensor)
{
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1,
                "Expected all tensors to be on the same device. "
                "Expected NPU tensor, please check whether the input tensor "
                "device is correct.",
                OPS_ERROR(ErrCode::TYPE));
    if (NpuUtils::check_match(&tensor) || NpuUtils::check_5d_5d_match(tensor)) {
        const torch_npu::NPUStorageDesc &tensor_desc = torch_npu::NPUBridge::GetNpuStorageImpl(tensor)->npu_desc_;
        return tensor_desc.npu_format_;
    } else if (tensor.data_ptr() == nullptr) {
        // transforming faketensor into realtensor and assigning format ND
        return ACL_FORMAT_ND;
    } else {
        return InferFormat::GuessFormatWhenContiguous(tensor);
    }
}

void CalcuOpUtil::CheckMemoryOverLaps(c10::ArrayRef<at::Tensor> inputs, c10::ArrayRef<at::Tensor> outputs)
{
    for (const auto i : c10::irange(outputs.size())) {
        if (!outputs[i].defined()) {
            continue;
        }

        assert_no_internal_overlap(outputs[i]);

        for (const auto j : c10::irange(inputs.size())) {
            assert_no_partial_overlap(outputs[i], inputs[j]);
        }
    }
}

bool CalcuOpUtil::IsScalarWrappedToTensor(const at::Tensor &tensor)
{
    return tensor.unsafeGetTensorImpl()->is_wrapped_number() && (!torch_npu::utils::is_npu(tensor));
}

float CalcuOpUtil::GetScalarFloatValue(const c10::Scalar &scalar)
{
    float value;
    if (scalar.isFloatingPoint()) {
        value = scalar.toFloat();
    } else {
        value = static_cast<float>(scalar.toInt());
    }

    return value;
}

c10::SmallVector<int64_t, SHAPE_SIZE> CalcuOpUtil::ConvertIntArrayRefToSmallVector(c10::IntArrayRef intArray)
{
    c10::SmallVector<int64_t, SHAPE_SIZE> intVec;
    for (const auto i : c10::irange(intArray.size())) {
        intVec.emplace_back(intArray[i]);
    }

    return intVec;
}

using aclCubeMathType = enum : int8_t {
    KEEP_DTYPE = 0,
    ALLOW_FP32_DOWN_PRECISION = 1,
    USE_FP16 = 2,
    USE_HF32 = 3,
};

static std::unordered_map<uint8_t, aclCubeMathType> ACL_CUBE_MATH_TYPE_MAP = {
    {0b00, KEEP_DTYPE}, {0b01, USE_FP16}, {0b10, USE_HF32}, {0b11, ALLOW_FP32_DOWN_PRECISION}};

int8_t CalcuOpUtil::GetCubeMathType(bool allowHf32)
{
    bool allowFp32ToFp16 = native::env::IsAllowFP32ToFP16();
    uint8_t CubeMathTypeCode = (static_cast<uint8_t>(allowHf32) << 1) + static_cast<uint8_t>(allowFp32ToFp16);
    auto iter = ACL_CUBE_MATH_TYPE_MAP.find(CubeMathTypeCode);
    if (iter == ACL_CUBE_MATH_TYPE_MAP.end()) {
        return ALLOW_FP32_DOWN_PRECISION;
    }
    return iter->second;
}

at::ScalarType CalcuOpUtil::ConvertToScalarType(const aclDataType data_type)
{
    auto iter = ACL_TYPE_TO_SCALAR_TYPE_MAP.find(data_type);
    if (iter == ACL_TYPE_TO_SCALAR_TYPE_MAP.end()) {
        TORCH_CHECK(false,
            std::string("aclDataType:") + std::to_string(data_type) + " has not been supported",
            OPS_ERROR(ErrCode::NOT_SUPPORT))
    }
    
    return iter->second;
}

} // namespace native
} // namespace at_npu
