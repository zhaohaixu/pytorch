#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"

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

// Host wrappers exported by the encryption AscendC kernels.
extern "C" void aes_cube_generate_mask(uint32_t blockDim, void *stream,
                                  void *roundKeysPadded192,
                                  void *input, void *output,
                                  void *te0, void *te1, void *te2,
                                  void *te3, void *sbox,
                                  void *b_workspace, void *c_workspace,
                                  void *workspace, void *tiling,
                                  uint32_t nounce, uint32_t dataSize);

extern "C" void aes_vec_generate_mask(uint32_t blockDim, void *stream,
                                       void *roundKeysPadded192,
                                       void *input, void *output,
                                       uint32_t nounce1, uint32_t nounce2,
                                       uint32_t nounce3, uint32_t dataSize);

extern "C" void xor_do(uint32_t blockDim, void *stream, void *state,
                       void *input, void *output, uint32_t dataSize,
                       uint32_t workspace);
extern "C" void chacha20_naive_generate_mask(uint32_t blockDim, void *stream,
                                               void *state, void *output,
                                               uint32_t dataSize);

enum class TorchEncryptionAlgorithm : uint8_t {
    DISABLED = 0,
    CHACHA20_NAIVE,
    AES_CUBE,
    AES_VEC,
};

constexpr const char *TORCH_ENC_ENV = "TORCH_ENC_ENABLE";
constexpr const char *CHACHA20_NAIVE_NAME = "chacha20-naive";
constexpr const char *AES_CUBE_NAME = "aes-cube";
constexpr const char *AES_VEC_NAME = "aes-vec";

const char *GetEncryptionAlgorithmName(TorchEncryptionAlgorithm algorithm)
{
    switch (algorithm) {
        case TorchEncryptionAlgorithm::CHACHA20_NAIVE:
            return CHACHA20_NAIVE_NAME;
        case TorchEncryptionAlgorithm::AES_CUBE:
            return AES_CUBE_NAME;
        case TorchEncryptionAlgorithm::AES_VEC:
            return AES_VEC_NAME;
        case TorchEncryptionAlgorithm::DISABLED:
            return "disabled";
    }
    return "unknown";
}

std::string NormalizeEncryptionAlgorithm(const char *configuredValue)
{
    std::string algorithm(configuredValue == nullptr ? "" : configuredValue);

    size_t begin = 0;
    while (begin < algorithm.size() &&
           std::isspace(static_cast<unsigned char>(algorithm[begin]))) {
        ++begin;
    }

    size_t end = algorithm.size();
    while (end > begin &&
           std::isspace(static_cast<unsigned char>(algorithm[end - 1]))) {
        --end;
    }

    algorithm = algorithm.substr(begin, end - begin);
    std::transform(algorithm.begin(), algorithm.end(), algorithm.begin(),
                   [](unsigned char ch) {
                       return static_cast<char>(std::tolower(ch));
                   });
    return algorithm;
}

// Parse and log TORCH_ENC_ENABLE once per process. The memcpy hot path only
// reads this function-local static and performs one branch.
TorchEncryptionAlgorithm GetTorchEncryptionAlgorithm()
{
    static const TorchEncryptionAlgorithm algorithm = []() {
        const char *configuredValue = std::getenv(TORCH_ENC_ENV);
        if (configuredValue == nullptr) {
            std::fprintf(stderr,
                         "[torch-npu encryption] disabled: %s is not set.\n",
                         TORCH_ENC_ENV);
            return TorchEncryptionAlgorithm::DISABLED;
        }

        const std::string normalized =
            NormalizeEncryptionAlgorithm(configuredValue);
        TorchEncryptionAlgorithm selected = TorchEncryptionAlgorithm::DISABLED;
        if (normalized == CHACHA20_NAIVE_NAME) {
            selected = TorchEncryptionAlgorithm::CHACHA20_NAIVE;
        } else if (normalized == AES_CUBE_NAME) {
            selected = TorchEncryptionAlgorithm::AES_CUBE;
        } else if (normalized == AES_VEC_NAME) {
            selected = TorchEncryptionAlgorithm::AES_VEC;
        } else {
            std::fprintf(stderr,
                         "[torch-npu encryption] unsupported %s=%s; encryption "
                         "is disabled. Supported values: %s, %s, %s.\n",
                         TORCH_ENC_ENV, configuredValue,
                         CHACHA20_NAIVE_NAME, AES_CUBE_NAME, AES_VEC_NAME);
            return TorchEncryptionAlgorithm::DISABLED;
        }

        std::fprintf(stderr,
                     "[torch-npu encryption] enabled: algorithm=%s.\n",
                     GetEncryptionAlgorithmName(selected));
        return selected;
    }();
    return algorithm;
}


using namespace matmul_tiling;

static const uint8_t Te0_cube[1024] = {
    0xa5U, 0x84U, 0x99U, 0x8dU, 0x0dU, 0xbdU, 0xb1U, 0x54U,
    0x50U, 0x03U, 0xa9U, 0x7dU, 0x19U, 0x62U, 0xe6U, 0x9aU,
    0x45U, 0x9dU, 0x40U, 0x87U, 0x15U, 0xebU, 0xc9U, 0x0bU,
    0xecU, 0x67U, 0xfdU, 0xeaU, 0xbfU, 0xf7U, 0x96U, 0x5bU,
    0xc2U, 0x1cU, 0xaeU, 0x6aU, 0x5aU, 0x41U, 0x02U, 0x4fU,
    0x5cU, 0xf4U, 0x34U, 0x08U, 0x93U, 0x73U, 0x53U, 0x3fU,
    0x0cU, 0x52U, 0x65U, 0x5eU, 0x28U, 0xa1U, 0x0fU, 0xb5U,
    0x09U, 0x36U, 0x9bU, 0x3dU, 0x26U, 0x69U, 0xcdU, 0x9fU,
    0x1bU, 0x9eU, 0x74U, 0x2eU, 0x2dU, 0xb2U, 0xeeU, 0xfbU,
    0xf6U, 0x4dU, 0x61U, 0xceU, 0x7bU, 0x3eU, 0x71U, 0x97U,
    0xf5U, 0x68U, 0x00U, 0x2cU, 0x60U, 0x1fU, 0xc8U, 0xedU,
    0xbeU, 0x46U, 0xd9U, 0x4bU, 0xdeU, 0xd4U, 0xe8U, 0x4aU,
    0x6bU, 0x2aU, 0xe5U, 0x16U, 0xc5U, 0xd7U, 0x55U, 0x94U,
    0xcfU, 0x10U, 0x06U, 0x81U, 0xf0U, 0x44U, 0xbaU, 0xe3U,
    0xf3U, 0xfeU, 0xc0U, 0x8aU, 0xadU, 0xbcU, 0x48U, 0x04U,
    0xdfU, 0xc1U, 0x75U, 0x63U, 0x30U, 0x1aU, 0x0eU, 0x6dU,
    0x4cU, 0x14U, 0x35U, 0x2fU, 0xe1U, 0xa2U, 0xccU, 0x39U,
    0x57U, 0xf2U, 0x82U, 0x47U, 0xacU, 0xe7U, 0x2bU, 0x95U,
    0xa0U, 0x98U, 0xd1U, 0x7fU, 0x66U, 0x7eU, 0xabU, 0x83U,
    0xcaU, 0x29U, 0xd3U, 0x3cU, 0x79U, 0xe2U, 0x1dU, 0x76U,
    0x3bU, 0x56U, 0x4eU, 0x1eU, 0xdbU, 0x0aU, 0x6cU, 0xe4U,
    0x5dU, 0x6eU, 0xefU, 0xa6U, 0xa8U, 0xa4U, 0x37U, 0x8bU,
    0x32U, 0x43U, 0x59U, 0xb7U, 0x8cU, 0x64U, 0xd2U, 0xe0U,
    0xb4U, 0xfaU, 0x07U, 0x25U, 0xafU, 0x8eU, 0xe9U, 0x18U,
    0xd5U, 0x88U, 0x6fU, 0x72U, 0x24U, 0xf1U, 0xc7U, 0x51U,
    0x23U, 0x7cU, 0x9cU, 0x21U, 0xddU, 0xdcU, 0x86U, 0x85U,
    0x90U, 0x42U, 0xc4U, 0xaaU, 0xd8U, 0x05U, 0x01U, 0x12U,
    0xa3U, 0x5fU, 0xf9U, 0xd0U, 0x91U, 0x58U, 0x27U, 0xb9U,
    0x38U, 0x13U, 0xb3U, 0x33U, 0xbbU, 0x70U, 0x89U, 0xa7U,
    0xb6U, 0x22U, 0x92U, 0x20U, 0x49U, 0xffU, 0x78U, 0x7aU,
    0x8fU, 0xf8U, 0x80U, 0x17U, 0xdaU, 0x31U, 0xc6U, 0xb8U,
    0xc3U, 0xb0U, 0x77U, 0x11U, 0xcbU, 0xfcU, 0xd6U, 0x3aU,

    0x63U, 0x7cU, 0x77U, 0x7bU, 0xf2U, 0x6bU, 0x6fU, 0xc5U,
    0x30U, 0x01U, 0x67U, 0x2bU, 0xfeU, 0xd7U, 0xabU, 0x76U,
    0xcaU, 0x82U, 0xc9U, 0x7dU, 0xfaU, 0x59U, 0x47U, 0xf0U,
    0xadU, 0xd4U, 0xa2U, 0xafU, 0x9cU, 0xa4U, 0x72U, 0xc0U,
    0xb7U, 0xfdU, 0x93U, 0x26U, 0x36U, 0x3fU, 0xf7U, 0xccU,
    0x34U, 0xa5U, 0xe5U, 0xf1U, 0x71U, 0xd8U, 0x31U, 0x15U,
    0x04U, 0xc7U, 0x23U, 0xc3U, 0x18U, 0x96U, 0x05U, 0x9aU,
    0x07U, 0x12U, 0x80U, 0xe2U, 0xebU, 0x27U, 0xb2U, 0x75U,
    0x09U, 0x83U, 0x2cU, 0x1aU, 0x1bU, 0x6eU, 0x5aU, 0xa0U,
    0x52U, 0x3bU, 0xd6U, 0xb3U, 0x29U, 0xe3U, 0x2fU, 0x84U,
    0x53U, 0xd1U, 0x00U, 0xedU, 0x20U, 0xfcU, 0xb1U, 0x5bU,
    0x6aU, 0xcbU, 0xbeU, 0x39U, 0x4aU, 0x4cU, 0x58U, 0xcfU,
    0xd0U, 0xefU, 0xaaU, 0xfbU, 0x43U, 0x4dU, 0x33U, 0x85U,
    0x45U, 0xf9U, 0x02U, 0x7fU, 0x50U, 0x3cU, 0x9fU, 0xa8U,
    0x51U, 0xa3U, 0x40U, 0x8fU, 0x92U, 0x9dU, 0x38U, 0xf5U,
    0xbcU, 0xb6U, 0xdaU, 0x21U, 0x10U, 0xffU, 0xf3U, 0xd2U,
    0xcdU, 0x0cU, 0x13U, 0xecU, 0x5fU, 0x97U, 0x44U, 0x17U,
    0xc4U, 0xa7U, 0x7eU, 0x3dU, 0x64U, 0x5dU, 0x19U, 0x73U,
    0x60U, 0x81U, 0x4fU, 0xdcU, 0x22U, 0x2aU, 0x90U, 0x88U,
    0x46U, 0xeeU, 0xb8U, 0x14U, 0xdeU, 0x5eU, 0x0bU, 0xdbU,
    0xe0U, 0x32U, 0x3aU, 0x0aU, 0x49U, 0x06U, 0x24U, 0x5cU,
    0xc2U, 0xd3U, 0xacU, 0x62U, 0x91U, 0x95U, 0xe4U, 0x79U,
    0xe7U, 0xc8U, 0x37U, 0x6dU, 0x8dU, 0xd5U, 0x4eU, 0xa9U,
    0x6cU, 0x56U, 0xf4U, 0xeaU, 0x65U, 0x7aU, 0xaeU, 0x08U,
    0xbaU, 0x78U, 0x25U, 0x2eU, 0x1cU, 0xa6U, 0xb4U, 0xc6U,
    0xe8U, 0xddU, 0x74U, 0x1fU, 0x4bU, 0xbdU, 0x8bU, 0x8aU,
    0x70U, 0x3eU, 0xb5U, 0x66U, 0x48U, 0x03U, 0xf6U, 0x0eU,
    0x61U, 0x35U, 0x57U, 0xb9U, 0x86U, 0xc1U, 0x1dU, 0x9eU,
    0xe1U, 0xf8U, 0x98U, 0x11U, 0x69U, 0xd9U, 0x8eU, 0x94U,
    0x9bU, 0x1eU, 0x87U, 0xe9U, 0xceU, 0x55U, 0x28U, 0xdfU,
    0x8cU, 0xa1U, 0x89U, 0x0dU, 0xbfU, 0xe6U, 0x42U, 0x68U,
    0x41U, 0x99U, 0x2dU, 0x0fU, 0xb0U, 0x54U, 0xbbU, 0x16U,

    0x63U, 0x7cU, 0x77U, 0x7bU, 0xf2U, 0x6bU, 0x6fU, 0xc5U,
    0x30U, 0x01U, 0x67U, 0x2bU, 0xfeU, 0xd7U, 0xabU, 0x76U,
    0xcaU, 0x82U, 0xc9U, 0x7dU, 0xfaU, 0x59U, 0x47U, 0xf0U,
    0xadU, 0xd4U, 0xa2U, 0xafU, 0x9cU, 0xa4U, 0x72U, 0xc0U,
    0xb7U, 0xfdU, 0x93U, 0x26U, 0x36U, 0x3fU, 0xf7U, 0xccU,
    0x34U, 0xa5U, 0xe5U, 0xf1U, 0x71U, 0xd8U, 0x31U, 0x15U,
    0x04U, 0xc7U, 0x23U, 0xc3U, 0x18U, 0x96U, 0x05U, 0x9aU,
    0x07U, 0x12U, 0x80U, 0xe2U, 0xebU, 0x27U, 0xb2U, 0x75U,
    0x09U, 0x83U, 0x2cU, 0x1aU, 0x1bU, 0x6eU, 0x5aU, 0xa0U,
    0x52U, 0x3bU, 0xd6U, 0xb3U, 0x29U, 0xe3U, 0x2fU, 0x84U,
    0x53U, 0xd1U, 0x00U, 0xedU, 0x20U, 0xfcU, 0xb1U, 0x5bU,
    0x6aU, 0xcbU, 0xbeU, 0x39U, 0x4aU, 0x4cU, 0x58U, 0xcfU,
    0xd0U, 0xefU, 0xaaU, 0xfbU, 0x43U, 0x4dU, 0x33U, 0x85U,
    0x45U, 0xf9U, 0x02U, 0x7fU, 0x50U, 0x3cU, 0x9fU, 0xa8U,
    0x51U, 0xa3U, 0x40U, 0x8fU, 0x92U, 0x9dU, 0x38U, 0xf5U,
    0xbcU, 0xb6U, 0xdaU, 0x21U, 0x10U, 0xffU, 0xf3U, 0xd2U,
    0xcdU, 0x0cU, 0x13U, 0xecU, 0x5fU, 0x97U, 0x44U, 0x17U,
    0xc4U, 0xa7U, 0x7eU, 0x3dU, 0x64U, 0x5dU, 0x19U, 0x73U,
    0x60U, 0x81U, 0x4fU, 0xdcU, 0x22U, 0x2aU, 0x90U, 0x88U,
    0x46U, 0xeeU, 0xb8U, 0x14U, 0xdeU, 0x5eU, 0x0bU, 0xdbU,
    0xe0U, 0x32U, 0x3aU, 0x0aU, 0x49U, 0x06U, 0x24U, 0x5cU,
    0xc2U, 0xd3U, 0xacU, 0x62U, 0x91U, 0x95U, 0xe4U, 0x79U,
    0xe7U, 0xc8U, 0x37U, 0x6dU, 0x8dU, 0xd5U, 0x4eU, 0xa9U,
    0x6cU, 0x56U, 0xf4U, 0xeaU, 0x65U, 0x7aU, 0xaeU, 0x08U,
    0xbaU, 0x78U, 0x25U, 0x2eU, 0x1cU, 0xa6U, 0xb4U, 0xc6U,
    0xe8U, 0xddU, 0x74U, 0x1fU, 0x4bU, 0xbdU, 0x8bU, 0x8aU,
    0x70U, 0x3eU, 0xb5U, 0x66U, 0x48U, 0x03U, 0xf6U, 0x0eU,
    0x61U, 0x35U, 0x57U, 0xb9U, 0x86U, 0xc1U, 0x1dU, 0x9eU,
    0xe1U, 0xf8U, 0x98U, 0x11U, 0x69U, 0xd9U, 0x8eU, 0x94U,
    0x9bU, 0x1eU, 0x87U, 0xe9U, 0xceU, 0x55U, 0x28U, 0xdfU,
    0x8cU, 0xa1U, 0x89U, 0x0dU, 0xbfU, 0xe6U, 0x42U, 0x68U,
    0x41U, 0x99U, 0x2dU, 0x0fU, 0xb0U, 0x54U, 0xbbU, 0x16U,

    0xc6U, 0xf8U, 0xeeU, 0xf6U, 0xffU, 0xd6U, 0xdeU, 0x91U,
    0x60U, 0x02U, 0xceU, 0x56U, 0xe7U, 0xb5U, 0x4dU, 0xecU,
    0x8fU, 0x1fU, 0x89U, 0xfaU, 0xefU, 0xb2U, 0x8eU, 0xfbU,
    0x41U, 0xb3U, 0x5fU, 0x45U, 0x23U, 0x53U, 0xe4U, 0x9bU,
    0x75U, 0xe1U, 0x3dU, 0x4cU, 0x6cU, 0x7eU, 0xf5U, 0x83U,
    0x68U, 0x51U, 0xd1U, 0xf9U, 0xe2U, 0xabU, 0x62U, 0x2aU,
    0x08U, 0x95U, 0x46U, 0x9dU, 0x30U, 0x37U, 0x0aU, 0x2fU,
    0x0eU, 0x24U, 0x1bU, 0xdfU, 0xcdU, 0x4eU, 0x7fU, 0xeaU,
    0x12U, 0x1dU, 0x58U, 0x34U, 0x36U, 0xdcU, 0xb4U, 0x5bU,
    0xa4U, 0x76U, 0xb7U, 0x7dU, 0x52U, 0xddU, 0x5eU, 0x13U,
    0xa6U, 0xb9U, 0x00U, 0xc1U, 0x40U, 0xe3U, 0x79U, 0xb6U,
    0xd4U, 0x8dU, 0x67U, 0x72U, 0x94U, 0x98U, 0xb0U, 0x85U,
    0xbbU, 0xc5U, 0x4fU, 0xedU, 0x86U, 0x9aU, 0x66U, 0x11U,
    0x8aU, 0xe9U, 0x04U, 0xfeU, 0xa0U, 0x78U, 0x25U, 0x4bU,
    0xa2U, 0x5dU, 0x80U, 0x05U, 0x3fU, 0x21U, 0x70U, 0xf1U,
    0x63U, 0x77U, 0xafU, 0x42U, 0x20U, 0xe5U, 0xfdU, 0xbfU,
    0x81U, 0x18U, 0x26U, 0xc3U, 0xbeU, 0x35U, 0x88U, 0x2eU,
    0x93U, 0x55U, 0xfcU, 0x7aU, 0xc8U, 0xbaU, 0x32U, 0xe6U,
    0xc0U, 0x19U, 0x9eU, 0xa3U, 0x44U, 0x54U, 0x3bU, 0x0bU,
    0x8cU, 0xc7U, 0x6bU, 0x28U, 0xa7U, 0xbcU, 0x16U, 0xadU,
    0xdbU, 0x64U, 0x74U, 0x14U, 0x92U, 0x0cU, 0x48U, 0xb8U,
    0x9fU, 0xbdU, 0x43U, 0xc4U, 0x39U, 0x31U, 0xd3U, 0xf2U,
    0xd5U, 0x8bU, 0x6eU, 0xdaU, 0x01U, 0xb1U, 0x9cU, 0x49U,
    0xd8U, 0xacU, 0xf3U, 0xcfU, 0xcaU, 0xf4U, 0x47U, 0x10U,
    0x6fU, 0xf0U, 0x4aU, 0x5cU, 0x38U, 0x57U, 0x73U, 0x97U,
    0xcbU, 0xa1U, 0xe8U, 0x3eU, 0x96U, 0x61U, 0x0dU, 0x0fU,
    0xe0U, 0x7cU, 0x71U, 0xccU, 0x90U, 0x06U, 0xf7U, 0x1cU,
    0xc2U, 0x6aU, 0xaeU, 0x69U, 0x17U, 0x99U, 0x3aU, 0x27U,
    0xd9U, 0xebU, 0x2bU, 0x22U, 0xd2U, 0xa9U, 0x07U, 0x33U,
    0x2dU, 0x3cU, 0x15U, 0xc9U, 0x87U, 0xaaU, 0x50U, 0xa5U,
    0x03U, 0x59U, 0x09U, 0x1aU, 0x65U, 0xd7U, 0x84U, 0xd0U,
    0x82U, 0x29U, 0x5aU, 0x1eU, 0x7bU, 0xa8U, 0x6dU, 0x2cU};

static const uint8_t Te1_cube[1024] = {
    0x63U, 0x7cU, 0x77U, 0x7bU, 0xf2U, 0x6bU, 0x6fU, 0xc5U,
    0x30U, 0x01U, 0x67U, 0x2bU, 0xfeU, 0xd7U, 0xabU, 0x76U,
    0xcaU, 0x82U, 0xc9U, 0x7dU, 0xfaU, 0x59U, 0x47U, 0xf0U,
    0xadU, 0xd4U, 0xa2U, 0xafU, 0x9cU, 0xa4U, 0x72U, 0xc0U,
    0xb7U, 0xfdU, 0x93U, 0x26U, 0x36U, 0x3fU, 0xf7U, 0xccU,
    0x34U, 0xa5U, 0xe5U, 0xf1U, 0x71U, 0xd8U, 0x31U, 0x15U,
    0x04U, 0xc7U, 0x23U, 0xc3U, 0x18U, 0x96U, 0x05U, 0x9aU,
    0x07U, 0x12U, 0x80U, 0xe2U, 0xebU, 0x27U, 0xb2U, 0x75U,
    0x09U, 0x83U, 0x2cU, 0x1aU, 0x1bU, 0x6eU, 0x5aU, 0xa0U,
    0x52U, 0x3bU, 0xd6U, 0xb3U, 0x29U, 0xe3U, 0x2fU, 0x84U,
    0x53U, 0xd1U, 0x00U, 0xedU, 0x20U, 0xfcU, 0xb1U, 0x5bU,
    0x6aU, 0xcbU, 0xbeU, 0x39U, 0x4aU, 0x4cU, 0x58U, 0xcfU,
    0xd0U, 0xefU, 0xaaU, 0xfbU, 0x43U, 0x4dU, 0x33U, 0x85U,
    0x45U, 0xf9U, 0x02U, 0x7fU, 0x50U, 0x3cU, 0x9fU, 0xa8U,
    0x51U, 0xa3U, 0x40U, 0x8fU, 0x92U, 0x9dU, 0x38U, 0xf5U,
    0xbcU, 0xb6U, 0xdaU, 0x21U, 0x10U, 0xffU, 0xf3U, 0xd2U,
    0xcdU, 0x0cU, 0x13U, 0xecU, 0x5fU, 0x97U, 0x44U, 0x17U,
    0xc4U, 0xa7U, 0x7eU, 0x3dU, 0x64U, 0x5dU, 0x19U, 0x73U,
    0x60U, 0x81U, 0x4fU, 0xdcU, 0x22U, 0x2aU, 0x90U, 0x88U,
    0x46U, 0xeeU, 0xb8U, 0x14U, 0xdeU, 0x5eU, 0x0bU, 0xdbU,
    0xe0U, 0x32U, 0x3aU, 0x0aU, 0x49U, 0x06U, 0x24U, 0x5cU,
    0xc2U, 0xd3U, 0xacU, 0x62U, 0x91U, 0x95U, 0xe4U, 0x79U,
    0xe7U, 0xc8U, 0x37U, 0x6dU, 0x8dU, 0xd5U, 0x4eU, 0xa9U,
    0x6cU, 0x56U, 0xf4U, 0xeaU, 0x65U, 0x7aU, 0xaeU, 0x08U,
    0xbaU, 0x78U, 0x25U, 0x2eU, 0x1cU, 0xa6U, 0xb4U, 0xc6U,
    0xe8U, 0xddU, 0x74U, 0x1fU, 0x4bU, 0xbdU, 0x8bU, 0x8aU,
    0x70U, 0x3eU, 0xb5U, 0x66U, 0x48U, 0x03U, 0xf6U, 0x0eU,
    0x61U, 0x35U, 0x57U, 0xb9U, 0x86U, 0xc1U, 0x1dU, 0x9eU,
    0xe1U, 0xf8U, 0x98U, 0x11U, 0x69U, 0xd9U, 0x8eU, 0x94U,
    0x9bU, 0x1eU, 0x87U, 0xe9U, 0xceU, 0x55U, 0x28U, 0xdfU,
    0x8cU, 0xa1U, 0x89U, 0x0dU, 0xbfU, 0xe6U, 0x42U, 0x68U,
    0x41U, 0x99U, 0x2dU, 0x0fU, 0xb0U, 0x54U, 0xbbU, 0x16U,

    0x63U, 0x7cU, 0x77U, 0x7bU, 0xf2U, 0x6bU, 0x6fU, 0xc5U,
    0x30U, 0x01U, 0x67U, 0x2bU, 0xfeU, 0xd7U, 0xabU, 0x76U,
    0xcaU, 0x82U, 0xc9U, 0x7dU, 0xfaU, 0x59U, 0x47U, 0xf0U,
    0xadU, 0xd4U, 0xa2U, 0xafU, 0x9cU, 0xa4U, 0x72U, 0xc0U,
    0xb7U, 0xfdU, 0x93U, 0x26U, 0x36U, 0x3fU, 0xf7U, 0xccU,
    0x34U, 0xa5U, 0xe5U, 0xf1U, 0x71U, 0xd8U, 0x31U, 0x15U,
    0x04U, 0xc7U, 0x23U, 0xc3U, 0x18U, 0x96U, 0x05U, 0x9aU,
    0x07U, 0x12U, 0x80U, 0xe2U, 0xebU, 0x27U, 0xb2U, 0x75U,
    0x09U, 0x83U, 0x2cU, 0x1aU, 0x1bU, 0x6eU, 0x5aU, 0xa0U,
    0x52U, 0x3bU, 0xd6U, 0xb3U, 0x29U, 0xe3U, 0x2fU, 0x84U,
    0x53U, 0xd1U, 0x00U, 0xedU, 0x20U, 0xfcU, 0xb1U, 0x5bU,
    0x6aU, 0xcbU, 0xbeU, 0x39U, 0x4aU, 0x4cU, 0x58U, 0xcfU,
    0xd0U, 0xefU, 0xaaU, 0xfbU, 0x43U, 0x4dU, 0x33U, 0x85U,
    0x45U, 0xf9U, 0x02U, 0x7fU, 0x50U, 0x3cU, 0x9fU, 0xa8U,
    0x51U, 0xa3U, 0x40U, 0x8fU, 0x92U, 0x9dU, 0x38U, 0xf5U,
    0xbcU, 0xb6U, 0xdaU, 0x21U, 0x10U, 0xffU, 0xf3U, 0xd2U,
    0xcdU, 0x0cU, 0x13U, 0xecU, 0x5fU, 0x97U, 0x44U, 0x17U,
    0xc4U, 0xa7U, 0x7eU, 0x3dU, 0x64U, 0x5dU, 0x19U, 0x73U,
    0x60U, 0x81U, 0x4fU, 0xdcU, 0x22U, 0x2aU, 0x90U, 0x88U,
    0x46U, 0xeeU, 0xb8U, 0x14U, 0xdeU, 0x5eU, 0x0bU, 0xdbU,
    0xe0U, 0x32U, 0x3aU, 0x0aU, 0x49U, 0x06U, 0x24U, 0x5cU,
    0xc2U, 0xd3U, 0xacU, 0x62U, 0x91U, 0x95U, 0xe4U, 0x79U,
    0xe7U, 0xc8U, 0x37U, 0x6dU, 0x8dU, 0xd5U, 0x4eU, 0xa9U,
    0x6cU, 0x56U, 0xf4U, 0xeaU, 0x65U, 0x7aU, 0xaeU, 0x08U,
    0xbaU, 0x78U, 0x25U, 0x2eU, 0x1cU, 0xa6U, 0xb4U, 0xc6U,
    0xe8U, 0xddU, 0x74U, 0x1fU, 0x4bU, 0xbdU, 0x8bU, 0x8aU,
    0x70U, 0x3eU, 0xb5U, 0x66U, 0x48U, 0x03U, 0xf6U, 0x0eU,
    0x61U, 0x35U, 0x57U, 0xb9U, 0x86U, 0xc1U, 0x1dU, 0x9eU,
    0xe1U, 0xf8U, 0x98U, 0x11U, 0x69U, 0xd9U, 0x8eU, 0x94U,
    0x9bU, 0x1eU, 0x87U, 0xe9U, 0xceU, 0x55U, 0x28U, 0xdfU,
    0x8cU, 0xa1U, 0x89U, 0x0dU, 0xbfU, 0xe6U, 0x42U, 0x68U,
    0x41U, 0x99U, 0x2dU, 0x0fU, 0xb0U, 0x54U, 0xbbU, 0x16U,

    0xc6U, 0xf8U, 0xeeU, 0xf6U, 0xffU, 0xd6U, 0xdeU, 0x91U,
    0x60U, 0x02U, 0xceU, 0x56U, 0xe7U, 0xb5U, 0x4dU, 0xecU,
    0x8fU, 0x1fU, 0x89U, 0xfaU, 0xefU, 0xb2U, 0x8eU, 0xfbU,
    0x41U, 0xb3U, 0x5fU, 0x45U, 0x23U, 0x53U, 0xe4U, 0x9bU,
    0x75U, 0xe1U, 0x3dU, 0x4cU, 0x6cU, 0x7eU, 0xf5U, 0x83U,
    0x68U, 0x51U, 0xd1U, 0xf9U, 0xe2U, 0xabU, 0x62U, 0x2aU,
    0x08U, 0x95U, 0x46U, 0x9dU, 0x30U, 0x37U, 0x0aU, 0x2fU,
    0x0eU, 0x24U, 0x1bU, 0xdfU, 0xcdU, 0x4eU, 0x7fU, 0xeaU,
    0x12U, 0x1dU, 0x58U, 0x34U, 0x36U, 0xdcU, 0xb4U, 0x5bU,
    0xa4U, 0x76U, 0xb7U, 0x7dU, 0x52U, 0xddU, 0x5eU, 0x13U,
    0xa6U, 0xb9U, 0x00U, 0xc1U, 0x40U, 0xe3U, 0x79U, 0xb6U,
    0xd4U, 0x8dU, 0x67U, 0x72U, 0x94U, 0x98U, 0xb0U, 0x85U,
    0xbbU, 0xc5U, 0x4fU, 0xedU, 0x86U, 0x9aU, 0x66U, 0x11U,
    0x8aU, 0xe9U, 0x04U, 0xfeU, 0xa0U, 0x78U, 0x25U, 0x4bU,
    0xa2U, 0x5dU, 0x80U, 0x05U, 0x3fU, 0x21U, 0x70U, 0xf1U,
    0x63U, 0x77U, 0xafU, 0x42U, 0x20U, 0xe5U, 0xfdU, 0xbfU,
    0x81U, 0x18U, 0x26U, 0xc3U, 0xbeU, 0x35U, 0x88U, 0x2eU,
    0x93U, 0x55U, 0xfcU, 0x7aU, 0xc8U, 0xbaU, 0x32U, 0xe6U,
    0xc0U, 0x19U, 0x9eU, 0xa3U, 0x44U, 0x54U, 0x3bU, 0x0bU,
    0x8cU, 0xc7U, 0x6bU, 0x28U, 0xa7U, 0xbcU, 0x16U, 0xadU,
    0xdbU, 0x64U, 0x74U, 0x14U, 0x92U, 0x0cU, 0x48U, 0xb8U,
    0x9fU, 0xbdU, 0x43U, 0xc4U, 0x39U, 0x31U, 0xd3U, 0xf2U,
    0xd5U, 0x8bU, 0x6eU, 0xdaU, 0x01U, 0xb1U, 0x9cU, 0x49U,
    0xd8U, 0xacU, 0xf3U, 0xcfU, 0xcaU, 0xf4U, 0x47U, 0x10U,
    0x6fU, 0xf0U, 0x4aU, 0x5cU, 0x38U, 0x57U, 0x73U, 0x97U,
    0xcbU, 0xa1U, 0xe8U, 0x3eU, 0x96U, 0x61U, 0x0dU, 0x0fU,
    0xe0U, 0x7cU, 0x71U, 0xccU, 0x90U, 0x06U, 0xf7U, 0x1cU,
    0xc2U, 0x6aU, 0xaeU, 0x69U, 0x17U, 0x99U, 0x3aU, 0x27U,
    0xd9U, 0xebU, 0x2bU, 0x22U, 0xd2U, 0xa9U, 0x07U, 0x33U,
    0x2dU, 0x3cU, 0x15U, 0xc9U, 0x87U, 0xaaU, 0x50U, 0xa5U,
    0x03U, 0x59U, 0x09U, 0x1aU, 0x65U, 0xd7U, 0x84U, 0xd0U,
    0x82U, 0x29U, 0x5aU, 0x1eU, 0x7bU, 0xa8U, 0x6dU, 0x2cU,

    0xa5U, 0x84U, 0x99U, 0x8dU, 0x0dU, 0xbdU, 0xb1U, 0x54U,
    0x50U, 0x03U, 0xa9U, 0x7dU, 0x19U, 0x62U, 0xe6U, 0x9aU,
    0x45U, 0x9dU, 0x40U, 0x87U, 0x15U, 0xebU, 0xc9U, 0x0bU,
    0xecU, 0x67U, 0xfdU, 0xeaU, 0xbfU, 0xf7U, 0x96U, 0x5bU,
    0xc2U, 0x1cU, 0xaeU, 0x6aU, 0x5aU, 0x41U, 0x02U, 0x4fU,
    0x5cU, 0xf4U, 0x34U, 0x08U, 0x93U, 0x73U, 0x53U, 0x3fU,
    0x0cU, 0x52U, 0x65U, 0x5eU, 0x28U, 0xa1U, 0x0fU, 0xb5U,
    0x09U, 0x36U, 0x9bU, 0x3dU, 0x26U, 0x69U, 0xcdU, 0x9fU,
    0x1bU, 0x9eU, 0x74U, 0x2eU, 0x2dU, 0xb2U, 0xeeU, 0xfbU,
    0xf6U, 0x4dU, 0x61U, 0xceU, 0x7bU, 0x3eU, 0x71U, 0x97U,
    0xf5U, 0x68U, 0x00U, 0x2cU, 0x60U, 0x1fU, 0xc8U, 0xedU,
    0xbeU, 0x46U, 0xd9U, 0x4bU, 0xdeU, 0xd4U, 0xe8U, 0x4aU,
    0x6bU, 0x2aU, 0xe5U, 0x16U, 0xc5U, 0xd7U, 0x55U, 0x94U,
    0xcfU, 0x10U, 0x06U, 0x81U, 0xf0U, 0x44U, 0xbaU, 0xe3U,
    0xf3U, 0xfeU, 0xc0U, 0x8aU, 0xadU, 0xbcU, 0x48U, 0x04U,
    0xdfU, 0xc1U, 0x75U, 0x63U, 0x30U, 0x1aU, 0x0eU, 0x6dU,
    0x4cU, 0x14U, 0x35U, 0x2fU, 0xe1U, 0xa2U, 0xccU, 0x39U,
    0x57U, 0xf2U, 0x82U, 0x47U, 0xacU, 0xe7U, 0x2bU, 0x95U,
    0xa0U, 0x98U, 0xd1U, 0x7fU, 0x66U, 0x7eU, 0xabU, 0x83U,
    0xcaU, 0x29U, 0xd3U, 0x3cU, 0x79U, 0xe2U, 0x1dU, 0x76U,
    0x3bU, 0x56U, 0x4eU, 0x1eU, 0xdbU, 0x0aU, 0x6cU, 0xe4U,
    0x5dU, 0x6eU, 0xefU, 0xa6U, 0xa8U, 0xa4U, 0x37U, 0x8bU,
    0x32U, 0x43U, 0x59U, 0xb7U, 0x8cU, 0x64U, 0xd2U, 0xe0U,
    0xb4U, 0xfaU, 0x07U, 0x25U, 0xafU, 0x8eU, 0xe9U, 0x18U,
    0xd5U, 0x88U, 0x6fU, 0x72U, 0x24U, 0xf1U, 0xc7U, 0x51U,
    0x23U, 0x7cU, 0x9cU, 0x21U, 0xddU, 0xdcU, 0x86U, 0x85U,
    0x90U, 0x42U, 0xc4U, 0xaaU, 0xd8U, 0x05U, 0x01U, 0x12U,
    0xa3U, 0x5fU, 0xf9U, 0xd0U, 0x91U, 0x58U, 0x27U, 0xb9U,
    0x38U, 0x13U, 0xb3U, 0x33U, 0xbbU, 0x70U, 0x89U, 0xa7U,
    0xb6U, 0x22U, 0x92U, 0x20U, 0x49U, 0xffU, 0x78U, 0x7aU,
    0x8fU, 0xf8U, 0x80U, 0x17U, 0xdaU, 0x31U, 0xc6U, 0xb8U,
    0xc3U, 0xb0U, 0x77U, 0x11U, 0xcbU, 0xfcU, 0xd6U, 0x3aU};

static const uint8_t Te2_cube[1024] = {
    0x63U, 0x7cU, 0x77U, 0x7bU, 0xf2U, 0x6bU, 0x6fU, 0xc5U,
    0x30U, 0x01U, 0x67U, 0x2bU, 0xfeU, 0xd7U, 0xabU, 0x76U,
    0xcaU, 0x82U, 0xc9U, 0x7dU, 0xfaU, 0x59U, 0x47U, 0xf0U,
    0xadU, 0xd4U, 0xa2U, 0xafU, 0x9cU, 0xa4U, 0x72U, 0xc0U,
    0xb7U, 0xfdU, 0x93U, 0x26U, 0x36U, 0x3fU, 0xf7U, 0xccU,
    0x34U, 0xa5U, 0xe5U, 0xf1U, 0x71U, 0xd8U, 0x31U, 0x15U,
    0x04U, 0xc7U, 0x23U, 0xc3U, 0x18U, 0x96U, 0x05U, 0x9aU,
    0x07U, 0x12U, 0x80U, 0xe2U, 0xebU, 0x27U, 0xb2U, 0x75U,
    0x09U, 0x83U, 0x2cU, 0x1aU, 0x1bU, 0x6eU, 0x5aU, 0xa0U,
    0x52U, 0x3bU, 0xd6U, 0xb3U, 0x29U, 0xe3U, 0x2fU, 0x84U,
    0x53U, 0xd1U, 0x00U, 0xedU, 0x20U, 0xfcU, 0xb1U, 0x5bU,
    0x6aU, 0xcbU, 0xbeU, 0x39U, 0x4aU, 0x4cU, 0x58U, 0xcfU,
    0xd0U, 0xefU, 0xaaU, 0xfbU, 0x43U, 0x4dU, 0x33U, 0x85U,
    0x45U, 0xf9U, 0x02U, 0x7fU, 0x50U, 0x3cU, 0x9fU, 0xa8U,
    0x51U, 0xa3U, 0x40U, 0x8fU, 0x92U, 0x9dU, 0x38U, 0xf5U,
    0xbcU, 0xb6U, 0xdaU, 0x21U, 0x10U, 0xffU, 0xf3U, 0xd2U,
    0xcdU, 0x0cU, 0x13U, 0xecU, 0x5fU, 0x97U, 0x44U, 0x17U,
    0xc4U, 0xa7U, 0x7eU, 0x3dU, 0x64U, 0x5dU, 0x19U, 0x73U,
    0x60U, 0x81U, 0x4fU, 0xdcU, 0x22U, 0x2aU, 0x90U, 0x88U,
    0x46U, 0xeeU, 0xb8U, 0x14U, 0xdeU, 0x5eU, 0x0bU, 0xdbU,
    0xe0U, 0x32U, 0x3aU, 0x0aU, 0x49U, 0x06U, 0x24U, 0x5cU,
    0xc2U, 0xd3U, 0xacU, 0x62U, 0x91U, 0x95U, 0xe4U, 0x79U,
    0xe7U, 0xc8U, 0x37U, 0x6dU, 0x8dU, 0xd5U, 0x4eU, 0xa9U,
    0x6cU, 0x56U, 0xf4U, 0xeaU, 0x65U, 0x7aU, 0xaeU, 0x08U,
    0xbaU, 0x78U, 0x25U, 0x2eU, 0x1cU, 0xa6U, 0xb4U, 0xc6U,
    0xe8U, 0xddU, 0x74U, 0x1fU, 0x4bU, 0xbdU, 0x8bU, 0x8aU,
    0x70U, 0x3eU, 0xb5U, 0x66U, 0x48U, 0x03U, 0xf6U, 0x0eU,
    0x61U, 0x35U, 0x57U, 0xb9U, 0x86U, 0xc1U, 0x1dU, 0x9eU,
    0xe1U, 0xf8U, 0x98U, 0x11U, 0x69U, 0xd9U, 0x8eU, 0x94U,
    0x9bU, 0x1eU, 0x87U, 0xe9U, 0xceU, 0x55U, 0x28U, 0xdfU,
    0x8cU, 0xa1U, 0x89U, 0x0dU, 0xbfU, 0xe6U, 0x42U, 0x68U,
    0x41U, 0x99U, 0x2dU, 0x0fU, 0xb0U, 0x54U, 0xbbU, 0x16U,

    0xc6U, 0xf8U, 0xeeU, 0xf6U, 0xffU, 0xd6U, 0xdeU, 0x91U,
    0x60U, 0x02U, 0xceU, 0x56U, 0xe7U, 0xb5U, 0x4dU, 0xecU,
    0x8fU, 0x1fU, 0x89U, 0xfaU, 0xefU, 0xb2U, 0x8eU, 0xfbU,
    0x41U, 0xb3U, 0x5fU, 0x45U, 0x23U, 0x53U, 0xe4U, 0x9bU,
    0x75U, 0xe1U, 0x3dU, 0x4cU, 0x6cU, 0x7eU, 0xf5U, 0x83U,
    0x68U, 0x51U, 0xd1U, 0xf9U, 0xe2U, 0xabU, 0x62U, 0x2aU,
    0x08U, 0x95U, 0x46U, 0x9dU, 0x30U, 0x37U, 0x0aU, 0x2fU,
    0x0eU, 0x24U, 0x1bU, 0xdfU, 0xcdU, 0x4eU, 0x7fU, 0xeaU,
    0x12U, 0x1dU, 0x58U, 0x34U, 0x36U, 0xdcU, 0xb4U, 0x5bU,
    0xa4U, 0x76U, 0xb7U, 0x7dU, 0x52U, 0xddU, 0x5eU, 0x13U,
    0xa6U, 0xb9U, 0x00U, 0xc1U, 0x40U, 0xe3U, 0x79U, 0xb6U,
    0xd4U, 0x8dU, 0x67U, 0x72U, 0x94U, 0x98U, 0xb0U, 0x85U,
    0xbbU, 0xc5U, 0x4fU, 0xedU, 0x86U, 0x9aU, 0x66U, 0x11U,
    0x8aU, 0xe9U, 0x04U, 0xfeU, 0xa0U, 0x78U, 0x25U, 0x4bU,
    0xa2U, 0x5dU, 0x80U, 0x05U, 0x3fU, 0x21U, 0x70U, 0xf1U,
    0x63U, 0x77U, 0xafU, 0x42U, 0x20U, 0xe5U, 0xfdU, 0xbfU,
    0x81U, 0x18U, 0x26U, 0xc3U, 0xbeU, 0x35U, 0x88U, 0x2eU,
    0x93U, 0x55U, 0xfcU, 0x7aU, 0xc8U, 0xbaU, 0x32U, 0xe6U,
    0xc0U, 0x19U, 0x9eU, 0xa3U, 0x44U, 0x54U, 0x3bU, 0x0bU,
    0x8cU, 0xc7U, 0x6bU, 0x28U, 0xa7U, 0xbcU, 0x16U, 0xadU,
    0xdbU, 0x64U, 0x74U, 0x14U, 0x92U, 0x0cU, 0x48U, 0xb8U,
    0x9fU, 0xbdU, 0x43U, 0xc4U, 0x39U, 0x31U, 0xd3U, 0xf2U,
    0xd5U, 0x8bU, 0x6eU, 0xdaU, 0x01U, 0xb1U, 0x9cU, 0x49U,
    0xd8U, 0xacU, 0xf3U, 0xcfU, 0xcaU, 0xf4U, 0x47U, 0x10U,
    0x6fU, 0xf0U, 0x4aU, 0x5cU, 0x38U, 0x57U, 0x73U, 0x97U,
    0xcbU, 0xa1U, 0xe8U, 0x3eU, 0x96U, 0x61U, 0x0dU, 0x0fU,
    0xe0U, 0x7cU, 0x71U, 0xccU, 0x90U, 0x06U, 0xf7U, 0x1cU,
    0xc2U, 0x6aU, 0xaeU, 0x69U, 0x17U, 0x99U, 0x3aU, 0x27U,
    0xd9U, 0xebU, 0x2bU, 0x22U, 0xd2U, 0xa9U, 0x07U, 0x33U,
    0x2dU, 0x3cU, 0x15U, 0xc9U, 0x87U, 0xaaU, 0x50U, 0xa5U,
    0x03U, 0x59U, 0x09U, 0x1aU, 0x65U, 0xd7U, 0x84U, 0xd0U,
    0x82U, 0x29U, 0x5aU, 0x1eU, 0x7bU, 0xa8U, 0x6dU, 0x2cU,

    0xa5U, 0x84U, 0x99U, 0x8dU, 0x0dU, 0xbdU, 0xb1U, 0x54U,
    0x50U, 0x03U, 0xa9U, 0x7dU, 0x19U, 0x62U, 0xe6U, 0x9aU,
    0x45U, 0x9dU, 0x40U, 0x87U, 0x15U, 0xebU, 0xc9U, 0x0bU,
    0xecU, 0x67U, 0xfdU, 0xeaU, 0xbfU, 0xf7U, 0x96U, 0x5bU,
    0xc2U, 0x1cU, 0xaeU, 0x6aU, 0x5aU, 0x41U, 0x02U, 0x4fU,
    0x5cU, 0xf4U, 0x34U, 0x08U, 0x93U, 0x73U, 0x53U, 0x3fU,
    0x0cU, 0x52U, 0x65U, 0x5eU, 0x28U, 0xa1U, 0x0fU, 0xb5U,
    0x09U, 0x36U, 0x9bU, 0x3dU, 0x26U, 0x69U, 0xcdU, 0x9fU,
    0x1bU, 0x9eU, 0x74U, 0x2eU, 0x2dU, 0xb2U, 0xeeU, 0xfbU,
    0xf6U, 0x4dU, 0x61U, 0xceU, 0x7bU, 0x3eU, 0x71U, 0x97U,
    0xf5U, 0x68U, 0x00U, 0x2cU, 0x60U, 0x1fU, 0xc8U, 0xedU,
    0xbeU, 0x46U, 0xd9U, 0x4bU, 0xdeU, 0xd4U, 0xe8U, 0x4aU,
    0x6bU, 0x2aU, 0xe5U, 0x16U, 0xc5U, 0xd7U, 0x55U, 0x94U,
    0xcfU, 0x10U, 0x06U, 0x81U, 0xf0U, 0x44U, 0xbaU, 0xe3U,
    0xf3U, 0xfeU, 0xc0U, 0x8aU, 0xadU, 0xbcU, 0x48U, 0x04U,
    0xdfU, 0xc1U, 0x75U, 0x63U, 0x30U, 0x1aU, 0x0eU, 0x6dU,
    0x4cU, 0x14U, 0x35U, 0x2fU, 0xe1U, 0xa2U, 0xccU, 0x39U,
    0x57U, 0xf2U, 0x82U, 0x47U, 0xacU, 0xe7U, 0x2bU, 0x95U,
    0xa0U, 0x98U, 0xd1U, 0x7fU, 0x66U, 0x7eU, 0xabU, 0x83U,
    0xcaU, 0x29U, 0xd3U, 0x3cU, 0x79U, 0xe2U, 0x1dU, 0x76U,
    0x3bU, 0x56U, 0x4eU, 0x1eU, 0xdbU, 0x0aU, 0x6cU, 0xe4U,
    0x5dU, 0x6eU, 0xefU, 0xa6U, 0xa8U, 0xa4U, 0x37U, 0x8bU,
    0x32U, 0x43U, 0x59U, 0xb7U, 0x8cU, 0x64U, 0xd2U, 0xe0U,
    0xb4U, 0xfaU, 0x07U, 0x25U, 0xafU, 0x8eU, 0xe9U, 0x18U,
    0xd5U, 0x88U, 0x6fU, 0x72U, 0x24U, 0xf1U, 0xc7U, 0x51U,
    0x23U, 0x7cU, 0x9cU, 0x21U, 0xddU, 0xdcU, 0x86U, 0x85U,
    0x90U, 0x42U, 0xc4U, 0xaaU, 0xd8U, 0x05U, 0x01U, 0x12U,
    0xa3U, 0x5fU, 0xf9U, 0xd0U, 0x91U, 0x58U, 0x27U, 0xb9U,
    0x38U, 0x13U, 0xb3U, 0x33U, 0xbbU, 0x70U, 0x89U, 0xa7U,
    0xb6U, 0x22U, 0x92U, 0x20U, 0x49U, 0xffU, 0x78U, 0x7aU,
    0x8fU, 0xf8U, 0x80U, 0x17U, 0xdaU, 0x31U, 0xc6U, 0xb8U,
    0xc3U, 0xb0U, 0x77U, 0x11U, 0xcbU, 0xfcU, 0xd6U, 0x3aU,

    0x63U, 0x7cU, 0x77U, 0x7bU, 0xf2U, 0x6bU, 0x6fU, 0xc5U,
    0x30U, 0x01U, 0x67U, 0x2bU, 0xfeU, 0xd7U, 0xabU, 0x76U,
    0xcaU, 0x82U, 0xc9U, 0x7dU, 0xfaU, 0x59U, 0x47U, 0xf0U,
    0xadU, 0xd4U, 0xa2U, 0xafU, 0x9cU, 0xa4U, 0x72U, 0xc0U,
    0xb7U, 0xfdU, 0x93U, 0x26U, 0x36U, 0x3fU, 0xf7U, 0xccU,
    0x34U, 0xa5U, 0xe5U, 0xf1U, 0x71U, 0xd8U, 0x31U, 0x15U,
    0x04U, 0xc7U, 0x23U, 0xc3U, 0x18U, 0x96U, 0x05U, 0x9aU,
    0x07U, 0x12U, 0x80U, 0xe2U, 0xebU, 0x27U, 0xb2U, 0x75U,
    0x09U, 0x83U, 0x2cU, 0x1aU, 0x1bU, 0x6eU, 0x5aU, 0xa0U,
    0x52U, 0x3bU, 0xd6U, 0xb3U, 0x29U, 0xe3U, 0x2fU, 0x84U,
    0x53U, 0xd1U, 0x00U, 0xedU, 0x20U, 0xfcU, 0xb1U, 0x5bU,
    0x6aU, 0xcbU, 0xbeU, 0x39U, 0x4aU, 0x4cU, 0x58U, 0xcfU,
    0xd0U, 0xefU, 0xaaU, 0xfbU, 0x43U, 0x4dU, 0x33U, 0x85U,
    0x45U, 0xf9U, 0x02U, 0x7fU, 0x50U, 0x3cU, 0x9fU, 0xa8U,
    0x51U, 0xa3U, 0x40U, 0x8fU, 0x92U, 0x9dU, 0x38U, 0xf5U,
    0xbcU, 0xb6U, 0xdaU, 0x21U, 0x10U, 0xffU, 0xf3U, 0xd2U,
    0xcdU, 0x0cU, 0x13U, 0xecU, 0x5fU, 0x97U, 0x44U, 0x17U,
    0xc4U, 0xa7U, 0x7eU, 0x3dU, 0x64U, 0x5dU, 0x19U, 0x73U,
    0x60U, 0x81U, 0x4fU, 0xdcU, 0x22U, 0x2aU, 0x90U, 0x88U,
    0x46U, 0xeeU, 0xb8U, 0x14U, 0xdeU, 0x5eU, 0x0bU, 0xdbU,
    0xe0U, 0x32U, 0x3aU, 0x0aU, 0x49U, 0x06U, 0x24U, 0x5cU,
    0xc2U, 0xd3U, 0xacU, 0x62U, 0x91U, 0x95U, 0xe4U, 0x79U,
    0xe7U, 0xc8U, 0x37U, 0x6dU, 0x8dU, 0xd5U, 0x4eU, 0xa9U,
    0x6cU, 0x56U, 0xf4U, 0xeaU, 0x65U, 0x7aU, 0xaeU, 0x08U,
    0xbaU, 0x78U, 0x25U, 0x2eU, 0x1cU, 0xa6U, 0xb4U, 0xc6U,
    0xe8U, 0xddU, 0x74U, 0x1fU, 0x4bU, 0xbdU, 0x8bU, 0x8aU,
    0x70U, 0x3eU, 0xb5U, 0x66U, 0x48U, 0x03U, 0xf6U, 0x0eU,
    0x61U, 0x35U, 0x57U, 0xb9U, 0x86U, 0xc1U, 0x1dU, 0x9eU,
    0xe1U, 0xf8U, 0x98U, 0x11U, 0x69U, 0xd9U, 0x8eU, 0x94U,
    0x9bU, 0x1eU, 0x87U, 0xe9U, 0xceU, 0x55U, 0x28U, 0xdfU,
    0x8cU, 0xa1U, 0x89U, 0x0dU, 0xbfU, 0xe6U, 0x42U, 0x68U,
    0x41U, 0x99U, 0x2dU, 0x0fU, 0xb0U, 0x54U, 0xbbU, 0x16U};

static const uint8_t Te3_cube[1024] = {
    0xc6U, 0xf8U, 0xeeU, 0xf6U, 0xffU, 0xd6U, 0xdeU, 0x91U,
    0x60U, 0x02U, 0xceU, 0x56U, 0xe7U, 0xb5U, 0x4dU, 0xecU,
    0x8fU, 0x1fU, 0x89U, 0xfaU, 0xefU, 0xb2U, 0x8eU, 0xfbU,
    0x41U, 0xb3U, 0x5fU, 0x45U, 0x23U, 0x53U, 0xe4U, 0x9bU,
    0x75U, 0xe1U, 0x3dU, 0x4cU, 0x6cU, 0x7eU, 0xf5U, 0x83U,
    0x68U, 0x51U, 0xd1U, 0xf9U, 0xe2U, 0xabU, 0x62U, 0x2aU,
    0x08U, 0x95U, 0x46U, 0x9dU, 0x30U, 0x37U, 0x0aU, 0x2fU,
    0x0eU, 0x24U, 0x1bU, 0xdfU, 0xcdU, 0x4eU, 0x7fU, 0xeaU,
    0x12U, 0x1dU, 0x58U, 0x34U, 0x36U, 0xdcU, 0xb4U, 0x5bU,
    0xa4U, 0x76U, 0xb7U, 0x7dU, 0x52U, 0xddU, 0x5eU, 0x13U,
    0xa6U, 0xb9U, 0x00U, 0xc1U, 0x40U, 0xe3U, 0x79U, 0xb6U,
    0xd4U, 0x8dU, 0x67U, 0x72U, 0x94U, 0x98U, 0xb0U, 0x85U,
    0xbbU, 0xc5U, 0x4fU, 0xedU, 0x86U, 0x9aU, 0x66U, 0x11U,
    0x8aU, 0xe9U, 0x04U, 0xfeU, 0xa0U, 0x78U, 0x25U, 0x4bU,
    0xa2U, 0x5dU, 0x80U, 0x05U, 0x3fU, 0x21U, 0x70U, 0xf1U,
    0x63U, 0x77U, 0xafU, 0x42U, 0x20U, 0xe5U, 0xfdU, 0xbfU,
    0x81U, 0x18U, 0x26U, 0xc3U, 0xbeU, 0x35U, 0x88U, 0x2eU,
    0x93U, 0x55U, 0xfcU, 0x7aU, 0xc8U, 0xbaU, 0x32U, 0xe6U,
    0xc0U, 0x19U, 0x9eU, 0xa3U, 0x44U, 0x54U, 0x3bU, 0x0bU,
    0x8cU, 0xc7U, 0x6bU, 0x28U, 0xa7U, 0xbcU, 0x16U, 0xadU,
    0xdbU, 0x64U, 0x74U, 0x14U, 0x92U, 0x0cU, 0x48U, 0xb8U,
    0x9fU, 0xbdU, 0x43U, 0xc4U, 0x39U, 0x31U, 0xd3U, 0xf2U,
    0xd5U, 0x8bU, 0x6eU, 0xdaU, 0x01U, 0xb1U, 0x9cU, 0x49U,
    0xd8U, 0xacU, 0xf3U, 0xcfU, 0xcaU, 0xf4U, 0x47U, 0x10U,
    0x6fU, 0xf0U, 0x4aU, 0x5cU, 0x38U, 0x57U, 0x73U, 0x97U,
    0xcbU, 0xa1U, 0xe8U, 0x3eU, 0x96U, 0x61U, 0x0dU, 0x0fU,
    0xe0U, 0x7cU, 0x71U, 0xccU, 0x90U, 0x06U, 0xf7U, 0x1cU,
    0xc2U, 0x6aU, 0xaeU, 0x69U, 0x17U, 0x99U, 0x3aU, 0x27U,
    0xd9U, 0xebU, 0x2bU, 0x22U, 0xd2U, 0xa9U, 0x07U, 0x33U,
    0x2dU, 0x3cU, 0x15U, 0xc9U, 0x87U, 0xaaU, 0x50U, 0xa5U,
    0x03U, 0x59U, 0x09U, 0x1aU, 0x65U, 0xd7U, 0x84U, 0xd0U,
    0x82U, 0x29U, 0x5aU, 0x1eU, 0x7bU, 0xa8U, 0x6dU, 0x2cU,

    0xa5U, 0x84U, 0x99U, 0x8dU, 0x0dU, 0xbdU, 0xb1U, 0x54U,
    0x50U, 0x03U, 0xa9U, 0x7dU, 0x19U, 0x62U, 0xe6U, 0x9aU,
    0x45U, 0x9dU, 0x40U, 0x87U, 0x15U, 0xebU, 0xc9U, 0x0bU,
    0xecU, 0x67U, 0xfdU, 0xeaU, 0xbfU, 0xf7U, 0x96U, 0x5bU,
    0xc2U, 0x1cU, 0xaeU, 0x6aU, 0x5aU, 0x41U, 0x02U, 0x4fU,
    0x5cU, 0xf4U, 0x34U, 0x08U, 0x93U, 0x73U, 0x53U, 0x3fU,
    0x0cU, 0x52U, 0x65U, 0x5eU, 0x28U, 0xa1U, 0x0fU, 0xb5U,
    0x09U, 0x36U, 0x9bU, 0x3dU, 0x26U, 0x69U, 0xcdU, 0x9fU,
    0x1bU, 0x9eU, 0x74U, 0x2eU, 0x2dU, 0xb2U, 0xeeU, 0xfbU,
    0xf6U, 0x4dU, 0x61U, 0xceU, 0x7bU, 0x3eU, 0x71U, 0x97U,
    0xf5U, 0x68U, 0x00U, 0x2cU, 0x60U, 0x1fU, 0xc8U, 0xedU,
    0xbeU, 0x46U, 0xd9U, 0x4bU, 0xdeU, 0xd4U, 0xe8U, 0x4aU,
    0x6bU, 0x2aU, 0xe5U, 0x16U, 0xc5U, 0xd7U, 0x55U, 0x94U,
    0xcfU, 0x10U, 0x06U, 0x81U, 0xf0U, 0x44U, 0xbaU, 0xe3U,
    0xf3U, 0xfeU, 0xc0U, 0x8aU, 0xadU, 0xbcU, 0x48U, 0x04U,
    0xdfU, 0xc1U, 0x75U, 0x63U, 0x30U, 0x1aU, 0x0eU, 0x6dU,
    0x4cU, 0x14U, 0x35U, 0x2fU, 0xe1U, 0xa2U, 0xccU, 0x39U,
    0x57U, 0xf2U, 0x82U, 0x47U, 0xacU, 0xe7U, 0x2bU, 0x95U,
    0xa0U, 0x98U, 0xd1U, 0x7fU, 0x66U, 0x7eU, 0xabU, 0x83U,
    0xcaU, 0x29U, 0xd3U, 0x3cU, 0x79U, 0xe2U, 0x1dU, 0x76U,
    0x3bU, 0x56U, 0x4eU, 0x1eU, 0xdbU, 0x0aU, 0x6cU, 0xe4U,
    0x5dU, 0x6eU, 0xefU, 0xa6U, 0xa8U, 0xa4U, 0x37U, 0x8bU,
    0x32U, 0x43U, 0x59U, 0xb7U, 0x8cU, 0x64U, 0xd2U, 0xe0U,
    0xb4U, 0xfaU, 0x07U, 0x25U, 0xafU, 0x8eU, 0xe9U, 0x18U,
    0xd5U, 0x88U, 0x6fU, 0x72U, 0x24U, 0xf1U, 0xc7U, 0x51U,
    0x23U, 0x7cU, 0x9cU, 0x21U, 0xddU, 0xdcU, 0x86U, 0x85U,
    0x90U, 0x42U, 0xc4U, 0xaaU, 0xd8U, 0x05U, 0x01U, 0x12U,
    0xa3U, 0x5fU, 0xf9U, 0xd0U, 0x91U, 0x58U, 0x27U, 0xb9U,
    0x38U, 0x13U, 0xb3U, 0x33U, 0xbbU, 0x70U, 0x89U, 0xa7U,
    0xb6U, 0x22U, 0x92U, 0x20U, 0x49U, 0xffU, 0x78U, 0x7aU,
    0x8fU, 0xf8U, 0x80U, 0x17U, 0xdaU, 0x31U, 0xc6U, 0xb8U,
    0xc3U, 0xb0U, 0x77U, 0x11U, 0xcbU, 0xfcU, 0xd6U, 0x3aU,

    0x63U, 0x7cU, 0x77U, 0x7bU, 0xf2U, 0x6bU, 0x6fU, 0xc5U,
    0x30U, 0x01U, 0x67U, 0x2bU, 0xfeU, 0xd7U, 0xabU, 0x76U,
    0xcaU, 0x82U, 0xc9U, 0x7dU, 0xfaU, 0x59U, 0x47U, 0xf0U,
    0xadU, 0xd4U, 0xa2U, 0xafU, 0x9cU, 0xa4U, 0x72U, 0xc0U,
    0xb7U, 0xfdU, 0x93U, 0x26U, 0x36U, 0x3fU, 0xf7U, 0xccU,
    0x34U, 0xa5U, 0xe5U, 0xf1U, 0x71U, 0xd8U, 0x31U, 0x15U,
    0x04U, 0xc7U, 0x23U, 0xc3U, 0x18U, 0x96U, 0x05U, 0x9aU,
    0x07U, 0x12U, 0x80U, 0xe2U, 0xebU, 0x27U, 0xb2U, 0x75U,
    0x09U, 0x83U, 0x2cU, 0x1aU, 0x1bU, 0x6eU, 0x5aU, 0xa0U,
    0x52U, 0x3bU, 0xd6U, 0xb3U, 0x29U, 0xe3U, 0x2fU, 0x84U,
    0x53U, 0xd1U, 0x00U, 0xedU, 0x20U, 0xfcU, 0xb1U, 0x5bU,
    0x6aU, 0xcbU, 0xbeU, 0x39U, 0x4aU, 0x4cU, 0x58U, 0xcfU,
    0xd0U, 0xefU, 0xaaU, 0xfbU, 0x43U, 0x4dU, 0x33U, 0x85U,
    0x45U, 0xf9U, 0x02U, 0x7fU, 0x50U, 0x3cU, 0x9fU, 0xa8U,
    0x51U, 0xa3U, 0x40U, 0x8fU, 0x92U, 0x9dU, 0x38U, 0xf5U,
    0xbcU, 0xb6U, 0xdaU, 0x21U, 0x10U, 0xffU, 0xf3U, 0xd2U,
    0xcdU, 0x0cU, 0x13U, 0xecU, 0x5fU, 0x97U, 0x44U, 0x17U,
    0xc4U, 0xa7U, 0x7eU, 0x3dU, 0x64U, 0x5dU, 0x19U, 0x73U,
    0x60U, 0x81U, 0x4fU, 0xdcU, 0x22U, 0x2aU, 0x90U, 0x88U,
    0x46U, 0xeeU, 0xb8U, 0x14U, 0xdeU, 0x5eU, 0x0bU, 0xdbU,
    0xe0U, 0x32U, 0x3aU, 0x0aU, 0x49U, 0x06U, 0x24U, 0x5cU,
    0xc2U, 0xd3U, 0xacU, 0x62U, 0x91U, 0x95U, 0xe4U, 0x79U,
    0xe7U, 0xc8U, 0x37U, 0x6dU, 0x8dU, 0xd5U, 0x4eU, 0xa9U,
    0x6cU, 0x56U, 0xf4U, 0xeaU, 0x65U, 0x7aU, 0xaeU, 0x08U,
    0xbaU, 0x78U, 0x25U, 0x2eU, 0x1cU, 0xa6U, 0xb4U, 0xc6U,
    0xe8U, 0xddU, 0x74U, 0x1fU, 0x4bU, 0xbdU, 0x8bU, 0x8aU,
    0x70U, 0x3eU, 0xb5U, 0x66U, 0x48U, 0x03U, 0xf6U, 0x0eU,
    0x61U, 0x35U, 0x57U, 0xb9U, 0x86U, 0xc1U, 0x1dU, 0x9eU,
    0xe1U, 0xf8U, 0x98U, 0x11U, 0x69U, 0xd9U, 0x8eU, 0x94U,
    0x9bU, 0x1eU, 0x87U, 0xe9U, 0xceU, 0x55U, 0x28U, 0xdfU,
    0x8cU, 0xa1U, 0x89U, 0x0dU, 0xbfU, 0xe6U, 0x42U, 0x68U,
    0x41U, 0x99U, 0x2dU, 0x0fU, 0xb0U, 0x54U, 0xbbU, 0x16U,

    0x63U, 0x7cU, 0x77U, 0x7bU, 0xf2U, 0x6bU, 0x6fU, 0xc5U,
    0x30U, 0x01U, 0x67U, 0x2bU, 0xfeU, 0xd7U, 0xabU, 0x76U,
    0xcaU, 0x82U, 0xc9U, 0x7dU, 0xfaU, 0x59U, 0x47U, 0xf0U,
    0xadU, 0xd4U, 0xa2U, 0xafU, 0x9cU, 0xa4U, 0x72U, 0xc0U,
    0xb7U, 0xfdU, 0x93U, 0x26U, 0x36U, 0x3fU, 0xf7U, 0xccU,
    0x34U, 0xa5U, 0xe5U, 0xf1U, 0x71U, 0xd8U, 0x31U, 0x15U,
    0x04U, 0xc7U, 0x23U, 0xc3U, 0x18U, 0x96U, 0x05U, 0x9aU,
    0x07U, 0x12U, 0x80U, 0xe2U, 0xebU, 0x27U, 0xb2U, 0x75U,
    0x09U, 0x83U, 0x2cU, 0x1aU, 0x1bU, 0x6eU, 0x5aU, 0xa0U,
    0x52U, 0x3bU, 0xd6U, 0xb3U, 0x29U, 0xe3U, 0x2fU, 0x84U,
    0x53U, 0xd1U, 0x00U, 0xedU, 0x20U, 0xfcU, 0xb1U, 0x5bU,
    0x6aU, 0xcbU, 0xbeU, 0x39U, 0x4aU, 0x4cU, 0x58U, 0xcfU,
    0xd0U, 0xefU, 0xaaU, 0xfbU, 0x43U, 0x4dU, 0x33U, 0x85U,
    0x45U, 0xf9U, 0x02U, 0x7fU, 0x50U, 0x3cU, 0x9fU, 0xa8U,
    0x51U, 0xa3U, 0x40U, 0x8fU, 0x92U, 0x9dU, 0x38U, 0xf5U,
    0xbcU, 0xb6U, 0xdaU, 0x21U, 0x10U, 0xffU, 0xf3U, 0xd2U,
    0xcdU, 0x0cU, 0x13U, 0xecU, 0x5fU, 0x97U, 0x44U, 0x17U,
    0xc4U, 0xa7U, 0x7eU, 0x3dU, 0x64U, 0x5dU, 0x19U, 0x73U,
    0x60U, 0x81U, 0x4fU, 0xdcU, 0x22U, 0x2aU, 0x90U, 0x88U,
    0x46U, 0xeeU, 0xb8U, 0x14U, 0xdeU, 0x5eU, 0x0bU, 0xdbU,
    0xe0U, 0x32U, 0x3aU, 0x0aU, 0x49U, 0x06U, 0x24U, 0x5cU,
    0xc2U, 0xd3U, 0xacU, 0x62U, 0x91U, 0x95U, 0xe4U, 0x79U,
    0xe7U, 0xc8U, 0x37U, 0x6dU, 0x8dU, 0xd5U, 0x4eU, 0xa9U,
    0x6cU, 0x56U, 0xf4U, 0xeaU, 0x65U, 0x7aU, 0xaeU, 0x08U,
    0xbaU, 0x78U, 0x25U, 0x2eU, 0x1cU, 0xa6U, 0xb4U, 0xc6U,
    0xe8U, 0xddU, 0x74U, 0x1fU, 0x4bU, 0xbdU, 0x8bU, 0x8aU,
    0x70U, 0x3eU, 0xb5U, 0x66U, 0x48U, 0x03U, 0xf6U, 0x0eU,
    0x61U, 0x35U, 0x57U, 0xb9U, 0x86U, 0xc1U, 0x1dU, 0x9eU,
    0xe1U, 0xf8U, 0x98U, 0x11U, 0x69U, 0xd9U, 0x8eU, 0x94U,
    0x9bU, 0x1eU, 0x87U, 0xe9U, 0xceU, 0x55U, 0x28U, 0xdfU,
    0x8cU, 0xa1U, 0x89U, 0x0dU, 0xbfU, 0xe6U, 0x42U, 0x68U,
    0x41U, 0x99U, 0x2dU, 0x0fU, 0xb0U, 0x54U, 0xbbU, 0x16U};

static const uint8_t AES_SBOX[256] = {
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16};


constexpr size_t AES_STREAM_BUFFER_SIZE = 1024ULL * 1024ULL * 1024ULL;
constexpr uint32_t AES_BLOCK_SIZE = 16;
constexpr size_t AES128_RK_BYTES = 176;
constexpr size_t AES128_RK_PAD_BYTES = 192;
constexpr uint32_t AES_CUBE_VEC_BLOCKS = 32;
constexpr size_t AES_WORKSPACE_B_BYTES_PER_CORE =
    AES_CUBE_VEC_BLOCKS * 4ULL * 256ULL;
constexpr size_t AES_WORKSPACE_C_BYTES_PER_CORE =
    16ULL * AES_CUBE_VEC_BLOCKS * 4ULL * sizeof(int32_t);
constexpr uint32_t AES_CUBE_KERNEL_BLOCKS = 1024;
// Keep the launch configuration identical to the AES-Vec demo.
constexpr uint32_t AES_VEC_KERNEL_BLOCKS = 1024;
constexpr int AES_CUBE_M = 16;
constexpr int AES_CUBE_N = AES_CUBE_VEC_BLOCKS * 4;
constexpr int AES_CUBE_K = 256;

static_assert(AES_STREAM_BUFFER_SIZE <= std::numeric_limits<uint32_t>::max(),
              "AES kernel dataSize is uint32_t");
static_assert(AES_STREAM_BUFFER_SIZE % AES_BLOCK_SIZE == 0,
              "AES keystream buffer must contain whole AES blocks");

inline size_t AlignUp32(size_t value)
{
    return (value + 31ULL) & ~31ULL;
}

void CheckAcl(aclError ret, const char *operation)
{
    if (ret != ACL_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed, aclError=" +
                                 std::to_string(static_cast<int>(ret)));
    }
}

void ReadRandomBytes(void *dst, size_t size)
{
    int fd = open("/dev/urandom", O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error("Failed to open /dev/urandom");
    }

    uint8_t *out = static_cast<uint8_t *>(dst);
    size_t done = 0;
    while (done < size) {
        ssize_t n = read(fd, out + done, size - done);
        if (n < 0 && errno == EINTR) {
            continue;
        }
        if (n <= 0) {
            close(fd);
            throw std::runtime_error("Failed to read /dev/urandom");
        }
        done += static_cast<size_t>(n);
    }
    close(fd);
}

void GenerateAesTiling(
    platform_ascendc::PlatformAscendC &platform,
    uint8_t *tilingBuf)
{
    matmul_tiling::MatmulApiTiling cubeTiling(platform);

    cubeTiling.SetAType(
        TPosition::GM, CubeFormat::ND, DataType::DT_INT8, false);
    cubeTiling.SetBType(
        TPosition::GM, CubeFormat::ND, DataType::DT_INT8, true);
    cubeTiling.SetCType(
        TPosition::GM, CubeFormat::ND, DataType::DT_INT32);
    cubeTiling.EnableBias(false);
    cubeTiling.SetShape(AES_CUBE_M, AES_CUBE_N, AES_CUBE_K);
    cubeTiling.SetOrgShape(AES_CUBE_M, AES_CUBE_N, AES_CUBE_K);
    cubeTiling.SetBufferSpace(-1, -1, 64 * 1024);

    optiling::TCubeTiling tilingData;
    if (cubeTiling.GetTiling(tilingData) == -1) {
        throw std::runtime_error("Failed to generate AES Matmul tiling");
    }

    const uint32_t tilingSize = tilingData.GetDataSize();
    tilingData.SaveToBuffer(tilingBuf, tilingSize);

    uint64_t localMemSize = 0;
    platform.GetCoreMemSize(
        platform_ascendc::CoreMemType::UB, localMemSize);
    std::memcpy(
        tilingBuf + tilingSize,
        &localMemSize,
        sizeof(localMemSize));
}

void ExpandAes128Key(const uint8_t key[16], uint8_t expanded[AES128_RK_BYTES])
{
    static constexpr uint8_t RCON[11] = {
        0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36};

    std::memcpy(expanded, key, 16);
    uint32_t generated = 16;
    uint8_t rconIndex = 1;
    uint8_t temp[4];

    while (generated < AES128_RK_BYTES) {
        for (int i = 0; i < 4; ++i) {
            temp[i] = expanded[generated - 4 + i];
        }

        if ((generated % 16) == 0) {
            uint8_t first = temp[0];
            temp[0] = AES_SBOX[temp[1]];
            temp[1] = AES_SBOX[temp[2]];
            temp[2] = AES_SBOX[temp[3]];
            temp[3] = AES_SBOX[first];
            temp[0] ^= RCON[rconIndex++];
        }

        for (int i = 0; i < 4; ++i) {
            expanded[generated] = expanded[generated - 16] ^ temp[i];
            ++generated;
        }
    }
}

void Aes128EncryptBlockCpu(const uint8_t input[16], uint8_t output[16],
                           const uint8_t roundKeys[AES128_RK_BYTES])
{
    auto addRoundKey = [](uint8_t state[16], const uint8_t *roundKey) {
        for (int i = 0; i < 16; ++i) {
            state[i] ^= roundKey[i];
        }
    };

    auto shiftRows = [](uint8_t state[16]) {
        uint8_t tmp[16];
        tmp[0] = state[0];   tmp[1] = state[5];   tmp[2] = state[10];  tmp[3] = state[15];
        tmp[4] = state[4];   tmp[5] = state[9];   tmp[6] = state[14];  tmp[7] = state[3];
        tmp[8] = state[8];   tmp[9] = state[13];  tmp[10] = state[2];  tmp[11] = state[7];
        tmp[12] = state[12]; tmp[13] = state[1];  tmp[14] = state[6];  tmp[15] = state[11];
        std::memcpy(state, tmp, sizeof(tmp));
    };

    auto xtime = [](uint8_t value) -> uint8_t {
        return static_cast<uint8_t>((value << 1) ^ ((value & 0x80U) ? 0x1bU : 0x00U));
    };

    auto mixColumns = [&](uint8_t state[16]) {
        for (int column = 0; column < 4; ++column) {
            uint8_t *c = state + column * 4;
            uint8_t a0 = c[0], a1 = c[1], a2 = c[2], a3 = c[3];
            c[0] = static_cast<uint8_t>(xtime(a0) ^ (xtime(a1) ^ a1) ^ a2 ^ a3);
            c[1] = static_cast<uint8_t>(a0 ^ xtime(a1) ^ (xtime(a2) ^ a2) ^ a3);
            c[2] = static_cast<uint8_t>(a0 ^ a1 ^ xtime(a2) ^ (xtime(a3) ^ a3));
            c[3] = static_cast<uint8_t>((xtime(a0) ^ a0) ^ a1 ^ a2 ^ xtime(a3));
        }
    };

    uint8_t state[16];
    std::memcpy(state, input, sizeof(state));
    addRoundKey(state, roundKeys);

    for (int round = 1; round <= 9; ++round) {
        for (uint8_t &value : state) {
            value = AES_SBOX[value];
        }
        shiftRows(state);
        mixColumns(state);
        addRoundKey(state, roundKeys + round * 16);
    }

    for (uint8_t &value : state) {
        value = AES_SBOX[value];
    }
    shiftRows(state);
    addRoundKey(state, roundKeys + 160);
    std::memcpy(output, state, sizeof(state));
}

// Generate exactly the same counter blocks as the AscendC AES kernels:
// [nonce1, nonce2, nonce3, block_index * 4], represented as host-endian
// uint32 words. AES-Cube passes the same nonce for all three nonce words.
void Aes128CtrGenerateMaskCpu(const uint8_t roundKeys[AES128_RK_BYTES],
                              uint32_t nonce1, uint32_t nonce2,
                              uint32_t nonce3, uint8_t *output,
                              size_t dataSize)
{
    const size_t totalBlocks = dataSize / AES_BLOCK_SIZE;
    for (size_t block = 0; block < totalBlocks; ++block) {
        uint32_t words[4] = {
            nonce1,
            nonce2,
            nonce3,
            static_cast<uint32_t>(block) * 4UL,
        };
        uint8_t counterBlock[16];
        std::memcpy(counterBlock, words, sizeof(counterBlock));
        Aes128EncryptBlockCpu(counterBlock, output + block * AES_BLOCK_SIZE, roundKeys);
    }
}

void XorKeyStreamCpu(const uint8_t *keyStream, const uint8_t *input,
                     uint8_t *output, size_t dataSize)
{
    for (size_t i = 0; i < dataSize; ++i) {
        output[i] = input[i] ^ keyStream[i];
    }
}

class Singleton {
public:
    static Singleton &getInstance()
    {
        static Singleton instance;
        return instance;
    }

    // A complete encrypt -> memcpy -> decrypt operation must be serialized.
    // Locking enc() and dec() separately allows another thread to consume the
    // opposite half of the keystream between them.
    std::unique_lock<std::mutex> acquireTransactionLock()
    {
        return std::unique_lock<std::mutex>(mutex_);
    }

    void invalidateKeyStream()
    {
        hostCurrentPos_ = streamBufferSize_;
        deviceCurrentPos_ = streamBufferSize_;
    }

    void enc(void *inputPtr, void *outputPtr, size_t dataSize, bool isH2d)
    {
        EnsureStream();
        CheckRequestSize(dataSize);

        size_t &producerPosition =
            isH2d ? hostCurrentPos_ : deviceCurrentPos_;
        if (producerPosition + dataSize > streamBufferSize_) {
            GenerateKeyStream();
        }

        if (isH2d) {
            const auto *keyStream =
                static_cast<const uint8_t *>(hostStreamBuffer_) +
                hostCurrentPos_;
            XorKeyStreamCpu(keyStream,
                            static_cast<const uint8_t *>(inputPtr),
                            static_cast<uint8_t *>(outputPtr), dataSize);
            hostCurrentPos_ += AlignUp32(dataSize);
        } else {
            void *keyStream =
                static_cast<uint8_t *>(deviceStreamBuffer_) +
                deviceCurrentPos_;
            // D2H flow: the encrypted temporary device buffer is consumed by
            // the following synchronous D2H memcpy, so the device-side XOR
            // must finish before that copy starts. This matches the original
            // integration flow.
            XorKeyStreamDevice(keyStream, inputPtr, outputPtr, dataSize,
                               true);
            deviceCurrentPos_ += AlignUp32(dataSize);
        }
    }

    void dec(void *inputPtr, void *outputPtr, size_t dataSize, bool isH2d)
    {
        EnsureStream();
        CheckRequestSize(dataSize);

        if (!isH2d) {
            const auto *keyStream =
                static_cast<const uint8_t *>(hostStreamBuffer_) +
                hostCurrentPos_;
            XorKeyStreamCpu(keyStream,
                            static_cast<const uint8_t *>(inputPtr),
                            static_cast<uint8_t *>(outputPtr), dataSize);
            hostCurrentPos_ += AlignUp32(dataSize);
        } else {
            void *keyStream =
                static_cast<uint8_t *>(deviceStreamBuffer_) +
                deviceCurrentPos_;
            // H2D flow: the synchronous H2D memcpy has already completed.
            // Submit decryption to encStream_ and let subsequent work on the
            // same stream preserve ordering, as in the original integration.
            XorKeyStreamDevice(keyStream, inputPtr, outputPtr, dataSize,
                               false);
            deviceCurrentPos_ += AlignUp32(dataSize);
        }

        if (deviceCurrentPos_ != hostCurrentPos_) {
            throw std::runtime_error(
                "H2D/D2H keystream positions diverged: host=" +
                std::to_string(hostCurrentPos_) + ", device=" +
                std::to_string(deviceCurrentPos_));
        }
    }

private:
    static constexpr size_t CHACHA_STREAM_BUFFER_SIZE =
        1024ULL * 1024ULL * 1024ULL;
    static constexpr size_t CHACHA_STATE_BYTES = 64;

    Singleton()
        : algorithm_(GetTorchEncryptionAlgorithm()),
          streamBufferSize_((algorithm_ == TorchEncryptionAlgorithm::AES_CUBE ||
                             algorithm_ == TorchEncryptionAlgorithm::AES_VEC)
                                ? AES_STREAM_BUFFER_SIZE
                                : CHACHA_STREAM_BUFFER_SIZE)
    {
        if (algorithm_ == TorchEncryptionAlgorithm::DISABLED) {
            throw std::logic_error(
                "Encryption manager was created while encryption is disabled");
        }

        try {
            AllocateCommonBuffers();
            switch (algorithm_) {
                case TorchEncryptionAlgorithm::CHACHA20_NAIVE:
                    AllocateChachaResources();
                    break;
                case TorchEncryptionAlgorithm::AES_CUBE:
                    AllocateAesRoundKeyResources();
                    AllocateAesCubeResources();
                    InitializeAesCubeResources();
                    break;
                case TorchEncryptionAlgorithm::AES_VEC:
                    AllocateAesRoundKeyResources();
                    break;
                case TorchEncryptionAlgorithm::DISABLED:
                    throw std::logic_error("Unexpected disabled algorithm");
            }

            hostCurrentPos_ = streamBufferSize_;
            deviceCurrentPos_ = streamBufferSize_;
        } catch (...) {
            ReleaseResources();
            throw;
        }
    }

    ~Singleton()
    {
        ReleaseResources();
    }

    Singleton(const Singleton &) = delete;
    Singleton &operator=(const Singleton &) = delete;

    void EnsureStream()
    {
        if (encStream_ == nullptr) {
            encStream_ = c10_npu::getCurrentNPUStream();
        }
    }

    void CheckRequestSize(size_t dataSize) const
    {
        if (dataSize > streamBufferSize_) {
            throw std::runtime_error(
                "Single memcpy size exceeds " +
                std::string(GetEncryptionAlgorithmName(algorithm_)) +
                " keystream buffer: size=" + std::to_string(dataSize) +
                ", capacity=" + std::to_string(streamBufferSize_));
        }
        if (dataSize > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error(
                "Encryption/XOR kernel dataSize exceeds uint32_t");
        }
    }

    void XorKeyStreamDevice(void *keyStream, void *inputPtr,
                            void *outputPtr, size_t dataSize,
                            bool synchronizeAfterXor)
    {
        xor_do(32, encStream_, keyStream, inputPtr, outputPtr,
               static_cast<uint32_t>(dataSize), 4096);

        if (synchronizeAfterXor) {
            CheckAcl(aclrtSynchronizeStream(encStream_),
                     "aclrtSynchronizeStream after device encryption");
        }
    }

    void GenerateKeyStream()
    {
        std::fprintf(stderr,
                     "[torch-npu encryption] generating memcpy keystream: "
                     "algorithm=%s, capacity=%zu bytes.\n",
                     GetEncryptionAlgorithmName(algorithm_),
                     streamBufferSize_);

        switch (algorithm_) {
            case TorchEncryptionAlgorithm::CHACHA20_NAIVE:
                GenerateChachaKeyStream();
                break;
            case TorchEncryptionAlgorithm::AES_CUBE:
                GenerateAesCubeKeyStream();
                break;
            case TorchEncryptionAlgorithm::AES_VEC:
                GenerateAesVecKeyStream();
                break;
            case TorchEncryptionAlgorithm::DISABLED:
                throw std::logic_error(
                    "Cannot generate a keystream while encryption is disabled");
        }

        hostCurrentPos_ = 0;
        deviceCurrentPos_ = 0;
    }

    static inline uint32_t RotateLeft(uint32_t value, int shift)
    {
        return (value << shift) | (value >> (32 - shift));
    }

    static inline void ChachaQuarterRound(uint32_t state[16], int a, int b,
                                         int c, int d)
    {
        state[a] += state[b];
        state[d] ^= state[a];
        state[d] = RotateLeft(state[d], 16);

        state[c] += state[d];
        state[b] ^= state[c];
        state[b] = RotateLeft(state[b], 12);

        state[a] += state[b];
        state[d] ^= state[a];
        state[d] = RotateLeft(state[d], 8);

        state[c] += state[d];
        state[b] ^= state[c];
        state[b] = RotateLeft(state[b], 7);
    }

    static void Chacha20GenerateMaskCpu(const uint32_t state[16],
                                        uint32_t *output,
                                        uint32_t dataSize)
    {
        constexpr size_t BLOCK_SIZE = 64;
        const size_t totalBlocks =
            (static_cast<size_t>(dataSize) + BLOCK_SIZE - 1) / BLOCK_SIZE;

        for (size_t block = 0; block < totalBlocks; ++block) {
            uint32_t initial[16];
            uint32_t working[16];
            std::memcpy(initial, state, sizeof(initial));
            initial[12] += static_cast<uint32_t>(block);
            std::memcpy(working, initial, sizeof(working));

            for (int round = 0; round < 10; ++round) {
                ChachaQuarterRound(working, 0, 4, 8, 12);
                ChachaQuarterRound(working, 1, 5, 9, 13);
                ChachaQuarterRound(working, 2, 6, 10, 14);
                ChachaQuarterRound(working, 3, 7, 11, 15);
                ChachaQuarterRound(working, 0, 5, 10, 15);
                ChachaQuarterRound(working, 1, 6, 11, 12);
                ChachaQuarterRound(working, 2, 7, 8, 13);
                ChachaQuarterRound(working, 3, 4, 9, 14);
            }

            for (int word = 0; word < 16; ++word) {
                output[block * 16 + word] = initial[word] + working[word];
            }
        }
    }

    void GenerateChachaKeyStream()
    {
        ReadRandomBytes(hostStateBuffer_, CHACHA_STATE_BYTES);
        CheckAcl(aclrtMemcpyAsync(deviceStateBuffer_, CHACHA_STATE_BYTES,
                                 hostStateBuffer_, CHACHA_STATE_BYTES,
                                 ACL_MEMCPY_HOST_TO_DEVICE, encStream_),
                 "aclrtMemcpyAsync ChaCha20 state");

        const uint32_t blockDim = static_cast<uint32_t>(
            streamBufferSize_ / 64 / 2048);
        chacha20_naive_generate_mask(
            blockDim, encStream_, deviceStateBuffer_, deviceStreamBuffer_,
            static_cast<uint32_t>(streamBufferSize_));

        Chacha20GenerateMaskCpu(
            static_cast<const uint32_t *>(hostStateBuffer_),
            static_cast<uint32_t *>(hostStreamBuffer_),
            static_cast<uint32_t>(streamBufferSize_));
    }

    void GenerateAesCubeKeyStream()
    {
        std::array<uint8_t, 20> randomMaterial{};
        ReadRandomBytes(randomMaterial.data(), randomMaterial.size());

        uint8_t key[16];
        std::memcpy(key, randomMaterial.data(), sizeof(key));
        uint32_t nonce = 0;
        std::memcpy(&nonce, randomMaterial.data() + sizeof(key),
                    sizeof(nonce));

        uint8_t expanded[AES128_RK_BYTES];
        ExpandAes128Key(key, expanded);

        std::memset(hostRoundKeys_, 0, AES128_RK_PAD_BYTES);
        std::memcpy(hostRoundKeys_, expanded, AES128_RK_BYTES);

        // Match the original ChaCha20 integration: enqueue the state/key copy
        // and device mask generation on encStream_, then generate the host mask
        // concurrently on the CPU.
        CheckAcl(aclrtMemcpyAsync(deviceRoundKeys_, AES128_RK_PAD_BYTES,
                                  hostRoundKeys_, AES128_RK_PAD_BYTES,
                                  ACL_MEMCPY_HOST_TO_DEVICE, encStream_),
                 "aclrtMemcpyAsync AES-Cube round keys");

        // The AES-Cube kernel constructs CTR input blocks internally and does
        // not read input. Reusing output as input avoids another large buffer.
        aes_cube_generate_mask(
            AES_CUBE_KERNEL_BLOCKS, encStream_,
            deviceRoundKeys_, deviceStreamBuffer_, deviceStreamBuffer_,
            deviceTe0_, deviceTe1_, deviceTe2_, deviceTe3_, deviceSbox_,
            deviceBWorkspace_, deviceCWorkspace_,
            deviceSystemWorkspace_, deviceTiling_, nonce,
            static_cast<uint32_t>(streamBufferSize_));

        Aes128CtrGenerateMaskCpu(
            expanded, nonce, nonce, nonce,
            static_cast<uint8_t *>(hostStreamBuffer_),
            streamBufferSize_);
    }

    void GenerateAesVecKeyStream()
    {
        // AES-Vec uses a 128-bit key and three independent 32-bit nonce words.
        std::array<uint8_t, 28> randomMaterial{};
        ReadRandomBytes(randomMaterial.data(), randomMaterial.size());

        uint8_t key[16];
        std::memcpy(key, randomMaterial.data(), sizeof(key));

        uint32_t nonce1 = 0;
        uint32_t nonce2 = 0;
        uint32_t nonce3 = 0;
        std::memcpy(&nonce1, randomMaterial.data() + sizeof(key),
                    sizeof(nonce1));
        std::memcpy(&nonce2,
                    randomMaterial.data() + sizeof(key) + sizeof(nonce1),
                    sizeof(nonce2));
        std::memcpy(&nonce3,
                    randomMaterial.data() + sizeof(key) + sizeof(nonce1) +
                        sizeof(nonce2),
                    sizeof(nonce3));

        uint8_t expanded[AES128_RK_BYTES];
        ExpandAes128Key(key, expanded);

        std::memset(hostRoundKeys_, 0, AES128_RK_PAD_BYTES);
        std::memcpy(hostRoundKeys_, expanded, AES128_RK_BYTES);

        CheckAcl(aclrtMemcpyAsync(deviceRoundKeys_, AES128_RK_PAD_BYTES,
                                  hostRoundKeys_, AES128_RK_PAD_BYTES,
                                  ACL_MEMCPY_HOST_TO_DEVICE, encStream_),
                 "aclrtMemcpyAsync AES-Vec round keys");

        // The AES-Vec kernel constructs CTR input blocks internally and does
        // not read input. Reusing output as input avoids another 1 GiB buffer.
        aes_vec_generate_mask(
            AES_VEC_KERNEL_BLOCKS, encStream_, deviceRoundKeys_,
            deviceStreamBuffer_, deviceStreamBuffer_, nonce1, nonce2, nonce3,
            static_cast<uint32_t>(streamBufferSize_));

        // Keep the original integration sequence: submit device keystream
        // generation asynchronously first, then generate the identical host
        // keystream while the device kernel is running.
        Aes128CtrGenerateMaskCpu(
            expanded, nonce1, nonce2, nonce3,
            static_cast<uint8_t *>(hostStreamBuffer_), streamBufferSize_);
    }

    void AllocateCommonBuffers()
    {
        CheckAcl(aclrtMalloc(&deviceStreamBuffer_, streamBufferSize_,
                             ACL_MEM_MALLOC_HUGE_FIRST),
                 "aclrtMalloc device keystream");
        hostStreamBuffer_ = std::malloc(streamBufferSize_);
        if (hostStreamBuffer_ == nullptr) {
            throw std::bad_alloc();
        }
    }

    void AllocateChachaResources()
    {
        CheckAcl(aclrtMalloc(&deviceStateBuffer_, CHACHA_STATE_BYTES,
                             ACL_MEM_MALLOC_HUGE_FIRST),
                 "aclrtMalloc ChaCha20 state");
        hostStateBuffer_ = std::malloc(CHACHA_STATE_BYTES);
        if (hostStateBuffer_ == nullptr) {
            throw std::bad_alloc();
        }
    }

    void AllocateAesRoundKeyResources()
    {
        CheckAcl(aclrtMalloc(&deviceRoundKeys_, AES128_RK_PAD_BYTES,
                             ACL_MEM_MALLOC_HUGE_FIRST),
                 "aclrtMalloc AES round keys");
        hostRoundKeys_ = std::malloc(AES128_RK_PAD_BYTES);
        if (hostRoundKeys_ == nullptr) {
            throw std::bad_alloc();
        }
    }

    void AllocateAesCubeResources()
    {
        CheckAcl(aclrtMalloc(&deviceTe0_, 4096,
                             ACL_MEM_MALLOC_HUGE_FIRST),
                 "aclrtMalloc AES-Cube Te0");
        CheckAcl(aclrtMalloc(&deviceTe1_, 1024,
                             ACL_MEM_MALLOC_HUGE_FIRST),
                 "aclrtMalloc AES-Cube Te1");
        CheckAcl(aclrtMalloc(&deviceTe2_, 1024,
                             ACL_MEM_MALLOC_HUGE_FIRST),
                 "aclrtMalloc AES-Cube Te2");
        CheckAcl(aclrtMalloc(&deviceTe3_, 1024,
                             ACL_MEM_MALLOC_HUGE_FIRST),
                 "aclrtMalloc AES-Cube Te3");
        CheckAcl(aclrtMalloc(&deviceSbox_, 256,
                             ACL_MEM_MALLOC_HUGE_FIRST),
                 "aclrtMalloc AES-Cube S-box");

        const size_t bWorkspaceBytes =
            AES_WORKSPACE_B_BYTES_PER_CORE * AES_CUBE_KERNEL_BLOCKS;
        const size_t cWorkspaceBytes =
            AES_WORKSPACE_C_BYTES_PER_CORE * AES_CUBE_KERNEL_BLOCKS;
        CheckAcl(aclrtMalloc(&deviceBWorkspace_, bWorkspaceBytes,
                             ACL_MEM_MALLOC_HUGE_FIRST),
                 "aclrtMalloc AES-Cube B workspace");
        CheckAcl(aclrtMalloc(&deviceCWorkspace_, cWorkspaceBytes,
                             ACL_MEM_MALLOC_HUGE_FIRST),
                 "aclrtMalloc AES-Cube C workspace");
    }

    void InitializeAesCubeResources()
    {
        auto *platform =
            platform_ascendc::PlatformAscendCManager::GetInstance();

        if (platform == nullptr) {
            throw std::runtime_error(
                "Failed to get current AscendC platform");
        }

        systemWorkspaceSize_ =
            static_cast<size_t>(platform->GetLibApiWorkSpaceSize());

        if (systemWorkspaceSize_ != 0) {
            CheckAcl(
                aclrtMalloc(
                    &deviceSystemWorkspace_,
                    systemWorkspaceSize_,
                    ACL_MEM_MALLOC_HUGE_FIRST),
                "aclrtMalloc AES-Cube system workspace");

            CheckAcl(
                aclrtMemset(
                    deviceSystemWorkspace_,
                    systemWorkspaceSize_,
                    0,
                    systemWorkspaceSize_),
                "aclrtMemset AES-Cube system workspace");
        }

        tilingSize_ = sizeof(TCubeTiling) + sizeof(uint64_t);
        std::vector<uint8_t> tilingHost(tilingSize_, 0);

        GenerateAesTiling(*platform, tilingHost.data());
        CheckAcl(aclrtMalloc(&deviceTiling_, tilingSize_,
                             ACL_MEM_MALLOC_HUGE_FIRST),
                 "aclrtMalloc AES-Cube tiling");
        CheckAcl(aclrtMemcpy(deviceTiling_, tilingSize_,
                             tilingHost.data(), tilingSize_,
                             ACL_MEMCPY_HOST_TO_DEVICE),
                 "aclrtMemcpy AES-Cube tiling");

        std::vector<int8_t> te0Cube(16 * 256, 0);
        for (int i = 0; i < 256; ++i) {
            te0Cube[0 * 256 + i] =
                static_cast<int8_t>(Te0_cube[0 * 256 + i]);
            te0Cube[1 * 256 + i] =
                static_cast<int8_t>(Te0_cube[1 * 256 + i]);
            te0Cube[2 * 256 + i] =
                static_cast<int8_t>(Te0_cube[2 * 256 + i]);
            te0Cube[3 * 256 + i] =
                static_cast<int8_t>(Te0_cube[3 * 256 + i]);
        }

        CheckAcl(aclrtMemcpy(deviceTe0_, 4096, te0Cube.data(), 4096,
                             ACL_MEMCPY_HOST_TO_DEVICE),
                 "aclrtMemcpy AES-Cube Te0");
        CheckAcl(aclrtMemcpy(deviceTe1_, 1024, Te1_cube, 1024,
                             ACL_MEMCPY_HOST_TO_DEVICE),
                 "aclrtMemcpy AES-Cube Te1");
        CheckAcl(aclrtMemcpy(deviceTe2_, 1024, Te2_cube, 1024,
                             ACL_MEMCPY_HOST_TO_DEVICE),
                 "aclrtMemcpy AES-Cube Te2");
        CheckAcl(aclrtMemcpy(deviceTe3_, 1024, Te3_cube, 1024,
                             ACL_MEMCPY_HOST_TO_DEVICE),
                 "aclrtMemcpy AES-Cube Te3");
        CheckAcl(aclrtMemcpy(deviceSbox_, 256, AES_SBOX, 256,
                             ACL_MEMCPY_HOST_TO_DEVICE),
                 "aclrtMemcpy AES-Cube S-box");
    }

    void ReleaseResources() noexcept
    {
        if (deviceTiling_ != nullptr) aclrtFree(deviceTiling_);
        if (deviceSystemWorkspace_ != nullptr)
            aclrtFree(deviceSystemWorkspace_);
        if (deviceCWorkspace_ != nullptr) aclrtFree(deviceCWorkspace_);
        if (deviceBWorkspace_ != nullptr) aclrtFree(deviceBWorkspace_);
        if (deviceSbox_ != nullptr) aclrtFree(deviceSbox_);
        if (deviceTe3_ != nullptr) aclrtFree(deviceTe3_);
        if (deviceTe2_ != nullptr) aclrtFree(deviceTe2_);
        if (deviceTe1_ != nullptr) aclrtFree(deviceTe1_);
        if (deviceTe0_ != nullptr) aclrtFree(deviceTe0_);
        if (deviceRoundKeys_ != nullptr) aclrtFree(deviceRoundKeys_);
        if (hostRoundKeys_ != nullptr) std::free(hostRoundKeys_);

        if (deviceStateBuffer_ != nullptr) aclrtFree(deviceStateBuffer_);
        if (hostStateBuffer_ != nullptr) std::free(hostStateBuffer_);

        if (deviceStreamBuffer_ != nullptr)
            aclrtFree(deviceStreamBuffer_);
        if (hostStreamBuffer_ != nullptr)
            std::free(hostStreamBuffer_);

        deviceTiling_ = nullptr;
        deviceSystemWorkspace_ = nullptr;
        deviceCWorkspace_ = nullptr;
        deviceBWorkspace_ = nullptr;
        deviceSbox_ = nullptr;
        deviceTe3_ = nullptr;
        deviceTe2_ = nullptr;
        deviceTe1_ = nullptr;
        deviceTe0_ = nullptr;
        deviceRoundKeys_ = nullptr;
        hostRoundKeys_ = nullptr;
        deviceStateBuffer_ = nullptr;
        hostStateBuffer_ = nullptr;
        deviceStreamBuffer_ = nullptr;
        hostStreamBuffer_ = nullptr;
    }

    TorchEncryptionAlgorithm algorithm_ =
        TorchEncryptionAlgorithm::DISABLED;
    size_t streamBufferSize_ = 0;
    size_t hostCurrentPos_ = 0;
    size_t deviceCurrentPos_ = 0;
    aclrtStream encStream_ = nullptr;

    void *hostStreamBuffer_ = nullptr;
    void *deviceStreamBuffer_ = nullptr;

    void *hostStateBuffer_ = nullptr;
    void *deviceStateBuffer_ = nullptr;

    void *hostRoundKeys_ = nullptr;
    void *deviceRoundKeys_ = nullptr;
    void *deviceTe0_ = nullptr;
    void *deviceTe1_ = nullptr;
    void *deviceTe2_ = nullptr;
    void *deviceTe3_ = nullptr;
    void *deviceSbox_ = nullptr;
    void *deviceBWorkspace_ = nullptr;
    void *deviceCWorkspace_ = nullptr;
    void *deviceSystemWorkspace_ = nullptr;
    void *deviceTiling_ = nullptr;
    size_t systemWorkspaceSize_ = 0;
    size_t tilingSize_ = 0;

    mutable std::mutex mutex_;
};

aclError AclrtMemcpyAsyncParamCheck(
    void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind, aclrtStream stream)
{
    auto ret = aclrtMemcpyAsync(dst, destMax, src, count, kind, stream);
    return ret;
}

aclError AclrtMemcpyParamCheck(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind)
{
    // Disabled and invalid configurations use the original memcpy path. The
    // singleton is therefore not constructed and no keystream/AES resources
    // are allocated in this branch.
    if (GetTorchEncryptionAlgorithm() ==
            TorchEncryptionAlgorithm::DISABLED ||
        count == 0) {
        return aclrtMemcpy(dst, destMax, src, count, kind);
    }

    if (ACL_MEMCPY_HOST_TO_DEVICE == kind) {
        Singleton &encryption = Singleton::getInstance();
        auto transactionLock = encryption.acquireTransactionLock();

        std::unique_ptr<void, decltype(&std::free)> srcTmp(
            std::malloc(count), &std::free);
        if (srcTmp == nullptr) {
            throw std::bad_alloc();
        }

        encryption.enc(const_cast<void *>(src), srcTmp.get(), count, true);
        const aclError ret =
            aclrtMemcpy(dst, destMax, srcTmp.get(), count, kind);
        if (ret == ACL_SUCCESS) {
            encryption.dec(dst, dst, count, true);
        } else {
            encryption.invalidateKeyStream();
        }

        return ret;
    } else if (ACL_MEMCPY_DEVICE_TO_HOST == kind) {
        Singleton &encryption = Singleton::getInstance();
        auto transactionLock = encryption.acquireTransactionLock();

        void *rawDeviceTmp = nullptr;
        CheckAcl(aclrtMalloc(&rawDeviceTmp, count,
                             ACL_MEM_MALLOC_HUGE_FIRST),
                 "aclrtMalloc encrypted D2H temporary buffer");
        struct AclDeviceBufferDeleter {
            void operator()(void *ptr) const noexcept
            {
                if (ptr != nullptr) {
                    aclrtFree(ptr);
                }
            }
        };
        std::unique_ptr<void, AclDeviceBufferDeleter> srcTmp(rawDeviceTmp);

        encryption.enc(const_cast<void *>(src), srcTmp.get(), count, false);
        const aclError ret =
            aclrtMemcpy(dst, destMax, srcTmp.get(), count, kind);
        if (ret == ACL_SUCCESS) {
            encryption.dec(dst, dst, count, false);
        } else {
            encryption.invalidateKeyStream();
        }

        return ret;
    }

    return aclrtMemcpy(dst, destMax, src, count, kind);
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