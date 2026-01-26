#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda/std/cstdint>
#include <cuda/std/utility>
#include <cute/container/tuple.hpp>

#include "cute_tie.cuh"

#ifdef __CLION_IDE__

__host__ __device__ __forceinline__ void host_device_printf(const char* format, ...) {
    asm volatile("trap;");
}

#define printf host_device_printf
#endif

#ifndef DG_DEVICE_ASSERT
#define DG_DEVICE_ASSERT(cond) \
do { \
    if (not (cond)) { \
        printf("Assertion failed: %s:%d, condition: %s\n", __FILE__, __LINE__, #cond); \
        asm("trap;"); \
    } \
} while (0)
#endif

#ifndef DG_TRAP_ONLY_DEVICE_ASSERT
#define DG_TRAP_ONLY_DEVICE_ASSERT(cond) \
do { \
    if (not (cond)) \
        asm("trap;"); \
} while (0)
#endif

#ifndef DG_STATIC_ASSERT
#define DG_STATIC_ASSERT(cond, ...) static_assert(cond, __VA_ARGS__)
#endif

namespace deep_gemm {

template <typename FuncT>
struct PatternVisitor {
    FuncT func;

    __device__ __host__
    explicit PatternVisitor(FuncT&& func): func(std::forward<FuncT>(func)) {}

    __device__ __host__
    auto operator [](const uint32_t& i) {
        return func(i);
    }
};

template <typename T>
__device__ __host__ T ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

template <typename T>
__device__ __host__ constexpr T constexpr_ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

template <typename T>
__device__ __host__ T align(T a, T b) {
    return ceil_div(a, b) * b;
}

template <typename T>
__device__ __host__ constexpr T constexpr_align(T a, T b) {
    return constexpr_ceil_div(a, b) * b;
}

template <typename T>
__device__ __host__ constexpr T constexpr_gcd(T a, T b) {
    return b == 0 ? a : constexpr_gcd(b, a % b);
}

template<typename T>
__forceinline__ __device__ void swap(T& a, T& b) {
    T temp = a;
    a = b;
    b = temp;
}

__forceinline__ __device__ uint32_t get_sm_idx() {
    uint32_t sm_idx;
    asm ("mov.u32 %0, %%smid;" : "=r"(sm_idx));
    return sm_idx;
}

__forceinline__ __device__ uint32_t get_lane_idx() {
    uint32_t lane_id;
    asm ("mov.u32 %0, %laneid;" : "=r"(lane_id));
    return lane_id;
}

__device__  __forceinline__ uint32_t ld_shared(const uint32_t* ptr) {
    uint32_t ret;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(ret) : "l"(__cvta_generic_to_shared(ptr)));
    return ret;
}

__device__  __forceinline__ float2 ld_shared(const float2* ptr) {
    float2 ret;
    asm volatile("ld.shared.v2.f32 {%0, %1}, [%2];" : "=f"(ret.x), "=f"(ret.y) : "l"(__cvta_generic_to_shared(ptr)));
    return ret;
}

__device__  __forceinline__ float4 ld_shared(const float4* ptr) {
    float4 ret;
    asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w) : "l"(__cvta_generic_to_shared(ptr)));
    return ret;
}

__device__  __forceinline__ uint4 ld_shared(const uint4* ptr) {
    uint4 ret;
    asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : "l"(__cvta_generic_to_shared(ptr)));
    return ret;
}

__device__  __forceinline__ float ld_shared(const float* ptr) {
    float ret;
    asm volatile("ld.shared.f32 %0, [%1];" : "=f"(ret) : "l"(__cvta_generic_to_shared(ptr)));
    return ret;
}

__device__ __forceinline__ void st_shared(const float* ptr, float val) {
    asm volatile("st.shared.f32 [%0], %1;" :: "l"(__cvta_generic_to_shared(ptr)), "f"(val));
}

__device__ __forceinline__ void st_shared(const float2* ptr, float2 val) {
    asm volatile("st.shared.v2.f32 [%0], {%1, %2};" :: "l"(__cvta_generic_to_shared(ptr)), "f"(val.x), "f"(val.y));
}

__device__ __forceinline__ void st_shared(const uint32_t* ptr, uint32_t val) {
    asm volatile("st.shared.u32 [%0], %1;" :: "l"(__cvta_generic_to_shared(ptr)), "r"(val));
}

__device__  __forceinline__ void st_shared(const void* ptr, uint32_t x, uint32_t y) {
    asm volatile("st.shared.v2.u32 [%0], {%1, %2};" :: "l"(__cvta_generic_to_shared(ptr)), "r"(x), "r"(y));
}

__device__  __forceinline__ void st_shared(const void* ptr, uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
    asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"(__cvta_generic_to_shared(ptr)), "r"(x), "r"(y), "r"(z), "r"(w));
}

template <typename old_t>
__device__ __forceinline__ int cast_into_bf16_and_pack(old_t& x, old_t& y) {
    auto bf16x2 = __float22bfloat162_rn({*reinterpret_cast<float*>(&x), *reinterpret_cast<float*>(&y)});
    return *reinterpret_cast<int*>(&bf16x2);
}

__device__ __forceinline__ void prefetch_l1(void *ptr) {
    asm volatile("prefetch.global.L1 [%0];" :: "l"(ptr));
}

__device__ __forceinline__ uint32_t atomic_add_release_global_32(uint32_t* addr, uint32_t value) {
    uint32_t ret;  // 返回值：原子操作前的原始值
    asm volatile (
        "atom.add.release.gpu.global.u32 %0, [%1], %2;" 
        : "=r"(ret)                // 输出约束：%0 对应 ret，r=32位寄存器（匹配 uint32_t）
        : "l"(addr), "r"(value)    // 输入约束：%1=addr（指针，64位），%2=value（32位值)
    );
    return ret;
}

__device__ __forceinline__ uint64_t atomic_add_release_global_64(uint64_t* addr, uint64_t value) {
    uint64_t ret;  // 返回值：原子操作前的原始值（与输入类型一致，均为 uint64_t）
    asm volatile (
        "atom.add.release.gpu.global.u64 %0, [%1], %2;" 
        : "=l"(ret)                // 输出约束：%0 对应 ret，l=64位寄存器（匹配 uint64_t）
        : "l"(addr), "l"(value)    // 输入约束：%1=addr（指针，64位），%2=value（64位值)
    );
    return ret;
}

__device__ __forceinline__ uint32_t ld_acquire_global_32(const uint32_t* addr) {
    uint32_t val;
    asm volatile (
        "ld.acquire.gpu.global.u32 %0, [%1];"  // 匹配.global内存、u32类型、acquire语义
        : "=r"(val)                              // 输出：32位寄存器（匹配uint32_t）
        : "l"(addr)                               // 输入：目标地址（64位指针，用l约束）
    );
    return val;
}

__device__ __forceinline__ uint64_t ld_acquire_global_64(const uint64_t* addr) {
    uint64_t val;
    asm volatile (
        "ld.acquire.gpu.global.u64 %0, [%1];"  // 匹配.global内存、u64类型、acquire语义
        : "=l"(val)                              // 输出：64位寄存器（匹配uint64_t）
        : "l"(addr)                               // 输入：目标地址（64位指针，用l约束）
    );
    return val;
}

template <uint32_t kNumBytes>
struct Vectorized {
    static auto zeros() {
        // TODO: add `ulonglong4` for SM100 once `__ldg` support this
        if constexpr (kNumBytes > 0 and kNumBytes % 16 == 0) {
            return make_uint4(0, 0, 0, 0);
        } else if constexpr (kNumBytes > 0 and kNumBytes % 8 == 0) {
            return make_uint2(0, 0);
        } else if constexpr (kNumBytes > 0 and kNumBytes % 4 == 0) {
            return 0;
        } else {
            DG_STATIC_ASSERT(kNumBytes > 0 and kNumBytes % 4 == 0, "Invalid vectorization");
        }
    }

    using vec_t = decltype(zeros());
};

} // namespace `deep_gemm`
