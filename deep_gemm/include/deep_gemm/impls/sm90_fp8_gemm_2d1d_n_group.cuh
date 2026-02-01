#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

#include <deep_gemm/common/epilogue_utils.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/scheduler.cuh>
#include <deep_gemm/common/sm90_utils.cuh>

namespace deep_gemm {

using namespace deep_gemm::sm90;

template <uint32_t kNumFormerIters, uint32_t kGap, uint32_t kEnd, typename func_t>
__device__ void dispatch_num_former_iters(uint32_t num_former_iters, const func_t& func) {
    if (num_former_iters == kNumFormerIters) {
        func(cute::Int<kNumFormerIters>{});
        return;
    }

    if constexpr (kNumFormerIters + kGap <= kEnd)
        dispatch_num_former_iters<kNumFormerIters + kGap, kGap, kEnd>(num_former_iters, func);
}

template <uint32_t kSignalMode,
          cute::UMMA::Major kMajorSFA,
          uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t kNumGroups,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode, uint32_t kSwizzleDMode,
          uint32_t kNumStages, uint32_t kNumLastStages,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads,
          uint32_t kNumTMAMulticast, bool kIsTMAMulticastOnA,
          uint32_t kNumSMs, GemmType kGemmType,
          typename epilogue_type_t>
__global__ __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1) void
sm90_fp8_gemm_2d1d_n_group_impl(float* sfa, int* grouped_layout,
                        uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_d,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_sfb) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900)) or defined(__CLION_IDE__)
    // Scaling checks
    DG_STATIC_ASSERT(BLOCK_K == 128, "Only support per-128-channel FP8 scaling");
    DG_STATIC_ASSERT(constexpr_ceil_div(BLOCK_M, BLOCK_K) == 1 or (constexpr_gcd(BLOCK_M, BLOCK_K) == BLOCK_M - BLOCK_K), "Too much A scales in a single block");
    // Types
    using WGMMA = typename FP8MMASelector<BLOCK_N>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    DG_STATIC_ASSERT(BLOCK_M % WGMMA::M == 0 or BLOCK_M < WGMMA::M, "Invalid block size");

    // Overwrite shape constants if the compiler gives
    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;

    // Shared memory
    static constexpr bool kMustUseUniformedScaleA = (BLOCK_K % BLOCK_M == 0); // 判断矩阵B的缩放因子是否可以 “统一使用”（即单个缩放因子覆盖整个B块）
    static constexpr uint32_t SMEM_D_SIZE = constexpr_align(BLOCK_M * BLOCK_N * static_cast<uint32_t>(sizeof(__nv_bfloat16)), 1024u); // 存储矩阵D（计算结果）的共享内存大小
    static constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(__nv_fp8_e4m3);// 每个流水线阶段（stage）中，矩阵A的共享内存大小。A的块大小是BLOCK_M × BLOCK_K（M×K维度），元素类型是__nv_fp8_e4m3（8 位浮点数），用于存储单次加载的A块数据
    static constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(__nv_fp8_e4m3);// 每个流水线阶段中，矩阵B的共享内存大小。B的块大小是BLOCK_N × BLOCK_K（N×K维度），元素类型同样是 8 位浮点数，存储单次加载的B块数据
    static constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE = BLOCK_N * sizeof(float);// 每个阶段中，矩阵A的缩放因子（SFA）的共享内存大小。BLOCK_M中的每一行对应一个缩放因子，类型是float（32 位），因此大小为BLOCK_M × 4字节
    static constexpr uint32_t ALIGNED_SMEM_SFB_SIZE_PER_STAGE = constexpr_align(SMEM_SFB_SIZE_PER_STAGE, 128u);
    const uint32_t& shape_k_scales = ceil_div(shape_k, BLOCK_K);// 计算K维度上缩放因子的总数量
    const uint32_t& shape_n_sfa = ceil_div(shape_m, BLOCK_K);
    const uint32_t& smem_sfa_size = align<uint32_t>(shape_k_scales * (kMustUseUniformedScaleA ? 1 : 2) * sizeof(float), sizeof(Barrier)); // 计算共享内存中存储B的缩放因子（SFB）所需的空间，并确保内存对齐
    // NOTES: Make sure we have enough shared memory for WGMMA padding
    static constexpr uint32_t WGMMA_A_SIZE_PER_STAGE = WGMMA::M * BLOCK_K * sizeof(__nv_fp8_e4m3);
    DG_STATIC_ASSERT(WGMMA_A_SIZE_PER_STAGE <= SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE * kNumStages, "Memory Out of bound for WGMMA");
    // Configs
    const uint32_t num_total_k_blocks = ceil_div(shape_k, BLOCK_K);
    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = get_lane_idx();

    // Prefetch TMA descriptors at the very beginning
    if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_sfb);
        cute::prefetch_tma_descriptor(&tensor_map_d);
    }
    __syncwarp();

    // Align to 1024 bytes for swizzle-128B
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_D_SIZE % 1024 == 0, "Shared memory of A/B must be aligned to 1024 bytes");

    // Data on shared memory
    auto smem_d = reinterpret_cast<__nv_bfloat16*>(smem_buffer);
    auto smem_a = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_D_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_D_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });
    constexpr uint32_t SMEM_SF_OFFSET = SMEM_D_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
    auto smem_sfb = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + SMEM_SF_OFFSET + i * ALIGNED_SMEM_SFB_SIZE_PER_STAGE);
    });
    auto smem_sfa = reinterpret_cast<float*>(smem_buffer + SMEM_SF_OFFSET + kNumStages * ALIGNED_SMEM_SFB_SIZE_PER_STAGE);

    // Fill barriers
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(reinterpret_cast<uint8_t*>(smem_sfa) + smem_sfa_size);
    auto full_barriers     = PatternVisitor([&](const uint32_t& i) { return barrier_start_ptr + i; });
    auto empty_barriers    = PatternVisitor([&](const uint32_t& i) { return barrier_start_ptr + kNumStages + i; });

    // Initialize barriers
    DG_STATIC_ASSERT(kNumTMAMulticast <= 32, "Too many TMA multicast");
    if (warp_idx == kNumMathThreads / 32 + 1 and cute::elect_one_sync()) {
        // NOTES: we always use `lane_idx` to arrive for the `lane_idx`-th CTA in the cluster,
        // even with TMA multicast disabled, we want to make the behavior aligned
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(kNumTMAMulticast * kNumMathThreads / 32);
        }

        // Make initialized barrier visible in async proxy  内存屏障（fence），用于确保屏障的初始化状态被所有线程（包括其他 CTA 的线程）可见
        cutlass::arch::fence_barrier_init();
    }

    // Synchronize all threads to make barrier visible in normal memory model
    (kNumTMAMulticast > 1) ? cute::cluster_sync() : __syncthreads();

    // Register reconfigurations
    constexpr uint32_t kNumTMARegisters = 40;
    constexpr uint32_t kNumMathRegisters = kNumMathThreads == 128 ? 248 : 232;

    // Block scheduler
    uint32_t m_block_idx, n_block_idx; // 用于存储当前线程块（CTA）被分配到的矩阵块在 M 维度（行）和 N 维度（列）上的索引（后续会由调度器赋值）
    auto scheduler = Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, kNumTMAMulticast, kIsTMAMulticastOnA, kNumSMs,
                               512u,  // SF_K_ALIGNMENT (default)
                               get_num_1d_blocks_per_group<kGemmType, BLOCK_M, BLOCK_N, kNumSMs, kIsTMAMulticastOnA>(),  // kNum1DBlocksPerGroup (default)
                               true   // kIsNGroupMasked = true
                               >(shape_m, shape_n, shape_k, grouped_layout);

    // Pipeline and TMA phases
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;

        // Flip phases only if reach the next first stage
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    if (warp_idx >= kNumMathThreads / 32) {
        // TMA warp-group for loading data
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();

        // NOTES: only one thread (or warp) will be used
        // We use the third warp, as warp 0/1 may be doing WGMMA with `BLOCK_M == 32`
        if (warp_idx == kNumMathThreads / 32 + 2 and cute::elect_one_sync()) {
            // Persistently schedule over blocks
            while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
                // Assign TMA multicast number into A and B
                // NOTES: there may be additional odd rows/columns or cases where multicast is not possible.
                const bool is_tma_multicast_valid = scheduler.is_tma_multicast_valid(m_block_idx);
                const uint32_t num_tma_multicast_a = (kIsTMAMulticastOnA and is_tma_multicast_valid) ? kNumTMAMulticast : 1;
                const uint32_t num_tma_multicast_b = (not kIsTMAMulticastOnA and is_tma_multicast_valid) ? kNumTMAMulticast : 1;
                DG_STATIC_ASSERT(kNumTMAMulticast <= 2, "Scheduler does not support > 2 TMA multicast");

                for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                    // Wait consumer release
                    empty_barriers[stage_idx]->wait(phase ^ 1);

                    // Issue TMA A
                    constexpr bool kWithGroupOffsetB = kGemmType == GemmType::MGroupedMasked;
                    auto& full_barrier = *full_barriers[stage_idx];
                    const uint32_t k_idx = k_block_idx * BLOCK_K;
                    tma_copy<BLOCK_K, BLOCK_M, kSwizzleAMode>(&tensor_map_a, &full_barrier,
                             smem_a[stage_idx], k_idx, scheduler.get_global_idx<true>(shape_m, BLOCK_M, m_block_idx),
                             num_tma_multicast_a);

                    // Issue TMA B
                    tma_copy<BLOCK_K, BLOCK_N, kSwizzleBMode>(&tensor_map_b, &full_barrier,
                             smem_b[stage_idx], k_idx, scheduler.get_global_idx<kWithGroupOffsetB>(shape_n, BLOCK_N, n_block_idx, m_block_idx),
                             num_tma_multicast_b);
                    // Issue TMA SFB: smem_outer_dim = 1 in descriptor, so BLOCK_OUTER = 1
                    tma_copy<BLOCK_N, BLOCK_K, 0>(&tensor_map_sfb, &full_barrier,
                             smem_sfb[stage_idx], n_block_idx * BLOCK_N, scheduler.get_global_idx<kWithGroupOffsetB>(shape_k_scales, 1, k_block_idx),
                             num_tma_multicast_b);
                    full_barrier.arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE + SMEM_SFB_SIZE_PER_STAGE);
                }
            }

            // To safely deconstruct distributed shared barriers, we need another round of empty waits
            if constexpr (kNumTMAMulticast > 1) {
                for (uint32_t i = 0; i < kNumStages; advance_pipeline(i))
                    empty_barriers[stage_idx]->wait(phase ^ 1);
            }
        }
    } else {
        // Math warp-groups for WGMMA
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        // NOTES: use `__shfl_sync` to encourage NVCC to use unified registers
        const auto math_wg_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0); // 计算当前线程所属的 “计算线程组索引”，并通过线程束洗牌（shuffle）操作让组内所有线程共享该索引
        const auto r_0 = warp_idx * 16 + lane_idx / 4, r_1 = r_0 + 8; // A 矩阵的缩放因子（smem_sfa）偏移量，4个线程使用相同两个缩放因子，每个warp就需要16个缩放因子

        auto a_desc = make_smem_desc(smem_a[0] + math_wg_idx * WGMMA::M * BLOCK_K, 1); // a_desc是smem_a的完整描述符，stride是每个warpgroup矩阵乘处理的数据大小
        auto b_desc = make_smem_desc(smem_b[0], 1);// b_desc是smem_b的完整描述符，stride是每个warpgroup矩阵乘处理的数据大小
        const uint32_t a_desc_lo = __shfl_sync(0xffffffff, a_desc.reg32_[0], 0);// 是 a_desc 描述符的低32位部分（reg32_[0]）包含基地址,广播到warp（为什么可以取低32位可以搜“union GmmaDescriptor”看它的结构）
        const uint32_t b_desc_lo = __shfl_sync(0xffffffff, b_desc.reg32_[0], 0);

        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            // Decide the number of scales A to load
            DG_TRAP_ONLY_DEVICE_ASSERT(shape_m % 8 == 0); // 在设备端（GPU）断言 N 维度的总大小（shape_n）必须是 8 的倍数，这个断言与Bscale无关？
            uint32_t num_former_iters = BLOCK_M / 8, num_full_iters = num_former_iters; // BLOCK_N是每个线程块（CTA）负责的 N 维度大小（例如 256）， 除以 8 的原因：后续处理 B 矩阵时，通常以 8 个元素为一组（如 WGMMA 指令一次处理 8 列）
            if constexpr (not kMustUseUniformedScaleA) {
                num_former_iters = min(BLOCK_M, BLOCK_K - m_block_idx * BLOCK_M % BLOCK_K) / 8;
                num_full_iters = min(shape_m - m_block_idx * BLOCK_M, BLOCK_M) / 8;
            }
            uint32_t num_sfa = shape_k_scales * (num_former_iters >= num_full_iters ? 1 : 2); // 计算需要加载的 B 矩阵缩放因子（sfb）的总数量

            // Load B scales with math warp-groups
            // NOTES: except the first warp, we want to overlap loading B scales with TMA stores between tasks 第一个warp负责TMA存储（在while循环的最后），与此处其余warp加载sfb操作是overlap的
            if (threadIdx.x >= 32) {
                auto previous_group_offset = scheduler.get_global_idx<true>(shape_n_sfa * shape_k_scales, 0, 0, n_block_idx);
                const uint32_t stride_n_sfa = kMajorSFA == cute::UMMA::Major::MN ? 1 : shape_k_scales;
                const uint32_t stride_k_sfa = kMajorSFA == cute::UMMA::Major::MN ? shape_n_sfa : 1;
                auto local_sfa = sfa + previous_group_offset + ((m_block_idx * BLOCK_M) / BLOCK_K) * stride_n_sfa;

                #pragma unroll
                for (uint32_t i = threadIdx.x - 32; i < num_sfa; i += kNumMathThreads - 32)
                    st_shared(smem_sfa + i, __ldg(i < shape_k_scales ? local_sfa + i * stride_k_sfa : local_sfa + (i - shape_k_scales) * stride_k_sfa + stride_n_sfa));
            }
            cutlass::arch::NamedBarrier::sync(kNumMathThreads, 0); // 通过命名屏障（NamedBarrier）同步所有计算线程（共kNumMathThreads个），确保所有sfb数据都已加载到共享内存后，再执行后续的矩阵乘法计算。

            // Accumulation for WGMMA or CUDA promotion
            constexpr uint32_t WAVE_BLOCK_M = BLOCK_M <= WGMMA::M ? BLOCK_M : WGMMA::M * 2;
            DG_STATIC_ASSERT(BLOCK_M % WAVE_BLOCK_M == 0, "Invalid block sizes");
            float accum[WGMMA::kNumAccum], final_accum[WGMMA::kNumAccum * (BLOCK_M / WAVE_BLOCK_M)] = {0};
            float2 scale_b[WGMMA::N / 8];
            // Pick threads whose WGMMA results are to be stored in shared memory
            DG_STATIC_ASSERT(BLOCK_M >= 64 or kNumMathThreads == 128, "Only one math warp group for `BLOCK_M < 64`");
            constexpr uint32_t kNumWGMMAStoreThreads = WAVE_BLOCK_M * (128 / WGMMA::M);
            const bool do_wgmma_store = BLOCK_M >= WGMMA::M or warp_idx < kNumWGMMAStoreThreads / 32;

            // Empty barrier arrival
            auto empty_barrier_arrive = [&]() {
                if constexpr (kNumTMAMulticast == 1) {
                    lane_idx == 0 ? empty_barriers[stage_idx]->arrive() : void();
                } else {
                    auto target_cta = scheduler.is_peer_cta_alive ? lane_idx : cute::block_rank_in_cluster();
                    lane_idx < kNumTMAMulticast ? empty_barriers[stage_idx]->arrive(target_cta) : void();
                }
            };

            // Skip useless computations
            if (scheduler.is_computation_valid(n_block_idx, 0)) {
                // The compiler must know the dynamic variable `num_former_iters`'s real value
                constexpr bool kShouldOptimize = BLOCK_K / constexpr_gcd(BLOCK_K, BLOCK_N) <= 4 and not kMustUseUniformedScaleA;
                constexpr uint32_t kGap = constexpr_gcd(BLOCK_K, BLOCK_N) / 8;
                constexpr uint32_t kEnd = kShouldOptimize ? BLOCK_K / 8 : 0;

                // Dispatch `num_former_iters` and launch MMAs
                dispatch_num_former_iters<0, kGap, kEnd>(kShouldOptimize ? num_former_iters : 0, [&](auto _) {
                    #pragma unroll 8
                    for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                        const auto& a_desc_base_lo = a_desc_lo + stage_idx * (SMEM_A_SIZE_PER_STAGE / 16);// 这里除以16的原因应该是描述符取reg32_[0]后，导致 4LSB not included
                        const auto& b_desc_base_lo = b_desc_lo + stage_idx * (SMEM_B_SIZE_PER_STAGE / 16);

                        // Read A scales
                        float scale_a_0 = ld_shared(smem_sfa + k_block_idx), scale_a_1;
                        // NOTES: even some blocks do not need to read the second row, but we still load one to align with other blocks
                        if constexpr (not kMustUseUniformedScaleA)
                            scale_a_1 = ld_shared(smem_sfa + k_block_idx + shape_k_scales);

                        // Wait TMA arrivals
                        full_barriers[stage_idx]->wait(phase);
                        // 耗时重灾区
                        #pragma unroll
                        for (uint32_t i = 0; i < WGMMA::N/8; ++ i)
                            scale_b[i] = ld_shared(reinterpret_cast<float2*>(smem_sfb[stage_idx] + i*8 + (lane_idx%4)*2));
                        // TODO: remove some useless computation for unaligned Ms
                        #pragma unroll
                        for (uint32_t local_idx = 0; local_idx < BLOCK_M / WAVE_BLOCK_M; ++ local_idx) { // BLOCK_M的基础上，拆分成 WAVE_BLOCK_M 的大小，然后再做计算
                            auto m_offset = local_idx * WAVE_BLOCK_M;

                            // 这里缺一个do_wgmma_store判断？但我不知道怎么做

                            // Commit WGMMA instructions
                            #pragma unroll
                            for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                                warpgroup_fence_operand(accum[i]); // 确保累加器（accum）的寄存器操作顺序，避免乱序执行导致的数据错误。
                            warpgroup_arrive(); // 线程组（warpgroup）内的线程同步，确保所有线程准备好执行矩阵乘法
                            #pragma unroll
                            for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) { // 将 K 维度的子块（BLOCK_K）拆分为 WGMMA 指令可处理的粒度（WGMMA::K)
                                a_desc.reg32_[0] = a_desc_base_lo + (m_offset * BLOCK_K + k * WGMMA::K) / 16;
                                b_desc.reg32_[0] = b_desc_base_lo + k * WGMMA::K / 16; // 这里忽略了BLOCK_N的遍历，原因在于BLOCK_N = WGMMA::N
                                WGMMA::wgmma(a_desc, b_desc, accum, k);
                            }
                            warpgroup_commit_batch(); // 提交计算任务
                            #pragma unroll
                            for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                                warpgroup_fence_operand(accum[i]); // // 确保累加器（accum）的寄存器操作顺序，避免乱序执行导致的数据错误。
                            warpgroup_wait<0>(); // 等待，确保所有线程的 WGMMA 指令执行完毕

                            // Notify barrier arrival at the last warpgroup wave
                            if (local_idx == BLOCK_M / WAVE_BLOCK_M - 1)
                                empty_barrier_arrive(); // 告知 TMA 线程 “当前阶段的共享内存数据已处理完毕，可安全加载新数据”

                            // Skip promotion for the unfilled parts
                            if (not do_wgmma_store)
                                continue;

                            // Promote with scales 将accum中的中间结果（WGMMA 计算结果）按缩放因子缩放后，累加到final_accum中（final_accum是存储最终结果的累加器数组，按 M 子块划分）
                            // NOTES: making it as predicates is very important for performance, comparing to two loops
                            float cloumn_scale_0, cloumn_scale_1;

                            auto shifted_accum = final_accum + WGMMA::kNumAccum * local_idx;
                            #pragma unroll
                            for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                                // NOTES: for unrolled `num_former_iters` cases, we expect the compiler to automatically make it a constant
                                bool predicate = kMustUseUniformedScaleA or i < num_former_iters;
                                cloumn_scale_0 = predicate ? scale_b[i].x * scale_a_0 : scale_b[i].x * scale_a_1;
                                cloumn_scale_1 = predicate ? scale_b[i].y * scale_a_0 : scale_b[i].y * scale_a_1;
                                shifted_accum[i * 4 + 0] += cloumn_scale_0 * accum[i * 4 + 0];
                                shifted_accum[i * 4 + 1] += cloumn_scale_1 * accum[i * 4 + 1];
                                shifted_accum[i * 4 + 2] += cloumn_scale_0 * accum[i * 4 + 2];
                                shifted_accum[i * 4 + 3] += cloumn_scale_1 * accum[i * 4 + 3];
                            }
                        }
                    }
                });
            } else {
                #pragma unroll
                for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                    full_barriers[stage_idx]->wait(phase);
                    empty_barrier_arrive();
                }
            }

            // TMA checks
            constexpr uint32_t kNumElemBytes = sizeof(nv_bfloat16); // 目标数据类型（nv_bfloat16）的字节大小
            constexpr uint32_t TMA_D_BLOCK_N = kSwizzleDMode == 0 ? BLOCK_N : (kSwizzleDMode / kNumElemBytes); // TMA 存储时 N 维度的块大小（每次 TMA 操作处理的列数）
            constexpr uint32_t WGMMA_M_PER_WARP = WGMMA::M / 4; // 每个 warp 负责的 M 维度子块大小
            DG_STATIC_ASSERT(BLOCK_M % 8 == 0, "Invalid swizzling atom");
            DG_STATIC_ASSERT(BLOCK_N % 8 == 0, "Invalid swizzling atom");
            DG_STATIC_ASSERT(BLOCK_N % TMA_D_BLOCK_N == 0 and BLOCK_N / TMA_D_BLOCK_N <= 32,
                            "Unaligned TMA store or too many TMA store instructions");
            DG_STATIC_ASSERT(TMA_D_BLOCK_N % 8 == 0, "Invalid TMA block N");

            // Skip WGMMA store for the unfilled parts
            if (not do_wgmma_store)
                continue;

            // Wait last TMA store to be finished
            if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N) 
                cute::tma_store_wait<0>();
            cutlass::arch::NamedBarrier::sync(kNumWGMMAStoreThreads, 1);

            // Write back to shared memory using STSM and issue TMA stores
            DG_STATIC_ASSERT(WGMMA::kNumAccum % 4 == 0, "Invalid STSM x2 vectorization");
            #pragma unroll
            for (uint32_t local_idx = 0; local_idx < BLOCK_M / WAVE_BLOCK_M; ++ local_idx) { // 按WAVE_BLOCK_M划分BLOCK_M，每个子块对应final_accum中的一段结果（shifted_accum）
                auto m_offset = local_idx * WAVE_BLOCK_M;
                auto shifted_accum = final_accum + WGMMA::kNumAccum * local_idx;
                #pragma unroll
                for (auto i = 0; i < WGMMA::kNumAccum / 4; ++ i) { // 按 4 个累加器为一组处理（WGMMA::kNumAccum / 4），匹配向量指令的处理粒度（一次处理 4 个元素）
                    // Swizzle or padding into the correct address
                    uint8_t* smem_ptr = nullptr;
                    if constexpr (kSwizzleDMode > 0) {
                        // Calculate the swizzling atom offset and in-atom offset
                        constexpr uint32_t kNumBankGroupBytes = 16;
                        auto atom_offset = i / (TMA_D_BLOCK_N / 8), in_atom_offset = i % (TMA_D_BLOCK_N / 8);

                        // Calculate the index of the bank group to be written in the atom
                        auto bank_group_index = in_atom_offset + lane_idx * (kSwizzleDMode / kNumBankGroupBytes);

                        // Reshape the atom in another view and swizzle
                        //  - original: `(BLOCK_M, kSwizzleDMode / kNumBankGroupBytes)`
                        //  - new: `(BLOCK_M * kSwizzleDMode / kNumBankGroupBytes / 8, 8)`
                        constexpr bool kHasShortcut = (kSwizzleDMode / kNumBankGroupBytes) == 8;
                        auto row = kHasShortcut ? (in_atom_offset / 8 + lane_idx) : (bank_group_index / 8);
                        auto col = kHasShortcut ? (in_atom_offset) : (bank_group_index % 8);
                        col ^= row % (kSwizzleDMode / 16);

                        // Add back into the base pointer
                        // NOTES: think twice before modifying this, as changes may affect the number of instructions
                        smem_ptr = reinterpret_cast<uint8_t*>(smem_d) +                // Base pointer
                            warp_idx * (WGMMA_M_PER_WARP * kSwizzleDMode) +            // Warp offset
                            m_offset * kSwizzleDMode +                                 // Wave offset
                            atom_offset * BLOCK_M * kSwizzleDMode +                    // Swizzle atom offset (constants)
                            row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes; // In-atom offset
                    } else {
                        // No swizzling, just padding
                        smem_ptr = reinterpret_cast<uint8_t*>(smem_d + (m_offset + warp_idx * WGMMA_M_PER_WARP + lane_idx) * BLOCK_N + i * 8);
                    }

                    // NOTES: only 16 lanes' addresses are used
                    SM90_U32x2_STSM_N<nv_bfloat162>::copy( // 用 SM90 架构的专用向量存储指令（STSM），将转换后的 bfloat16 向量写入smem_ptr指向的共享内存位置，一次操作完成 4 个元素的存储
                        __float22bfloat162_rn({shifted_accum[i * 4 + 0], shifted_accum[i * 4 + 1]}), //将 float 类型的累加结果（shifted_accum中的 4 个元素）转换为 bfloat16 类型（2 字节），并打包为向量（每次转换 2 对元素，共 4 个元素）
                        __float22bfloat162_rn({shifted_accum[i * 4 + 2], shifted_accum[i * 4 + 3]}),
                        smem_ptr
                    );
                }
            }
            cute::tma_store_fence(); // TMA 存储的内存屏障，确保所有 TMA 存储操作完成
            cutlass::arch::NamedBarrier::sync(kNumWGMMAStoreThreads, 1); // 确保所有计算线程（共kNumMathThreads个）都完成结果写入，统一进入 TMA 存储阶段

            // Use TMA store to write back to global memory
            // TODO: compatible with FP32 output
            constexpr bool kWithGroupOffsetD = kGemmType == GemmType::MGroupedMasked;
            DG_STATIC_ASSERT(kNumWGMMAStoreThreads >= BLOCK_N / TMA_D_BLOCK_N, "Too many TMA blocks");
            if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N) { // 仅threadIdx.x < BLOCK_N / TMA_D_BLOCK_N的线程参与 TMA 存储
                auto in_block_n_offset = threadIdx.x * TMA_D_BLOCK_N; // 当前线程负责的 N 维度在本地块内的偏移
                auto smem_ptr = smem_d + in_block_n_offset * BLOCK_M; // 共享内存中待存储数据的起始地址
                cute::SM90_TMA_STORE_2D::copy(&tensor_map_d, smem_ptr,
                                              epilogue_type_t::apply_index_n<TMA_D_BLOCK_N>(n_block_idx * BLOCK_N + in_block_n_offset),
                                              scheduler.get_global_idx<kWithGroupOffsetD>(shape_m, BLOCK_M, m_block_idx)); // 使用 TMA 存储指令（TMA_STORE_2D），将数据从共享内存中存储到全局内存中
                cute::tma_store_arrive();
            }
            __syncwarp();
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_90a");
#endif
}

};  // namespace deep_gemm

#pragma clang diagnostic pop
