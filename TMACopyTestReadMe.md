注意，仅针对的normal矩阵乘的 sm90_1D2D进行了修改。
本实验的目的是测试多次TMA Copy对矩阵乘的影响。
因此需要将原本矩阵乘中的一次TMA Copy，拆分为多次。


/csrc/jit_kernels/heuristics/common.hpp
line: 288
best_multicast_config = {1, is_multicast_on_a};
说明：
强制使多播数量为1（关闭多播），因为我们TMA Copy的Size不是Block_M 与 BLOCK_N，为避免问题，所以不采用多播。


/csrc/jit_kernels/impls/runtime_utils.hpp
line: 145~164 修改为
static CUtensorMap make_tma_a_desc(const cute::UMMA::Major& major,
                                   const torch::Tensor& t,
                                   const int& shape_m, const int& shape_k,
                                   const int& block_m, const int& block_k,
                                   const int& outer_stride,
                                   const int& num_groups,
                                   const int& swizzle_mode, const int& swizzle_base = 0,
                                   const bool& allow_tf32 = false,
                                   const int& gps_tma_copy_size = 1) {
    if (num_groups > 1)
        DG_HOST_ASSERT(major == cute::UMMA::Major::K);
    const auto& [gmem_inner_dim, gmem_outer_dim] = get_inner_outer_dims(major, shape_k, shape_m * num_groups);
    const auto& [smem_inner_dim, smem_outer_dim] = get_inner_outer_dims(major, block_k, block_m / gps_tma_copy_size);
    return make_tma_2d_desc(t,
                            gmem_inner_dim, gmem_outer_dim,
                            smem_inner_dim, smem_outer_dim,
                            outer_stride,
                            swizzle_mode, swizzle_base,
                            allow_tf32);
}
说明：
新增参数 gps_tma_copy_size，表示一次TMA Copy中，shared memory的大小为 block_m / gps_tma_copy_size * block_k;




/csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp
line：96～100 修改为
const auto& tensor_map_a = make_tma_a_desc(major_a, a, m, k,
    SM90ArchSpec::get_ab_load_block_m(config.multicast_config, config.block_m),
    config.block_k,
    static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), 1,
    config.smem_config.swizzle_a_mode, 0, false, 2);
说明：
修改支持参数 gps_tma_copy_size。 此处 gps_tma_copy_size=2




/deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh
line：169～174 修改为
#pragma unroll
for(int i=0; i<2; ++i) {
    tma_copy(&tensor_map_a, reinterpret_cast<uint64_t*>(&full_barrier),
            smem_a[stage_idx]+i*BLOCK_M/2*BLOCK_K, k_idx, scheduler.get_global_idx<kWithGroupOffsetA>(shape_m, BLOCK_M, m_block_idx)+i*BLOCK_M/2,
            num_tma_multicast_a);
}
说明：
注意“i<2”、“i*BLOCK_M/2”，这个 2 需要与 gps_tma_copy_size 保持一致。
通过这个修改，拆分单次TMA Copy为多次，每次Copy数据量减少为原本的 1/gps_tma_copy_size。


Attention:
目前还没有做自动化，因此每次测试需要手动修改 sm90_fp8_gemm_1d2d.hpp 与 sm90_fp8_gemm_1d2d.cuh 中的值，并重新编译。

