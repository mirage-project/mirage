// clang-format off
#pragma once

#include "../../lib.h"

ADD_TESTCASE(Testcase("tb_matmul", {"threadblock", "correctness", "perf"}, "threadblock-level matmul op test", [](Testcase* this_testcase) {
	struct SubcaseConfig {
		string subcase_name;
		std::tuple<int, int, int> mnk;
		std::tuple<int, int, int> smem_mnk;
		bool is_perf_test;
		bool is_A_k_innermost;
		bool is_B_k_innermost;
		dim3 block_dim;
	};
	vector<SubcaseConfig> shape_subcases = {
		// Correctness tests
		{
			"16x8x16 + 16x8x16",
			{16, 8, 16},
			{16, 8, 16},
			false
		},
		{
			"64x64x64 + 16x8x16",
			{64, 64, 64},
			{16, 8, 16},
			false
		},
		{
			"256x256x64 + 32x32x32",
			{256, 256, 64},
			{32, 32, 32},
			false
		},
		{
			"128x128x64 + 128x8x32",
			{128, 128, 64},
			{128, 8, 32},
			false
		},
		{
			"128x128x64 + 16x128x32",
			{128, 128, 64},
			{16, 128, 32},
			false
		},
		{
			// Really small matrix, should trigger SM80_16x8x8 & oob protection
			"15x15x24 + 5x5x8",
			{15, 15, 24},
			{5, 5, 8},
			false
		},
		{
			// Really small matrix, should trigger SM80_16x8x8 & oob protection & K-0-padding
			"15x15x15 + 5x5x5",
			{15, 15, 15},
			{5, 5, 5},
			false
		},
		{
			// Should trigger oob protection
			"96x96x64 + 24x24x32",
			{96, 96, 64},
			{24, 24, 32},
			false
		},
		{
			// Should trigger oob protection & K-0-padding
			"96x96x57 + 24x24x19",
			{96, 96, 57},
			{24, 24, 19},
			false
		},
		{
			// Should trigger oob protection & K-0-padding
			"96x96x63 + 24x24x21",
			{96, 96, 63},
			{24, 24, 21},
			false
		},
		{
			// Large matrix, should trigger oob protection & K-0-padding
			"405x435x63 + 27x29x21",
			{405, 435, 63},
			{27, 29, 21},
			false
		},
		{
			// Large matrix, large block
			"381x504x62 + 127x126x31",
			{381, 504, 62},
			{127, 126, 31},
			false
		},
		{
			// Should trigger shift swizzle
			"92x94x64 + 46x47x32",
			{92, 94, 64},
			{46, 47, 32},
			false
		},

		// Performance tests
		{
			"4096x4096x1024 + 128x128x32",
			{4096, 4096, 1024},
			{128, 128, 32},
			true
		},
	};
	vector<pair<bool, bool>> layout_subcases = {
		{true, true}, {true, false}, {false, true}, {false, false}
	};
	vector<dim3> block_dim_subcases = {
		{32, 1, 1}, {32, 2, 1}, {128, 1, 1}, {8, 8, 4}
	};
	vector<SubcaseConfig> subcases;
	for (const SubcaseConfig &shape_subcase : shape_subcases)
		for (const auto &[is_A_k_innermost, is_B_k_innermost] : layout_subcases)
			for (const dim3 &block_dim_subcase : block_dim_subcases) {
				int num_threads = block_dim_subcase.x * block_dim_subcase.y * block_dim_subcase.z;
				SubcaseConfig subcase {
					shape_subcase.subcase_name + ", " + "MK"[is_A_k_innermost] + "NK"[is_B_k_innermost] + ", " + std::string(3-std::to_string(num_threads).length(), ' ') + std::to_string(num_threads) + " thrs",
					shape_subcase.mnk,
					shape_subcase.smem_mnk,
					shape_subcase.is_perf_test,
					is_A_k_innermost,
					is_B_k_innermost,
					block_dim_subcase
				};
				subcases.push_back(subcase);
			}
	for (const SubcaseConfig &subcase : subcases) {
		auto [m, n, k] = subcase.mnk;
		auto [tile_m, tile_n, tile_k] = subcase.smem_mnk;
		assert(m%tile_m == 0);
		assert(n%tile_n == 0);
		assert(k%tile_k == 0);
		Subcase t({
			subcase.is_perf_test ? 2 : 1,
			subcase.is_perf_test ? 4 : 0,
			!subcase.is_perf_test,
			{
				80
			}
		}, subcase.subcase_name, subcase.is_perf_test ? optional([=](const Subcase::RunResult &res) -> string {
			float gflops = (float)m*n*k*2 / res.avg_time_ms / 1e6;
			float mem_read_vol = ((float)m*k*(n/tile_n) + (float)k*n*(m/tile_m)) * sizeof(half);
			float mem_write_vol = (float)m*n * sizeof(half);
			float mem_bw_gBps = (mem_read_vol + mem_write_vol) / res.avg_time_ms / 1e6;
			return "GFLOPS: " + std::to_string(gflops) + ", Mem BW (GB/s): " + std::to_string(mem_bw_gBps);
		}) : nullopt);
		size_t m_r8 = round_to_multiple(m, 8);
		size_t n_r8 = round_to_multiple(n, 8);
		size_t k_r8 = round_to_multiple(k, 8);
		vector<size_t> layout_A = subcase.is_A_k_innermost ? vector<size_t>{k_r8, 1} : vector<size_t>{1, m_r8};
		vector<size_t> layout_B = subcase.is_B_k_innermost ? vector<size_t>{1, k_r8} : vector<size_t>{n_r8, 1};
		kn::DTensor i0 = t.new_input({m, k}, Gen::ARange(), layout_A);
		kn::DTensor i1 = t.new_input({k, n}, Gen::ARange(), layout_B);
		auto [o0] = t.add_custom_op<1>({i0, i1}, [&](const vector<kn::DTensor> &inputs) -> Subcase::add_custom_op_ret_t<1> {
			dim3 grid_shape = dim3(m / tile_m, n / tile_n, 1);
			std::shared_ptr<tb::Graph> sg = std::make_shared<tb::Graph>(grid_shape, subcase.block_dim, k / tile_k, 1);
			tb::STensor si0 = sg->new_input(inputs[0], {0, -1, -1}, 1, layout::SmemRowMajor);
			tb::STensor si1 = sg->new_input(inputs[1], {-1, 1, -1}, 0, layout::SmemRowMajor);
			tb::STensor sx0 = sg->matmul(si0, si1);
			tb::STensor so0 = sg->forloop_accum(sx0, type::TB_FORLOOP_ACCUM_NO_RED_OP);
			kn::DTensor o0 = sg->mark_output(so0, {0, 1, -1}, -1, type::TB_EPILOGUE_NONE);
			return {sg, {o0}};
		});
		t.mark_output(o0, {m, n}, subcase.is_perf_test ? vector<half>{} : [m, n, k]() {
			auto half_vec2float_vec = [](const vector<half> &half_vec) -> vector<float> {
				vector<float> float_vec(half_vec.size());
				for (size_t i = 0; i < half_vec.size(); ++i) {
					float_vec[i] = (float)half_vec[i];
				}
				return float_vec;
			};
			vector<float> A = half_vec2float_vec(Gen::ARange()({m, k}));
			vector<float> B = half_vec2float_vec(Gen::ARange()({k, n}));
			vector<half> C(m*n);
			for (int i = 0; i < m; ++i)
				for (int j = 0; j < n; ++j) {
					float sum = 0.0;
					for (int l = 0; l < k; ++l) {
						sum += A[i*k+l] * B[l*n+j];
					}
					C[i*n+j] = (half)sum;
				}
			return C;
		}());
		this_testcase->add_subcase(t);
	}
}));
