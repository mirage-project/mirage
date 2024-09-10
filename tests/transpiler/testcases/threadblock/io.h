// clang-format off
#pragma once

#include "../../lib.h"
#include "mirage/threadblock/smem_tensor.h"

ADD_TESTCASE(Testcase("tb_input_output", {"threadblock", "correctness", "perf"}, "threadblock-level input & output op test", [](Testcase* this_testcase) {
	struct SubcaseConfig {
		string subcase_name;
		int b, m, n;
		int tile_m, tile_n;
		// The shape of the input will be [b, m, n], and the shape of the output will be [b, m, tile_n]. Each threadblock performs "reduction" along one row
		bool is_perf_test;
		bool is_n_innermost;
		bool is_perfect_layout;
		dim3 block_dim;
	};
	vector<SubcaseConfig> input_cases = {
		// Correctness tests
		{
			"normal",
			1, 16, 16,
			8, 8,
			false
		},
		{
			"n-non-divisible",
			1, 16, 18,
			8, 9,
			false
		},
		{
			"m-non-divisible",
			1, 18, 16,
			9, 8,
			false
		},
		{
			"mn-non-divisible",
			1, 18, 18,
			9, 9,
			false
		},
		{
			"random0",
			2, 68, 72,
			17, 6,
			false
		},
		// Performance tests
		{
			"large0",
			16, 4096, 4096,
			128, 128,
			true,
		}
	};
	vector<dim3> block_dim_subcases = {
		{32, 1, 1}, {32, 2, 1}, {128, 1, 1}, {8, 8, 4}
	};
	vector<SubcaseConfig> subcases;
	for (const SubcaseConfig &input_case : input_cases) {
		for (bool is_n_innermost : {true, false}) {
			for (bool is_perfect_layout : {true, false}) {
				if (!is_perfect_layout && input_case.m % 8 == 0 && input_case.n % 8 == 0) {
					continue;
				}
				for (const dim3 &block_dim_subcase : block_dim_subcases) {
					int num_threads = block_dim_subcase.x * block_dim_subcase.y * block_dim_subcase.z;
					SubcaseConfig subcase = input_case;
					subcase.subcase_name += is_n_innermost ? ", N" : ", M";
					subcase.subcase_name += is_perfect_layout ? ",  P" : ", NP";
					subcase.subcase_name += ", " + std::string(3-std::to_string(num_threads).length(), ' ') + std::to_string(num_threads) + " thrs";
					subcase.is_n_innermost = is_n_innermost;
					subcase.is_perfect_layout = is_perfect_layout;
					subcase.block_dim = block_dim_subcase;
					subcases.push_back(subcase);
				}
			}
		}
	}
	for (const SubcaseConfig &subcase : subcases) {
		Subcase t({
			subcase.is_perf_test ? 2 : 1,
			subcase.is_perf_test ? 5 : 0,
			!subcase.is_perf_test,
			{
				80
			}
		}, subcase.subcase_name, subcase.is_perf_test ? optional([=](const Subcase::RunResult &res) -> string {
			size_t mem_read_vol = subcase.b*subcase.m*subcase.n;
			size_t mem_write_vol = subcase.b*subcase.m*subcase.tile_n;
			float mem_bw_gBps = (mem_read_vol + mem_write_vol) * sizeof(half) / res.avg_time_ms / 1e6;
			return "Mem BW (GB/s): " + std::to_string(mem_bw_gBps);
		}) : nullopt);
		// auto [b, m, n] = std::tie(subcase.b, subcase.m, subcase.n);
		// auto [tile_m, tile_n] = std::tie(subcase.tile_m, subcase.tile_n);
		int b = subcase.b, m = subcase.m, n = subcase.n;
		int tile_m = subcase.tile_m, tile_n = subcase.tile_n;
		assert(m % tile_m == 0);
		assert(n % tile_n == 0);

		size_t m_stride = subcase.is_n_innermost ? (subcase.is_perfect_layout ? round_to_multiple(n, 8) : n) : 1;
		size_t n_stride = subcase.is_n_innermost ? 1 : (subcase.is_perfect_layout ? round_to_multiple(m, 8) : m);
		size_t b_stride = subcase.is_perfect_layout ? round_to_multiple(m, 8) * round_to_multiple(n, 8) : m * n;
		vector<size_t> strides = {b_stride, m_stride, n_stride};

		kn::DTensor input = t.new_input({b, m, n}, Gen::ARange(-1.0, 1.0), strides);
		auto [output] = t.add_custom_op<1>({input}, [&](const vector<kn::DTensor> &inputs) -> Subcase::add_custom_op_ret_t<1> {
			dim3 grid_dim = dim3(b, m/tile_m, 1);
			int forloop_range = n / tile_n;
			std::shared_ptr<tb::Graph> sg = std::make_shared<tb::Graph>(grid_dim, subcase.block_dim, forloop_range, 1);
			tb::STensor sinput = sg->new_input(inputs[0], {0, 1, -1}, 2, layout::SmemRowMajor);
			tb::STensor soutput = sg->forloop_accum(sinput, type::TB_FORLOOP_ACCUM_NO_RED_OP);
			kn::DTensor output = sg->mark_output(soutput, {0, 1, -1}, -1, type::TB_EPILOGUE_NONE);
			return {sg, {output}};
		});

		t.mark_output(output, {b, m, tile_n}, subcase.is_perf_test ? vector<half>{} : [&]() {
			vector<half> input = Gen::ARange()({b, m, n});
			vector<half> output(b*m*tile_n);
			for (int b_i = 0; b_i < b; b_i++) {
				for (int m_i = 0; m_i < m; m_i++) {
					for (int n_i = 0; n_i < tile_n; n_i++) {
						half ans = 0;
						for (int n_offset = n_i; n_offset < n; n_offset += tile_n) {
							ans += input[b_i*m*n + m_i*n + n_offset];
						}
						output[b_i*m*tile_n + m_i*tile_n + n_i] = ans;
					}
				}
			}
			return output;
		}());
		this_testcase->add_subcase(t);
	}
}));
