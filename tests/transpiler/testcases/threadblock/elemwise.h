// clang-format off
#pragma once

#include "../../lib.h"
#include "mirage/threadblock/smem_tensor.h"

ADD_TESTCASE(Testcase("tb_elemwise_correctness", {"threadblock", "correctness"}, "threadblock-level elementwise op correctness test", [](Testcase* this_testcase) {
	struct SubcaseConfig {
		string subcase_name;
		vector<int> dims;
		vector<size_t> layout_i0, layout_i1;
		dim3 grid_dim, block_dim;
		int3 io_map;
		int forloop_dim;
		int forloop_range;
	};
	vector<int> dims = {10, 12, 14, 16};
	vector<SubcaseConfig> input_cases = {
		{
			"4D row major",
			dims,
			{3315, 255, 17, 1}, {4096, 256, 16, 1}
		},
		{
			"4D col major",
			dims,
			{1, 10, 131, 2000}, {1, 16, 256, 4096}
		},
		{
			"4D mixed",
			dims,
			{1, 16, 256, 4096}, {3315, 255, 17, 1}
		}
	};
	vector<SubcaseConfig> partition_cases = {
		{
			"no forloop, 2d grid",
			{}, {}, {},
			dim3(dims[0], dims[1], 1),
			dim3(2, 3, 4),
			{0, 1, -1},
			-1, 1
		},
		{
			"no forloop, 3d grid 1",
			{}, {}, {},
			dim3(dims[0], dims[1], dims[2]),
			dim3(2, 3, 4),
			{0, 1, 2},
			-1, 1
		},
		{
			"no forloop, 3d grid 2",
			{}, {}, {},
			dim3(dims[0], dims[1], dims[2]/7),
			dim3(2, 3, 4),
			{0, 1, 2},
			-1, 1	// The forloop dim can be anything since forloop_range is 1
		},
		{
			"no forloop, 3d grid 3",
			{}, {}, {},
			dim3(dims[3]/4, dims[1], dims[0]),
			dim3(3, 1, 1),
			{3, 1, 0},
			-1, 1
		},
		// Commented out because of currently Mirage does support output forloop
		// dim other than -1
		// {
		// 	"forloop, 2d grid",
		// 	{}, {}, {},
		// 	dim3(dims[0], dims[1], 1),
		// 	dim3(2, 3, 4),
		// 	{0, 1, -1},
		// 	3, 4
		// },
		// {
		// 	"forloop, 3d grid 1",
		// 	{}, {}, {},
		// 	dim3(dims[0], dims[1], dims[2]),
		// 	dim3(2, 3, 4),
		// 	{0, 1, 2},
		// 	3, 4
		// },
		// {
		// 	"forloop, 3d grid 2",
		// 	{}, {}, {},
		// 	dim3(dims[0], dims[1], dims[2]/7),
		// 	dim3(2, 3, 4),
		// 	{0, 1, 2},
		// 	3, 4
		// },
		// {
		// 	"forloop, 3d grid 3",
		// 	{}, {}, {},
		// 	dim3(dims[3]/4, dims[1], dims[0]),
		// 	dim3(3, 1, 1),
		// 	{3, 1, 0},
		// 	2, 7
		// }
	};
	vector<SubcaseConfig> subcases;
	for (const SubcaseConfig &input_case : input_cases)
		for (const SubcaseConfig &partition_case : partition_cases) {
			SubcaseConfig subcase {
				input_case.subcase_name + " x " + partition_case.subcase_name,
				input_case.dims,
				input_case.layout_i0, input_case.layout_i1,
				partition_case.grid_dim, partition_case.block_dim,
				partition_case.io_map,
				partition_case.forloop_dim, partition_case.forloop_range
			};
			subcases.push_back(subcase);
		}
	subcases.push_back({
		"Large 3D mixed, no forloop, 2d grid",
		{100, 120, 140},
		{1, 100, 12000}, {18200, 140, 1},
		dim3(100, 3, 1),
		dim3(128, 1, 1),
		{0, 1, -1},
		-1, 1
	});
	for (const SubcaseConfig &subcase : subcases) {
		Subcase t({
			1,
			0,
			true,
			{
				80
			}
		}, subcase.subcase_name);
		kn::DTensor i0 = t.new_input(subcase.dims, Gen::ARange(-1.0, 1.0), subcase.layout_i0);
		kn::DTensor i1 = t.new_input(subcase.dims, Gen::ARange(1.0, 2.0), subcase.layout_i1);
		auto [o0] = t.add_custom_op<1>({i0, i1}, [&](const vector<kn::DTensor> &inputs) -> Subcase::add_custom_op_ret_t<1> {
			std::shared_ptr<tb::Graph> sg = std::make_shared<tb::Graph>(subcase.grid_dim, subcase.block_dim, subcase.forloop_range, 1);
			tb::STensor si0 = sg->new_input(inputs[0], subcase.io_map, subcase.forloop_dim, layout::SmemRowMajor);
			tb::STensor si1 = sg->new_input(inputs[1], subcase.io_map, subcase.forloop_dim, layout::SmemRowMajor);
			tb::STensor sx0 = sg->add(si0, si1);
			tb::STensor sx1 = sg->div(sx0, si1);
			tb::STensor sx2 = sg->exp(sx1);
			tb::STensor so0 = sg->forloop_accum(sx2, type::TB_FORLOOP_ACCUM_NO_RED_OP);
			kn::DTensor o0 = sg->mark_output(so0, subcase.io_map, subcase.forloop_dim, type::TB_EPILOGUE_NONE);
			return {sg, {o0}};
		});

		t.mark_output(o0, subcase.dims, [&subcase]() {
			size_t numel = 1;
			for (int dim : subcase.dims) numel *= dim;
			float step0 = 2.0 / (float)numel;
			float step1 = 1.0 / (float)numel;
			vector<half> ret(numel);
			for (size_t i = 0; i < numel; ++i) {
				float i0 = (-1.0 + (float)i * step0);
				float i1 = (1.0 + (float)i * step1);
				float o0 = exp((i0 + i1) / i1);
				ret[i] = (half)o0;
			}
			return ret;
		}());
		this_testcase->add_subcase(t);
	}
}));
