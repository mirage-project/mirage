// clang-format off
#pragma once

#include "../../lib.h"
#include "mirage/threadblock/smem_tensor.h"
#include "mirage/type.h"

ADD_TESTCASE(Testcase("tb_elemwise_bcast_correctness", {"threadblock", "correctness"}, "threadblock-level elementwise op broadcasting correctness test", [](Testcase* this_testcase) {
	struct SubcaseConfig {
		vector<int> shape0;
		vector<int> shape1;
		bool is_0_row_major;
		bool is_1_row_major;
		string name;
	};
	vector<SubcaseConfig> shape_cases = {
		{
			{12, 15},
			{12, 15}
		},
		{
			{1, 15},
			{12, 15}
		},
		{
			{12, 1},
			{12, 15}
		},
		{
			{1, 15},
			{12, 1}
		},
		{
			{12, 1},
			{1, 15}
		}
	};
	vector<SubcaseConfig> subcases;
	for (SubcaseConfig const& subcase_config : shape_cases) {
		for (bool is_0_row_major : vector<bool>{false, true}) {
			for (bool is_1_row_major : vector<bool>{false, true}) {
				subcases.push_back({
					subcase_config.shape0,
					subcase_config.shape1,
					is_0_row_major,
					is_1_row_major,
					std::to_string(subcase_config.shape0[0]) + "x" + std::to_string(subcase_config.shape0[1]) + ", " + std::to_string(subcase_config.shape1[0]) + "x" + std::to_string(subcase_config.shape1[1]) + ", " + "CR"[is_0_row_major] + "CR"[is_1_row_major]
				});
			}
		}
	}
	for (const SubcaseConfig &subcase : subcases) {
		Subcase t({
			1,
			0,
			true,
			{
				80
			}
		}, subcase.name);
		size_t num_dims = subcase.shape0.size();
		assert(subcase.shape0.size() == num_dims);
		assert(subcase.shape1.size() == num_dims);
		vector<int> output_shape(num_dims);
		for (size_t i = 0; i < num_dims; i++) {
			int dim0 = subcase.shape0[i];
			int dim1 = subcase.shape1[i];
			if (dim0 != dim1) {
				assert(dim0 == 1 || dim1 == 1);
			}
			output_shape[i] = std::max(dim0, dim1);
		}

		auto get_layout = [](vector<int> shape, bool is_row_major) {
			assert(shape.size() == 2);
			if (is_row_major) {
				return vector<size_t>{(size_t)shape[1], 1};
			} else {
				return vector<size_t>{1, (size_t)shape[0]};
			}
		};
		kn::DTensor i0 = t.new_input(subcase.shape0, Gen::ARange(), get_layout(subcase.shape0, subcase.is_0_row_major));
		kn::DTensor i1 = t.new_input(subcase.shape1, Gen::ARange(), get_layout(subcase.shape1, subcase.is_1_row_major));

		auto [o0] = t.add_custom_op<1>({i0, i1}, [&](const vector<kn::DTensor> &inputs) -> Subcase::add_custom_op_ret_t<1> {
			std::shared_ptr<tb::Graph> sg = std::make_shared<tb::Graph>(dim3(1, 1, 1), dim3(128, 1, 1), 1, 1);
			tb::STensor si0 = sg->new_input(inputs[0], {-1, -1}, -1, layout::SmemRowMajor);
			tb::STensor si1 = sg->new_input(inputs[1], {-1, -1}, -1, layout::SmemRowMajor);
			tb::STensor sx0 = sg->add(si0, si1);
			tb::STensor so0 = sg->forloop_accum(sx0, type::TB_FORLOOP_ACCUM_NO_RED_OP);
			kn::DTensor o0 = sg->mark_output(so0, {-1, -1}, -1, type::TB_EPILOGUE_NONE);
			return {sg, {o0}};
		});

		t.mark_output(o0, output_shape, [&]() {
			assert(num_dims == 2);
			vector<half> input0 = Gen::ARange()(subcase.shape0);
			vector<half> input1 = Gen::ARange()(subcase.shape1);
			vector<half> output(output_shape[0] * output_shape[1]);
			for (int i = 0; i < output_shape[0]; i++) {
				for (int j = 0; j < output_shape[1]; j++) {
					int input0_i = subcase.shape0[0] == 1 ? 0 : i;
					int input0_j = subcase.shape0[1] == 1 ? 0 : j;
					int input1_i = subcase.shape1[0] == 1 ? 0 : i;
					int input1_j = subcase.shape1[1] == 1 ? 0 : j;
					output[i * output_shape[1] + j] = input0[input0_i * subcase.shape0[1] + input0_j] + input1[input1_i * subcase.shape1[1] + input1_j];
				}
			}
			return output;
		}());
		this_testcase->add_subcase(t);
	}
}));
