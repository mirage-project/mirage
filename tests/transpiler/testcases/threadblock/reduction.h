// clang-format off
#pragma once

#include "../../lib.h"

ADD_TESTCASE(Testcase("tb_reduction_correctness", {"threadblock", "correctness"}, "threadblock-level reduction op correctness test", [](Testcase* this_testcase) {
	auto dimvec2str = [&](const vector<int> &dims) {
		// Convert a vector like {a, b, c} to "axbxc"
		string ret = "";
		for (int i = 0; i < (int)dims.size(); ++i) {
			ret += std::to_string(dims[i]);
			if (i + 1 < (int)dims.size()) {
				ret += "x";
			}
		}
		return ret;
	};

	struct SubcaseConfig {
		string subcase_name;
		vector<int> src_dims;
		vector<int> dst_dims;
		int reduction_dim;
		int reduction_dimx;	// The size of the dim after reduction
		dim3 block_dim;
	};
	vector<SubcaseConfig> subcases;

	vector<int> src_dims = {12, 24};	// Currently Mirage core only support 2D STensor
	for (int reduction_dim = 0; reduction_dim < (int)src_dims.size(); ++reduction_dim) {
		for (int reduction_dimx : vector<int>{1, 2, 4, 6}) {
			for (dim3 block_dim : vector<dim3>{{32, 1, 1}, {16, 4, 2}}) {
				vector<int> dst_dims = src_dims;
				dst_dims[reduction_dim] = reduction_dimx;
				int num_threads = block_dim.x * block_dim.y * block_dim.z;
				string subcase_name = dimvec2str(src_dims) + " -> " + dimvec2str(dst_dims) + ", " + std::to_string(num_threads) + " thrs";
				subcases.push_back({
					subcase_name,
					src_dims,
					dst_dims,
					reduction_dim,
					reduction_dimx,
					block_dim
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
		}, subcase.subcase_name);
		kn::DTensor i0 = t.new_input(subcase.src_dims, Gen::ARange());
		auto [o0] = t.add_custom_op<1>({i0}, [&](const vector<kn::DTensor> &inputs) -> Subcase::add_custom_op_ret_t<1> {
			dim3 grid_shape = dim3(1, 1, 1);
			std::shared_ptr<tb::Graph> sg = std::make_shared<tb::Graph>(grid_shape, subcase.block_dim, 1, subcase.reduction_dimx);
			tb::STensor si0 = sg->new_input(inputs[0], {-1, -1, -1}, -1, layout::SmemRowMajor);
			tb::STensor sx0 = subcase.reduction_dimx == 1 ? sg->reduction(si0, subcase.reduction_dim) : sg->reduction_to_dimx(si0, subcase.reduction_dim);
			tb::STensor so0 = sg->forloop_accum(sx0, type::TB_FORLOOP_ACCUM_NO_RED_OP);
			kn::DTensor o0 = sg->mark_output(so0, {-1, -1, -1}, -1, type::TB_EPILOGUE_NONE);
			return {sg, {o0}};
		});
		t.mark_output(o0, subcase.dst_dims, [=]() {
			auto get_logical_coord = [&](const vector<int> &dims, int idx) {
				vector<int> ret(dims.size());
				for (int i = (int)dims.size() - 1; i >= 0; --i) {
					ret[i] = idx % dims[i];
					idx /= dims[i];
				}
				return ret;
			};
			auto get_logical_offset = [&](const vector<int> &dims, const vector<int> &coord) {
				int ret = 0;
				for (int i = 0; i < (int)dims.size(); ++i) {
					ret = ret * dims[i] + coord[i];
				}
				return ret;
			};
			int dst_numel = 1;
			for (int i = 0; i < (int)subcase.dst_dims.size(); ++i) {
				dst_numel *= subcase.dst_dims[i];
			}
			int reduc_dim = subcase.reduction_dim;
			int reduction_factor = subcase.src_dims[reduc_dim] / subcase.dst_dims[reduc_dim];
			vector<half> input = Gen::ARange()(subcase.src_dims);
			vector<half> output(dst_numel);
			int reduction_dim_stride = 1;
			for (int i = reduc_dim + 1; i < (int)subcase.src_dims.size(); ++i) {
				reduction_dim_stride *= subcase.src_dims[i];
			}
			for (int offset_dst = 0; offset_dst < dst_numel; ++offset_dst) {
				vector<int> logical_coord = get_logical_coord(subcase.dst_dims, offset_dst);
				logical_coord[reduc_dim] *= reduction_factor;
				int offset_src = get_logical_offset(subcase.src_dims, logical_coord);
				half res = (half)0.0f;
				for (int j = 0; j < reduction_factor; ++j) {
					res += input[offset_src + j * reduction_dim_stride];
				}
				output[offset_dst] = res;
			}
			return output;
		}());
		this_testcase->add_subcase(t);
	}
}));
