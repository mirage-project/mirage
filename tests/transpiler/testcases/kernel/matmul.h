// clang-format off
#pragma once

#include "../../lib.h"

ADD_TESTCASE(Testcase("matmul_correctness", {"kernel", "correctness"}, "kernel-level matmul correctness test", [](Testcase* this_testcase) {
	Subcase t({
		1,
		0,
		true,
		{
			80
		}
	});
	kn::DTensor i0 = t.new_input({3, 6, 9}, Gen::ARange(-1.0, 1.0));
	kn::DTensor i1 = t.new_input({3, 9, 4}, Gen::ARange(1.0, 2.0));
	kn::DTensor o0 = t.g->matmul(i0, i1);
	t.mark_output(o0, {3, 6, 4}, {-9.798,-9.873,-9.955,-10.032,-8.649,-8.716,-8.788,-8.856,-7.501,-7.558,-7.621,-7.680,-6.353,-6.402,-6.455,-6.505,-5.205,-5.245,-5.289,-5.330,-4.056,-4.087,-4.121,-4.153,-3.761,-3.784,-3.807,-3.832,-2.278,-2.293,-2.307,-2.322,-0.797,-0.803,-0.808,-0.813,0.684,0.688,0.692,0.697,2.167,2.181,2.194,2.208,3.648,3.670,3.693,3.717,6.275,6.307,6.341,6.372,8.091,8.133,8.176,8.216,9.906,9.957,10.011,10.059,11.719,11.780,11.843,11.901,13.534,13.604,13.677,13.744,15.348,15.428,15.511,15.587});
	this_testcase->add_subcase(t);
}));

ADD_TESTCASE(Testcase("matmul_perf", {"kernel", "perf"}, "kernel-level matmul performance test", [](Testcase* this_testcase) {
	const int batch_size = 1;
	const int m = 4096;
	const int n = 4096;
	const int k = 1024;
	auto epilogue = [=](const Subcase::RunResult &res) -> string {
		return "GFLOPS: " + std::to_string((double)batch_size*m*n*k*2 / res.avg_time_ms / 1e6);
	};
	Subcase t({
		2,
		10,
		false,
		{
			80
		}
	}, nullopt, epilogue);
	kn::DTensor i0 = t.new_input({batch_size, m, k}, Gen::ARange());
	kn::DTensor i1 = t.new_input({batch_size, k, n}, Gen::ARange());
	kn::DTensor o0 = t.g->matmul(i0, i1);
	t.mark_output(o0, {batch_size, m, n}, {});
	this_testcase->add_subcase(t);
}));
