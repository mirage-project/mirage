// clang-format off
#pragma once

#include "../../lib.h"

ADD_TESTCASE(Testcase("reduction_correctness", {"kernel", "correctness"}, "kernel-level reduction correctness test", [](Testcase* this_testcase) {
	Subcase t({
		1,
		0,
		true,
		{
			80
		}
	});
	kn::DTensor i0 = t.new_input({10, 8, 3}, Gen::ARange(-1.0, 1.0));
	kn::DTensor o0 = t.g->reduction(i0, 1, 2);
	t.mark_output(o0, {10, 2, 3}, {-3.850,-3.817,-3.783,-3.450,-3.417,-3.383,-3.050,-3.017,-2.983,-2.650,-2.617,-2.583,-2.250,-2.217,-2.183,-1.850,-1.817,-1.783,-1.450,-1.417,-1.383,-1.050,-1.017,-0.983,-0.650,-0.617,-0.583,-0.250,-0.217,-0.183,0.150,0.183,0.217,0.550,0.583,0.617,0.950,0.983,1.017,1.350,1.383,1.417,1.750,1.783,1.817,2.150,2.183,2.217,2.550,2.583,2.617,2.950,2.983,3.017,3.350,3.383,3.417,3.750,3.783,3.817});
	this_testcase->add_subcase(t);
}));
