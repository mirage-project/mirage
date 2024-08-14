#include "mirage/transpiler/structs.h"
#include "mirage/transpiler/transpiler.h"

#include <unordered_set>
#include <functional>

namespace mirage {
namespace transpiler {

TBMemoryPlan Transpiler::get_threadblock_memory_plan(tb::Graph const& tb_graph, const TBSched& tb_sched) {
	// Currently we use a simple allocation-only strategy. In the future we may
	// incorporate more advanced strategies like memory reuse, etc.
	static constexpr size_t ALIGN = 16;
  class MemoryPlanner {
  private:
    size_t cur_addr = 0;

  public:
    // size is in bytes
    size_t allocate(size_t size) {
      size_t addr = cur_addr;
      assert(addr % ALIGN == 0);
      cur_addr += size;
      cur_addr = round_to_multiple(cur_addr, ALIGN);
      return addr;
    }

    // Get the needed size of the buffer
    size_t get_buf_size() {
      return cur_addr;
    }
  };
	TBMemoryPlan plan;
	MemoryPlanner planner;

	planner.allocate(16);	// The first 16 bytes will be filled with zero as required by matmul
	std::unordered_set<sguid_t> processed_sguids;
	for (tb::TBOperator const* op : tb_graph.operators) {
		if (is_fused_with_next[op]) {
			// Skip the output tensors of the fused operators
			continue;
		}
		for (tb::STensor const& stensor : op->output_tensors) {
			sguid_t guid = stensor.guid;
			if (processed_sguids.count(guid) != 0) {
				// Allocated before, skip it
				continue;
			}
			processed_sguids.insert(guid);
			STensorMeta& meta = stensor_metas[guid];
			size_t phy_size = meta.num_phy_elems * type::get_datatype_size(stensor.data_type);
			size_t addr = planner.allocate(phy_size);
			plan.addrs[guid] = addr;
		}
	}

	plan.smem_size = planner.get_buf_size();
	return plan;
}

}
}

