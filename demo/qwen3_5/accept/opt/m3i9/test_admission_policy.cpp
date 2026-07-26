// M3-I9 -- host-side test of the MODE_OFFLINE admission arithmetic.
//
// It includes the REAL header the megakernel uses
// (include/mirage/persistent_kernel/admission_policy.h) and calls the REAL
// function both device call sites call, so the thing under test is the changed
// code, not a copy of it. The surrounding loop mirrors `prepare_next_batch`
// steps 1/3 + the admission loop (persistent_kernel.cuh:296-400); it is the
// scaffolding, and `test_admission_policy.py` pins it to the .cuh by asserting
// the call sites use the helper and nothing else clamps.
//
// Emits one line per iteration: "<n_live> <tok0> <tok1> ...", which
// test_admission_policy.py compares against protocol_sim.py -- the replay
// already validated against the profiler's own BEGIN_TASK_GRAPH count at all
// five batch sizes (109/109/109/111/203).
//
// No CUDA, no build system: g++ -std=c++17 -I<include> this file.

#include <cstdio>
#include <cstdlib>
#include <vector>

#ifndef MPK_MAX_NUM_BATCHED_TOKENS
#define MPK_MAX_NUM_BATCHED_TOKENS 16
#endif

#include "mirage/persistent_kernel/admission_policy.h"

int main(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(
        stderr, "usage: %s <max_seq_length> <plen> [plen ...]\n", argv[0]);
    return 2;
  }
  int const msl = std::atoi(argv[1]);
  std::vector<int> plen;
  for (int i = 2; i < argc; i++) {
    plen.push_back(std::atoi(argv[i]));
  }
  int const n_req = static_cast<int>(plen.size());
  int const mbr = n_req; // the wave protocol: total_num_requests == mbr

  std::vector<int> step(n_req, 0);
  std::vector<int> slot_of(mbr, -1); // config.request_ids[]
  int next_request_id = 0;           // *config.next_request_id

  for (int guard = 0; guard < 500000; guard++) {
    // ---- step 3: compact survivors toward slot 0 and fill the budget -------
    int num_reqs = 0, num_tokens = 0;
    std::vector<int> chunk, req;
    for (int i = 0; i < mbr; i++) {
      int const r = slot_of[i];
      if (r == -1) {
        continue;
      }
      int num_new_tokens = plen[r] - step[r];
      if (num_new_tokens > 0) {
        num_new_tokens = mirage::mpk::admission_prefill_tokens(
            num_new_tokens,
            MPK_MAX_NUM_BATCHED_TOKENS - num_tokens,
            MPK_MAX_TOKENS_PER_REQUEST);
      } else {
        num_new_tokens = (1 < MPK_MAX_NUM_BATCHED_TOKENS - num_tokens)
                             ? 1
                             : MPK_MAX_NUM_BATCHED_TOKENS - num_tokens;
      }
      req.push_back(r);
      chunk.push_back(num_new_tokens);
      num_tokens += num_new_tokens;
      num_reqs++;
    }
    // ---- admit new prefill requests until we reach capacity ----------------
    while (num_reqs < mbr && num_tokens < MPK_MAX_NUM_BATCHED_TOKENS &&
           next_request_id < n_req) {
      int const r = next_request_id++;
      int const num_new_tokens = mirage::mpk::admission_prefill_tokens(
          plen[r],
          MPK_MAX_NUM_BATCHED_TOKENS - num_tokens,
          MPK_MAX_TOKENS_PER_REQUEST);
      req.push_back(r);
      chunk.push_back(num_new_tokens);
      num_tokens += num_new_tokens;
      num_reqs++;
    }
    for (int i = 0; i < mbr; i++) {
      slot_of[i] = (i < num_reqs) ? req[i] : -1;
    }
    if (num_tokens == 0) {
      break;
    }
    std::printf("%d", num_reqs);
    for (int i = 0; i < num_reqs; i++) {
      std::printf(" %d", chunk[i]);
    }
    std::printf("\n");
    // ---- step 1 of the next prepare: advance, then retire in place ---------
    for (int i = 0; i < num_reqs; i++) {
      int const r = req[i];
      step[r] += chunk[i];
      if (step[r] + 1 >= msl) {
        slot_of[i] = -1;
      }
    }
  }
  return 0;
}
