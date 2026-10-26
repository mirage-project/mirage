"""The megakernel loop (docs/megakernel_dev_loop.md): the code that is the same for every model part.

  part.py      a part: its part.yaml and its hooks module (load the tensors, build the graph, check the outputs)
  run.py       the generic harness run of one candidate: plan -> schedules -> layer.cu -> check -> timed launches (-> cost records)
  candidate.py a candidate: (template variants, split, placement, post-processing choices) as one file
  lints.py     the automatic checks (contract, source, build, SASS, plan, records)
  model.py     cost records with context, the model's prediction of a plan, its error per phase
  explore.py   plans for measuring: random plans, perturbations of a plan
  triage.py    the critical path of a timed launch and the attribution of its time to classes
  passes.py    the post-processing passes (apply / features / constraints / estimate) and their ablations
  driver.py    the round: state.json, ledger.jsonl, candidate directories, the ladder, promotion
  box.py       the GPU box: sync with md5 proof, detached jobs, a waiter that stops on the first lost connection

Nothing here imports the compiled Mirage extension at import time, so the loop's logic runs on a machine without a GPU.
"""
