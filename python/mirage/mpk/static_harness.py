"""Test harness for a built static megakernel (static_schedule.StaticKernel, any graph). The kernel's launcher only runs it (launch,
wait) and hands out its instrumentation buffers (StaticKernel.read, layout); everything that measures or checks a launch is here.

  Harness(sk)
    flush_l2(nbytes)                 read nbytes on every GPU: L2 emptied and left clean, as before a layer in a real model
    launch(flush_bytes, timeout_s)   flush_l2, one launch, wait; timeout_s > 0: report() and TimeoutError after that long
    timing() / span_us()             per GPU (last SM past the start barrier, latest SM end) of the last launch
    report()                         per GPU the counters per line and how many SMs passed each stamp (also while a launch hangs)
    task_times()                     timing build: per GPU, SM and list entry (start, last warp's end, stage stamps 0 and 1)
    repeat(reps, watch, ...)         reps launches: the span of each, and per watched tensor the values whose bits changed
"""
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch


class Harness:
    def __init__(self, sk):
        self.sk = sk
        self.layout = sk.layout()
        self._flush: List[Optional[torch.Tensor]] = [None] * sk.num_gpus

    def flush_l2(self, nbytes: int) -> None:
        """Read nbytes on every GPU, then wait for every GPU. Read, not written: a write leaves L2 full of dirty lines, and the
        layer's reads would then also pay their write-back to memory, which a layer in a real model does not see."""
        for g in range(self.sk.num_gpus):
            if self._flush[g] is None or self._flush[g].numel() * 4 < nbytes:
                self._flush[g] = torch.zeros(nbytes // 4, dtype=torch.int32, device=f"cuda:{g}")
            torch.amax(self._flush[g][: nbytes // 4])
        for g in range(self.sk.num_gpus):
            torch.cuda.synchronize(g)

    def launch(self, flush_bytes: int = 0, timeout_s: float = 0.0) -> None:
        """flush_bytes > 0: flush_l2 first. One launch on every GPU, then wait for it. timeout_s > 0: a launch not finished after
        that long: report(), then TimeoutError (the kernel is still running). A failed kernel: report(), then RuntimeError."""
        if flush_bytes > 0:
            self.flush_l2(flush_bytes)
        self.sk.launch()
        try:
            if timeout_s <= 0:
                self.sk.wait()
                return
            t0 = time.monotonic()
            while not self.sk.done():
                if time.monotonic() - t0 > timeout_s:
                    raise TimeoutError(f"layer kernel not finished after {timeout_s:g} s")
        except (TimeoutError, RuntimeError):
            self.report()
            raise

    def _array(self, g: int, name: str, dtype) -> np.ndarray:
        return np.frombuffer(self.sk.read(g, name), dtype=dtype)

    def timing(self) -> List[Tuple[int, int]]:
        """Per GPU (last SM past the start barrier, latest SM end) of the last launch, globaltimer ns."""
        lay, out = self.layout, []
        for g in range(self.sk.num_gpus):
            stamps = self._array(g, "stamps", np.int64).reshape(lay["num_sms"], lay["num_stamps"])
            out.append((int(self._array(g, "start_barrier", np.int64)[0]), int(stamps[:, lay["stamp_end"]].max())))
        return out

    def span_us(self) -> float:
        """Max over GPUs of (latest SM end - last SM past the start barrier) of the last launch, in microseconds."""
        return max(end - start for start, end in self.timing()) / 1e3

    def report(self) -> None:
        """Per GPU: the sum of each counter line that is not 0, and how many SMs passed each stamp."""
        lay = self.layout
        for g in range(self.sk.num_gpus):
            try:
                cnt = self._array(g, "counters", np.uint32).reshape(lay["counter_lines"], lay["counters_per_line"])
                stamps = self._array(g, "stamps", np.int64).reshape(lay["num_sms"], lay["num_stamps"])
            except RuntimeError as e:
                print(f"layer GPU {g}: diagnostics not readable ({e})")
                continue
            sums = cnt.sum(axis=1, dtype=np.uint64)
            past = (stamps != 0).sum(axis=0)
            lines = "".join(f" {line}: {int(s)}" for line, s in enumerate(sums) if s)
            print(f"layer GPU {g}: counters (line: sum):{lines} | SMs past: start {past[lay['stamp_start']]}, task stamp 0 "
                  f"{past[lay['stamp_task0']]}, task stamp 1 {past[lay['stamp_task1']]}, end {past[lay['stamp_end']]}", flush=True)

    def task_times(self) -> List[List[List[Tuple[int, int, int, int]]]]:
        """Timing build: [gpu][sm][list entry] = (start, the last warp's end, stage stamp 0, stage stamp 1) of the last launch,
        globaltimer ns (0: no task / no stamp). MoE GEMM tasks: stage 0 = first load issued, 1 = first stage landed."""
        lay = self.layout
        if not lay["profile"]:
            raise RuntimeError("not a timing build (compile_static(profile=True))")
        w, out = lay["warps"], []
        for g in range(self.sk.num_gpus):
            t = self._array(g, "task_times", np.uint64).reshape(lay["num_sms"], lay["max_tasks"], lay["time_slots"])
            entries = np.stack([t[..., 0], t[..., 1:w + 1].max(axis=-1), t[..., w + 1], t[..., w + 2]], axis=-1)
            out.append([[tuple(e) for e in sm] for sm in entries.tolist()])
        return out

    def repeat(self, reps: int, watch: Optional[Dict[str, torch.Tensor]] = None, flush_bytes: int = 0, timeout_s: float = 0.0,
               after_launch: Optional[Callable[[int], None]] = None) -> Tuple[List[float], List[Dict[str, torch.Tensor]]]:
        """reps launches (launch(flush_bytes, timeout_s)). Returns the span of each (span_us) and, per launch, for each watched
        tensor a bool tensor (CPU, its shape): the values whose bits differ from the tensor before the first of these launches
        (all False when every launch gives the same result). after_launch(i): called after launch i (e.g. to read task_times)."""
        ints = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}

        def bits(t: torch.Tensor) -> torch.Tensor:   # a copy on the CPU, its values as integers of the same size
            return t.detach().to("cpu", copy=True).view(ints[t.element_size()])
        watch = watch or {}
        before = {name: bits(t) for name, t in watch.items()}
        spans, changed = [], []
        for i in range(reps):
            self.launch(flush_bytes, timeout_s)
            spans.append(self.span_us())
            if after_launch:
                after_launch(i)
            changed.append({name: bits(t) != before[name] for name, t in watch.items()})
        return spans, changed
