"""CPU-side prefix KV-cache for the Mirage engine.

Design
------
Pages are the unit of KV-cache sharing.  A *page* holds page_size consecutive
tokens' worth of KV data.  Only complete pages are cached; trailing partial
pages are always recomputed.

Lifecycle per generate() call
------------------------------
1. LLMEngine.generate() calls manager.admit() for every request.
   admit() calls PrefixCache.lookup() → (prefix_len, page_indices).
   prefix_len is written into step[token_row] so the kernel skips those tokens.
   page_indices are written into paged_kv_indices_buffer so the kernel reuses
   the cached KV data.

2. After cuda.synchronize(), LLMEngine reads final page assignments from
   final_paged_kv_indices / final_paged_kv_num_pages tensors (written by the
   kernel on request completion) and calls manager.register_completed_pages()
   to insert them into the cache.

Kernel patches required
-----------------------
Three changes to persistent_kernel.cuh (all in MODE_OFFLINE / init path):

  P1. Externalize scheduling tensors — use Python-provided pointers instead of
      gpu_malloc for: request_ids, next_request_id, page_queue,
      page_queue_head, page_queue_tail.  Remove their gpu_free() calls from
      finalize_persistent_kernel().

  P2. New-request admission: read initial_step = config.step[next_request_id]
      and start token copying from that offset:
          int initial_step = config.step[next_request_id];
          int remaining = config.prompt_length[next_request_id] - initial_step;
          int num_new_tokens = min(remaining, budget);
          for j: input_tokens[...] = tokens[request_id*MAX_SEQ + initial_step + j]
          // prefix pages already in paged_kv_indices_buffer (written by Python)
          int num_prefix_pages = (initial_step + PAGE_SIZE - 1) / PAGE_SIZE;
          // copy prefix pages (pre-populated by Python into paged_kv_indices_buffer)
          for j < num_prefix_pages: paged_kv_indices_buffer[num_pages+j] = existing[j]
          // allocate new pages for suffix
          int num_new_pages = ceil((initial_step+num_new_tokens)/PAGE_SIZE)-num_prefix_pages;
          ...

  P3. On request completion (when request_ids[i] set to -1), write final pages:
          int n = paged_kv_indptr_buffer[i+1] - paged_kv_indptr_buffer[i];
          config.final_paged_kv_num_pages[request_id] = n;
          for j < n:
              config.final_paged_kv_indices[request_id * MAX_PAGES_PER_REQ + j]
                  = smem_kv_indices[paged_kv_indptr_buffer[i] + j];
"""

from __future__ import annotations
from collections import OrderedDict


class PagePool:
    """CPU-side tracker of free and prefix-pinned GPU page indices.

    After init_func resets the GPU page_queue to [0..max_pages-1], Python
    must overwrite it with only the free pages (excluding those pinned by the
    prefix cache).  PagePool tracks this on the CPU side.
    """

    def __init__(self, max_num_pages: int) -> None:
        self.max_num_pages = max_num_pages
        self._free: list[int] = list(range(max_num_pages))
        self._pinned: set[int] = set()

    def allocate(self, n: int) -> list[int]:
        if len(self._free) < n:
            raise RuntimeError(
                f"PagePool exhausted: need {n} pages, have {len(self._free)}"
            )
        pages, self._free = self._free[:n], self._free[n:]
        return pages

    def free(self, pages: list[int]) -> None:
        for p in pages:
            self._pinned.discard(p)
            self._free.append(p)

    def pin(self, pages: list[int]) -> None:
        """Remove pages from the allocatable pool (held by prefix cache)."""
        for p in pages:
            if p in self._free:
                self._free.remove(p)
            self._pinned.add(p)

    def free_queue(self) -> list[int]:
        """Ordered free-page list for writing into the kernel's page_queue tensor."""
        return list(self._free)

    @property
    def num_free(self) -> int:
        return len(self._free)


class PrefixCache:
    """LRU cache mapping aligned token prefix → GPU page indices.

    Only full page_size-aligned prefixes are cached.  LRU eviction frees the
    pinned pages back to the PagePool when the cache is full.
    """

    def __init__(
        self,
        page_pool: PagePool,
        page_size: int,
        max_entries: int = 128,
    ) -> None:
        self.page_pool = page_pool
        self.page_size = page_size
        self.max_entries = max_entries
        # LRU order: least-recently-used at front
        self._store: OrderedDict[tuple[int, ...], list[int]] = OrderedDict()

    # ── Public API ────────────────────────────────────────────────────────────

    def lookup(self, token_ids: list[int]) -> tuple[int, list[int]]:
        """Return (prefix_len, page_indices) for the longest cached prefix.

        prefix_len is always a multiple of page_size.  Returns (0, []) on miss.
        """
        n = (len(token_ids) // self.page_size) * self.page_size
        while n > 0:
            key = tuple(token_ids[:n])
            if key in self._store:
                self._store.move_to_end(key)   # refresh LRU position
                return n, list(self._store[key])
            n -= self.page_size
        return 0, []

    def insert(self, token_ids: list[int], page_indices: list[int]) -> None:
        """Cache the KV pages for a completed request's full-block prefix.

        Only the first floor(len(token_ids)/page_size) pages are cached;
        the trailing partial block is intentionally excluded.
        """
        n = (len(token_ids) // self.page_size) * self.page_size
        if n == 0:
            return
        key = tuple(token_ids[:n])
        if key in self._store:
            return  # already cached
        num_full_blocks = n // self.page_size
        pages = list(page_indices[:num_full_blocks])
        # Evict LRU entry if at capacity
        while len(self._store) >= self.max_entries:
            _, evicted = self._store.popitem(last=False)
            self.page_pool.free(evicted)
        self._store[key] = pages
        self.page_pool.pin(pages)

    @property
    def num_entries(self) -> int:
        return len(self._store)
