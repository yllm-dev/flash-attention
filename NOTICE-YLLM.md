# NOTICE (yLLM modifications)

This repository is jasl's fork of FlashAttention (Dao-AILab/flash-attention, BSD 3-Clause,
see `LICENSE`), maintained under github.com/yllm-dev for the yLLM inference engine. The `yllm`
branch is rebased periodically onto the upstream's `main`; every file it changes carries a
`Modified by yLLM:` line under the original header naming the change and its source.

## Paged KV pages of 16 tokens in the FlashAttention-2 varlen / kvcache forward

Files: `csrc/flash_attn/src/utils.h`, `csrc/flash_attn/src/kernel_traits.h`,
`csrc/flash_attn/src/flash_fwd_kernel.h`, `csrc/flash_attn/flash_api.cpp`.

The upstream's split-KV forward walks one `kBlockN` key tile inside one KV page, so the paged
path requires `page_block_size % 256 == 0`. The port lets a KV page be smaller than the tile:
each thread loads a contiguous row tile of K/V (`kGmemRowsPerThread` rows, the `*Paged` tiled
copies in `kernel_traits.h`) and `flash::resolve_thread_kv_page_slice_offset` (`utils.h`) maps
(thread, n_block) to the page and the row inside it through `block_table`, the final partial tile
bounded by `final_block_size`; `mha_varlen_fwd` and `mha_fwd_kvcache` check `page_block_size % 16 == 0`.

Source: vLLM's fork of FlashAttention, https://github.com/vllm-project/flash-attention
- `90eacc1af2a7c3de62ea249e929ed5faccf38954` "vllm-sqaushed-changes + fa3 building"
  (Lucas Wilkinson, 2025-01-06): the resolver, the paged tiled copies, the kernel sites.
- `720c94869cf2e0ff5a706e9c7f1dce0939686ade` "[Bugfix] fix illegal memory access (#42)"
  (Lucas Wilkinson, 2025-02-06): the `partial_block_size` clamp and the `n_block` indexing fix.
- Read at that fork's commit `f3e1a4f74c99145c0717709860bf765de1703779` (the one vLLM 0.28.0 pins).

License of the ported code: that fork's `LICENSE` is the BSD 3-Clause License, "Copyright (c) 2022,
the respective contributors, as shown by the AUTHORS file" (its `setup.py` classifier says
"BSD License"); the copyright of the ported lines is their authors' (Lucas Wilkinson / Neural Magic,
the vLLM project's contributors). The BSD 3-Clause notice and disclaimer of this repository's
`LICENSE` apply to them unchanged.

Not ported from that fork: its sparse kernels (`flash_api_sparse.cpp`, `flash_fwd_sparse_*`),
the torch stable-ABI layer (`flash_api_torch_lib.cpp`, `cuda_check.h`, `STD_TORCH_CHECK`), the
dropout removal, and its split-KV `n_blocks_per_split` change over `actual_seqlen_k`.
