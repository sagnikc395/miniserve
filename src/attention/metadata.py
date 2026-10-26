from dataclasses import dataclass
import torch


@dataclass
class AttnMetadata:
    # batch descriptor for one engine step, shared by every attention layer.

    # flat cache addresses for each of the T new tokens.
    # slot = block_id * block_size + (t % block_size)
    slot_mapping: torch.Tensor

    # prefix sums of this step query lengths. request i owns q[qsl[i]:qsl[i+1]]
    query_start_loc: torch.Tensor

    # prefix sums of full context lengths (kv_len)
    cu_seqlens_k: torch.Tensor

    # page table , Row i is req.block_ids padded to max_blocks
    block_table: torch.Tensor

    # live rows of block_table / entries in the cu_seqlens arrays
    num_seqs: int

    # max this step
    max_query_len: int

    # max(kv_len_i) this step
    max_seq_len: int

    # host copies so torch_ref can index in python without a GPU sync per request
    query_start_loc_cpu: list[int]

    # per request kv_len on host , not prefix summed
    # seq_lens_cpu[i] = cu_seqlens_k[i+1] - cu_seqlens_k[i]
    seq_lens_cpu: list[int]
