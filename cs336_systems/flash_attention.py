import torch
import triton
import triton.language as tl

@triton.jit
def flash_attention_fwd():
