import torch
import triton
import triton.language as tl
import math

'''
forward pass :
S = QK^T / sqrt(d_k)
P = Softmax(S)
O = PV

backward pass:
dV = P^TdO
dp = dOV^T
dS = dsoftmax(dP) = (diag(P) - PP^T)dP
dQ = dSK/ sqrt(d_k)
dK = dS^TQ/sqrt(d_k)
'''

@triton.jit
def flash_attention_kernel(
        Q_ptr, K_ptr, V_ptr,
        O_ptr, L_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_ob, stride_oq, stride_od,
        stride_lb, stride_lq,
        N_QUERIES, N_KEYS,
        scale,
        D : tl.constexpr,
        Q_TILE_SIZE : tl.constexpr,
        K_TILE_SIZE : tl.constexpr,
        is_causal: tl.constexpr):
    
    q_tile_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    kv_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)

    Q_block_ptr = tl.make_block_ptr(Q_ptr + batch_idx * stride_qb,
        shape = (N_QUERIES,D),
        strides = (stride_qq, stride_qd),
        offsets = (q_tile_idx * Q_TILE_SIZE,0),
        block_shape = (Q_TILE_SIZE,D),
        order = (1,0))

    K_block_ptr = tl.make_block_ptr(K_ptr + batch_idx * stride_kb,
        shape = (N_KEYS,D),
        strides = (stride_kk, stride_kd),
        offsets = (0,0),
        block_shape = (K_TILE_SIZE, D),
        order = (1,0))

    V_block_ptr = tl.make_block_ptr(V_ptr + batch_idx * stride_vb,
        shape = (N_KEYS,D),
        strides = (stride_vk, stride_vd),
        offsets = (0,0),
        block_shape = (K_TILE_SIZE,D),
        order = (1,0))
    
    O_block_ptr = tl.make_block_ptr(O_ptr + batch_idx * stride_ob,
        shape = (N_QUERIES, D),
        offsets = (q_tile_idx * Q_TILE_SIZE,0),
        strides = (stride_oq, stride_od),
        block_shape = (Q_TILE_SIZE, D),
        order = (1,0))

    L_block_ptr = tl.make_block_ptr(L_ptr + batch_idx * stride_lb,
        shape = (N_QUERIES,),
        strides = (stride_lq, ),
        offsets = (q_tile_idx * Q_TILE_SIZE,),
        block_shape = (Q_TILE_SIZE,),
        order = (0,))

    Qi = tl.load(Q_block_ptr, boundary_check = (0,), padding_option="zero")
    Oi = tl.zeros((Q_TILE_SIZE,D), dtype = tl.float32)
    Li = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    Mi = tl.full((Q_TILE_SIZE,), value=float("-inf"), dtype=tl.float32)

    for j in range(kv_tiles):
        Ki = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero")
        Vi = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
        
        Si = tl.dot(Qi, tl.trans(Ki)) * scale 

        if is_causal:
            q_pos = q_tile_idx * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)   
            k_pos = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)            
            causal_mask = q_pos[:, None] >= k_pos[None, :]                 
            Si = tl.where(causal_mask, Si, float("-inf"))

        m_new = tl.maximum(Mi, tl.max(Si,axis = 1))
        Pi = tl.exp(Si - m_new[:, None])
        alpha = tl.exp(Mi - m_new)
        Li = alpha * Li + tl.sum(Pi, axis=1)
        Oi = alpha[:, None] * Oi + tl.dot(Pi.to(Vi.dtype), Vi)

        Mi = m_new

        K_block_ptr = tl.advance(K_block_ptr, offsets=(K_TILE_SIZE,0))
        V_block_ptr = tl.advance(V_block_ptr, offsets=(K_TILE_SIZE,0))

    Oi = Oi / Li[:, None]
    Li_final = Mi + tl.log(Li)

    tl.store(O_block_ptr, Oi.to(O_ptr.dtype.element_ty), boundary_check=(0,))
    tl.store(L_block_ptr, Li_final, boundary_check=(0,))

class TritonFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q,K,V, is_causal = False):
        assert Q.shape[-1] == V.shape[-1] == K.shape[-1], "Embedding dim must be the same"
        assert K.shape[-2] == V.shape[-2], "Sequence length must be the same"
        assert Q.is_cuda and K.is_cuda and V.is_cuda, "Inputs must be on GPU"

        orig_shape = Q.shape
        B, Nq, D = Q.shape
        Nk = K.shape[-2]

        O_ = torch.empty_like(Q)
        L_ = torch.empty((B, Nq), device=Q.device, dtype=torch.float32)
        scale = 1.0 / math.sqrt(D)
        Q_TILE_SIZE = 16
        K_TILE_SIZE = 16

        q_tiles = triton.cdiv(Nq, Q_TILE_SIZE)
        grid = (q_tiles, B)
        
        flash_attention_kernel[grid](
            Q, K, V,
            O_, L_,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O_.stride(0), O_.stride(1), O_.stride(2),
            L_.stride(0), L_.stride(1),
            Nq, Nk,
            scale,
            D=D,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal
        )

        O = O_.reshape(orig_shape)     # B, H, Nq, D
        L = L_.reshape(B, Nq)

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward():
        raise NotImplementedError
    
