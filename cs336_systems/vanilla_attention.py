import torch
import triton

class PytorchFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        assert Q.shape[-1] == V.shape[-1] == K.shape[-1], "Embedding dim must be the same"
        assert K.shape[-2] == V.shape[-2], "Sequence length must be the same"

        Nq = Q.shape[-2]
        Nk = K.shape[-2]
        Dk = Q.shape[-1]
        Bq = 16
        Bk = 16
        q_tiles = triton.cdiv(Nq, Bq)
        kv_tiles = triton.cdiv(Nk, Bk)

        O = torch.zeros_like(Q)
        L = torch.zeros(Q.shape[:-1], device=Q.device, dtype=Q.dtype)

        for i in range(q_tiles):
            start_q = i * Bq
            end_q = min(start_q + Bq, Nq)
            Qi = Q[..., start_q:end_q, :]

            Oi = torch.zeros_like(Qi)
            Li = torch.zeros(Qi.shape[:-1], device=Qi.device, dtype=Qi.dtype)
            Mi = torch.full(Qi.shape[:-1], float('-inf'), device=Qi.device, dtype=Qi.dtype)

            for j in range(kv_tiles):
                start_k = j * Bk
                end_k = min(start_k + Bk, Nk)
                Ki = K[..., start_k:end_k, :]
                Vi = V[..., start_k:end_k, :]

                Si = (Qi @ Ki.transpose(-2, -1)) / (Dk ** 0.5)

                m_new = torch.maximum(Mi, Si.max(dim=-1).values)
                Pi = torch.exp(Si - m_new[..., None])

                Li = torch.exp(Mi - m_new) * Li + Pi.sum(dim=-1)
                Oi = torch.exp(Mi - m_new)[..., None] * Oi + Pi @ Vi

                Mi = m_new

            Oi = Oi / Li[..., None]
            Li_final = Mi + torch.log(Li)

            O[..., start_q:end_q, :] = Oi
            L[..., start_q:end_q] = Li_final

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        Nq = Q.shape[-2]
        Nk = K.shape[-2]
        Dk = Q.shape[-1]
        Bq = 16
        Bk = 16
        q_tiles = triton.cdiv(Nq, Bq)
        kv_tiles = triton.cdiv(Nk, Bk)
        scale = Dk ** 0.5

        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)

        D = torch.sum(dO * O, dim=-1)

        for i in range(q_tiles):
            start_q = i * Bq
            end_q = min(start_q + Bq, Nq)
            Qi = Q[..., start_q:end_q, :]
            Oi = O[..., start_q:end_q, :]
            dOi = dO[..., start_q:end_q, :]
            Li = L[..., start_q:end_q]
            Di = D[..., start_q:end_q]

            dQi = torch.zeros_like(Qi)

            for j in range(kv_tiles):
                start_k = j * Bk
                end_k = min(start_k + Bk, Nk)
                Kj = K[..., start_k:end_k, :]
                Vj = V[..., start_k:end_k, :]

                Sij = (Qi @ Kj.transpose(-2, -1)) / scale
                Pij = torch.exp(Sij - Li[..., None])

                dVj = Pij.transpose(-2, -1) @ dOi
                dPij = dOi @ Vj.transpose(-2, -1)
                dSij = Pij * (dPij - Di[..., None])

                dQi += dSij @ Kj / scale
                dKj = dSij.transpose(-2, -1) @ Qi / scale

                dK[..., start_k:end_k, :] += dKj
                dV[..., start_k:end_k, :] += dVj

            dQ[..., start_q:end_q, :] = dQi

        return dQ, dK, dV, None



        

