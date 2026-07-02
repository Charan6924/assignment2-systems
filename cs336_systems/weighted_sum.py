import triton
import triton.language as tl
import torch
from einops import rearrange

@triton.jit
def weighted_sum_fwd(x_ptr, weight_ptr,
                    output_ptr,
                    x_stride_row, x_stride_dim,
                    weight_stride_dim, output_stride_row,
                    NUM_ROWS, D, 
                    ROWS_TILE_SIZE:tl.constexpr, D_TILE_SIZE : tl.constexpr):
    row_tile_idx = tl.program_id(0)

    #x is of shape (NUM_ROWS,D)
    #wieght is of shape (D,)
    #output is of shape (NUM_ROWS)

    x_block_ptr = tl.make_block_ptr(x_ptr, shape = (NUM_ROWS, D),
    strides = (x_stride_row, x_stride_dim),
    offsets = (row_tile_idx * ROWS_TILE_SIZE,0),
    block_shape = (ROWS_TILE_SIZE,D_TILE_SIZE),
    order = (1,0))

    weight_block_ptr = tl.make_block_ptr(
        weight_ptr, shape=(D,),
        strides = (weight_stride_dim,),
        offsets = (0,),
        block_shape=(D_TILE_SIZE,),
        order = (0,)
    )

    output_block_ptr = tl.make_block_ptr(
        output_ptr, shape = (NUM_ROWS,),
        strides = (output_stride_row,),
        offsets = (row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order = (0,),
    )

    output = tl.zeros((ROWS_TILE_SIZE,), dtype = tl.float32)

    for _ in range(tl.cdiv(D, D_TILE_SIZE)):
        row = tl.load(x_block_ptr, boundary_check=(0,1), padding_option="zero")
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")
        
        output += tl.sum(row * weight[None, :], axis = 1)

        x_block_ptr = x_block_ptr.advance((0,D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))

    tl.store(output_block_ptr,output, boundary_check=(0,))

@triton.jit
def weighted_sum_bwd(x_ptr, weight_ptr,
                     grad_output_ptr,
                     grad_x_ptr, partial_grad_weight_ptr,
                     stride_xr, stride_xd,
                     stride_wd,
                     stride_gr,
                     stride_gxr, stride_gxd,
                     stride_gwb, stride_gwd,
                     NUM_ROWS, D,
                     ROWS_TILE_SIZE : tl.constexpr, D_TILE_SIZE : tl.constexpr):
    
    row_tile_idx = tl.program_id(0)
    n_row_tiles = tl.num_programs(0)

    grad_output_block_ptr = tl.make_block_ptr(
        grad_output_ptr, shape=(NUM_ROWS,),
        strides = (stride_gr,), offsets = (row_tile_idx * ROWS_TILE_SIZE,),
        block_shape = (ROWS_TILE_SIZE,),
        order = (0,)
    )

    x_block_ptr = tl.make_block_ptr(
        x_ptr, shape=(NUM_ROWS, D), strides = (stride_xr,stride_xd),
        offsets = (row_tile_idx * ROWS_TILE_SIZE,0,),
        block_shape=(ROWS_TILE_SIZE,D_TILE_SIZE),
        order = (1,0)
    )

    weight_block_ptr = tl.make_block_ptr(
        weight_ptr, shape = (D,),
        strides = (stride_wd,), offsets = (0,),
        block_shape = (D_TILE_SIZE,),
        order = (0,)
    )

    grad_x_block_ptr = tl.make_block_ptr(
        grad_x_ptr, shape = (NUM_ROWS,D), strides = (stride_gxr,stride_gxd),
        offsets = (row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape = (ROWS_TILE_SIZE,D_TILE_SIZE),
        order = (1,0)
    )

    partial_grad_weight_block_ptr = tl.make_block_ptr(
        partial_grad_weight_ptr, shape = (n_row_tiles, D),
        strides = (stride_gwb, stride_gwd),
        offsets = (row_tile_idx,0),
        block_shape = (1,D_TILE_SIZE),
        order = (1,0)
    )

    for _ in range(tl.cdiv(D, D_TILE_SIZE)):
        grad_output = tl.load(grad_output_block_ptr, boundary_check = (0,), padding_option="zero")

        weight = tl.load(weight_block_ptr, boundary_check = (0,), padding_option = "zero")
        grad_x_row = grad_output[:,None] * weight[None, :]
        tl.store(grad_x_block_ptr, grad_x_row, boundary_check = (0,1))

        row = tl.load(x_block_ptr, boundary_check = (0,1), padding_option="zero")
        grad_weight_row = tl.sum(row * grad_output[:,None], axis = 0, keep_dims = True)
        tl.store(partial_grad_weight_block_ptr, grad_weight_row, boundary_check = (1,))

        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))
        grad_x_block_ptr = grad_x_block_ptr.advance((0,D_TILE_SIZE))
        partial_grad_weight_block_ptr = partial_grad_weight_block_ptr.advance((0, D_TILE_SIZE))



class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,weight):
        D, output_dims = x.shape[-1], x.shape[:-1]
        input_shape = x.shape
        x = rearrange(x,"... d -> (...) d")

        ctx.save_for_backward(x,weight)

        assert len(weight.shape) == 1 and weight.shape[0] == D, "Dimension mismatch"
        assert x.is_cuda and weight.is_cuda, "Tensors have to be on CUDA"
        assert x.is_contiguous()

        ctx.D_TILE_SIZE = triton.next_power_of_2(D) // 16
        ctx.ROWS_TILE_SIZE = 16
        ctx.input_shape = input_shape

        y = torch.empty(output_dims, device = x.device,dtype=x.dtype)

        n_rows = y.numel()
        weighted_sum_fwd[(triton.cdiv(n_rows, ctx.ROWS_TILE_SIZE),)](
            x, weight,y,
            x.stride(0),x.stride(1),
            weight.stride(0),
            y.stride(0),
            NUM_ROWS=n_rows, D=D,
            ROWS_TILE_SIZE=ctx.ROWS_TILE_SIZE, D_TILE_SIZE = ctx.D_TILE_SIZE, #type: ignore
        )

        return y.view(input_shape[:-1])

    @staticmethod
    def backward(ctx,grad_out): #type: ignore
        x, weight = ctx.saved_tensors
        ROWS_TILE_SIZE, D_TILE_SIZE = ctx.ROWS_TILE_SIZE, ctx.D_TILE_SIZE
        n_rows, D = x.shape

        partial_grad_weight = torch.empty((triton.cdiv(n_rows, ROWS_TILE_SIZE), D), device = x.device, dtype = x.dtype) #type: ignore
        x_grad = torch.empty_like(x)

        weighted_sum_bwd[(triton.cdiv(n_rows, ROWS_TILE_SIZE),)](
            x, weight,
            grad_out, x_grad,
            partial_grad_weight,
            x.stride(0),x.stride(1),
            weight.stride(0),
            grad_out.stride(0),
            x_grad.stride(0),x_grad.stride(1),
            partial_grad_weight.stride(0),partial_grad_weight.stride(1),
            NUM_ROWS=n_rows, D=D,
            ROWS_TILE_SIZE=ROWS_TILE_SIZE, D_TILE_SIZE=D_TILE_SIZE
        )

        grad_weight = partial_grad_weight.sum(axis=0)
        return x_grad, grad_weight


def test_weighted_sum():
    torch.manual_seed(0)
    device = "cuda"
    seq, D =  8, 64 

    x = torch.randn(seq,  D, device=device, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(D, device=device, dtype=torch.float32, requires_grad=True)

    x_ref = x.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()
    y_ref = (x_ref * weight_ref).sum(dim=-1)
    y_ref.sum().backward()

    x_triton = x.detach().clone().requires_grad_()
    weight_triton = weight.detach().clone().requires_grad_()
    y_triton = WeightedSumFunc.apply(x_triton, weight_triton)
    y_triton.sum().backward()

    fwd_ok = torch.allclose(y_triton, y_ref, atol=1e-3, rtol=1e-3)
    print("Forward match:", fwd_ok, "| max diff:", (y_triton - y_ref).abs().max().item())

    grad_x_ok = torch.allclose(x_triton.grad, x_ref.grad, atol=1e-3, rtol=1e-3)
    grad_w_ok = torch.allclose(weight_triton.grad, weight_ref.grad, atol=1e-3, rtol=1e-3)

    print("grad_x match:", grad_x_ok, "| max diff:", (x_triton.grad - x_ref.grad).abs().max().item())
    print("grad_weight match:", grad_w_ok, "| max diff:", (weight_triton.grad - weight_ref.grad).abs().max().item())

    assert fwd_ok and grad_x_ok and grad_w_ok, "Mismatch between Triton and reference implementation!"

if __name__ == "__main__":
    test_weighted_sum()
