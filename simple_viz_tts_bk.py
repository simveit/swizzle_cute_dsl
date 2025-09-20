import cutlass
import cutlass.cute as cute
from visualize import layout_svg


@cute.jit
def simple_swizzle(
    S: cute.Shape, D: cute.Stride, bms: cute.IntTuple, S_trg: cute.Shape
):
    L = cute.make_layout(S, stride=D)
    b, m, s = bms[0], bms[1], bms[2]
    sw = cute.make_swizzle(b, m, s)
    L_swizzled = cute.make_composed_layout(sw, 0, L)
    L_tiled = cute.tile_to_shape(L_swizzled, S_trg, order=(1, 0))
    M, N = cute.size(L_tiled, mode=[0]), cute.size(L_tiled, mode=[1])
    return (
        M,
        N,
        {
            (i, j): (cute.crd2idx((i, j), L_tiled)//2) % 32
            for i in range(M)
            for j in range(N)
        },
    )


if __name__ == "__main__":
    S = (8, 64)
    D = (64, 1)
    b, m, s = (3, 4, 3)
    S_trg = (64, 64)

    M, N, layout = simple_swizzle(S, D, (b, m, s), S_trg)
    layout_svg(
        M,
        N,
        layout,
        name=f"images/{b}_{m}_{s}_K_major_{S_trg[0]}_{S_trg[1]}_Bank_Conflict",
        mode="bank_conflict",
        mul=8,
    )
