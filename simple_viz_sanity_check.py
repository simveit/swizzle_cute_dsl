import cutlass
import cutlass.cute as cute
from visualize import layout_svg


@cute.jit
def simple_swizzle(S: cute.Shape, D: cute.Stride, bms: cute.IntTuple):
    L = cute.make_layout(S, stride=D)
    b, m, s = bms[0], bms[1], bms[2]
    sw = cute.make_swizzle(b, m, s)
    L_swizzled = cute.make_composed_layout(sw, 0, L)
    M, N = cute.size(L, mode=[0]), cute.size(L, mode=[1])
    return (
        M,
        N,
        {(i, j): cute.crd2idx((i, j), L_swizzled) for i in range(M) for j in range(N)},
    )


if __name__ == "__main__":
    S = ((2, 4, 2), (8,2))
    D = ((8,64,32), (1,16))
    b, m, s = (3, 3, 3)

    M, N, layout = simple_swizzle(S, D, (b, m, s))
    layout_svg(M, N, layout, name=f"images/{b}_{m}_{s}_sanity", mode="color", mul=8)
