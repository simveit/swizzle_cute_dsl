import cutlass
import cutlass.cute as cute
from visualize import layout_svg


@cute.jit
def bank_conflict(S: cute.Shape, D: cute.Stride, bms: cute.IntTuple):
    L = cute.make_layout(S, stride=D)
    b, m, s = bms[0], bms[1], bms[2]
    sw = cute.make_swizzle(b, m, s)
    L_swizzled = cute.make_composed_layout(sw, 0, L)
    M, N = cute.size(L, mode=[0]), cute.size(L, mode=[1])
    # Use unswizzled Layout here to compare with swizzled version.
    # Use swizzled Layout to see there (note here not tiled covered with atom!)
    return (
        M,
        N,
        {
            (i, j): (cute.crd2idx((i, j), L) // 2) % 32
            for i in range(M)
            for j in range(N)
        },
    )


if __name__ == "__main__":
    S = (64, 64)
    D = (64, 1)
    b, m, s = (3, 4, 3)

    M, N, layout = bank_conflict(S, D, (b, m, s))
    layout_svg(
        M,
        N,
        layout,
        name=f"images/Unswizzled_{M}_{N}_K_major_Bank_Conflict",
        mode="bank_conflict",
    )
