import cutlass
import cutlass.cute as cute


@cute.jit
def simple_swizzle(
    S: cute.Shape, D: cute.Stride, bms: cute.IntTuple, coord: cute.IntTuple
):
    L = cute.make_layout(S, stride=D)
    b, m, s = bms[0], bms[1], bms[2]
    sw = cute.make_swizzle(b, m, s)
    L_swizzled = cute.make_composed_layout(sw, 0, L)
    print(coord)
    print(cute.crd2idx(coord, L))
    print(cute.crd2idx(coord, L_swizzled))


if __name__ == "__main__":
    S = (8, 32)
    D = (32, 1)
    bms = (2, 4, 3)
    coord = (7, 25)
    simple_swizzle(S, D, bms, coord)
