# Adapted from https://github.com/NVIDIA/cutlass/issues/2453#issuecomment-3133409976

import svgwrite


def layout_svg(M, N, layout, name="", mode="simple", mul=0):
    # 8 RBG-255 Greyscale colors
    if mode == "simple":
        rgb_255_colors = [
            (255, 255, 255),
            (153, 153, 153),
            (204, 204, 204),
            (102, 102, 102),
            (230, 230, 230),
            (128, 128, 128),
            (179, 179, 179),
            (77, 77, 77),
        ]
    elif mode == "bank_conflict":
        # 32 RGB-255 Greyscale colors (white to medium-dark grey)
        # Note: In this mode you should normalize the layout values
        # by 32 for optimal experience.
        start = 255
        stop = 80
        rgb_255_colors = [
            (
                start - i * (start - stop) // 32,
                start - i * (start - stop) // 32,
                start - i * (start - stop) // 32,
            )
            for i in range(32)
        ]
    else:
        rgb_255_colors = (
            [(160, 160, 255)] * mul  # dark blue
            + [(160, 255, 160)] * mul  # dark green
            + [(255, 255, 160)] * mul  # dark yellow
            + [(255, 160, 160)] * mul  # dark red
            + [(190, 190, 255)] * mul  # light blue
            + [(190, 255, 190)] * mul  # light green
            + [(255, 255, 190)] * mul  # light yellow
            + [(255, 190, 190)] * mul  # light red
        )

    # Cell size in pixels
    cell_size = 20

    # Create SVG canvas
    dwg = svgwrite.Drawing(f"{name}.svg", size=(N * cell_size, M * cell_size))

    # Draw grid cells
    for i in range(M):
        for j in range(N):
            idx = layout[(i, j)]
            x = j * cell_size
            y = i * cell_size

            # Draw rectangle
            dwg.add(
                dwg.rect(
                    insert=(x, y),
                    size=(cell_size, cell_size),
                    fill=svgwrite.rgb(
                        *rgb_255_colors[idx % len(rgb_255_colors)],
                        mode="RGB",
                        # *get_rgb_color(idx),
                        # mode="RGB",
                    ),
                    stroke="black",
                )
            )

            # Add label text
            dwg.add(
                dwg.text(
                    str(idx),
                    insert=(x + cell_size // 2, y + cell_size // 2),
                    text_anchor="middle",
                    alignment_baseline="central",
                    font_size="8px",
                )
            )

    dwg.save()
