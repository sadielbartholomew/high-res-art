"""Art pieces based around the stopping time pattern of the Collatz conjecture.

Created by Sadie L. Bartholomew, July 2024.

Artistic description for Design 1, Collatz Kaleidoscope:

    Overlaid plots of the scatter pattern formed by the number of iterations
    required for convergence to one in the Collatz conjecture (i.e. the
    '3n + 1' problem). The pattern is plotted multiple times, cycling through
    marker colors all displayed at very low transparency, with a relatively
    small linear translation applied each time. This design parameterizes
    this overall concept by displaying the window with bounds on the initial
    integer 'n' from 3000 to 23,000 (x axis) and on the corresponding steps
    taken to converge 's' from 4 to 219 (y axis), for six linear shifts of
    order of magnitude n=0 and s=1, using oversized square markers and a
    prismatic-like seven-component color scheme. Such a configuration was
    refined to engender the blending of markers to form an almost painterly
    coalescence, intended as a vibrant and textural spectacle of color that
    suggests the structure of the underlying pattern without having it as
    the focus.

Artistic description for Design 1, Collatz Residuals:

    Overlaid plots of the scatter pattern formed by the number of iterations
    required for convergence to one in the Collatz conjecture (i.e. the
    '3n + 1' problem). The pattern is plotted multiple times, cycling
    through marker colors all displayed at very low transparency, with a
    relatively small linear translation applied each time. This design
    parameterizes this overall concept by displaying the window
    with bounds on the initial integer 'n' from 9001 to 30,333
    (x axis) and on the corresponding steps taken to converge 's'
    from 80 to 270 (y axis), for six linear shifts of order of magnitude
    n=0 and s=1. It is shown with medium-sized hexagonal markers on a
    sea-green background to produce an effect reminiscent of light reflections
    off a body of water at night, fading from a more densely-packed area of
    the pattern in the foreground to a sparsely-packed part in the background
    to accentuate the complex subtle repetitiveness within the pattern and
    also bring to mind the notion of residuals through the apparent gradual
    fading and blurring of specific recurring motifs across the design.

Technical description (for both designs):

    The image is generated wholly from code written in Python (version 3.11.5)
    using the visualization library 'matplotlib' (version 3.8.3) to
    programmatically and procedurally define the design. This underlying code
    has, along with the generated output design, been open-sourced under the
    'CC BY 4.0' license and is available to view from a public GitHub
    repository 'high-res-art' under the artist's personal space, namely
    github.com/sadielbartholomew/high-res-art/blob/main/collatz_hi_res.py.
    High-performance computing was indispensable towards refining the
    parameters encoding the precise design, in particular the translation
    factors and set of hex-triplet color codes defining the background and
    marker colors, through batch processing of configurations of the parameters
    starting from exploratory values, followed by inspection of the generated
    outcomes and honing in on parameter sets producing promising outcomes in
    several iterations until finally this design emerged as a visual favorite.
    Specifically, the Slurm workload manager was used for such batch
    computing on the UK's JASMIN supercomputer.

Based upon code and earlier designs made by the same author in February 2022,
but using new designs with different overall dimensions considered and with
further thought into colour schemes and pattern shifts, etc., then plotted
at ultra HD resolution (3840 x 2160 pixels and 72 dpi).

For the original code and designs, see:

    https://github.com/sadielbartholomew/creative-matplotlib/
    tree/master/collatz-pattern-textures

"""

from itertools import cycle
from os.path import join

import matplotlib.pyplot as plt


# As per formatting requirements, 3840 wide x 2160 high pixels and 72 dpi
SET_DPI = 72
PIXEL_SIZE = (3840, 2160)


def collatz():
    """Return a list of Collatz 'stopping time' steps for start 1 to 32,000."""
    steps = []
    for n in range(1, 32000):
        count = 0
        while not n == 1:
            if n % 2 == 0:
                n /= 2
            else:
                n = 3 * n + 1
            count += 1
        steps.append(count)

    return steps


def shift_sequence(seq, m, c):
    """Translate a Collatz sequence by a linear shift encoded by y=mx+c."""
    return [m * x + c for x in seq]


def create_formatted_figure(xy_limits, background_col):
    """Create the blank matplotlib figure and axes objects for an art piece."""
    pixel_use = tuple([p / SET_DPI for p in PIXEL_SIZE])
    fig, ax = plt.subplots(figsize=pixel_use, dpi=SET_DPI)

    fig.patch.set_facecolor(background_col)

    # To remove the frame around the design, without affecting the pixel size
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(xy_limits[0])
    ax.set_ylim(xy_limits[1])
    ax.set_axis_off()
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)

    return fig, ax


def create_and_save_design(
    seq,
    index,
    pattern_shifts,
    xy_limits,
    background_col,
    foreground_cols,
    marker_type,
    marker_size,
    marker_alpha,
):
    """Plot and save a specific Collatz pattern design."""
    fig, axes = create_formatted_figure(xy_limits, background_col)

    for pattern_shift in pattern_shifts:
        axes.plot(
            shift_sequence(seq, *pattern_shift),
            marker_type,
            color=next(foreground_cols),
            markersize=marker_size,
            alpha=marker_alpha,
        )

    fig.savefig(
        join(f"pieces/collatz_hires_{index}.png"),
        format="png",
        pad_inches=0.0,
    )
    plt.show()


if __name__ == "__main__":
    """Define parameters to encode the designs and then plot and save them."""
    # For efficiency, calculate this only once, to re-use, since it is static.
    collatz_iterations = collatz()

    # Parameters and plotting for Design 1, Collatz Kaleidoscope
    PATTERN_SHIFT_1 = [
        (1, 0),
        (1.1, -5),
        (0.89, -2),
        (0.95, -20),
        (0.99, 4),
        (0.9, 12),
    ]
    BACKGROUND_COL_1 = "#BB0A21"  # bright red
    FOREGOUND_COLOURS_1 = cycle(
        [
            "#1A3BFF",
            "#FFB509",
            "#33C2CC",
            "#FCDC4D",
            "#8CF2A6",
            "#FF9582",
        ]
    )
    WINDOW_1 = ((3000, 23000), (4, 219))
    create_and_save_design(
        collatz_iterations,
        1,
        PATTERN_SHIFT_1,
        WINDOW_1,
        BACKGROUND_COL_1,
        FOREGOUND_COLOURS_1,
        "s",
        65,
        0.03,
    )

    # Parameters and plotting for Design 2, Collatz Residuals
    PATTERN_SHIFT_2 = [
        (0.96, 10),
        (1.15, -3),
        (0.98, 5),
        (1.03, 0),
        (0.94, -10),
        (1, 3),
    ]
    BACKGROUND_COL_2 = "#022D31"  # very dark green, sea-like colour
    FOREGOUND_COLOURS_2 = cycle(
        [
            "#577399",
            "#1BE7FF",
            "#139BC4",
            "#40C9A2",
            "#FF312E",
            "#FF5B7C",
        ]
    )
    WINDOW_2 = ((9001, 30333), (80, 270))

    create_and_save_design(
        collatz_iterations,
        2,
        PATTERN_SHIFT_2,
        WINDOW_2,
        BACKGROUND_COL_2,
        FOREGOUND_COLOURS_2,
        "h",
        44,
        0.04,
    )
