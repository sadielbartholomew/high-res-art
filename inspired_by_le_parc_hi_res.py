"""Art pieces inspired by rotational designs made in 1959 by Julio Le Parc.

Created by Sadie L. Bartholomew, July 2024.

Based upon code and earlier designs made by the same author in 2020-2021,
but using original patch designs, rotation arrays and colour schemes, plotted
at ultra HD resolution (3840 x 2160 pixels and 72 dpi).

For the original code and replcated and original (inspired-by) designs, see:

    https://github.com/sadielbartholomew/creative-matplotlib/tree/master/
    julio-le-parc-replications

"""

from abc import ABCMeta, abstractmethod
import itertools
import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms

import numpy as np


SET_DPI = 72
PIXEL_SIZE = (3840, 2160)


class LeParcDesign(metaclass=ABCMeta):
    """Abstract Base Class to replicate a 1959 Julio Le Parc design.

    For more detail on the artist Julio Le Parc, see:
    http://www.julioleparc.org/

    """

    def __init__(self, design_name, gridpoints, colours):
        """Set up a new 1959 Julio Le Parc design instance."""
        self.design_name = design_name

        self.gridpoints = gridpoints
        self.grid_indices = (range(self.gridpoints[0]), range(self.gridpoints[1]))

        self.colours = colours
        self.background_colour = None

        # Setting formatting specs here
        # In order to get req pixels with matplotlib, need to use figsize
        # as multiplier of dpi that give number of pixels, hence
        # <pixel size>/dpi.
        pixel_use = tuple([p / SET_DPI for p in PIXEL_SIZE])
        fig, ax = plt.subplots(figsize=pixel_use, dpi=SET_DPI)

        self.fig = fig
        self.axes = ax
        self.fig.set_canvas(plt.gcf().canvas)
        self.format_plt()
        self.patches = None

        self.save_to_dir = "pieces"

        # Create directory if it does not exist
        os.makedirs(self.save_to_dir, exist_ok=True)

        self.angles_array = self.create_design_angles_array()

        self.ABC_error_msg = "Designs must be created by subclassing."

    @abstractmethod
    def create_design_patches_per_gridpoint(self):
        """Create the underlying patches to rotate at each gridpoint."""
        raise NotImplementedError(self.ABC_error_msg)

    @abstractmethod
    def create_design_angles_array(self):
        """Create the array of patch rotation angles per gridpoint."""
        raise NotImplementedError(self.ABC_error_msg)

    @abstractmethod
    def create_design(self):
        """Create a design by placing and rotating relevant patches."""
        raise NotImplementedError(self.ABC_error_msg)

    def format_canvas(self):
        """Format canvas to centre and remove any axes markings."""
        self.background_colour = self.colours["COLOUR 1"]
        self.fig.patch.set_facecolor(self.background_colour)
        padding_per_side = 2
        limits = (
            self.gridpoints[0] - padding_per_side,
            self.gridpoints[1] + padding_per_side,
        )

        self.axes.set_xlim(*limits)
        self.axes.set_ylim(*limits)

    def format_plt(self):
        """Format the plot to add padding and hide axes components."""
        # Note: use this instead of plt.axis('equal') since due to the
        # rotation of outer patches in the animation, 'equal' will vary
        # somewhat and therefore the whole animation will shift and re-size
        # slightly otherwise, but want all patches fixed in position.
        min_point = 1
        # +2 to pad by 1 on each side
        max_points = (self.gridpoints[0] + 2, self.gridpoints[1] + 2)
        plt.axis([min_point, max_points[0], min_point, max_points[1]])

        plt.axis("off")
        plt.xticks([])
        plt.yticks([])

    def plot_and_save_design(self, number_output):
        """Plot and save a static replication of a Le Parc design."""
        self.format_canvas()
        self.create_design()
        self.format_plt()

        plt.tight_layout()

        plt.savefig(
            f"pieces/inspired_by_leparc_design_{number_output}.png",
            format="png",
            facecolor=self.background_colour,
        )


class RotationsLikePatchDesign(LeParcDesign):
    """A rotational design using the mechanics of Le Parc's Rotations.

    The parameters set here produce a design called 'Undulations In Rotation'.
    """

    def __init__(self):
        """Set up a new 'Rotations' replication instance."""
        higher_grid_spec = 78
        super().__init__(
            "Undulations In Rotation",
            (higher_grid_spec, round(higher_grid_spec / (3840 / 2160))),
            {
                "COLOUR 1": "#171123",  # purple-black
                "COLOUR 3": "#55D6BE",
                "COLOUR 2": "#E86252",
            },
        )

    def create_design_patches_per_gridpoint(
        self,
        centre,
        rect_angle,
        foreground_colour,
        background_colour,
        radius,
        padding,
    ):
        """Create the underlying patches to rotate at each gridpoint."""
        offset_amount = 0.00

        patch = mpatches.Circle(
            centre,
            radius,
            facecolor=foreground_colour,
            edgecolor=background_colour,
        )
        # A clipping rectangle, rotated appropriately.
        clip_patch = mpatches.Rectangle(
            (centre[0] + offset_amount, centre[1] - radius),
            radius - offset_amount + padding,
            2 * radius,
            color=background_colour,
            transform=mtransforms.Affine2D().rotate_deg_around(*centre, rect_angle)
            + self.axes.transData,
        )
        return (patch, clip_patch)

    def create_design_angles_array(self):
        """Create the array of patch rotation angles per gridpoint."""
        angles_array = np.zeros((self.gridpoints[0], self.gridpoints[1]), dtype=float)
        spaced_thetas_A = np.linspace(-147, 653, self.gridpoints[0])
        spaced_thetas_B = np.linspace(-47, 353, self.gridpoints[1])

        # 1. Make first and last column correct
        for j in self.grid_indices[1]:
            angles_array[0][j] = spaced_thetas_A[j]
            angles_array[-1][j] = spaced_thetas_B[-j - 1]
        # 2. Create rows linearly-spaced based on first and last columns
        for i in self.grid_indices[1]:
            row_angles = np.linspace(
                -1 * angles_array[0][i],
                angles_array[-1][i],
                self.gridpoints[0],
            )
            angles_array[:, i] = row_angles

        return angles_array

    def create_design(self, angles_array=None, image_pad_points=2):
        """Create a design by placing and rotating relevant patches."""
        if angles_array is None:
            angles_array = self.angles_array

        for i, j in itertools.product(self.grid_indices[0], self.grid_indices[1]):
            position_xy = (image_pad_points + i, image_pad_points + j)

            # First
            circle, clip_rectangle = self.create_design_patches_per_gridpoint(
                position_xy,
                self.angles_array[i][j],
                self.colours["COLOUR 3"],
                self.colours["COLOUR 1"],
                radius=0.56,
                padding=0.5,
            )
            self.axes.add_patch(circle)
            clip_rectangle.set_clip_path(circle)
            self.axes.add_patch(clip_rectangle)

            # Second
            circle, clip_rectangle = self.create_design_patches_per_gridpoint(
                position_xy,
                self.angles_array[i][j],
                self.colours["COLOUR 2"],
                self.colours["COLOUR 1"],
                radius=0.35,
                padding=0.1,
            )
            self.axes.add_patch(circle)
            clip_rectangle.set_clip_path(circle)
            self.axes.add_patch(clip_rectangle)


class MutationOfFormsLikePatchDesign(LeParcDesign):
    """A rotational design using the mechanics of Le Parc's Mutation of Forms.

    For more detail on the original piece, see:
      https://www.metmuseum.org/art/collection/search/815337

    The parameters set here produce a design called 'Connections In Rotation'.
    """

    def __init__(self):
        """Set up a new 'Mutation of Forms' inspired instance."""
        super().__init__(
            "Connections In Rotation",
            (72, 36),
            {
                "COLOUR 1": "#14161f",  # dark background now: off-black
                "COLOUR 2": "#ffc266",
                "COLOUR 3": "#b30059",
            },
        )

        self.red_angles_array = self.angles_array
        self.blue_angles_array = self.create_design_angles_array(is_red=False)

    @staticmethod
    def plot_mutations_wedge(centre, theta1, theta2, colour, zorder):
        """Create a single wedge patch with given angular coverage."""
        # 0.5 radius means the circles containing the wedges just touch their
        # neighbours. Use 0.475 to provide a small gap as per the design.
        return mpatches.Wedge(
            centre,
            0.58,
            theta1,
            theta2,
            color=colour,
            width=0.15,  # this makes it dougnut-like
            alpha=0.85,
            zorder=zorder,
        )

    def create_design_patches_per_gridpoint(
        self, position, wedge_1_thetas, wedge_2_thetas, colour_1, colour_2
    ):
        """Create the underlying patches to rotate at each gridpoint."""
        wedge_1 = self.plot_mutations_wedge(
            position,
            *wedge_1_thetas,
            colour_1,
            zorder=1,
        )
        wedge_2 = self.plot_mutations_wedge(
            position,
            *wedge_2_thetas,
            colour_2,
            zorder=2,
        )
        return wedge_1, wedge_2

    def create_mutations_linspaced_angles(self, max_coverage, min_coverage, num):
        """Create a 1D array of evenly-spaced wedge coverage angles.

        The even spacing between the leftmost and rightmost angles for the
        start, and separately, end, wedge angles is achieved using the NumPy
        function 'linspace'.
        """
        theta1_min_to_max = np.linspace(max_coverage[0], min_coverage[0], num=num)
        theta2_min_to_max = np.linspace(max_coverage[1], min_coverage[1], num=num)

        return np.column_stack((theta1_min_to_max, theta2_min_to_max))

    def create_design_angles_array(self, is_red=True):
        """Create the array of patch rotation angles per gridpoint."""
        a = 110
        b = -180 - a
        c = -b
        d = -a

        red_max = (a, a + 180)
        red_min = (b, b)
        blue_max = (c, c + 180)
        blue_min = (d, d)

        angles_array = np.zeros(
            (self.gridpoints[0], self.gridpoints[1]), dtype=(float, 2)
        )

        if is_red:
            index = 1
            spaced_thetas = self.create_mutations_linspaced_angles(
                max_coverage=red_max, min_coverage=red_min, num=self.gridpoints[1]
            )
        else:
            index = -1  # to reverse the spaced_thetas array later, via [::-1]
            spaced_thetas = self.create_mutations_linspaced_angles(
                max_coverage=blue_max, min_coverage=blue_min, num=self.gridpoints[1]
            )

        # 1. Make first and last column correct:
        for j in self.grid_indices[1]:
            angles_array[0][j] = spaced_thetas[::index][j]
            angles_array[-1][j] = spaced_thetas[::index][-j - 1]

        # 2. Create rows linearly-spaced based on first and last columns:
        for i in self.grid_indices[1]:
            row_angles = self.create_mutations_linspaced_angles(
                max_coverage=angles_array[0][i],
                min_coverage=angles_array[-1][i],
                num=self.gridpoints[0],
            )
            angles_array[:, i] = row_angles

        return angles_array

    def create_design(self):
        """Create a design by placing and rotating relevant patches."""
        for i, j in itertools.product(self.grid_indices[0], self.grid_indices[1]):
            # Get angles:
            red_thetas = self.red_angles_array[i][j]
            blue_thetas = self.blue_angles_array[i][j]

            # Now create and plot the wedges onto the canvas:
            image_pad_points = 2
            position_xy = (image_pad_points + i, image_pad_points + j)
            red_wedge, blue_wedge = self.create_design_patches_per_gridpoint(
                position_xy,
                red_thetas,
                blue_thetas,
                colour_1=self.colours["COLOUR 2"],
                colour_2=self.colours["COLOUR 3"],
            )
            self.axes.add_patch(red_wedge)
            self.axes.add_patch(blue_wedge)


if __name__ == "__main__":
    for index, design_class in enumerate(
        [
            MutationOfFormsLikePatchDesign,
            RotationsLikePatchDesign,
        ]
    ):
        design_class().plot_and_save_design(index)
        plt.show()
