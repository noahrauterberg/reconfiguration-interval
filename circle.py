#
# Copyright (c) 2025 Noah Rauterberg. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times"],
    }
)
plt.rcParams["font.size"] = 18
# Create figure and axis
fig = plt.figure(figsize=(10, 10))

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_aspect("equal")

# Draw the circle
circle = plt.Circle((0, 0), 1, fill=False, color="black")
ax.add_patch(circle)

# Set axis limits and remove ticks
ax.set_xlim(-1.25, 1.25)
ax.set_ylim(-1.25, 1.25)
ax.set_xticks([])
ax.set_yticks([])


def get_coordinates(degrees):
    degrees = np.array(degrees)
    # Add 90 to rotate so 0° is at top
    adjusted_degrees = degrees + 90
    radians = np.radians(adjusted_degrees)
    x = np.cos(radians)
    y = np.sin(radians)
    return x, y


# First group of degrees
group1_degrees = [
    -170,
    -150,
    -100,
    -90,
    -80,
    -60,
    -10,
    0,
    10,
    30,
    80,
    90,
    100,
    120,
    170,
    180,
]

all_degrees = np.arange(-170, 181, 10)
group2_degrees = [d for d in all_degrees if d not in group1_degrees]

# Plot points and labels for both groups
for degrees, color in [(group1_degrees, "darkorange"), (group2_degrees, "blue")]:
    x, y = get_coordinates(degrees)
    # Plot points
    plt.scatter(x, y, c=color, s=150, zorder=2)
    # Add degree labels
    for deg in degrees:
        x, y = get_coordinates([deg])
        x_offset = 0.1
        y_offset = 0.1

        plt.text(
            x + x * x_offset, y + y * y_offset, f"{deg}°E", va="center", ha="center"
        )

# Add legend
from matplotlib.lines import Line2D

legend_elements = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="darkorange",
        label="168 Switches",
        markersize=10,
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="blue",
        label="170 Switches",
        markersize=10,
    ),
]
ax.legend(handles=legend_elements, loc="upper right")

# Set title
plt.title("GSL-Switches for Ground Stations on the Equator")
plt.tight_layout()
plt.savefig("circle_plot.pdf")
plt.close()
