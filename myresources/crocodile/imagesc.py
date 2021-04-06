# Copyright 2019 Google LLC.
# SPDX-License-Identifier: Apache-2.0

# Author: Anton Mikhailov

import matplotlib.cm
from matplotlib.colors import ListedColormap


# The look-up table contains 256 entries. Each entry is a floating point sRGB triplet. To use it with matplotlib,
# pass cmap=ListedColormap(turbo_colormap_data) as an arg to imshow() (don't forget "from matplotlib.colors import
# ListedColormap"). If you have a typical 8-bit greyscale image, you can use the 8-bit value to index into this LUT
# directly. The floating point color values can be converted to 8-bit sRGB via multiplying by 255 and
# casting/flooring to an integer. Saturation should not be required for IEEE-754 compliant arithmetic. If you have a
# floating point value in the range [0,1], you can use interpolate() to linearly interpolate between the entries. If
# you have 16-bit or 32-bit integer values, convert them to floating point values on the [0,1] range and then use
# interpolate(). Doing the interpolation in floating point will reduce banding. If some of your values may lie
# outside the [0,1] range, use interpolate_or_clip() to highlight them.


def interpolate(colormap, x):
    x = max(0.0, min(1.0, x))
    a = int(x * 255.0)
    b = min(255, a + 1)
    f = x * 255.0 - a
    return [colormap[a][0] + (colormap[b][0] - colormap[a][0]) * f,
            colormap[a][1] + (colormap[b][1] - colormap[a][1]) * f,
            colormap[a][2] + (colormap[b][2] - colormap[a][2]) * f]


def interpolate_or_clip(colormap, x):
    if x < 0.0:
        return [0.0, 0.0, 0.0]
    elif x > 1.0:
        return [1.0, 1.0, 1.0]
    else:
        return interpolate(colormap, x)


# matplotlib.cm.register_cmap('turbo', cmap=ListedColormap(turbo_colormap_data))

if __name__ == '__main__':
    pass
