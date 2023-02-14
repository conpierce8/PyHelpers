#! python3
# -*- coding: utf-8 -*-
#
# __init__.py
#
# init file for the `plots` package. Utilities for setting up figures and
# automatically captioning figures in IPython notebooks.
#
# Author:   Connor D. Pierce
# Created:  2019-03-28 12:46
# Modified: 2023-02-14 05:34:29
#
# Copyright (c) 2019-2023 Connor D. Pierce
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT


"""General utilities for managing matplotlib figures in IPython notebooks."""


## Imports
from helpers.utils import get_style, nudge, FigureRegistry


__all__ = [
    "get_style",
    "nudge",
    "FigureRegistry",
    "default_style",
    "presentation_style",
]


## Matplotlib styles
default_style = {
    "axes.labelpad": 2,
    "axes.labelsize": 10,
    "axes.linewidth": 1.0,
    "axes.prop_cycle": cycler(
        color=[
            "1f77b4",
            "ff7f0e",
            "2ca02c",
            "d62728",
            "9467bd",
            "8c564b",
            "e377c2",
            "7f7f7f",
            "bcbd22",
            "17becf",
        ],
        marker=["d", "^", "s", "D", "P", "v", "o", "<", ">", "x"],
    ),
    "axes.titlelocation": "left",
    "axes.titlepad": 0,
    "axes.titley": 0.8,
    "figure.dpi": 768,
    "figure.figsize": (1.75, 1.666666666667),  # Width & height to allocate per subplot
    "figure.subplot.left": 0.20,  # Axis left position within subplot grid cell
    "figure.subplot.right": 0.94,  # Axis right position within subplot grid cell
    "figure.subplot.bottom": 0.17,  # Axis bottom position within subplot grid cell
    "figure.subplot.top": 0.90,  # Axis top position within subplot grid cell
    "font.size": 10,
    "font.family": "serif",
    "lines.linewidth": 1.5,
    "lines.markersize": 2,
    "text.usetex": True,
    "xtick.direction": "in",
    "xtick.labelsize": 10,
    "xtick.major.pad": 2,
    "xtick.major.size": 6,
    "xtick.major.width": 0.5,
    "xtick.minor.size": 3,
    "xtick.minor.visible": True,
    "xtick.minor.width": 0.5,
    "xtick.top": True,
    "ytick.direction": "in",
    "ytick.labelsize": 10,
    "ytick.major.pad": 2,
    "ytick.major.size": 6,
    "ytick.major.width": 0.5,
    "ytick.minor.size": 3,
    "ytick.minor.visible": True,
    "ytick.minor.width": 0.5,
    "ytick.right": True,
}

presentation_style = default_style.copy()
presentation_style.update(
    {
        "axes.labelsize": 16,
        "font.size": 14,
        "xtick.direction": "in",
        "xtick.labelsize": 14,
        "xtick.major.size": 6,
        "xtick.minor.size": 4,
        "ytick.direction": "in",
        "ytick.labelsize": 14,
        "ytick.major.size": 6,
        "ytick.minor.size": 4,
    }
)
