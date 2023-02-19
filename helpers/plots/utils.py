#! python3
# -*- coding: utf-8 -*-
#
# utils.py
#
# Utilities for setting up figures and automatically captioning figures in IPython
# notebooks.
#
# Author:   Connor D. Pierce
# Created:  2019-03-28 12:46
# Modified: 2023-02-18 15:50:13
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
import cycler
import logging
import numpy as np
import scipy as sp

from helpers.units import ureg, Qty
from helpers.utils import EmptyObject
from IPython.display import display, Image, Markdown
from matplotlib import cm, rcParams, pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import signal
from scipy.linalg import lstsq
from scipy.stats import linregress


__all__ = [
    "linewidth_cycler",
    "linestyle_cycler",
    "tab20_cycler",
    "get_style",
    "nudge",
    "FigureRegistry",
]


# Some pre-defined cyclers
linewidth_cycler = cycler.cycler(
    linewidth=(1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0)
)
linestyle_cycler = cycler.cycler(
    linestyle=(
        (0, (1.0,)),  # solid line
        (0, (3.7, 1.6)),  # normal dashed
        (0, (6.4, 1.6, 1.0, 1.6)),  # normal dashed-dotted
        (0, (1.0, 1.65)),  # normal dotted
        (0, (3.7, 0.5)),  # tightly dashed
        (0, (6.4, 0.5, 1.0, 0.5)),  # tightly dashed-dotted
        (0, (3.7, 3.7)),  # equal dashed
        (0, (6.4, 3.7, 1.0, 3.7)),  # equal dashed-dotted
        (0, (3.7, 7.4)),  # loosely dashed
        (0, (6.4, 7.4, 1.0, 7.4)),  # loosely dashed-dotted
        (0, (1.0, 3.7)),  # loosely dotted
    ),
)
tab20_cycler = cycler.cycler(
    color=[
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#aec7e8",
        "#ffbb78",
        "#98df8a",
        "#ff9896",
        "#c5b0d5",
        "#c49c94",
        "#f7b6d2",
        "#c7c7c7",
        "#dbdb8d",
        "#9edae5",
    ],  # modified "tab20"
)

## Functions
def get_style(j, params):
    """
    Returns the plot color and linestyle for the `j`th call to plot, given the
    set of linestyles and line colors provided in `params`.
    """

    if params["Color"]:
        n_colors = len(params["Color"])
        plotcolor = params["Color"][j % n_colors]
    else:
        n_colors = 10
        plotcolor = cm.tab10(j % n_colors)

    if params["LineStyle"]:
        plotstyle = params["LineStyle"][(j // n_colors) % len(params["LineStyle"])]
    else:
        plotstyle = "-"
    return plotcolor, plotstyle


def nudge(ax, x=0, y=0, w=0, h=0):
    """
    Moves `ax` by the specified amounts (in figure units).
    """

    x0, y0, w0, h0 = ax.get_position().bounds

    ax.set_position([x0 + x, y0 + y, w0 + w, h0 + h])
    return


## Classes
class FigureRegistry:
    """
    Maintains a database of figures, automates the display of figures in the
    Jupyter notebook, and saves figures to file with a uniform naming
    convention.
    """

    def __init__(self, directory, filetypes={"latex": "pdf", "jupyter": "png"}):
        """
        Initializes the `FigureRegistry`.

        Parameters:
        -----------
        `directory` : `str`
            Location where newly-created figures will be saved.
        `filetypes`: `dict`
            File types in which to save the figures. Keys denote the application
            for which the file is intended (e.g. "latex" or "jupyter") while the
            values give the file extension/type. If the key "jupyter" is not
            specified, "jupyter":"png" is automatically added.
        """

        self.figureMeta = {"currNum": 0}
        self.figures = {}
        self.filetypes = filetypes
        if "jupyter" not in filetypes:
            self.filetypes["jupyter"] = "png"
        self.logger = logging.getLogger("FigureRegistry")
        self._dir = directory

    def _registerFigure(self, caption, iD, filenames):
        """
        Registers a figure with caption given by `caption` and a unique identifier
        given by `iD`, with source files located at `filenames`. If a figure with
        identifier `iD` already exists, the figure is assigned the same number as
        the previously-registered figure.

        This method does not output anything to the notebook. This
        function should not be called directly; rather, use registerPyplotFigure()
        for figures plotted using pyplot, and registerExternalFigure() for figures
        generated using another program.

        === RETURNS ===
            dict containing the figure details
        """

        # Check to see if a figure with this ID already exists. If not, register a
        # figure with this name.
        if iD not in self.figures:
            self.figures[iD] = {"number": self.figureMeta["currNum"] + 1}
            self.figureMeta["currNum"] += 1
        else:
            self.logger.info("Re-using figure {0}".format(iD))

        # Complete the entry for this figure by updating the "caption", "file",
        # and "label" fields.
        self.figures[iD]["caption"] = caption
        self.figures[iD]["label"] = iD
        self.figures[iD]["files"] = filenames

        return self.figures[iD]

    def _displayFigure(self, figDetails):
        """
        Displays the figure specified in `figDetails`.
        """

        # List the figure title, caption, and label
        # print("Figure {0:d}: {1:s}".format(figDetails["number"],
        #        figDetails["files"]["latex"]))
        # print("Label: {0:s}".format(figDetails["label"]))
        # print("Caption: {0:s}".format(figDetails["caption"]))

        # Display the figure image
        display(
            Image(
                filename=figDetails["files"]["jupyter"],
                metadata={
                    "latex_metadata": {
                        "caption": figDetails["caption"],
                        "label": figDetails["label"],
                        "file": figDetails["files"]["latex"],
                    }
                },
            )
        )
        display(
            Markdown(
                "__Figure {0:d}__: {1:s}".format(
                    figDetails["number"], figDetails["caption"]
                )
            ),
            metadata={"latex_metadata": {"jupyter_caption": True}},
        )
        return

    def registerPyplotFigure(self, fig, caption, iD):
        """
        Registers a figure with caption given by `caption` and a unique identifier
        given by `iD`. If a figure with identifier `iD` already exists, the figure
        is assigned the same number as the previously-registered figure. Displays
        the title, label, and caption of the figure in the Jupyter notebook via
        stdout using print(), and displays the figure in the notebook using
        `display(Image())`.

        Parameters
        ----------
        `fig` : matplotlib.figure.Figure
            Figure to be registered
        `caption` : `str`
            Caption for the figure
        `iD` : `str`
            Unique identifier for the figure. Subsequent calls to this method or
            to `registerOtherFigure` using the same ID will overwrite the
            previous figure.

        Returns
        -------
            `dict` containing metadata for the figure
        """

        # Register the figure. The filenames will depend on the number assigned to
        # the figure, so we will pass an empty dict to filenames and then set the
        # filenames within this method.
        figDetails = self._registerFigure(caption, iD, {})

        figStr = self._dir + "/Figure_{0:02d}.".format(figDetails["number"])
        figDetails["files"] = {}
        for k in self.filetypes:
            figDetails["files"][k] = figStr + self.filetypes[k]

        # Save the figures for all registered formats
        for figFormat in figDetails["files"]:
            fig.savefig(figDetails["files"][figFormat])

        # Display figure in the Jupyter notebook:
        self._displayFigure(figDetails)

        return figDetails

    def registerOtherFigure(self, caption, iD, files):
        """
        Registers a figure generated in another software with caption given by
        `caption` and a unique identifier given by `iD`, having source files
        `files`. Displays the title, label, and caption of the figure in the Jupyter
        notebook via stdout using print(), and displays the figure in the notebook
        using display(Image()).

        Parameters
        ----------
        `caption` - `str`
            The figure caption (Should NOT start with "Figure <N>:")
        `iD` - `str`
            Unique identifier for the figure. Subsequent calls to this method or
            to `registerPyplotFigure` using the same ID will overwrite the
            previous figure.
        `files` - `dict`
            Paths to the figure files for each application specified in
            `self.filetypes`. Must always include the key "jupyter".

        Returns
        -------
            `dict` containing metadata for the figure
        """

        # Register the figure. The filenames will depend on the number assigned to
        # the figure, so we will pass an empty dict to filenames and then set the
        # filenames within this method.
        figDetails = self._registerFigure(caption, iD, files)

        # TODO: copy files from figure src folder to figures folder, renaming as
        #       Figure_XX.png

        # Display figure in the Jupyter notebook:
        self._displayFigure(figDetails)

        return figDetails
