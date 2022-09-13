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
# Modified: 2022-09-01 15:54:04
#
# Copyright (c) 2019-2022 Connor D. Pierce
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: LGPL-3.0-or-later


"""General utilities for managing matplotlib figures in IPython notebooks."""


### Imports ====================================================================
import logging
import numpy as np
import scipy as sp

from helpers import ureg, Qty, EmptyObject
from IPython.display import display, Image, Markdown
from matplotlib import cm, rcParams
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import signal
from scipy.linalg import lstsq
from scipy.stats import linregress

### Function Definitions =======================================================


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


def mpl_setup(style="none"):
    """
    Configures matplotlib rcParams with preset styles that improve the figure
    appearance for certain applications, like embedding in presentations.

    Available styles:
        "none" : makes no changes to rcParams
        "presentation-14pt" : 14-pt font for general fonts and tick labels,
            16-pt font for axes labels, inward-pointing ticks with increased
            length
    """

    if style == "none":
        return
    elif style == "presentation-14pt":
        # Configure matplotlib
        rcParams["axes.labelsize"] = 16
        rcParams["xtick.labelsize"] = 14
        rcParams["ytick.labelsize"] = 14
        rcParams["font.size"] = 14
        rcParams["xtick.major.size"] = 6
        rcParams["xtick.minor.size"] = 4
        rcParams["ytick.major.size"] = 6
        rcParams["ytick.minor.size"] = 4
        rcParams["xtick.direction"] = "in"
        rcParams["ytick.direction"] = "in"
    else:
        raise ValueError("Unknown style: " + str(style))


### Classes ====================================================================


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


# Import modules from this package
from . import qs, fs
