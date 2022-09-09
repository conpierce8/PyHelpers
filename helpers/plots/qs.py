#! python3
# -*- coding: utf-8 -*-
#
# qs.py
#
# Functions for plotting frequency-sweep results.
#
# Author:   Connor D. Pierce
# Created:  2021-03-31 11:33
# Modified: 2022-09-01 16:05:51
#
# Copyright (c) 2021-2022 Connor D. Pierce
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


"""Functions for plotting quasi-static test results."""


### Imports ====================================================================
import matplotlib as mpl
import numpy as np

from helpers import ureg, Qty, EmptyObject
from helpers.plots import get_style
from IPython.display import display, Image, Markdown
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from scipy import signal
from scipy.linalg import lstsq
from scipy.stats import linregress

# Configure logging
import logging
logger = logging.getLogger("helpers.plots.qs")
#TODO: finish setting up logging

### Function Definitions =======================================================

def ss(ax, testData, label=None, units={}, loading_only=False):
    """
    Plots quasistatic stress-strain results.
    
    Parameters
    ----------
    `ax` : `matplotlib.axes.Axes`
        Axes object on which to plot the quasistatic test results.
    `testData` : `list` (or `list` of `list`) of `helpers.io.QuasistaticTest`
        Data for the quasi-static test(s). When a `list` of `list`s is provided,
        all `QuasistaticTest` objects grouped within a list will have the same
        line color and different line styles. Otherwise, all lines are plotted
        as solid with different colors.
    `label` : `str` or `None` (optional)
        format string for legend entries, or `None` to suppress the legend. May
        include the following keyword parameters:
            samplename: `str` (e.g. "Sample 2a")
            samplenum: `str` (e.g. "2a")
            testname: `str` (e.g. "Test1a")
            testnum: `str` (e.g. "1a")
            B: `float` magnetic field in mT (e.g. 171)
            B_T: `float` magnetic field in T (e.g. 0.171)
    `units` : `dict` (optional)
        desired units for the plot. Allowable keys: `strain`, `stress`, `B`. If
        a key is omitted, the default values are "" for strain, "MPa" for
        stress, and "T" for B.
    `loading_only` : `bool` (optional)
        Specifies whether to plot only the loading portion of the curve (`True`)
        or the entire curve (`False`). Default is `False`.
    """
    
    colorCounter = 0
    for t in testData:
        if isinstance(t, list):
            styleCounter = 0
            for tt in t:
                _plot_ss(ax, tt, label, units, colorCounter, styleCounter,
                        loading_only)
                styleCounter += 1
        else:
            _plot_ss(ax, t, label, units, colorCounter, 0, loading_only)
        colorCounter += 1
    return

def ss_seq(testData, label=None, units={}, loading_only=False):
    """
    Plots a sequence of quasistatic stress-strain tests. A new figure is created
    for each test in the sequence, which contains all previous tests (plotted in
    gray) and the i-th test plotted in color.
    
    Parameters
    ----------
    `ax` : `matplotlib.axes.Axes`
        Axes object on which to plot the quasistatic test results.
    `testData` : `list` (or `list` of `list`) of `helpers.io.QuasistaticTest`
        Data for the quasi-static test(s). When a `list` of `list`s is provided,
        all `QuasistaticTest` objects grouped within a list will have the same
        line color and different line styles. Otherwise, all lines are plotted
        as solid with different colors.
    `label` : `str` or `None` (optional)
        format string for legend entries, or `None` to suppress the legend. May
        include the following keyword parameters:
            samplename: `str` (e.g. "Sample 2a")
            samplenum: `str` (e.g. "2a")
            testname: `str` (e.g. "Test1a")
            testnum: `str` (e.g. "1a")
            B: `float` magnetic field in mT (e.g. 171)
            B_T: `float` magnetic field in T (e.g. 0.171)
    `units` : `dict` (optional)
        desired units for the plot. Allowable keys: `strain`, `stress`, `B`. If
        a key is omitted, the default values are "" for strain, "MPa" for
        stress, and "T" for B.
    `loading_only` : `bool` (optional)
        Specifies whether to plot only the loading portion of the curve (`True`)
        or the entire curve (`False`). Default is `False`.
        
    Returns
    -------
    `fig` : `list`
        A list of the figures created by this method.
    """
    
    totalPlots = 0 
    for t in testData:
        if isinstance(t, list):
            for tt in t:
                totalPlots += 1
        else:
            if t.test.load_reg is None:
                totalPlots += 1
            else:
                totalPlots += len(t.test.load_reg)
    
    fig = []
    for i in range(totalPlots):
        fig.append(Figure())
        ax = fig[-1].add_subplot(1,1,1)
        colorCounter = 0
        plotCounter = 0
        for t in testData:
            if isinstance(t, list):
                styleCounter = 0
                for tt in t:
                    if plotCounter == i:
                        # This is the last plot; plot it in color then break
                        _plot_ss(ax, tt, label, units, colorCounter,
                                styleCounter, loading_only)
                        plotCounter += 1
                        break
                    else:
                        # This is not the last plot; plot in gray and continue
                        _plot_ss(ax, tt, None, units, "grey",
                                styleCounter, loading_only)
                        plotCounter += 1
                        styleCounter += 1
            elif t.test.load_reg is None:
                if plotCounter == i:
                    _plot_ss(ax, t, label, units, colorCounter, 0, loading_only)
                    plotCounter += 1
                else:
                    _plot_ss(ax, t, None, units, "grey", 0, loading_only)
            else:
                styleCounter = 0
                for k in range(len(t.test.load_reg)):
                    if plotCounter == i:
                        # This is the last plot; plot it in color then break
                        _plot_ss(ax, t, label, units, colorCounter,
                                styleCounter, loading_only, k)
                        plotCounter += 1
                        break
                    else:
                        # This is not the last plot; plot in gray and continue
                        _plot_ss(ax, t, None, units, "grey",
                                styleCounter, loading_only, k)
                        plotCounter += 1
                        styleCounter += 1
            colorCounter += 1
            
            if plotCounter > i:
                break
    return fig

def _plot_ss(ax, qsTest, label, units, colorIdx, styleIdx, loading_only, i=0):
    strain_unit = units.get("strain", "")
    stress_unit = units.get("stress", "MPa")
    B_unit = units.get("B", "T")
    
    strain = qsTest.get_strain().to(strain_unit).magnitude
    stress = qsTest.get_stress().to(stress_unit).magnitude
    
    if loading_only:
        modulus = qsTest.get_modulus(tTol=0.1)
        if isinstance(modulus[0], list):
            idx = modulus[2]
        else:
            idx = ((modulus[2][1], modulus[2][2]), )
    else:
        idx = ((0, strain.shape[0]), )
    
    params = {}
    params["samplename"] = qsTest.specimen.name
    params["samplenum"] = qsTest.specimen.name[6:].strip()
    params["testname"] = qsTest.test.name
    params["testnum"] = qsTest.test.name[4:].strip()
    params["B"] = qsTest.test.magField.to(B_unit).magnitude
    
    if isinstance(colorIdx, int):
        c = cm.tab10(colorIdx % 10)
    elif isinstance(colorIdx, str):
        c = colorIdx
    ls = ("-", "--", "-.", ":")[styleIdx % 4]
    if label is None:
        h, = ax.plot(strain[idx[i][0]:idx[i][1]], stress[idx[i][0]:idx[i][1]], color=c,
                linestyle=ls)
    else:
        h, = ax.plot(strain[idx[i][0]:idx[i][1]], stress[idx[i][0]:idx[i][1]], 
                label=label.format(**params), color=c, linestyle=ls)
    if strain_unit == "":
        ax.set_xlabel("Strain [{u:s}]".format(u="-"))
    else:
        ax.set_xlabel("Strain [{u:s}]".format(u=strain_unit))
    ax.set_ylabel("Stress [{u:s}]".format(u=stress_unit))
    
#    if not idx is None:
#        reg = linregress(strain[idx[0]:idx[1]], stress[idx[0]:idx[1]])
#        
#        print(qsTest.specimen.name+", "+qsTest.test.name+":")
#        print("Modulus is:",reg.slope,"MPa")
#        print("Strain range:",strain[idx[1]]-strain[idx[0]])
#        
#        strainFit = qsTest.get_strain().to("")[idx]
#        stressFit = (reg.slope*ureg.MPa)*strainFit + (reg.intercept*ureg.MPa)
#        
#        ax.plot(strainFit.magnitude, stressFit.to("MPa").magnitude, "k-")
#        ax.annotate("{0:.2f} MPa".format(reg.slope),
#            xy=[strainFit.magnitude[0], stressFit.to("MPa").magnitude[0]],
#            ha="right", va="bottom", color=h.get_color())

class LivePlot:
    lock = None  # only one can be animated at a time

    def __init__(self, qsTest=None, var="strain"):
        self.fig = plt.figure()
        self.fig.canvas.mpl_disconnect(
                self.fig.canvas.manager.key_press_handler_id)
        self.fig.canvas.mpl_connect("key_press_event",
            lambda event: LivePlot.key_press_handler(
                event, self.fig.canvas, self.fig.canvas.toolbar
            )
        )
        
        self.rects = []
        self.activeRect = None
        self.activeEdge = np.zeros((2,2))
        
        self.press = None
        self.background = None
        self.liveAxes = self.fig.add_subplot(1,1,1)
        self.fig.tight_layout()
        
        self.qsTest = qsTest
        self.var = var
        self._draw_qs_data()
        
        self.cidpress = self.fig.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.fig.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.fig.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)
        self.cidkey = self.fig.canvas.mpl_connect(
            'key_release_event', self.on_key_release)
    
    def _draw_qs_data(self):
        if self.qsTest is None:
            return
        
        if self.var == "strain":
            y = self.qsTest.get_strain("raw")
        else:
            y = self.qsTest.get_stress("raw")
        x = np.arange(y.size)
        self.liveAxes.plot(x, y.magnitude)
        self.liveAxes.set_xlim(auto=True)
        self.liveAxes.set_ylim(auto=True)
        self.fig.canvas.draw()
    
    @staticmethod
    def key_press_handler(event, canvas, toolbar=None):
        try:
            idxL = rcParams["keymap.back"].index("left")
            rcParams["keymap.back"].remove("left")
        except ValueError:
            idxL = -1
        try:
            idxR = rcParams["keymap.forward"].index("right")
            rcParams["keymap.forward"].remove("right")
        except ValueError:
            idxR = -1
        key_press_handler(event, canvas, toolbar)
        if idxL >= 0:
            rcParams["keymap.back"].insert(idxL, "left")
        if idxR >= 0:
            rcParams["keymap.forward"].insert(idxR, "right")
    
    def connect(self):
        """Connect to all the events we need."""
        self.cidpress = self.fig.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.fig.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.fig.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)
    
    def on_key_release(self, event):
        # print(event.key)
        if LivePlot.lock is self:
            return
        
        if event.key == "enter":
            self.rects.append(self.activeRect)
            self.activeRect.set_animated(False)
            self.activeRect.set_zorder(10)
            self.activeRect.set_facecolor((0.8,0.8,0.8,0.5))
            self.activeRect = None
            self.fig.canvas.draw()
        elif event.key == "delete":
            # Remove most recent rectangle
            pass
        elif event.key in ("right", "left"):
            xlim = self.liveAxes.get_xlim()
            # print("\t", xlim)
            wd = xlim[1] - xlim[0]
            shift = 0.2*wd*(1 if event.key=="right" else -1)
            self.liveAxes.set_xlim([xlim[0]+shift,xlim[1]+shift])
            # print("\t", self.liveAxes.get_xlim())
            self.fig.canvas.draw()
            # print("\t", self.liveAxes.get_xlim())
        elif event.key in ("up", "down"):
            xlim = self.liveAxes.get_xlim()
            # print("\t", xlim)
            wd = xlim[1] - xlim[0]
            avgx = (xlim[0] + xlim[1])/2
            shift = wd*(0.5 if event.key=="up" else 2)
            self.liveAxes.set_xlim([avgx-shift/2,avgx+shift/2])
            # print("\t", self.liveAxes.get_xlim())
            self.fig.canvas.draw()
            # print("\t", self.liveAxes.get_xlim())
    
    def on_press(self, event):
        """Check whether mouse is over us; if so, store some data."""
        if (event.inaxes != self.liveAxes or LivePlot.lock is not None):
            return
        
        screenx = self.liveAxes.transData.transform((event.xdata,event.ydata))
        edgex = self.liveAxes.transData.transform(self.activeEdge)
        if self.activeRect and (abs(edgex[0,0]-screenx[0])<50 or
                abs(edgex[1,0]-screenx[0])<50):
            # Resize the active rectangle
            self.resize = True
            if abs(edgex[0,0]-screenx[0])<50:
                self.edge = "L"
                self.edgex = self.activeEdge[0,0]
                self.edgew = self.activeRect.get_width()
            else:
                self.edge = "R"
                self.edgex = self.activeEdge[1,0]
                self.edgew = self.activeRect.get_width()
        else:
            if self.activeRect:
                self.activeRect.remove()
            ylim = self.liveAxes.get_ylim()
            self.resize = False
            self.activeRect = Rectangle((event.xdata,ylim[0]),
                                        0,ylim[1]-ylim[0],
                                        zorder=10, facecolor=(1,0,0,0.5))
            self.activeEdge[0,0] = event.xdata
            self.activeEdge[1,0] = event.xdata
            self.liveAxes.add_patch(self.activeRect)
        
        self.xpress = event.xdata
        LivePlot.lock = self

        # draw everything but the selected rectangle and store the pixel buffer
        self.activeRect.set_animated(True)
        self.fig.canvas.draw()
        self.background = self.fig.canvas.copy_from_bbox(self.liveAxes.bbox)

        # now redraw just the rectangle
        self.liveAxes.draw_artist(self.activeRect)

        # and blit just the redrawn area
        self.fig.canvas.blit(self.liveAxes.bbox)

    def on_motion(self, event):
        """Move the rectangle if the mouse is over us."""
        if self.activeRect is None:
            return
        
        if (event.inaxes != self.activeRect.axes or LivePlot.lock is not self):
            return
        
        dx = event.xdata - self.xpress
        if self.resize:
            if self.edge == "L":
                self.activeRect.set_x(self.edgex + dx)
                self.activeRect.set_width(self.edgew - dx)
            else:
                self.activeRect.set_width(self.edgew + dx)
        else:
            self.activeRect.set_width(dx)
            self.activeEdge[1,0] = self.activeEdge[0,0] + dx

        # restore the background region
        self.fig.canvas.restore_region(self.background)

        # redraw just the current rectangle
        self.liveAxes.draw_artist(self.activeRect)

        # blit just the redrawn area
        self.fig.canvas.blit(self.liveAxes.bbox)

    def on_release(self, event):
        """Clear button press information."""
        if LivePlot.lock is not self:
            return
        
        dx = event.xdata - self.xpress
        if self.resize:
            if self.edge == "L":
                self.activeRect.set_x(self.edgex + dx)
                self.activeRect.set_width(self.edgew - dx)
            else:
                self.activeRect.set_width(self.edgew + dx)
            self.activeEdge[0,0] = self.activeRect.get_x()
            self.activeEdge[1,0] = (self.activeEdge[0,0]
                    +self.activeRect.get_width())
        else:
            self.activeRect.set_width(dx)
            self.activeEdge[1,0] = self.activeEdge[0,0] + dx
        
        self.edge = None
        self.edgex = None
        self.edgew = None
        self.resize = False
        
        self.xpress = None
        LivePlot.lock = None

        # turn off the rect animation property and reset the background
        self.activeRect.set_animated(False)
        self.background = None

        # redraw the full figure
        self.fig.canvas.draw()

    def disconnect(self):
        """Disconnect all callbacks."""
        self.activeRect.figure.canvas.mpl_disconnect(self.cidpress)
        self.activeRect.figure.canvas.mpl_disconnect(self.cidrelease)
        self.activeRect.figure.canvas.mpl_disconnect(self.cidmotion)
    
    def get_regions(self):
        reg = []
        for r in self.rects:
            reg.append( (r.get_x(), r.get_x()+r.get_width()) )
        return reg
    
    def set_qstest(self, qsTest):
        for r in self.rects:
            r.remove()
        self.rects.clear()
        
        if not self.activeRect is None:
            self.activeRect.remove()
            self.activeRect = None
        
        self.qsTest = qsTest
        self._draw_qs_data()