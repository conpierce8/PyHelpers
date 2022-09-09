#! python3
# -*- coding: utf-8 -*-
#
# fs.py
#
# Functions for plotting frequency-sweep results.
#
# Author:   Connor D. Pierce
# Created:  2021-03-31 11:40
# Modified: 2022-09-06 11:50:54
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


"""Functions for plotting frequency-sweep test results."""


### Imports ====================================================================
import cycler
import matplotlib as mpl
import numpy as np

from helpers import ureg, Qty, EmptyObject
from helpers.plots import FigureRegistry, nudge, get_style

from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.linalg import lstsq

# Configure logging
import logging
logger = logging.getLogger("helpers.plots.fs")
#TODO: finish configuration (i.e. check if this logger has a handler, if not,
#      add a handler; set level)


### Function Definitions =======================================================

def plot_a_to_v_transmission(ax, params):
    ''' Plots transmission on the specified axis for the tests specified in
    params, taking velocity as output and acceleration as input.
    
    ax - Matplotlib Axes object
    params - dictionary-based data structure having the following form:
             {
                "Input":<input file name>,
                "Output":<output file name>,
                "InputSens":<input sensitivity (volt/units) OR False>,
                "OutputSens":<output sensitivity (units/volt) OR False>,
                "Legend":<legend entry or False>,
                "LineStyle":<string line style or False>
             }
    '''
    
    if params["InputSens"] and params["OutputSens"]:
        vi_sens = params["InputSens" ]
        vo_sens = params["OutputSens"]
        normalize = False
    else:
        vi_sens = 1
        vo_sens = 1
        normalize = True
    
    datain, inheadings = load_data(params["Input"])
    dataout, outheadings = load_data(params["Output"])
    
    if isinstance(datain, str) or isinstance(dataout, str):
        return
    
    __plot_transmission(ax, datain, dataout, normalize,
            lambda freq, input, output: 2*np.pi*freq*a_sens*v_sens*output/input,
            params)

def plot_v_to_v_transmission(ax, params):
    ''' Plots transmission on the specified axis for the tests specified in
    params.
    
    ax - Matplotlib Axes object
    params - dictionary-based data structure having the following form:
             {
                "Input":<input file name>,
                "Output":<output file name>,
                "InputSens":<input sensitivity (units/volt) OR False>,
                "OutputSens":<output sensitivity (units/volt) OR False>,
                "Legend":[<legend entry 1>, ...] or False,
                "LineStyle":[<style1>, <style2>, ..., <styleN>] OR False,
                "Color":[<color1>, <color2>, ..., <colorM>] OR False
             }
    '''
    
    if params["InputSens"] and params["OutputSens"]:
        vi_sens = params["InputSens" ]
        vo_sens = params["OutputSens"]
        normalize = False
    else:
        vi_sens = 1
        vo_sens = 1
        normalize = True
    
    datain, inheadings = load_data(params["Input"])
    dataout, outheadings = load_data(params["Output"])
    
    if isinstance(datain, str) or isinstance(dataout, str):
        return dataout
    
    __plot_transmission(ax, datain, dataout, normalize,
            lambda freq, inp, outp: (vo_sens/vi_sens)*outp/inp, params)
    return False

def plot_phase(ax, params):
    ''' Plots transmission on the specified axis for the tests specified in
    params.
    
    ax - Matplotlib Axes object
    params - dictionary-based data structure having the following form:
             {
                "Input":<input file name>,
                "Legend":<legend entry or False>,
                "LineStyle":[<style1>, <style2>, ..., <styleN>] OR False,
                "Color":[<color1>, <color2>, ..., <colorM>] OR False
             }
    '''
    
    datain, inheadings = load_data(params["Input"])
    
    if isinstance(datain, str):
        return datain
    
    if datain.shape[1] == 3:
        fCol = 0
        pCol = 2
        parameterized = False
    else:
        fCol = 1
        pCol = 3
        parameterized = True
    
    plt.sca(ax)
    
    pIdx = find_pIdx(datain, parameterized)
    for j in range(0, len(pIdx)):
        plotcolor, plotstyle = get_style(j, params)
        
        sweepStartIdx = find_swpStartIdx(pIdx[j], datain, fCol)
       
        avg_in = getAvgData(pIdx[j], sweepStartIdx, datain, pCol)
        
        f = datain[sweepStartIdx[-1]:pIdx[j][1]+1,fCol]
        h,  = plt.plot(f, avg_in, plotstyle, color=plotcolor)
        
        if params["Legend"]:
            h.set_label(params["Legend"])
    
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Phase [${}^\circ$]')
    plt.grid(True)
    return False

def plot_raw_voltage(ax, params):
    f = lambda freq, data: data
    __plot_manipulated_voltage(ax, params, f)
    return

def plot_raw_time_dep(ax, params):
    datain, inheadings = load_data(params["Input"])
    if inheadings:
        dt = params["dt"]
        t = dt * np.arange(0, datain.shape[0])
        plt.sca(ax)
        i = 0
        while i < len(inheadings):
            if inheadings[i] == "R":
                break
            else:
                i += 1
        plt.plot(t, datain[:, i])
    else:
        return

def plot_vel(ax, params):
    v_sens = params["InputSens"]
    f = lambda freq, data: v_sens*data
    __plot_manipulated_voltage(ax, params, f)
    return

def plot_acc(ax, params):
    a_sens = params["InputSens"]
    f = lambda freq, data: data/a_sens
    __plot_manipulated_voltage(ax, params, f)
    return

def plot_vel_as_disp(ax, params):
    v_sens = params["InputSens"]
    f = lambda freq, data: v_sens*data/(2*np.pi*freq)
    __plot_manipulated_voltage(ax, params, f)
    return

def plot_acc_as_disp(ax, params):
    a_sens = params["InputSens"]
    f = lambda freq, data: data/(a_sens*(2*np.pi*freq)**2)
    __plot_manipulated_voltage(ax, params, f)
    return

def __plot_manipulated_voltage(ax, params, fn):
    datain, inheadings = load_data(params["Input"])
    
    plt.sca(ax)
    if inheadings:
        if datain.shape[1] == 3:
            fCol = 0
            parameterized = False
        else:
            fCol = 1
            parameterized = True
    else:
        return
    
    leg = params["Legend"]
    pIdx = find_pIdx(datain, parameterized)
    for i in range(len(pIdx)):
        sweepStartIdx = find_swpStartIdx(pIdx[i], datain, fCol)
        sweepStartIdx.append(pIdx[i][1]+1)
        for j in range(len(sweepStartIdx)-1):
            swpRange = np.arange(sweepStartIdx[j], sweepStartIdx[j+1])
            
            plotcolor, plotstyle = get_style(i, params)
            
            f = datain[swpRange, fCol]
            h, = plt.plot(f, fn(f, datain[swpRange, fCol+1]), plotstyle, color=plotcolor)
            if leg:
                h.set_label(leg[i % len(leg)])
    return

def __plot_transmission(ax, datain, dataout, normalize, T_fun, params):
    if datain.shape[1] == 3:
        fCol = 0
        RCol = 1
        parameterized = False
    else:
        fCol = 1
        RCol = 2
        parameterized = True
    
    plt.sca(ax)
    
    leg = params["Legend"]
    pIdx = find_pIdx(datain, parameterized)
    for j in range(0, len(pIdx)):
        plotcolor, plotstyle = get_style(j, params)
        
        sweepStartIdx = find_swpStartIdx(pIdx[j], datain, fCol)
       
        avg_in = getAvgData(pIdx[j], sweepStartIdx, datain, RCol)
        avg_out = getAvgData(pIdx[j], sweepStartIdx, dataout, RCol)
        
        f = datain[sweepStartIdx[-1]:pIdx[j][1]+1,fCol]
        avg_T   = T_fun(f, avg_in, avg_out)
        if normalize:
            avg_T = avg_T / avg_T[0]
        
        h = plt.plot(f, 20*np.log10(avg_T), plotstyle, color=plotcolor)
        
        if leg:
            h.set_label(leg[j % len(leg)])
    
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Transmission Amplitude [dB]')
    plt.grid(True)


def plotModeShapes(fig, files, sens, freqs, animate=False, decomp=None):
    # Check that each file contains the same frequencies
    freq = files[0][0][:,0]
    
    for file in files:
        if not np.all(freq == file[0][:,0]):
            raise ValueError("Frequency mismatch between files")
    
    # Check that all specified frequencies are in the files
    for f in freqs:
        if not f in freq:
            raise ValueError("Invalid frequency specified")
    
    # Plot mode shape for each frequency
    N = len(files)
    ampl = sens*np.stack([files[i][0][:,1] for i in range(N)], axis=1)
    phas = np.stack([files[i][0][:,2] for i in range(N)], axis=1)
    vel  = ampl * np.exp(1j * np.pi * phas / 180)
    count = 1
    x = np.arange(1, N+1)
    
    genFunc = lambda vecAmpl: ( lambda t: np.real(np.exp(1j*t)*vecAmpl) )
    generators = []
    curves     = []
    for f in freqs:
        for j in range(freq.size):
            if freq[j] == f:
                break
        
        ax = fig.add_subplot(len(freqs), 1, count)
        
        velR = genFunc(vel[j,:])
        generators.append(velR)
        
        # Format axes
        lim = 1.1*np.amax(ampl[j,:])
        ax.set_ylim([-lim, lim])
        ax.spines["bottom"].set_position(("data", 0))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylabel("{0:.0f} Hz".format(freq[j]))
        
        # Plot the mode shape at time t=0
        h, = ax.plot(x, velR(0), "k-o", linewidth=1.5, markersize=4)
        curves.append(h)
        
        # Decompose the mode using the vectors in `decomp`, if provided
        if isinstance(decomp, np.ndarray):
            modeAmpl, *_ = lstsq(decomp, vel[j,:])
            
            for i in range(modeAmpl.size):
                velMode = genFunc(modeAmpl[i]*decomp[:,i])
                h, = ax.plot(x, velMode(0), "-o", linewidth=1.0, markersize=2)
                
                generators.append(velMode)
                curves.append(h)
            
            velTot = genFunc(decomp.dot(modeAmpl))
            h, = ax.plot(x, velTot(0), "w--o", linewidth=1.0, markersize=1.5)
            
            generators.append(velTot)
            curves.append(h)
        
        count += 1
    fig.tight_layout()
    
    # Animate the plot, if desired
    if animate:
        def animFunc(t):
            for c, g in zip(curves, generators):
                c.set_ydata(g(t))
        
        anim = FuncAnimation(fig, animFunc, interval=10, blit=False, 
                             frames=np.arange(0,2*np.pi,np.pi/100))
        return anim

def plot_fs_test(
    fs_test,
    fig,
    qtys=("f", "E*"),
    units=("Hz", "MPa"),
    averaged=False,
    axes_types=("lin", "lin"),
    param_subplots=("A", None),
    param_cycler=cycler(
        color=[
            '1f77b4', 'ff7f0e', '2ca02c', 'd62728', '9467bd', '8c564b',
            'e377c2', '7f7f7f', 'bcbd22', '17becf', 'aec7e8', 'ffbb78',
            '98df8a', 'ff9896', 'c5b0d5', 'c49c94', 'f7b6d2', 'c7c7c7',
            'dbdb8d', '9edae5',
        ], # modified "tab20"
    ),
    trial_cyclers=(
        cycler.cycler(
            linestyle=(
                (0, (1.0, )),               # solid line
                (0, (3.7, 1.6)),            # normal dashed
                (0, (6.4, 1.6, 1.0, 1.6)),  # normal dashed-dotted
                (0, (1.0, 1.65)),           # normal dotted
                (0, (3.7, 0.5)),            # tightly dashed
                (0, (6.4, 0.5, 1.0, 0.5)),  # tightly dashed-dotted
                (0, (3.7, 3.7)),            # equal dashed
                (0, (6.4, 3.7, 1.0, 3.7)),  # equal dashed-dotted
                (0, (3.7, 7.4)),            # loosely dashed
                (0, (6.4, 7.4, 1.0, 7.4)),  # loosely dashed-dotted
                (0, (1.0, 3.7)),            # loosely dotted
            ),
        ),
        cycler.cycler(linewidth=(1.0, 1.5, 2.0, 3.0, 4.0, 5.0)),
    ),
):
    """Plots the results of a frequency-sweep test.
    
    Parameters
    ----------
    fs_test : FreqSweepTest
    fig : matplotlib.figure.Figure or matplotlib.axes.Axes
    qtys : (str, str)
        The quantities ("f", "A", "u", "F", "E") to plot on the x- and y-axes.
        The phasor values "u", "F", and "E" can be suffixed with "*", "'", or 
        "''" to plot the magnitude, real, or imaginary parts, respectively. 
        If no suffix is given, the magnitude is assumed.
    units : (<str or pint.Unit>, <str or pint.Unit>)
        The units in which to plot.
    subplots : None or tuple(str, dict(float: matplotlib.axes.Axes))
        The quantity to use for splitting into subplots, or `None` to plot all
        results on the same axes
    axes_types : tuple of (str, str)
        The type of axis ("lin" or "log") for the x- and y-axis
    """
    
    valid_x_qtys = ("f", "A")
    valid_y_qtys = (
        "E", "E*", "E'", "E''", "u", "u*", "u'", "u''",
        "F", "F*", "F'", "F''", "d",
    )
    x_qty, y_qty = qtys
    if not x_qty in valid_x_qtys:
        raise ValueError("Invalid x_qty")
    if not y_qty in valid_y_qtys:
        raise ValueError("Invalid y_qty")
    
    plt_cmd_str = None
    if axes_types == ("lin", "lin"):
        plt_cmd_str = "plot"
    elif axes_types == ("lin", "log"):
        plt_cmd_str = "semilogy"
    elif axes_types == ("log", "lin"):
        plt_cmd_str = "semilogx"
    elif axes_types == ("log", "log"):
        plt_cmd_str = "loglog"
    else:
        raise ValueError("Invalid plot type specified: " + str(axes_types))
    
    # Make sure the data for this test has been loaded
    fs_test.load_data()
    
    # Check that this test includes the required parameters
    if not _plot_fs_test_check_params(fs_test, x_qty):
        raise ValueError("fs_test does not have the required x_qty")
    if not _plot_fs_test_check_params(fs_test, y_qty):
        raise ValueError("fs_test does not have the required y_qty")
    
    # Extract the data to be plotted
    x_unit, y_unit = units
    x_data, x_ind, x_label = _plot_fs_test_get_data(fs_test, x_qty, averaged)
    y_data, y_ind, y_label = _plot_fs_test_get_data(fs_test, y_qty, averaged)
    x_data.ito(x_unit)
    y_data.ito(y_unit)
    
    handles = []
    plt_kwargs = {}
    
    if param_subplots is None:
        if isinstance(fig, mpl.figure.Figure):
            ax = fig.add_subplot(1,1,1)
        else:
            ax = fig
        
        plt_cmd = ax.__getattribute__(plt_cmd_str)
        
        if len(y_data.shape) == 1:
            handles.append(plt_cmd(x_data.magnitude, y_data.magnitude))
        else:
            for slc in _plot_fs_test_get_slices(
                y_data.shape,
                x_ind,
                param_cycler,
                trial_cyclers,
                plt_kwargs,
            ):
                logger.debug(slc)
                handles.append(
                    plt_cmd(
                        x_data.magnitude, y_data[slc].magnitude
                    )
                )
        
        ax.set_xlabel(x_label + " " + x_unit)
        ax.set_ylabel(y_label + " " + y_unit)
        #TODO: legend
    elif not subplots[0] in valid_qtys:
        raise ValueError("Unsupported quantity for subplotting")
    else:
        
        pass
    return handles

def _plot_fs_test_get_slices(
    shape,
    param_axis,
    param_cycler,
    trial_cyclers,
    plt_kwargs,
    axis=0,
    index=[]
):
    if axis == param_axis:
        if axis == len(shape)-1:
            yield tuple(index + [slice(None)])
        else:
            yield from _plot_fs_test_get_slices(
                shape,
                param_axis,
                param_cycler,
                trial_cyclers,
                plt_kwargs,
                axis+1, 
                index + [slice(None)]
            )
    else:
        for i in range(shape[axis]):
            if axis == len(shape)-1:
                yield tuple(index + [i])
            else:
                yield from _plot_fs_test_get_slices(
                    shape,
                    param_axis,
                    param_cycler,
                    trial_cyclers,
                    plt_kwargs
                    axis+1,
                    index + [i]
                )

def _plot_fs_test_check_params(fs_test, param):
    if param == "f":
        flag1 = (not fs_test._noDisp) or "Frequency" in fs_test.disp.params
        flag2 = (not fs_test._noForce) or "Frequency" in fs_test.force.params
        return (flag1 and flag2) and not (fs_test._noDisp and fs_test._noForce)
    elif param == "A":
        flag1 = (not fs_test._noDisp) or "Amplitude" in fs_test.disp.params
        flag2 = (not fs_test._noForce) or "Amplitude" in fs_test.force.params
        return (flag1 and flag2) and not (fs_test._noDisp and fs_test._noForce)
    elif param == "d":
        return not fs_test.specimen.noSS
    elif param.startswith("E"):
        return not fs_test.specimen.noSS
    elif param.startswith("u"):
        return not fs_test._noDisp
    elif param.startswith("F"):
        return not fs_test._noForce
    else:
        raise ValueError("Invalid param"+param)

def _plot_fs_test_get_data(fs_test, param, averaged):
    if param == "f":
        return fs_test.get_freq() + ("Frequency", )
    elif param == "A":
        return fs_test.get_ampl() + ("Amplitude", )
    elif param == "d":
        E_complex = fs_test.get_complex_modulus(averaged)
        return (
            (
                np.imag(E_complex.magnitude) / np.real(E_complex.magnitude)
                * ureg.dimensionless
            ),
            "Loss factor, $\\eta$",
        )
    elif param.startswith("E"):
        E_complex = fs_test.get_complex_modulus(averaged)
        if param == "E" or param == "E*":
            return (
                np.abs(E_complex) * E_complex.units,
                "Complex modulus magnitude, $|E^*|$",
            )
        elif param == "E'":
            return (
                np.real(E_complex.magnitude)*E_complex.units,
                "Storage Modulus, $E'$",
            )
        elif param == "E''":
            return (
                np.imag(E_complex.magnitude)*E_complex.units,
                "Loss modulus, $E''$",
            )
        else:
            raise ValueError("Unknown param: "+param)
    elif param.startswith("u"):
        u_complex = fs_test.get_disp(averaged)
        if param == "u" or param == "u*":
            return np.abs(u_complex) * u_complex.units, "Displacement magnitude"
        elif param == "u'":
            return (
                np.real(u_complex.magnitude) * u_complex.units,
                "Displacement (real)",
            )
        elif param == "u''":
            return (
                np.imag(u_complex.magnitude) * u_complex.units,
                "Displacement (imag)",
            )
        else:
            raise ValueError("Unknown param: "+param)
    elif param.startswith("F"):
        F_complex = fs_test.get_force(averaged)
        if param == "F" or param == "F*":
            return np.abs(F_complex), "Force magnitude"
        elif param == "F'":
            return (
                np.real(F_complex.magnitude)*F_complex.units,
                "Force (real)",
            )
        elif param == "F''":
            return (
                np.imag(F_complex.magnitude)*F_complex.units,
                "Force (imag)",
            )
        else:
            raise ValueError("Unknown param: "+param)
    else:
        raise ValueError("Invalid param"+param)
#! python3
# -*- coding: utf-8 -*-
#
# fs.py
#
# Functions for plotting frequency-sweep results.
#
# Author:   Connor D. Pierce
# Created:  2021-03-31 11:40
# Modified: 2022-09-06 11:50:54
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


"""Functions for plotting frequency-sweep test results."""


### Imports ====================================================================
import matplotlib as mpl
import numpy as np

from helpers import ureg, Qty, EmptyObject
from helpers.plots import FigureRegistry, nudge, get_style

from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.linalg import lstsq

# Configure logging
import logging
logger = logging.getLogger("helpers.plots.fs")
#TODO: finish configuration (i.e. check if this logger has a handler, if not,
#      add a handler; set level)


### Function Definitions =======================================================

def plot_a_to_v_transmission(ax, params):
    ''' Plots transmission on the specified axis for the tests specified in
    params, taking velocity as output and acceleration as input.
    
    ax - Matplotlib Axes object
    params - dictionary-based data structure having the following form:
             {
                "Input":<input file name>,
                "Output":<output file name>,
                "InputSens":<input sensitivity (volt/units) OR False>,
                "OutputSens":<output sensitivity (units/volt) OR False>,
                "Legend":<legend entry or False>,
                "LineStyle":<string line style or False>
             }
    '''
    
    if params["InputSens"] and params["OutputSens"]:
        vi_sens = params["InputSens" ]
        vo_sens = params["OutputSens"]
        normalize = False
    else:
        vi_sens = 1
        vo_sens = 1
        normalize = True
    
    datain, inheadings = load_data(params["Input"])
    dataout, outheadings = load_data(params["Output"])
    
    if isinstance(datain, str) or isinstance(dataout, str):
        return
    
    __plot_transmission(ax, datain, dataout, normalize,
            lambda freq, input, output: 2*np.pi*freq*a_sens*v_sens*output/input,
            params)

def plot_v_to_v_transmission(ax, params):
    ''' Plots transmission on the specified axis for the tests specified in
    params.
    
    ax - Matplotlib Axes object
    params - dictionary-based data structure having the following form:
             {
                "Input":<input file name>,
                "Output":<output file name>,
                "InputSens":<input sensitivity (units/volt) OR False>,
                "OutputSens":<output sensitivity (units/volt) OR False>,
                "Legend":[<legend entry 1>, ...] or False,
                "LineStyle":[<style1>, <style2>, ..., <styleN>] OR False,
                "Color":[<color1>, <color2>, ..., <colorM>] OR False
             }
    '''
    
    if params["InputSens"] and params["OutputSens"]:
        vi_sens = params["InputSens" ]
        vo_sens = params["OutputSens"]
        normalize = False
    else:
        vi_sens = 1
        vo_sens = 1
        normalize = True
    
    datain, inheadings = load_data(params["Input"])
    dataout, outheadings = load_data(params["Output"])
    
    if isinstance(datain, str) or isinstance(dataout, str):
        return dataout
    
    __plot_transmission(ax, datain, dataout, normalize,
            lambda freq, inp, outp: (vo_sens/vi_sens)*outp/inp, params)
    return False

def plot_phase(ax, params):
    ''' Plots transmission on the specified axis for the tests specified in
    params.
    
    ax - Matplotlib Axes object
    params - dictionary-based data structure having the following form:
             {
                "Input":<input file name>,
                "Legend":<legend entry or False>,
                "LineStyle":[<style1>, <style2>, ..., <styleN>] OR False,
                "Color":[<color1>, <color2>, ..., <colorM>] OR False
             }
    '''
    
    datain, inheadings = load_data(params["Input"])
    
    if isinstance(datain, str):
        return datain
    
    if datain.shape[1] == 3:
        fCol = 0
        pCol = 2
        parameterized = False
    else:
        fCol = 1
        pCol = 3
        parameterized = True
    
    plt.sca(ax)
    
    pIdx = find_pIdx(datain, parameterized)
    for j in range(0, len(pIdx)):
        plotcolor, plotstyle = get_style(j, params)
        
        sweepStartIdx = find_swpStartIdx(pIdx[j], datain, fCol)
       
        avg_in = getAvgData(pIdx[j], sweepStartIdx, datain, pCol)
        
        f = datain[sweepStartIdx[-1]:pIdx[j][1]+1,fCol]
        h,  = plt.plot(f, avg_in, plotstyle, color=plotcolor)
        
        if params["Legend"]:
            h.set_label(params["Legend"])
    
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Phase [${}^\circ$]')
    plt.grid(True)
    return False

def plot_raw_voltage(ax, params):
    f = lambda freq, data: data
    __plot_manipulated_voltage(ax, params, f)
    return

def plot_raw_time_dep(ax, params):
    datain, inheadings = load_data(params["Input"])
    if inheadings:
        dt = params["dt"]
        t = dt * np.arange(0, datain.shape[0])
        plt.sca(ax)
        i = 0
        while i < len(inheadings):
            if inheadings[i] == "R":
                break
            else:
                i += 1
        plt.plot(t, datain[:, i])
    else:
        return

def plot_vel(ax, params):
    v_sens = params["InputSens"]
    f = lambda freq, data: v_sens*data
    __plot_manipulated_voltage(ax, params, f)
    return

def plot_acc(ax, params):
    a_sens = params["InputSens"]
    f = lambda freq, data: data/a_sens
    __plot_manipulated_voltage(ax, params, f)
    return

def plot_vel_as_disp(ax, params):
    v_sens = params["InputSens"]
    f = lambda freq, data: v_sens*data/(2*np.pi*freq)
    __plot_manipulated_voltage(ax, params, f)
    return

def plot_acc_as_disp(ax, params):
    a_sens = params["InputSens"]
    f = lambda freq, data: data/(a_sens*(2*np.pi*freq)**2)
    __plot_manipulated_voltage(ax, params, f)
    return

def __plot_manipulated_voltage(ax, params, fn):
    datain, inheadings = load_data(params["Input"])
    
    plt.sca(ax)
    if inheadings:
        if datain.shape[1] == 3:
            fCol = 0
            parameterized = False
        else:
            fCol = 1
            parameterized = True
    else:
        return
    
    leg = params["Legend"]
    pIdx = find_pIdx(datain, parameterized)
    for i in range(len(pIdx)):
        sweepStartIdx = find_swpStartIdx(pIdx[i], datain, fCol)
        sweepStartIdx.append(pIdx[i][1]+1)
        for j in range(len(sweepStartIdx)-1):
            swpRange = np.arange(sweepStartIdx[j], sweepStartIdx[j+1])
            
            plotcolor, plotstyle = get_style(i, params)
            
            f = datain[swpRange, fCol]
            h, = plt.plot(f, fn(f, datain[swpRange, fCol+1]), plotstyle, color=plotcolor)
            if leg:
                h.set_label(leg[i % len(leg)])
    return

def __plot_transmission(ax, datain, dataout, normalize, T_fun, params):
    if datain.shape[1] == 3:
        fCol = 0
        RCol = 1
        parameterized = False
    else:
        fCol = 1
        RCol = 2
        parameterized = True
    
    plt.sca(ax)
    
    leg = params["Legend"]
    pIdx = find_pIdx(datain, parameterized)
    for j in range(0, len(pIdx)):
        plotcolor, plotstyle = get_style(j, params)
        
        sweepStartIdx = find_swpStartIdx(pIdx[j], datain, fCol)
       
        avg_in = getAvgData(pIdx[j], sweepStartIdx, datain, RCol)
        avg_out = getAvgData(pIdx[j], sweepStartIdx, dataout, RCol)
        
        f = datain[sweepStartIdx[-1]:pIdx[j][1]+1,fCol]
        avg_T   = T_fun(f, avg_in, avg_out)
        if normalize:
            avg_T = avg_T / avg_T[0]
        
        h = plt.plot(f, 20*np.log10(avg_T), plotstyle, color=plotcolor)
        
        if leg:
            h.set_label(leg[j % len(leg)])
    
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Transmission Amplitude [dB]')
    plt.grid(True)


def plotModeShapes(fig, files, sens, freqs, animate=False, decomp=None):
    # Check that each file contains the same frequencies
    freq = files[0][0][:,0]
    
    for file in files:
        if not np.all(freq == file[0][:,0]):
            raise ValueError("Frequency mismatch between files")
    
    # Check that all specified frequencies are in the files
    for f in freqs:
        if not f in freq:
            raise ValueError("Invalid frequency specified")
    
    # Plot mode shape for each frequency
    N = len(files)
    ampl = sens*np.stack([files[i][0][:,1] for i in range(N)], axis=1)
    phas = np.stack([files[i][0][:,2] for i in range(N)], axis=1)
    vel  = ampl * np.exp(1j * np.pi * phas / 180)
    count = 1
    x = np.arange(1, N+1)
    
    genFunc = lambda vecAmpl: ( lambda t: np.real(np.exp(1j*t)*vecAmpl) )
    generators = []
    curves     = []
    for f in freqs:
        for j in range(freq.size):
            if freq[j] == f:
                break
        
        ax = fig.add_subplot(len(freqs), 1, count)
        
        velR = genFunc(vel[j,:])
        generators.append(velR)
        
        # Format axes
        lim = 1.1*np.amax(ampl[j,:])
        ax.set_ylim([-lim, lim])
        ax.spines["bottom"].set_position(("data", 0))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylabel("{0:.0f} Hz".format(freq[j]))
        
        # Plot the mode shape at time t=0
        h, = ax.plot(x, velR(0), "k-o", linewidth=1.5, markersize=4)
        curves.append(h)
        
        # Decompose the mode using the vectors in `decomp`, if provided
        if isinstance(decomp, np.ndarray):
            modeAmpl, *_ = lstsq(decomp, vel[j,:])
            
            for i in range(modeAmpl.size):
                velMode = genFunc(modeAmpl[i]*decomp[:,i])
                h, = ax.plot(x, velMode(0), "-o", linewidth=1.0, markersize=2)
                
                generators.append(velMode)
                curves.append(h)
            
            velTot = genFunc(decomp.dot(modeAmpl))
            h, = ax.plot(x, velTot(0), "w--o", linewidth=1.0, markersize=1.5)
            
            generators.append(velTot)
            curves.append(h)
        
        count += 1
    fig.tight_layout()
    
    # Animate the plot, if desired
    if animate:
        def animFunc(t):
            for c, g in zip(curves, generators):
                c.set_ydata(g(t))
        
        anim = FuncAnimation(fig, animFunc, interval=10, blit=False, 
                             frames=np.arange(0,2*np.pi,np.pi/100))
        return anim

def plot_fs_test(
    fs_test,
    fig,
    qtys,
    units,
    subplots=None,
    axes_types=("lin","lin"),
    color=None,
    linestyle=None,
):
    """
    Plots the results of a frequency-sweep test.
    
    Parameters
    ----------
    fs_test : FreqSweepTest
    fig : matplotlib.figure.Figure or matplotlib.axes.Axes
    qtys : (str, str)
        The quantities ("f", "A", "u", "F", "E") to plot on the x- and y-axes.
        The phasor values "u", "F", and "E" can be suffixed with "*", "'", or 
        "''" to plot the magnitude, real, or imaginary parts, respectively. 
        If no suffix is given, the magnitude is assumed.
    units : (<str or pint.Unit>, <str or pint.Unit>)
        The units in which to plot.
    subplots : str or None
        The quantity to use for splitting into subplots, or `None` to plot all
        results on the same axes
    axes_types : tuple of (str, str)
        The type of axis ("lin" or "log") for the x- and y-axis
    """
    
    for s in axes_types:
        if s not in ("lin", "log"):
            raise ValueError("Invalid axes type")
    
    valid_qtys = ("f", "A", "E", "E*", "E'", "E''", "u", "u*", "u'", "u''",
            "F", "F*", "F'", "F''", "d")
    x_qty, y_qty = qtys
    if not x_qty in valid_qtys:
        raise ValueError("Invalid x_qty")
    if not y_qty in valid_qtys:
        raise ValueError("Invalid y_qty")
    if (x_qty == "f" and y_qty == "A") or (x_qty == "A" and y_qty == "f"):
        raise ValueError("Cannot plot freq vs ampl")
    
    plt_cmd_str = None
    if axes_types == ("lin", "lin"):
        plt_cmd_str = "plot"
    elif axes_types == ("lin", "log"):
        plt_cmd_str = "semilogy"
    elif axes_types == ("log", "lin"):
        plt_cmd_str = "semilogx"
    elif axes_types == ("log", "log"):
        plt_cmd_str = "loglog"
    else:
        raise ValueError("Invalid plot type specified: " + str(axes_types))
    
    # Make sure the data for this test has been loaded
    fs_test.load_data()
    
    # Check that this test includes the required parameters
    if not _plot_fs_test_check_params(fs_test, x_qty):
        raise ValueError("fs_test does not have the required x_qty")
    if not _plot_fs_test_check_params(fs_test, y_qty):
        raise ValueError("fs_test does not have the required y_qty")
    
    # Extract the data to be plotted
    x_unit, y_unit = units
    x_data, x_ind = _plot_fs_test_get_data(fs_test, x_qty)
    y_data, y_ind = _plot_fs_test_get_data(fs_test, y_qty)
    x_data.ito(x_unit)
    y_data.ito(y_unit)
    
    handles = []
    
    if subplots is None:
        fs_test.load_data()
        
        if isinstance(fig, mpl.figure.Figure):
            ax = fig.add_subplot(1,1,1)
        else:
            ax = fig
        
        plt_cmd = ax.__getattribute__(plt_cmd_str)
        
        if len(x_data.shape) == 1:
            if len(y_data.shape) == 1:
                handles.append(plt_cmd(x_data.magnitude, y_data.magnitude))
            else:
                for slc in _plot_fs_test_get_slices(y_data.shape, x_ind):
                    logger.debug(slc)
                    handles.append(plt_cmd(x_data.magnitude, y_data[slc].magnitude))
        else:
            if len(y_data.shape) == 1:
                for slc in _plot_fs_test_get_slices(x_data.shape, y_ind):
                    handles.append(ax.plot(x_data[slc].magnitude, y_data.magnitude))
            else:
                handles.append(ax.plot(x_data.flatten().magnitude, y_data.flatten().magnitude,
                        'o'))
    elif not subplots in valid_qtys:
        raise ValueError("Unsupported quantity for subplotting")
    else:
        raise NotImplementedError("Not implemented yet")
        pass
    return handles

def _plot_fs_test_get_slices(shape, param_axis, axis=0, index=[]):
    if axis == param_axis:
        if axis == len(shape)-1:
            yield tuple(index + [slice(None)])
        else:
            yield from _plot_fs_test_get_slices(shape, param_axis, axis+1, 
                    index+[slice(None)])
    else:
        for i in range(shape[axis]):
            if axis == len(shape)-1:
                yield tuple(index + [i])
            else:
                yield from _plot_fs_test_get_slices(shape, param_axis, axis+1,
                        index+[i])

def _plot_fs_test_check_params(fs_test, param):
    if param == "f":
        flag1 = (not fs_test._noDisp) or "Frequency" in fs_test.disp.params
        flag2 = (not fs_test._noForce) or "Frequency" in fs_test.force.params
        return (flag1 and flag2) and not (fs_test._noDisp and fs_test._noForce)
    elif param == "A":
        flag1 = (not fs_test._noDisp) or "Amplitude" in fs_test.disp.params
        flag2 = (not fs_test._noForce) or "Amplitude" in fs_test.force.params
        return (flag1 and flag2) and not (fs_test._noDisp and fs_test._noForce)
    elif param == "d":
        return not fs_test.specimen.noSS
    elif param.startswith("E"):
        return not fs_test.specimen.noSS
    elif param.startswith("u"):
        return not fs_test._noDisp
    elif param.startswith("F"):
        return not fs_test._noForce
    else:
        raise ValueError("Invalid param"+param)

def _plot_fs_test_get_data(fs_test, param):
    if param == "f":
        if fs_test._noDisp:
            return (fs_test.force.params["Frequency"].values * ureg.Hz,
                    fs_test.force.params["Frequency"].axis)
        else:
            return (fs_test.disp.params["Frequency"].values * ureg.Hz,
                    fs_test.disp.params["Frequency"].axis)
    elif param == "A":
        if not "A" in fs_test.disp.params:
            return (fs_test.force.params["Amplitude"].values * ureg.volt,
                    fs_test.force.params["Amplitude"].axis)
        else:
            return (fs_test.disp.params["Amplitude"].values * ureg.volt,
                    fs_test.disp.params["Amplitude"].axis)
    elif param == "d":
        E_complex = fs_test.get_complex_modulus()
        return (np.imag(E_complex.magnitude) / np.real(E_complex.magnitude) * ureg.dimensionless,
                None)
    elif param.startswith("E"):
        E_complex = fs_test.get_complex_modulus()
        if param == "E" or param == "E*":
            return np.abs(E_complex), None
        elif param == "E'":
            return np.real(E_complex.magnitude)*E_complex.units, None
        elif param == "E''":
            return np.imag(E_complex.magnitude)*E_complex.units, None
        else:
            raise ValueError("Unknown param: "+param)
    elif param.startswith("u"):
        u_complex = fs_test.get_disp()
        if param == "u" or param == "u*":
            return np.abs(u_complex), None
        elif param == "u'":
            return np.real(u_complex.magnitude)*u_complex.units, None
        elif param == "u''":
            return np.imag(u_complex.magnitude)*u_complex.units, None
        else:
            raise ValueError("Unknown param: "+param)
    elif param.startswith("F"):
        F_complex = fs_test.get_force()
        if param == "F" or param == "F*":
            return np.abs(F_complex), None
        elif param == "F'":
            return np.real(F_complex.magnitude)*F_complex.units, None
        elif param == "F''":
            return np.imag(F_complex.magnitude)*F_complex.units, None
        else:
            raise ValueError("Unknown param: "+param)
    else:
        raise ValueError("Invalid param"+param)
