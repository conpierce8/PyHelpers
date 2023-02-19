#! python3
# -*- coding: utf-8 -*-
#
# fs.py
#
# Functions for plotting frequency-sweep results.
#
# Author:   Connor D. Pierce
# Created:  2021-03-31 11:40
# Modified: 2023-02-18 15:20:08
#
# Copyright (c) 2021-2023 Connor D. Pierce
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


## Imports
import cycler
import helpers
import logging
import matplotlib as mpl
import numpy as np
import pint
import typing

from helpers.plots import FigureRegistry, nudge, get_style
from helpers.plots.utils import linestyle_cycler, linewidth_cycler
from helpers.units import ureg, Qty
from helpers.utils import EmptyObject
from matplotlib import cm, pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.linalg import lstsq


__all__ = [
    "plot_a_to_v_transmission",
    "plot_v_to_v_transmission",
    "plot_phase",
    "plot_raw_voltage",
    "plot_raw_time_dep",
    "plot_vel",
    "plot_acc",
    "plot_vel_as_disp",
    "plot_acc_as_disp",
    "plotModeShapes",
    "plot_fs_test",
]


## Configure logging
logger = logging.getLogger("helpers.plots.fs")
# TODO: finish configuration (i.e. check if this logger has a handler, if not,
#      add a handler; set level)


## Functions
def plot_a_to_v_transmission(ax, params):
    """Plots transmission on the specified axis for the tests specified in
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
    """

    if params["InputSens"] and params["OutputSens"]:
        vi_sens = params["InputSens"]
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

    __plot_transmission(
        ax,
        datain,
        dataout,
        normalize,
        lambda freq, input, output: 2 * np.pi * freq * a_sens * v_sens * output / input,
        params,
    )


def plot_v_to_v_transmission(ax, params):
    """Plots transmission on the specified axis for the tests specified in
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
    """

    if params["InputSens"] and params["OutputSens"]:
        vi_sens = params["InputSens"]
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

    __plot_transmission(
        ax,
        datain,
        dataout,
        normalize,
        lambda freq, inp, outp: (vo_sens / vi_sens) * outp / inp,
        params,
    )
    return False


def plot_phase(ax, params):
    """Plots transmission on the specified axis for the tests specified in
    params.

    ax - Matplotlib Axes object
    params - dictionary-based data structure having the following form:
             {
                "Input":<input file name>,
                "Legend":<legend entry or False>,
                "LineStyle":[<style1>, <style2>, ..., <styleN>] OR False,
                "Color":[<color1>, <color2>, ..., <colorM>] OR False
             }
    """

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

        f = datain[sweepStartIdx[-1] : pIdx[j][1] + 1, fCol]
        (h,) = plt.plot(f, avg_in, plotstyle, color=plotcolor)

        if params["Legend"]:
            h.set_label(params["Legend"])

    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Phase [${}^\circ$]")
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
    f = lambda freq, data: v_sens * data
    __plot_manipulated_voltage(ax, params, f)
    return


def plot_acc(ax, params):
    a_sens = params["InputSens"]
    f = lambda freq, data: data / a_sens
    __plot_manipulated_voltage(ax, params, f)
    return


def plot_vel_as_disp(ax, params):
    v_sens = params["InputSens"]
    f = lambda freq, data: v_sens * data / (2 * np.pi * freq)
    __plot_manipulated_voltage(ax, params, f)
    return


def plot_acc_as_disp(ax, params):
    a_sens = params["InputSens"]
    f = lambda freq, data: data / (a_sens * (2 * np.pi * freq) ** 2)
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
        sweepStartIdx.append(pIdx[i][1] + 1)
        for j in range(len(sweepStartIdx) - 1):
            swpRange = np.arange(sweepStartIdx[j], sweepStartIdx[j + 1])

            plotcolor, plotstyle = get_style(i, params)

            f = datain[swpRange, fCol]
            (h,) = plt.plot(
                f, fn(f, datain[swpRange, fCol + 1]), plotstyle, color=plotcolor
            )
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

        f = datain[sweepStartIdx[-1] : pIdx[j][1] + 1, fCol]
        avg_T = T_fun(f, avg_in, avg_out)
        if normalize:
            avg_T = avg_T / avg_T[0]

        h = plt.plot(f, 20 * np.log10(avg_T), plotstyle, color=plotcolor)

        if leg:
            h.set_label(leg[j % len(leg)])

    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Transmission Amplitude [dB]")
    plt.grid(True)


def plotModeShapes(fig, files, sens, freqs, animate=False, decomp=None):
    # Check that each file contains the same frequencies
    freq = files[0][0][:, 0]

    for file in files:
        if not np.all(freq == file[0][:, 0]):
            raise ValueError("Frequency mismatch between files")

    # Check that all specified frequencies are in the files
    for f in freqs:
        if not f in freq:
            raise ValueError("Invalid frequency specified")

    # Plot mode shape for each frequency
    N = len(files)
    ampl = sens * np.stack([files[i][0][:, 1] for i in range(N)], axis=1)
    phas = np.stack([files[i][0][:, 2] for i in range(N)], axis=1)
    vel = ampl * np.exp(1j * np.pi * phas / 180)
    count = 1
    x = np.arange(1, N + 1)

    genFunc = lambda vecAmpl: (lambda t: np.real(np.exp(1j * t) * vecAmpl))
    generators = []
    curves = []
    for f in freqs:
        for j in range(freq.size):
            if freq[j] == f:
                break

        ax = fig.add_subplot(len(freqs), 1, count)

        velR = genFunc(vel[j, :])
        generators.append(velR)

        # Format axes
        lim = 1.1 * np.amax(ampl[j, :])
        ax.set_ylim([-lim, lim])
        ax.spines["bottom"].set_position(("data", 0))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylabel("{0:.0f} Hz".format(freq[j]))

        # Plot the mode shape at time t=0
        (h,) = ax.plot(x, velR(0), "k-o", linewidth=1.5, markersize=4)
        curves.append(h)

        # Decompose the mode using the vectors in `decomp`, if provided
        if isinstance(decomp, np.ndarray):
            modeAmpl, *_ = lstsq(decomp, vel[j, :])

            for i in range(modeAmpl.size):
                velMode = genFunc(modeAmpl[i] * decomp[:, i])
                (h,) = ax.plot(x, velMode(0), "-o", linewidth=1.0, markersize=2)

                generators.append(velMode)
                curves.append(h)

            velTot = genFunc(decomp.dot(modeAmpl))
            (h,) = ax.plot(x, velTot(0), "w--o", linewidth=1.0, markersize=1.5)

            generators.append(velTot)
            curves.append(h)

        count += 1
    fig.tight_layout()

    # Animate the plot, if desired
    if animate:

        def animFunc(t):
            for c, g in zip(curves, generators):
                c.set_ydata(g(t))

        anim = FuncAnimation(
            fig,
            animFunc,
            interval=10,
            blit=False,
            frames=np.arange(0, 2 * np.pi, np.pi / 100),
        )
        return anim


def plot_fs_test(
    fs_test: helpers.io.FreqSweepTest,
    fig: typing.Union[mpl.figure.Figure, mpl.figure.Axes],
    qtys: tuple[str, str] = ("f", "E*"),
    units: tuple[str, str] = ("Hz", "MPa"),
    averaged: bool = False,
    axes_types: tuple[str, str] = ("lin", "lin"),
    subplot_params: typing.Union[bool, str, typing.Iterable[str]] = False,
    param_vals: None or tuple[tuple[str, ...], list[tuple[float, ...]]] = None,
    param_cycler: cycler.cycler = linestyle_cycler,
    trial_cycler: cycler.cycler = linewidth_cycler,
):
    """Plot the results of a frequency-sweep test.

    By default, this method plots all data contained in the FreqSweepTest, with
    the data separated so that each unique combination of parameters and each
    unique trial of that parameter combination is represented by its own curve.

    The data is plotted on a single `Axes` by default. If an `Figure` object is
    passed to the `fig` argument, the plots can be separated onto automatically-
    generated subplots by supplying `subplot_params=True`. If an `Axes` object
    is passed to the `fig` argument, `subplot_params` is ignored.

    The default is to plot the data for all parameters; however, the set of
    parameters plotted can be restricted by `param_vals` argument. The value
    passed to this argument should be a 2-`tuple`. `param_vals[0]` should be an
    `N_param`-tuple of `str` that names the parameters used to restrict the
    plotting. `param_vals[1]` should be a list of `N_param`-tuples of floats,
    where each entry identifies a combination of parameters to be plotted. All
    parameter values will be included for any parameter in `fs_test` not named
    in `param_vals[0]`.

    The format of the plots can be controlled with the `param_cycler` and
    `trial_cycler` arguments.

    Parameters
    ----------
    `fs_test` : data to be plotted
    `fig` : plot object where the data will be displayed
    `qtys` : The quantities to plot on the x- and y-axes. Allowable values are
             ("f", "A", "u", "F", "E", "d") for frequency, amplitude,
             displacement, force, modulus, and loss angle, respectively. The
             phasor values "u", "F", and "E" can be suffixed with "*", "'", or
             "''" to plot the magnitude, real, or imaginary parts, respectively.
             If no suffix is given, the magnitude is assumed.
    `units` : units in which to plot the data
    `averaged` : whether to take the average over all trials or trials of a
                 specified variable
    `axes_types` : scaling ("lin" or "log") for the x- and y-axis
    `subplot_params` : parameter(s) to use for splitting into subplots
    `param_vals` : restrict the data plotted to only these parameter value(s)
    `param_cycler` : style to distinguish between parameter values
    `trial_cycler` : style to distinguish between trial numbers
    """

    if averaged:
        raise NotImplementedError("averaging is not implemented yet")

    if subplot_params and not isinstance(fig, mpl.figure.Figure):
        raise ValueError("fig must be a Figure to use subplots")

    valid_x_qtys = ("f", "A")
    valid_y_qtys = (
        "E",
        "E*",
        "E'",
        "E''",
        "u",
        "u*",
        "u'",
        "u''",
        "F",
        "F*",
        "F'",
        "F''",
        "d",
    )
    x_qty, y_qty = qtys
    if not x_qty in valid_x_qtys:
        raise ValueError("Invalid quantity for x axis")
    if not y_qty in valid_y_qtys:
        raise ValueError("Invalid quantity for y axis")

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

    # Fill out the matrix of parameter value combinations if there are unspecified
    # parameters
    all_params = [iv.name for iv in fs_test.disp.raw.ind_vars]
    if x_qty == "f":
        all_params.remove("Frequency")
    else:
        all_params.remove("Amplitude")
    translation = {"f": "Frequency", "A": "Amplitude"}
    if param_vals is None:
        _param_vals = (tuple(all_params), _get_param_matrix(fs_test, all_params))
    else:
        if len(param_vals[0]) == len(all_params):
            _param_vals = (tuple([translation[s] for s in param_vals[0]]), param_vals[1])
        else:
            unspecified = all_params.copy()
            for p in param_vals[0]:
                unspecified.remove(p)
            unspecified_vals = _get_param_matrix(fs_test, unspecified)
            _param_vals = (tuple([translation[s] for s in param_vals[0]]) + tuple([translation[s] for s in unspecified]), [p + p2 for p in param_vals[0] for p2 in unspecified_vals])

    # Determine a suitable layout of subplots that makes each subplot as close to square
    # as possible
    if subplot_params:
        N_subplots = len(_param_vals[1])
        if fig.figsize[0] > fig.figsize[1]:
            layouts = [(n, int(np.ceil(N_subplots / n))) for n in range(1, int(np.floor(np.sqrt(N_subplots))) + 1)]
            ratios = np.array([x[1] / x[0] for x in layouts])
            layout_i = np.argmin(np.abs(ratios - fig.figsize[0] / fig.figsize[1]))
            layout = layouts[i]
        else:
            layouts = [(int(np.ceil(N_subplots / n)), n) for n in range(1, int(np.floor(np.sqrt(N_subplots))) + 1)]
            ratios = np.array([x[0] / x[1] for x in layouts])
            layout_i = np.argmin(np.abs(ratios - fig.figsize[1] / fig.figsize[0]))
            layout = layouts[i]
        ax = None
    elif isinstance(fig, mpl.figure.Figure):
        ax = fig.add_subplot(1, 1, 1)
    else:
        ax = fig

    # Create a matrix of trials
    _trials = _get_trial_matrix(fs_test, _param_vals[0])

    # Do the plotting
    for i, x, y, x_label, y_label, style in _plot_fs_test_get_slices(fs_test, qtys, units, _param_vals, _trials, param_cycler, trial_cycler):
        if subplot_params:
            ax = fig.add_subplot(layout[0], layout[1], i + 1)

        plt_cmd = ax.__getattribute__(plt_cmd_str)
        plt_cmd(x, y, **style)
        ax.set_xlabel(x_label.format(unit=units[0]))
        ax.set_ylabel(y_label.format(unit=units[1]))
    return

def _get_param_matrix(fs_test, params):
    def get_combos(fs_test, params, lv=0, prefix=tuple()):
        for v in fs_test.disp.raw.names[params[lv]].values:
            suffix = (v, )
            if lv == len(params) - 1:
                yield prefix + suffix
            else:
                yield from get_combos(fs_test, params, lv + 1, prefix + suffix)
    return [combo for combo in get_combos(fs_test, params)]

def _get_trial_matrix(fs_test, params):
    def get_combos(fs_test, params, lv=0, prefix=tuple()):
        for v in fs_test.disp.raw.names[params[lv]].trial.trials:
            suffix = (v, )
            if lv == len(params) - 1:
                yield prefix + suffix
            else:
                yield from get_combos(fs_test, params, lv + 1, prefix + suffix)
    return tuple([fs_test.disp.raw.names[p].trial.name for p in params]), [combo for combo in get_combos(fs_test, params)]

def _plot_fs_test_get_slices(fs_test, qtys, units, _param_vals, _trials, param_cycler, trial_cycler):
    p_iter = param_cycler.__iter__()
    for i, p_val in enumerate(_param_vals[1]):
        try:
            style = p_iter.__next__()
        except StopIteration:
            p_iter = param_cycler.__iter__()
            style = p_iter.__next__()
        t_iter = trial_cycler.__iter__()
        for t in _trials[1]:
            index = dict(zip(_param_vals[0], p_val))
            index.update(dict(zip(_trials[0], t)))
            # Extract the data to be plotted
            x_unit, y_unit = units
            x_data, x_label = _plot_fs_test_get_data(fs_test, index, qtys[0], False)
            y_data, y_label = _plot_fs_test_get_data(fs_test, index, qtys[1], False)
            x_data.ito(x_unit)
            y_data.ito(y_unit)
            try:
                style.update(t_iter.__next__())
            except StopIteration:
                t_iter = trial_cycler.__iter__()
                style.update(t_iter.__next__())
            yield i, x_data.magnitude, y_data.magnitude, x_label, y_label, style


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
        raise ValueError("Invalid param" + param)


def _plot_fs_test_get_data(fs_test, index, param, averaged):
    index_copy = index.copy()
    # print(index)
    for k in index:
        if fs_test.disp.raw.names[k].axis is None:
            del index_copy[k]
    # print(index_copy)

    if param == "f":
        return fs_test.get_freq(), "Frequency [{unit}]"
    elif param == "A":
        return fs_test.get_ampl(), "Amplitude [{unit}]"
    elif param == "d" or param.startswith("E"):
        E_complex = fs_test.get_complex_modulus(averaged)[index_copy]
        if param == "d":
            return (
                np.imag(E_complex.data[0].data.magnitude) / np.real(E_complex.data[0].data.magnitude) * ureg.dimensionless,
                "Loss factor, $\\tan{{\\delta}}$",
            )
        else:
            if param == "E" or param == "E*":
                return (
                    np.abs(E_complex.data[0].data),
                    "Complex modulus magnitude, $|E^*|$ [{unit}]",
                )
            elif param == "E'":
                return (
                    np.real(E_complex.data[0].data.magnitude) * E_complex.data[0].data.units,
                    "Storage Modulus, $E'$ [{unit}]",
                )
            elif param == "E''":
                return (
                    np.imag(E_complex.data[0].data.magnitude) * E_complex.data[0].data.units,
                    "Loss modulus, $E''$ [{unit}]",
                )
            else:
                raise ValueError("Unknown param: " + param)
    elif param.startswith("u"):
        u_complex = fs_test.get_disp(averaged)[index_copy]
        if param == "u" or param == "u*":
            return np.abs(u_complex.data[0].data), "Displacement magnitude [{unit}]"
        elif param == "u'":
            return (
                np.real(u_complex.data[0].data.magnitude) * u_complex.data[0].data.units,
                "Displacement (real) [{unit}]",
            )
        elif param == "u''":
            return (
                np.imag(u_complex.data[0].data.magnitude) * u_complex.data[0].data.units,
                "Displacement (imag) [{unit}]",
            )
        else:
            raise ValueError("Unknown param: " + param)
    elif param.startswith("F"):
        F_complex = fs_test.get_force(averaged)[index_copy]
        if param == "F" or param == "F*":
            return np.abs(F_complex.data[0].data), "Force magnitude [{unit}]"
        elif param == "F'":
            return (
                np.real(F_complex.data[0].data.magnitude) * F_complex.data[0].data.units,
                "Force (real) [{unit}]",
            )
        elif param == "F''":
            return (
                np.imag(F_complex.data[0].data.magnitude) * F_complex.data[0].data.units,
                "Force (imag) [{unit}]",
            )
        else:
            raise ValueError("Unknown param: " + param)
    else:
        raise ValueError("Invalid param" + param)
