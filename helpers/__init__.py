#! python3
# -*- coding: utf-8 -*-
#
# __init__.py
#
# init file for the `helpers` package. Sets up scientific unit handling via `pint`
# and defines some generic useful functions and classes.
#
# Author:   Connor D. Pierce
# Created:  2020-11-24 16:08
# Modified: 2023-02-13 15:29:57
#
# Copyright (c) 2023 Connor D. Pierce
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


"""
Utilities for scientific unit handling, input/output of experiment and
simulation data, and plotting results.
"""


# Imports
from helpers import io, plots, units
from helpers.units import ureg, Qty
from helpers.utils import factors, EmptyObject


__all__ = ["io", "plots", "units", "ureg", "Qty", "factors", "EmptyObject"]
