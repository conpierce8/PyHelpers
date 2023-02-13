#! python3
# -*- coding: utf-8 -*-
#
# units.py
#
# Customize `pint`'s handling of units.
#
# Author:   Connor D. Pierce
# Created:  2023-02-12 14:49:04
# Modified: 2023-02-12 21:48:50
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


"""Customize `pint` for automatic unit conversions."""


# Imports
import pint


# Customize `pint` for unit conversions in all `helpers` submodules. All modules should
# use the unit registry `ureg` to ensure that data loaded with units can be converted.
ureg = pint.UnitRegistry()
ureg.define("percent = 1/100 = pct")
# Create an abbreviation for `pint.UnitRegistry.Quantity`
Qty = ureg.Quantity
