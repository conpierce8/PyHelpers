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
# Modified: 2022-09-01 15:49:25
#
# Copyright (c) 2020-2022 Connor D. Pierce
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: LGPL-3.0-or-later


"""
Utilities for scientific unit handling, input/output of experiment and
simulation data, and plotting results.
"""


# Set up pint for unit conversions in submodules. All submodules should use the
# unit registry `ureg` to ensure that all data loaded with units can be
# converted.
import pint

ureg = pint.UnitRegistry()
ureg.define("percent = 1/100 = pct")

Qty = ureg.Quantity


### Functions ==================================================================


def factors(x):
    """
    Calculates all factors of the positive integer `x`.
    """

    return [i for i in range(1, x + 1) if x % i == 0]


### Classes ====================================================================


class EmptyObject:
    """Defines a class with no members that can be used as a generic data
    container. Use this class by creating an instance and assigning variables to
    it, e.g.:

    ```
    varName = EmptyObject()
    varName.member1 = 1
    varName.member2 = "Hello, world!"
    varName.member3 = ["It's", "nice", "to", "meet", "you."]
    ```

    """

    pass
