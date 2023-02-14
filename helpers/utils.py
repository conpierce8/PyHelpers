#! python3
# -*- coding: utf-8 -*-
#
# utils.py
#
# Collection of utility functions and classes.
#
# Author:   Connor D. Pierce
# Created:  2023-02-13 15:28:30
# Modified: 2023-02-13 15:29:22
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


__all__ = ["factors", "EmptyObject"]


## Functions
def factors(x):
    """Calculate all factors of the positive integer `x`."""

    return [i for i in range(1, x + 1) if x % i == 0]


## Classes
class EmptyObject:
    """Generic data container with no instance members.

    Use this class by creating an instance and assigning variables to it, e.g.:

    ```
    varName = EmptyObject()
    varName.member1 = 1
    varName.member2 = "Hello, world!"
    varName.member3 = ["It's", "nice", "to", "meet", "you."]
    ```
    """

    pass
