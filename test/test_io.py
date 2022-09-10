#!python3
# -*- coding: utf-8 -*-
#
# test_io.py
#
# Tests I/O functions provided by the `helpers.io` module.
#
# Author:   Connor D. Pierce
# Created:  2022-09-08 00:04:35
# Modified: 2022-09-08 10:02:36
#
# Copyright (C) 2022 Connor D. Pierce
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later


import logging
import numpy as np
import sys
import unittest

from helpers import io
# io.logger.setLevel(logging.DEBUG)
# io.logger.addHandler(logging.StreamHandler(sys.stderr))


class TestSweepTest(unittest.TestCase):
    
    def setUp(self):
        self.N0 = 2
        self.N1 = 3
        self.T0 = 4
        self.T1 = 5
        self.P = 6
        
        count = 0
        rows = []
        for t0 in range(self.T0):
            for n0 in range(self.N0):
                for t1 in range(self.T1):
                    for n1 in range(self.N1):
                        rows.append(
                            [n0, n1]
                            + [0.5*(count+j) for j in range(self.P)]
                        )
                        count += self.P
        self.raw_data = np.array(rows)
    
    def test_load(self):
        ## (1) Test that invalid column specs are rejected
        
        # (1.a) all columns explicitly numbered
        with self.subTest("Test (1.a.1)"):
            # incorrect number of columns for data provided
            self.assertRaises(
                ValueError,
                io.SweepTest().load,
                self.raw_data,
                "x0,y0",
            )
        with self.subTest("Test (1.a.2)"):
            # correct number of columns; repeated ind. var. numbering
            self.assertRaises(
                ValueError,
                io.SweepTest().load,
                self.raw_data,
                ",".join(["x0"]*2 + [f"y{i}" for i in range(self.P)]),
            )
        with self.subTest("Test (1.a.3)"):
            # correct number of columns; repeated dep. var. numbering
            self.assertRaises(
                ValueError,
                io.SweepTest().load,
                self.raw_data,
                ",".join(["x0", "x1"] + ["y0"]*self.P),
            )
        
        # (1.b) some columns not explicitly numbered
        with self.subTest("Test (1.b.1)"):
            # correct number of columns; repeated dep. var. numbers
            self.assertRaises(
                ValueError,
                io.SweepTest().load,
                self.raw_data,
                ",".join(["x", "x1"] + ["y0"]*self.P),
            )
        with self.subTest("Test (1.b.2)"):
            # correct number of columns; some repeated dep. var. numbers
            self.assertRaises(
                ValueError,
                io.SweepTest().load,
                self.raw_data,
                ",".join(["x", "x1", "y", "y"] + ["y0"]*(self.P-2)),
            )
        
        # (1.c) expanding specifiers used
        with self.subTest("Test (1.c.1)"):
            # correct number of cols with expanding spec; repeated ind. var.
            self.assertRaises(
                ValueError,
                io.SweepTest().load,
                self.raw_data,
                "x0,x0,:y"
            )
        with self.subTest("Test (1.c.2)"):
            # correct number of cols with expanding spec; repeated dep. var.
            self.assertRaises(
                ValueError,
                io.SweepTest().load,
                self.raw_data,
                "x0,x1,y0,y0,:y",
            )
        with self.subTest("Test (1.c.3)"):
            # multiple expanding specs given
            self.assertRaises(
                ValueError,
                io.SweepTest().load,
                self.raw_data,
                "x,:,y,:",
            )
        with self.subTest("Test (1.c.4)"):
            # invalid syntax for expanding spec
            self.assertRaises(
                ValueError,
                io.SweepTest().load,
                self.raw_data,
                "x,x,y:",
            )
        
        # (1.d) invalid variable specs
        with self.subTest("Test (1.d.1)"):
            # invalid variable type
            self.assertRaises(
                ValueError,
                io.SweepTest().load,
                self.raw_data,
                "a,x,:y",
            )
        with self.subTest("Test (1.d.2)"):
            # invalid variable number
            self.assertRaises(
                ValueError,
                io.SweepTest().load,
                self.raw_data,
                "x,xA,:y",
            )
        
        ## (2) Test that correct specifications are parsed correctly
        
        with self.subTest("Test (2.a)"):
            # Fully explicit numbering
            st = io.SweepTest(
                self.raw_data,
                ",".join(["x1,x0"]+[f"y{i}" for i in range(self.P)]),
            )
            self.assertEqual(
                st.shape,
                {
                    0: self.N1,
                    1: self.N0,
                    2: self.P,
                    3: self.T1,
                    4: self.T0,
                    "_Col1": self.N1,
                    "_Col0": self.N0,
                    io.DependentVariable: self.P,
                    "_Col1 trial": self.T1,
                    "_Col0 trial": self.T0,
                },
            )
            self.assertEqual(len(st.names), 2+self.P)
            for i, Tn in zip([1, 0], [self.T0, self.T1]):
                self.assertEqual(st._names[f"x{i}"].axis, i)
                self.assertEqual(st._names[f"x{i}"].t_axis, i + 3)
                self.assertEqual(st._names[f"x{i}"].Tn, Tn)
        with self.subTest("Test (2.b)"):
            # Implicit numbering; fully specified
            st = io.SweepTest(self.raw_data, ",".join(["x"]*2 + ["y"]*self.P))
            self.assertEqual(len(st.names), 2+self.P)
        with self.subTest("Test (2.c)"):
            # Expanding specs
            with self.subTest("Test (2.c.1)"):
                # Ind. var. expanded
                st = io.SweepTest(self.raw_data, ":x"+",y"*self.P)
                self.assertEqual(len(st.names), 2+self.P)
            with self.subTest("Test (2.c.2)"):
                # Dep. var. expanded
                st = io.SweepTest(self.raw_data, "x,x,:y")
                self.assertEqual(len(st.names), 2+self.P)
        
        #TODO: add other tests to ensure col spec was correctly parsed, e.g.:
        #
    
    def test_dim(self):
        st = io.SweepTest(self.raw_data, ",".join(["x"]*2 + ["y"]*self.P))
        self.assertEqual(st.dim, 5)
    
    def test_shape(self):
        st = io.SweepTest(self.raw_data, ",".join(["x"]*2 + ["y"]*self.P))
        shape = {
            0: self.N0,
            1: self.N1,
            2: self.P,
            3: self.T0,
            4: self.T1,
            "_Col0": self.N0,
            "_Col1": self.N1,
            "_Col0 trial": self.T0,
            "_Col1 trial": self.T1,
            io.DependentVariable: self.P
        }
        io.logger.debug("axes are " + str(st.axes))
        with self.subTest("Has all keys"):
            self.assertEqual(len(st.shape), len(shape))
        
        for key in st.shape:
            with self.subTest("key = " + str(key)):
                self.assertIn(key, shape)
                self.assertEqual(st.shape[key], shape[key])
    
    @unittest.skip
    def test_indexing(self):
        st = io.SweepTest(self.raw_data, ",".join(["x"]*2 + ["y"]*self.P))
        
        with self.subTest("Dependent variable scalar indexing"):
            var_name = "_Col4"
            st2 = st[var_name]
            self.assertEqual(st2.dim, st.dim-1)
            for dv in st.dep_vars:
                self.assertNotIn(dv.name, st2.names)
            self.assertEqual(len(st2.dep_vars), 1)
            self.assertEqual(st2.dep_vars[0].name, var_name)
            self.assertEqual(st2.dep_vars[0]._name, st.names[var_name]._name)
            self.assertIsNone(st2.dep_vars[0].axis)
            self.assertIsNone(st2.dep_vars[0].idx)
            
            st2 = st[st.dep_vars[0]]
            self.assertEqual(st2.dim, st.dim-1)
            for dv in st.dep_vars:
                self.assertNotIn(dv.name, st2.names)
            self.assertEqual(len(st2.dep_vars), 1)
            self.assertEqual(st2.dep_vars[0].name, st.dep_vars[0].name)
            self.assertEqual(st2.dep_vars[0]._name, st.dep_vars[0]._name)
            self.assertIsNone(st2.dep_vars[0].axis)
            self.assertIsNone(st2.dep_vars[0].idx)
        
        with self.subTest("Complex slicing"):
            st2 = st[:, 1, 3, 2:3, 2:6:2]
            self.assertIn("_Col0", st2.names)
            self.assertNotIn("_Col1", st2.names)
            for iv in st2.ind_vars:
                if iv.name == "_Col1":
                    self.assertIsNone(iv.axis)
            self.assertNotIn(io.DependentVariable, st2.names)
            for dv in st2.dep_vars:
                self.assertIsNone(dv.axis)
                self.assertIsNone(dv.idx)
            self.assertIn("_Col0 trial", st2.names)
            self.assertIn("_Col1 trial", st2.names)
            self.assertEqual(
                st2.shape,
                {
                    0: self.N0,
                    1: 1,
                    2: 2,
                    "_Col0": self.N0,
                    "_Col0 trial": 1,
                    "_Col1 trial": 2,
                }
            )
            
            st2 = st[{
                "_Col0": slice(None),
                "_Col1": 1,
                io.DependentVariable: 3,
                "_Col0 trial": slice(2,3),
                "_Col1 trial": slice(2,6,2),
            }]
            self.assertIn("_Col0", st2.names)
            self.assertNotIn("_Col1", st2.names)
            for iv in st2.ind_vars:
                if iv.name == "_Col1":
                    self.assertIsNone(iv.axis)
            self.assertNotIn(io.DependentVariable, st2.names)
            for dv in st2.dep_vars:
                self.assertIsNone(dv.axis)
                self.assertIsNone(dv.idx)
            self.assertIn("_Col0 trial", st2.names)
            self.assertIn("_Col1 trial", st2.names)
            self.assertEqual(
                st2.shape,
                {
                    0: self.N0,
                    1: 1,
                    2: 2,
                    "_Col0": self.N0,
                    "_Col0 trial": 1,
                    "_Col1 trial": 2
                }
            )