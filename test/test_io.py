#!python3
# -*- coding: utf-8 -*-
#
# test_io.py
#
# Tests I/O functions provided by the `helpers.io` module.
#
# Author:   Connor D. Pierce
# Created:  2022-09-08 00:04:35
# Modified: 2022-09-12 15:33:33
#
# Copyright (C) 2022 Connor D. Pierce
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
        
        with self.subTest("Test (2.a): Fully explicit numbering"):
            st = io.SweepTest(
                self.raw_data,
                ",".join(["x1,x0"]+[f"y{i}" for i in range(self.P)]),
            )
            self.assertEqual(st.N, 2)
            self.assertEqual(st.P, 6)
            self.assertEqual(
                st.shape,
                {
                    0: self.N1,
                    1: self.N0,
                    2: self.P,
                    3: self.T1,
                    4: self.T0,
                    "_Col0": self.N0,
                    "_Col1": self.N1,
                    io.DependentVariable: self.P,
                    "_Col0 trial": self.T0,
                    "_Col1 trial": self.T1,
                },
            )
            self.assertEqual(len(st.names), 4+self.P)
            for s in st.names:
                self.assertIn(s, ("_Col0", "_Col1", "_Col0 trial", "_Col1 trial") + tuple([f"_Col{i}" for i in range(2,8)]))
            self.assertEqual(len(st._names), 4+self.P)
            for s in st._names:
                self.assertIn(s, ("x0", "x1", "x0_trial", "x1_trial") + tuple([f"y{i}" for i in range(6)]))
            self.assertEqual(len(st.axes), 5)
            self.assertEqual(st.axes[0]._name, "x0")
            self.assertEqual(st.axes[1]._name, "x1")
            self.assertEqual(st.axes[2], io.DependentVariable)
            self.assertEqual(st.axes[3]._name, "x0_trial")
            self.assertEqual(st.axes[4]._name, "x1_trial")
            self.assertEqual(st.axes[0].name, "_Col1")
            self.assertEqual(st.axes[1].name, "_Col0")
            self.assertEqual(st.axes[3].name, "_Col1 trial")
            self.assertEqual(st.axes[4].name, "_Col0 trial")
            self.assertIsNotNone(st.axes[0].trial)
            self.assertIsNotNone(st.axes[1].trial)
            self.assertEqual(st.axes[0].trial.axis, 3)
            self.assertEqual(st.axes[1].trial.axis, 4)
            self.assertEqual(st.names["_Col0"].trial.axis, 4)
        
        def _test_default_ordered(st):
            self.assertEqual(st.N, 2)
            self.assertEqual(st.P, 6)
            self.assertEqual(
                st.shape,
                {
                    0: self.N0,
                    1: self.N1,
                    2: self.P,
                    3: self.T0,
                    4: self.T1,
                    "_Col0": self.N0,
                    "_Col1": self.N1,
                    io.DependentVariable: self.P,
                    "_Col0 trial": self.T0,
                    "_Col1 trial": self.T1,
                },
            )
            self.assertEqual(len(st.names), 4+self.P)
            for s in st.names:
                self.assertIn(s, ("_Col0", "_Col1", "_Col0 trial", "_Col1 trial") + tuple([f"_Col{i}" for i in range(2,8)]))
            self.assertEqual(len(st._names), 4+self.P)
            for s in st._names:
                self.assertIn(s, ("x0", "x1", "x0_trial", "x1_trial") + tuple([f"y{i}" for i in range(6)]))
            self.assertEqual(len(st.axes), 5)
            self.assertEqual(st.axes[0]._name, "x0")
            self.assertEqual(st.axes[1]._name, "x1")
            self.assertEqual(st.axes[2], io.DependentVariable)
            self.assertEqual(st.axes[3]._name, "x0_trial")
            self.assertEqual(st.axes[4]._name, "x1_trial")
            self.assertEqual(st.axes[0].name, "_Col0")
            self.assertEqual(st.axes[1].name, "_Col1")
            self.assertEqual(st.axes[3].name, "_Col0 trial")
            self.assertEqual(st.axes[4].name, "_Col1 trial")
            self.assertIsNotNone(st.axes[0].trial)
            self.assertIsNotNone(st.axes[1].trial)
            self.assertEqual(st.axes[0].trial.axis, 3)
            self.assertEqual(st.axes[1].trial.axis, 4)
            self.assertEqual(st.names["_Col0"].trial.axis, 3)
        with self.subTest("Test (2.b) Implicit numbering; fully specified"):
            st = io.SweepTest(self.raw_data, ",".join(["x"]*2 + ["y"]*self.P))
            _test_default_ordered(st)
        with self.subTest("Test (2.c)"):
            # Expanding specs
            with self.subTest("Test (2.c.1): expanded independent variable"):
                # Ind. var. expanded
                st = io.SweepTest(self.raw_data, ":"+",y"*self.P)
                _test_default_ordered(st)
            with self.subTest("Test (2.c.2)"):
                # Dep. var. expanded
                st = io.SweepTest(self.raw_data, "x,x,:y")
                _test_default_ordered(st)
    
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
        
        with self.subTest("Basic slicing (tuple)"):
            raise unittest.SkipTest("basic slicing (tuple)")
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
        
        with self.subTest("Basic slicing (dict)"):
            raise unittest.SkipTest("basic slicing (dict)")
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
        
        with self.subTest("Advanced slicing; ints only (tuple)"):
            st2 = st[:, 1, 3, 2:3, [2,4]]
            # Test correctness of st2.names
            for s in st2.names:
                self.assertIn(s, ("_Col0", "_Col1", "_Col5", "_Col0 trial", "_Col1 trial"))
            self.assertEqual(st2.names["_Col0"].axis, 0)
            self.assertIsNone(st2.names["_Col1"].axis)
            self.assertIsNone(st2.names["_Col5"].axis)
            self.assertEqual(st2.names["_Col0 trial"].axis, 1)
            self.assertEqual(st2.names["_Col1 trial"].axis, 2)
            # Test correctness of st2._names
            for s in st2._names:
                self.assertIn(s, ("x0", "x1", "y3", "x0_trial", "x1_trial"))
            self.assertEqual(st2._names["x0"].axis, 0)
            self.assertIsNone(st2._names["x1"].axis)
            self.assertIsNone(st2._names["y3"].axis)
            self.assertEqual(st2._names["x0_trial"].axis, 1)
            self.assertEqual(st2._names["x1_trial"].axis, 2)
            # Test correctness of axis assignments
            self.assertEqual(len(st2.axes), 3)
            self.assertEqual(st2.axes[0].name, "_Col0")
            self.assertEqual(st2.axes[1].name, "_Col0 trial")
            self.assertEqual(st2.axes[2].name, "_Col1 trial")
            self.assertTrue(np.allclose(st2.axes[0].values, np.array([0.0, 1.0])))
            self.assertTrue(np.allclose(st2.axes[1].trials, np.array([2])))
            self.assertTrue(np.allclose(st2.axes[2].trials, np.array([2, 4])))
            # Test correctness of dependent variables
            self.assertEqual(len(st2.dep_vars), 1)
            for dv in st2.dep_vars:
                self.assertIsNone(dv.axis)
                self.assertIsNone(dv.idx)
                self.assertEqual(dv.name, "_Col5")
                self.assertEqual(dv._name, "y3")
            # Test correctness of shape
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
        
        with self.subTest("Advanced slicing; floats (tuple)"):
            st2 = st[np.array([0.0, 1.0]), 1, 3, 2:3, [2,4]]
            # Test correctness of st2.names
            for s in st2.names:
                self.assertIn(s, ("_Col0", "_Col1", "_Col5", "_Col0 trial", "_Col1 trial"))
            self.assertEqual(len(st2.names), 5)
            self.assertEqual(st2.names["_Col0"].axis, 0)
            self.assertIsNone(st2.names["_Col1"].axis)
            self.assertIsNone(st2.names["_Col5"].axis)
            self.assertEqual(st2.names["_Col0 trial"].axis, 1)
            self.assertEqual(st2.names["_Col1 trial"].axis, 2)
            # Test correctness of st2._names
            for s in st2._names:
                self.assertIn(s, ("x0", "x1", "y3", "x0_trial", "x1_trial"))
            self.assertEqual(len(st2._names), 5)
            self.assertEqual(st2._names["x0"].axis, 0)
            self.assertIsNone(st2._names["x1"].axis)
            self.assertIsNone(st2._names["y3"].axis)
            self.assertEqual(st2._names["x0_trial"].axis, 1)
            self.assertEqual(st2._names["x1_trial"].axis, 2)
            # Test correctness of axis assignments
            self.assertEqual(len(st2.axes), 3)
            self.assertEqual(st2.axes[0].name, "_Col0")
            self.assertEqual(st2.axes[1].name, "_Col0 trial")
            self.assertEqual(st2.axes[2].name, "_Col1 trial")
            self.assertTrue(np.allclose(st2.axes[0].values, np.array([0.0, 1.0])))
            self.assertTrue(np.allclose(st2.axes[1].trials, np.array([2])))
            self.assertTrue(np.allclose(st2.axes[2].trials, np.array([2, 4])))
            # Test correctness of dependent variables
            self.assertEqual(len(st2.dep_vars), 1)
            for dv in st2.dep_vars:
                self.assertIsNone(dv.axis)
                self.assertIsNone(dv.idx)
                self.assertEqual(dv.name, "_Col5")
                self.assertEqual(dv._name, "y3")
            # Test correctness of shape
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
    
    def test_reshape_adv_idx(self):
        with self.subTest("N = 1, i = 0"):
            self.assertEqual(io.SweepTest.reshape_adv_idx([1,2,3], 1, 0), [1, 2, 3])
        with self.subTest("N = 2, i = 0"):
            self.assertEqual(io.SweepTest.reshape_adv_idx([1,2,3], 2, 0), [[1], [2], [3]])
        with self.subTest("N = 2, i = 1"):
            self.assertEqual(io.SweepTest.reshape_adv_idx([1,2,3], 2, 1), [1, 2, 3])
        with self.subTest("N = 3, i = 0"):
            self.assertEqual(io.SweepTest.reshape_adv_idx([1,2,3], 3, 0), [[[1]], [[2]], [[3]]])
        with self.subTest("N = 3, i = 1"):
            self.assertEqual(io.SweepTest.reshape_adv_idx([1,2,3], 3, 1), [[1], [2], [3]])
        with self.subTest("N = 3, i = 2"):
            self.assertEqual(io.SweepTest.reshape_adv_idx([1,2,3], 3, 2), [1, 2, 3])