#!python3
# -*- coding: utf-8 -*-
#
# test_io.py
#
# Tests I/O functions provided by the `helpers.io` module.
#
# Author:   Connor D. Pierce
# Created:  2022-09-08 00:04:35
# Modified: 2023-02-16 20:35:24
#
# Copyright (C) 2022 Connor D. Pierce
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


"""Test the `helpers.io` module.

This module uses the builtin `unittest` framework to test the classes exported
by the `helpers.io` module. Executing this module with

```
python -m unittest -v test/test_io.py
```

runs all the tests in this module and prints the results (pass/fail/error).

Classes:
--------
TestSweepTest(unittest.TestCase)
    Test the helpers.io.SweepTest class.
TestIndependentVariable(unittest.TestCase)
    Test the helpers.io.IndependentVariable class.
TestTrial(unittest.TestCase)
    Test the helpers.io.Trial class.
"""

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
                            [n0, n1] + [0.5 * (count + j) for j in range(self.P)]
                        )
                        count += self.P
        self.raw_data = np.array(rows)
        self.st = io.SweepTest(self.raw_data, ",".join(["x"] * 2 + ["y"] * self.P))

    def _verify_correct_loading(self, st, default_order=True):
        self.assertEqual(st.N, 2)
        self.assertEqual(st.P, 6)
        self.assertEqual(st.ids["x0"].name, "_Col0" if default_order else "_Col1")
        self.assertEqual(st.ids["x1"].name, "_Col1" if default_order else "_Col0")
        self.assertEqual(
            st.shape,
            {
                "_Col0": self.N0,
                "_Col1": self.N1,
                "Trial(_Col0)": self.T0,
                "Trial(_Col1)": self.T1,
                "x0": self.N0 if default_order else self.N1,
                "x1": self.N1 if default_order else self.N0,
                "x0_t": self.T0 if default_order else self.T1,
                "x1_t": self.T1 if default_order else self.T0,
                io.DependentVariable: self.P,
            },
        )
        self.assertEqual(len(st.names), 4 + self.P)
        for s in st.names:
            self.assertIn(
                s,
                ("_Col0", "_Col1", "Trial(_Col0)", "Trial(_Col1)")
                + tuple([f"_Col{i}" for i in range(2, 8)]),
            )
        self.assertEqual(len(st.ids), 4 + self.P)
        for s in st.ids:
            self.assertIn(
                s,
                ("x0", "x1", "x0_t", "x1_t")
                + tuple([f"y{i}" for i in range(6)]),
            )
        self.assertEqual(len(st.axes), 4)
        self.assertEqual(st.axes[0].id, "x0")
        self.assertEqual(st.axes[1].id, "x0_t")
        self.assertEqual(st.axes[2].id, "x1")
        self.assertEqual(st.axes[3].id, "x1_t")
        self.assertEqual(st.axes[0].name, "_Col0" if default_order else "_Col1")
        self.assertEqual(st.axes[1].name, "Trial(_Col0)" if default_order else "Trial(_Col1)")
        self.assertEqual(st.axes[2].name, "_Col1" if default_order else "_Col0")
        self.assertEqual(st.axes[3].name, "Trial(_Col1)" if default_order else "Trial(_Col0)")
        self.assertIsNotNone(st.axes[0].trial)
        self.assertIsNotNone(st.axes[2].trial)
        self.assertEqual(st.names["_Col0"].t_axis, 1 if default_order else 3)
        self.assertEqual(st.names["_Col1"].t_axis, 3 if default_order else 1)
        self.assertEqual(st.axes[0].t_axis, 1)
        self.assertEqual(st.axes[2].t_axis, 3)
        self.assertEqual(st.ids["x0"].t_axis, 1)
        self.assertEqual(st.ids["x1"].t_axis, 3)

    def test_load(self):
        """Test loading data into a SweepTest."""

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
                ",".join(["x0"] * 2 + [f"y{i}" for i in range(self.P)]),
            )
        with self.subTest("Test (1.a.3)"):
            # correct number of columns; repeated dep. var. numbering
            self.assertRaises(
                ValueError,
                io.SweepTest().load,
                self.raw_data,
                ",".join(["x0", "x1"] + ["y0"] * self.P),
            )

        # (1.b) some columns not explicitly numbered
        with self.subTest("Test (1.b.1)"):
            # correct number of columns; repeated dep. var. numbers
            self.assertRaises(
                ValueError,
                io.SweepTest().load,
                self.raw_data,
                ",".join(["x", "x1"] + ["y0"] * self.P),
            )
        with self.subTest("Test (1.b.2)"):
            # correct number of columns; some repeated dep. var. numbers
            self.assertRaises(
                ValueError,
                io.SweepTest().load,
                self.raw_data,
                ",".join(["x", "x1", "y", "y"] + ["y0"] * (self.P - 2)),
            )

        # (1.c) expanding specifiers used
        with self.subTest("Test (1.c.1)"):
            # correct number of cols with expanding spec; repeated ind. var.
            self.assertRaises(
                ValueError, io.SweepTest().load, self.raw_data, "x0,x0,:y"
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
                ",".join(["x1,x0"] + [f"y{i}" for i in range(self.P)]),
            )
            self._verify_correct_loading(st, False)

        with self.subTest("Test (2.b) Implicit numbering; fully specified"):
            st = io.SweepTest(self.raw_data, ",".join(["x"] * 2 + ["y"] * self.P))
            self._verify_correct_loading(st)
        with self.subTest("Test (2.c)"):
            # Expanding specs
            with self.subTest("Test (2.c.1): expanded independent variable"):
                # Ind. var. expanded
                st = io.SweepTest(self.raw_data, ":" + ",y" * self.P)
                self._verify_correct_loading(st)
            with self.subTest("Test (2.c.2)"):
                # Dep. var. expanded
                st = io.SweepTest(self.raw_data, "x,x,:y")
                self._verify_correct_loading(st)

    def test_dim(self):
        """Test `dim` property."""

        st = io.SweepTest(self.raw_data, ",".join(["x"] * 2 + ["y"] * self.P))
        self.assertEqual(st.dim, 4)

    def test_shape(self):
        """Test `shape` property."""

        st = io.SweepTest(self.raw_data, ",".join(["x"] * 2 + ["y"] * self.P))
        shape = {
            "_Col0": self.N0,
            "_Col1": self.N1,
            "Trial(_Col0)": self.T0,
            "Trial(_Col1)": self.T1,
            "x0": self.N0,
            "x1": self.N1,
            "x0_t": self.T0,
            "x1_t": self.T1,
            io.DependentVariable: self.P,
        }
        io.logger.debug("axes are " + str(st.axes))
        with self.subTest("Has all keys"):
            self.assertEqual(len(st.shape), len(shape))

        for key in st.shape:
            with self.subTest("key = " + str(key)):
                self.assertIn(key, shape)
                self.assertEqual(st.shape[key], shape[key])

    def _verify_dict_index_result(self, st2, scalar_dep=True):
        """Verify correctness of complex indexing with a dict object.

        Verifies that a SweepTest correctly performs dict indexing. Intended to be
        called repeatedly by `test_indexing`.

        Initial state of the SweepTest is assumed to have two independent variables:
        ```
        Name      ID       # of values   Values                     # of trials
        ----      --       -----------   ------                     -----------
        _Col0     x0       self.N0       np.arange(0.0, self.N0)    self.T0
        _Col1     x1       self.N1       np.arange(0.0, self.N1)    self.T1
        ```

        and six dependent variables:

        ```
        Name      ID
        ----      --
        _Col2     y0
        _Col3     y1
        _Col4     y2
        _Col5     y3
        _Col6     y4
        _Col7     y5
        ```

        The SweepTest `st2` is assumed to be the result of the following indexing:
        ```
        Variable        Index
        --------        -----
        _Col0           <all values>
        Trial(_Col0)    vector index (slice or list) selecting only position "2"
        _Col1           scalar index selecting trial 1
        Trial(_Col1)    vector index selecting positions 4 and 2 (in that order)
        <Dependent>     if `scalar_dep`: scalar index selecting position 3
                        if not `scalar_dep`: vector index selecting positions 3 and 4
        ```

        Parameters
        ==========
        `st2` : `SweepTest`
            Result of indexing
        `scalar_dep` : `bool`
            `True` if the dependent variable is indexed by a scalar, `False` if by a vec
        """

        self.assertEqual(st2.N, 2)
        if scalar_dep:
            self.assertEqual(st2.P, 1)
        else:
            self.assertEqual(st2.P, 2)

        # Test correctness of st2.names
        names = {"_Col0", "_Col1", "_Col5", "Trial(_Col0)", "Trial(_Col1)"}
        if not scalar_dep:
            names.update({"_Col6"})
        for s in st2.names:
            self.assertIn(s, names)
        for s in names:
            self.assertIn(s, st2.names)

        # Test correctness of st2.ids
        ids = {"x0", "x1", "y3", "x0_t", "x1_t"}
        if not scalar_dep:
            ids.update({"y4"})
        for s in st2.ids:
            self.assertIn(s, ids)
        for s in ids:
            self.assertIn(s, st2.ids)

        # Test correctness of axis assignments
        self.assertEqual(len(st2.axes), 3)
        #    by axis:
        self.assertEqual(st2.axes[0].name, "_Col0")
        self.assertEqual(st2.axes[1].name, "Trial(_Col0)")
        self.assertEqual(st2.axes[2].name, "Trial(_Col1)")
        #    by name:
        self.assertEqual(st2.names["_Col0"].axis, 0)
        self.assertEqual(st2.names["Trial(_Col0)"].axis, 1)
        self.assertIsNone(st2.names["_Col1"].axis)
        self.assertEqual(st2.names["Trial(_Col1)"].axis, 2)
        #    by id:
        self.assertEqual(st2.ids["x0"].axis, 0)
        self.assertEqual(st2.ids["x0_t"].axis, 1)
        self.assertIsNone(st2.ids["x1"].axis)
        self.assertEqual(st2.ids["x1_t"].axis, 2)

        # Test linking of trials and independent variables
        self.assertEqual(st2.names["_Col0"].t_axis, 1)
        self.assertEqual(st2.names["_Col1"].t_axis, 2)
        self.assertEqual(st2.names["Trial(_Col0)"].iv.axis, 0)
        self.assertIsNone(st2.names["Trial(_Col1)"].iv.axis)

        # Test correctness of indexed independent variable values
        self.assertTrue(np.allclose(st2.names["_Col0"].values, np.arange(0.0, self.N0)[:]))
        self.assertTrue(np.allclose(st2.names["_Col0"].trials, np.arange(0, self.T0)[2]))
        self.assertTrue(
            np.allclose(st2.names["_Col1"].values, np.arange(0.0, self.N1)[1])
        )
        self.assertTrue(
            np.allclose(st2.names["_Col1"].trials, np.arange(0, self.T1)[4:1:-2])
        )

        # Test correctness of dependent variables
        dep_names = ["_Col5"] if scalar_dep else ["_Col5", "_Col6"]
        dep_ids = ["y3"] if scalar_dep else ["y3", "y4"]
        self.assertEqual(len(st2.data), len(dep_names))
        for i, dv in enumerate(st2.data):
            self.assertEqual(dv.idx, i)
            self.assertEqual(dv.name, dep_names[i])
            self.assertEqual(dv.id, dep_ids[i])
            self.assertTrue(np.allclose(dv.data.shape, (self.N0, 1, 2)))

        # Test correctness of shape
        self.assertEqual(st2.dim, 3)
        shape = {
            "_Col0": self.N0,
            "_Col1": 1,
            "Trial(_Col0)": 1,
            "Trial(_Col1)": 2,
            "x0": self.N0,
            "x1": 1,
            "x0_t": 1,
            "x1_t": 2,
            io.DependentVariable: len(dep_names),
        }
        for s in st2.shape:
            self.assertEqual(st2.shape[s], shape[s])

    def test_indexing(self):
        """Test indexing a SweepTest."""

        st = self.st

        with self.subTest("Dependent variable scalar indexing"):
            var_name = "_Col4"
            def _verify_scalar_index_result(st2):
                self.assertEqual(st2.dim, st.dim)
                self.assertEqual(st2.P, 1)
                for dv in st.data:
                    if dv.name != var_name:
                        self.assertNotIn(dv.name, st2.names)
                self.assertEqual(len(st2.data), 1)
                self.assertEqual(st2.data[0].name, var_name)
                self.assertEqual(st2.data[0].id, st.names[var_name].id)
                self.assertTrue(
                    np.allclose(st.names[var_name].data.shape, st2.data[0].data.shape)
                )

            st2 = st[var_name]
            _verify_scalar_index_result(st2)

            st2 = st[st.names[var_name].id]
            _verify_scalar_index_result(st2)

            st2 = st[st.names[var_name]]
            _verify_scalar_index_result(st2)

        with self.subTest("Basic slicing (dict)"):
            st2 = st[
                {
                    "_Col0": slice(None),
                    "_Col1": 1,
                    io.DependentVariable: 3,
                    "Trial(_Col0)": slice(2, 3),
                    "Trial(_Col1)": slice(4, 1, -2),
                }
            ]
            self._verify_dict_index_result(st2)

        with self.subTest("Advanced slicing (dict)"):
            st2 = st[
                {
                    "_Col0": slice(None),
                    "_Col1": 1,
                    io.DependentVariable: [3, 4],
                    "Trial(_Col0)": slice(2, 3),
                    "Trial(_Col1)": [4, 2],
                }
            ]
            self._verify_dict_index_result(st2, False)

        with self.subTest("Advanced slicing, floats (dict)"):
            st2 = st[
                {
                    "_Col0": np.array([0.0, 1.0]),
                    "_Col1": 1.0,
                    io.DependentVariable: 3,
                    "Trial(_Col0)": slice(2, 3),
                    "Trial(_Col1)": [4, 2],
                }
            ]
            self._verify_dict_index_result(st2)

    def test_subsequent_indexing(self):
        st3 = self.st[
            {
                "_Col0": slice(None),
                "_Col1": slice(None),
                "Trial(_Col0)": slice(None),
                "Trial(_Col1)": slice(None),
                io.DependentVariable: [f"y{i}" for i in range(self.P)],
            }
        ]
        with self.subTest("Basic slicing (dict)"):
            st2 = st3[
                {
                    "_Col0": slice(None),
                    "_Col1": 1,
                    io.DependentVariable: 3,
                    "Trial(_Col0)": slice(2, 3),
                    "Trial(_Col1)": slice(4, 1, -2),
                }
            ]
            self._verify_dict_index_result(st2)

        with self.subTest("Advanced slicing (dict)"):
            st2 = st3[
                {
                    "_Col0": slice(None),
                    "_Col1": 1,
                    io.DependentVariable: [3, 4],
                    "Trial(_Col0)": slice(2, 3),
                    "Trial(_Col1)": [4, 2],
                }
            ]
            self._verify_dict_index_result(st2, False)

        with self.subTest("Advanced slicing, floats (dict)"):
            st2 = st3[
                {
                    "_Col0": np.array([0.0, 1.0]),
                    "_Col1": 1.0,
                    io.DependentVariable: 3,
                    "Trial(_Col0)": slice(2, 3),
                    "Trial(_Col1)": [4, 2],
                }
            ]
            self._verify_dict_index_result(st2)

    def test_reshape_adv_idx(self):
        """Test reshaping list of indices into a broadcastable shape."""

        with self.subTest("N = 1, i = 0"):
            self.assertTrue(
                np.allclose(io.reshape_adv_idx([1, 2, 3], 1, 0), [1, 2, 3])
            )
        with self.subTest("N = 2, i = 0"):
            self.assertTrue(
                np.allclose(io.reshape_adv_idx([1, 2, 3], 2, 0), [[1], [2], [3]])
            )
        with self.subTest("N = 2, i = 1"):
            self.assertTrue(
                np.allclose(io.reshape_adv_idx([1, 2, 3], 2, 1), [1, 2, 3])
            )
        with self.subTest("N = 3, i = 0"):
            self.assertTrue(
                np.allclose(io.reshape_adv_idx([1, 2, 3], 3, 0), [[[1]], [[2]], [[3]]])
            )
        with self.subTest("N = 3, i = 1"):
            self.assertTrue(
                np.allclose(io.reshape_adv_idx([1, 2, 3], 3, 1), [[1], [2], [3]])
            )
        with self.subTest("N = 3, i = 2"):
            self.assertTrue(
                np.allclose(io.reshape_adv_idx([1, 2, 3], 3, 2), [1, 2, 3])
            )


class TestIndependentVariable(unittest.TestCase):
    """Test helpers.io.IndependentVariable."""

    def setUp(self):
        self.iv = io.IndependentVariable()
        self.iv.values = 0.5 * np.arange(10)
        self.trial = io.Trial(self.iv)
        self.values = np.array([0, 1, 2, 3, 4, 5, 6])
        self.trial.trials = self.values
        self.trial.axis = 2
        self.iv.trial = self.trial

    def test_copy(self):
        """Test copying an IndependentVariable."""

        self.skipTest("TODO: add test code here")

    def test_getattr(self):
        """Test attributes inherited from the associated Trial."""

        self.assertEqual(self.iv.t_axis, 2)
        self.assertEqual(self.iv.Tn, 7)
        self.assertTrue(np.allclose(self.iv.trials, self.values))

    @unittest.skip
    def test_index(self):
        """Test indexing an IndependentVariable."""

        def _test_values(iv, values):
            self.assertTrue(np.allclose(iv.values, values))
            self.assertTrue(iv.values.flags["OWNDATA"])
            self.assertEqual(len(iv.values.shape), len(values.shape))
            self.assertTrue(np.allclose(iv.values.shape, values.shape))

        with self.subTest("Index by scalar int"):
            iv = self.iv.index(3)
            _test_values(iv, np.array([1.5]))

        with self.subTest("Index by slice"):
            iv = self.iv.index(slice())
            _test_values(iv, 0.5 * np.arange(10))
            iv = self.iv.index(slice(4, 9, 2))
            _test_values(iv, np.array([2.0, 3.0, 4.0]))

        with self.subTest("Index by list"):
            iv = self.iv.index([1, 7, 8, 4])
            _test_values(iv, np.array([0.5, 3.5, 4.0, 2.0]))
            iv = self.iv.index(np.array([[[1]], [[7]], [[8]], [[4]]]))
            _test_values(iv, np.array([0.5, 3.5, 4.0, 2.0]))

        with self.subTest("Index by value"):
            iv = self.iv.index([0.75])
            _test_values(iv, np.array([]))
            iv = self.iv.index(np.array([0.5, 1.0, 3.0, 3.5, 4.7]))
            _test_values(iv, np.array([0.5, 1.0, 3.0, 3.5]))

        with self.subTest("Sequential indexing"):
            iv = self.iv.index(0)
            iv2 = iv.index([1, 2, 3])
            _test_values(iv2, np.array([]))

            iv = self.iv.index([1, 4, 6])

            iv2 = iv.index(3)
            _test_values(iv2, np.array([]))

            iv2 = iv.index(1)
            _test_values(iv2, np.array([2.0]))

            iv2 = iv.index(slice(0, 2))
            _test_values(iv2, np.array([0.5, 2.0]))

            iv2 = iv.index(np.array([0.5, 3.0, 3.1]))
            _test_values(iv2, np.array([0.5, 3.0]))


class TestTrial(unittest.TestCase):
    """Test helpers.io.IndependentVariable."""

    def setUp(self):
        self.iv = io.IndependentVariable()
        self.iv.values = 0.5 * np.arange(10)
        self.iv.name = "VarName"
        self.iv.id = "x0"
        self.iv.col = 3
        self.trial = io.Trial(self.iv)
        self.trial_values = np.array([0, 1, 2, 3, 4, 5, 6])
        self.trial.values = self.trial_values

    def test_getattr(self):
        """Test attributes inherited from associated IndependentVariable."""

        self.assertEqual(self.trial.name, "Trial(VarName)")
        self.assertEqual(self.trial.id, "x0_t")
        self.assertEqual(self.trial.col, 3)

    @unittest.skip
    def test_index(self):
        """Test indexing a Trial."""

        def _test_values(iv, values):
            self.assertTrue(np.allclose(iv.values, values))
            self.assertTrue(iv.values.flags["OWNDATA"])
            self.assertEqual(len(iv.values.shape), len(values.shape))
            self.assertTrue(np.allclose(iv.values.shape, values.shape))

        with self.subTest("Index by scalar int"):
            iv = self.iv.index(3)
            _test_values(iv, np.array([1.5]))

        with self.subTest("Index by slice"):
            iv = self.iv.index(slice())
            _test_values(iv, 0.5 * np.arange(10))
            iv = self.iv.index(slice(4, 9, 2))
            _test_values(iv, np.array([2.0, 3.0, 4.0]))

        with self.subTest("Index by list"):
            iv = self.iv.index([1, 7, 8, 4])
            _test_values(iv, np.array([0.5, 3.5, 4.0, 2.0]))
            iv = self.iv.index(np.array([[[1]], [[7]], [[8]], [[4]]]))
            _test_values(iv, np.array([0.5, 3.5, 4.0, 2.0]))

        with self.subTest("Sequential indexing"):
            trial = self.trial.index(0)
            trial2 = trial.index([1, 2, 3])
            _test_values(trial2, np.array([]))

            trial = self.trial.index([1, 4, 6])

            trial2 = trial.index(3)
            _test_values(trial2, np.array([]))

            trial2 = trial.index(1)
            _test_values(trial2, np.array([4]))

            trial2 = trial.index(slice(0, 2))
            _test_values(trial2, np.array([1, 4]))

            trial2 = trial.index([0, 3])
            _test_values(trial2, np.array([1]))
