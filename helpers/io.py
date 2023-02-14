#! python3
# -*- coding: utf-8 -*-
#
# io.py
#
# Input/output utilities for loading data collected in experiments and exported
# from simulations.
#
# Author:   Connor D. Pierce
# Created:  2019-03-28 12:46
# Modified: 2023-02-14 05:35:43
#
# Copyright (c) 2019-2023 Connor D. Pierce
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
Input/output utilities for loading data collected in experiments and exported
from simulations.
"""


## Imports
import logging
import numpy as np
import os
import pint
import scipy as sp
import typing
import yaml

from helpers.units import ureg, Qty, EmptyObject
from helpers.utils import factors
from scipy import signal, stats


## Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

## For numpy type-checking
_FLOAT_TYPES = (np.float16, np.float32, np.float64, np.complex64, np.complex128)
_INT_TYPES = (np.int0, np.int16, np.int32, np.int64, np.int8)


## Functions
def augmentInput(inData, outData):
    """Augment input settings with output settings to make a full test definition."""
    tmp = inData.copy()
    for key in outData:
        tmp[key] = outData[key]
    return tmp


def convertComplexNumbers(s):
    return complex(s.decode().replace("i", "j"))


def load_data(fname, src="exp", complexCols=[]):
    """
    Loads an array of numbers from the given file.

    Parameters
    ----------
       fname - path to the file
       src   - the type of source which generated the data: "exp", "sim",
               "dma", "rheo", "osc", "mcz"
       complexCols - specifies which (if any) columns should be interpreted
                     using as complex numbers
    """

    # Extract data from files
    if src == "exp" or src == "osc":
        line_offset = 1
    else:
        line_offset = 0
    headings = ""

    fl = open(fname, encoding="utf-8")

    if src == "rheo":  # Process in a separate method
        return __loadRheometerData(fl)

    lastline = ""
    mcz_header_found = False
    for line in fl:
        if src == "exp":
            if line.find("R\tTheta") > -1:
                headings = line.split("\t")
                if headings[-1].strip() == "":
                    headings = headings[:-1]
                break
            else:
                line_offset += 1
        elif src == "osc":
            if line.find("TIME,") == 0:
                headings = line.strip().split(",")
                if headings[-1] == "":
                    headings = headings[:-1]
                break
            else:
                line_offset += 1
        elif src == "sim":
            if line.startswith("%"):
                lastline = line.strip()
                line_offset += 1
            else:
                headings = lastline
                break
        elif src == "dma":
            if line.find("StartOfData") > -1:
                headings = line.strip().split("\t")
                if headings[-1] == "":
                    headings = headings[:-1]
                line_offset += 1
                break
            else:
                line_offset += 1
        elif src == "mcz":
            line_offset += 1
            if line.find("Position ,") > -1:
                headings = line.strip().split(",")
                mcz_header_found = True
            elif mcz_header_found:
                units = line.split(",")
                break
        elif src == "mcz_raw":
            line_offset += 1
            if line.find("Position \t") > -1:
                headings = line.strip().split("\t")
                mcz_header_found = True
            elif mcz_header_found:
                units = line.strip().split("\t")
                break
    fl.close()
    logger.debug("Skipping " + str(line_offset) + " lines")

    # Attempt to read data
    if src == "exp" or src == "dma" or src == "mcz_raw":
        datain = np.loadtxt(fname, delimiter="\t", skiprows=line_offset)
    elif src == "osc" or src == "mcz":
        datain = np.loadtxt(fname, delimiter=",", skiprows=line_offset)
    elif src == "sim":
        cnvrtrs = {}
        for cnvrtr in complexCols:
            cnvrtrs[cnvrtr] = convertComplexNumbers
        datain = np.loadtxt(
            fname, skiprows=line_offset, converters=cnvrtrs, dtype=np.complex128
        )

    # If data was successfully read, we can exit the loop
    if src == "mcz" or src == "mcz_raw":
        return datain, headings, units
    else:
        return datain, headings


def __loadRheometerData(fl):
    allTests = []

    smplName = ""
    geomName = ""

    lineCt = 0
    currTest = None
    for line in fl:
        if line.startswith("Sample name"):
            substr = line.split("\t")
            smplName = substr[1]
        elif line.startswith("Geometry name"):
            substr = line.split("\t")
            geomName = substr[1]
        elif line.startswith("[step]"):
            currTest = {
                "lineNums": {
                    "name": lineCt + 1,
                    "header": lineCt + 2,
                    "units": lineCt + 3,
                    "dataStart": lineCt + 4,
                },
                "sampleName": smplName,
                "geometryName": geomName,
            }
        elif currTest != None:
            if lineCt == currTest["lineNums"]["name"]:
                currTest["name"] = line[:-1]
            elif lineCt == currTest["lineNums"]["header"]:
                substr = line[:-1].split("\t")
                idxs = [i - 1 for i in range(1, len(substr))]
                logger.debug("substr = " + substr)
                logger.debug("idxs = " + idxs)
                currTest["headers"] = dict(zip(substr[1:], idxs))
            elif lineCt == currTest["lineNums"]["units"]:
                substr = line[:-1].split("\t")
                currTest["units"] = substr[1:]
            else:
                if len(line.split("\t")) == 1:
                    currTest["lineNums"]["dataEnd"] = lineCt
                    logger.debug("start = " + str(currTest["lineNums"]["dataStart"]))
                    logger.debug("end = " + str(currTest["lineNums"]["dataEnd"]))
                    allTests.append(currTest)
                    currTest = None
                else:
                    substr = line.split("\t")
                    if lineCt == currTest["lineNums"]["dataStart"]:
                        currTest["dateTime"] = [substr[0]]
                    else:
                        currTest["dateTime"].append(substr[0])
        else:
            pass
        lineCt += 1

    for test in allTests:
        fl.seek(0, 0)
        logger.debug("Reading data for test " + test["name"])
        uc = [test["headers"][x] + 1 for x in test["headers"]]
        logger.debug("usecols = " + str(uc))
        test["data"] = sp.genfromtxt(
            fl,
            dtype=sp.float64,
            delimiter="\t",
            skip_header=test["lineNums"]["dataStart"],
            usecols=uc,
            max_rows=test["lineNums"]["dataEnd"] - test["lineNums"]["dataStart"],
            filling_values=np.NaN,
        )
    return allTests


def find_pIdx(datain, parameterized):
    currP_start = 0
    pIdx = []
    if parameterized:
        for i in range(1, datain.shape[0]):
            if datain[i, 0] != datain[i - 1, 0]:
                pIdx.append([currP_start, i - 1])
                currP_start = i
    pIdx.append([currP_start, datain.shape[0] - 1])
    return pIdx


def find_swpStartIdx(pIdx, datain, fCol):
    sweepStartIdx = [pIdx[0]]
    if datain[0, fCol] < datain[1, fCol]:
        sweepDir = 1
    else:
        sweepDir = -1

    for i in range(pIdx[0] + 1, pIdx[1] + 1):
        if (datain[i, fCol] - datain[i - 1, fCol]) * sweepDir < 0:
            sweepStartIdx.append(i)
    return sweepStartIdx


def getAvgData(pIdx, sweepStartIdx, rawData, RCol):
    plt_num = len(sweepStartIdx)
    avg = rawData[sweepStartIdx[-1] : (pIdx[1] + 1), RCol]

    for i in range(0, plt_num - 1):
        swpRange = sp.arange(sweepStartIdx[i], sweepStartIdx[i + 1])
        avg += rawData[swpRange, RCol]

    avg = avg / plt_num
    return avg


def apply_units(data, units):
    if units is None:
        return data * ureg.dimensionless
    elif isinstance(units, str):
        return ureg.Quantity(data, units)
    elif isinstance(units, pint.Unit):
        return raw_data[0 : var._blocksize : prev_multiplicity, var.col] * units


if True:

    def reshape_adv_idx(t: typing.Iterable, N: int, i: int, lv: int = 0):
        """Reshape advanced index to a broadcastable shape.

        Given a 1-D array `t` which acts as an index to axis `i` of an `N`-D array, reshape
        `t` into an `i`-D array of shape `(t.size, ) + (1, ) * (N - 1 - i)`.
        """

        if lv == 0:
            if lv == N - 1 - i:
                return t
            else:
                return [reshape_adv_idx(t_i, N, i, lv + 1) for t_i in t]
        else:
            if lv == N - 1 - i:
                return [t]
            else:
                return [reshape_adv_idx(t, N, i, lv + 1)]

else:

    def reshape_adv_idx(t: typing.Iterable, N: int, i: int):
        return np.array(t).reshape((t.size,) + (1,) * (N - i - 1))


## Exceptions
class DataNotLoadedError(Exception):
    """
    Raised by a test object (such as `QuasistaticTest`) when the user requests
    test data which has not been loaded.
    """

    # No additional functionality needed; this exception is defined to provide
    # more specificity about the type of error encountered.
    pass


class DataNotAvailableError(Exception):
    """
    Raised by a test object when the user requests data that cannot be computed
    (such as strain, when the initial length of the specimen was not specified).
    """

    pass


class SpecimenFileError(Exception):
    """
    Raised by a `Database` object if the user attempts to load a named test but
    has not provided the path to a Specimens.yml file.
    """

    pass


class DataInconsistentError(Exception):
    """
    Raised by a `FreqSweepTest` object if the data loaded for a frequency sweep
    test is inconsistent, e.g. if the parameters do not match between force and
    displacment sweeps.
    """

    pass


## Classes
class Database:
    """
    Manages the loading and storage of test data so that I/O operations (i.e.
    reading data from files) are not unnecessarily duplicated. This class deals
    only with loading and storing the raw data (e.g. voltages from transducers)
    and does not convert any signals into the physical quantities (e.g.
    displacement, force) that they represent.
    """

    def __init__(self, topLevelDir=None, specimens="Specimens.yml"):
        """
        Initialize the database from a top-level directory. Optionally with a
        path to a file `Specimens.yml` which defines a set of test
        specifications.

        Parameters
        ----------
        `topLevelDir` : `str`
            Location of the "top-level directory" for the project. Paths to test
            data (i.e. in Specimens.yml) are specified relative to this
            directory. This can be a relative or absolute path.
        `specimens` : `str` or `None`
            Path to the Specimens.yml file (relative to the top-level
            directory), or `None` if there is no Specimens.yml file for this
            project. Default is `"Specimens.yml"`.

        """

        if topLevelDir is None:
            self._tldir = os.path.abspath(os.getcwd())
        else:
            self._tldir = os.path.abspath(topLevelDir)

        # Initialize class fields
        self.specimens = {}
        self.specimenSpecLocs = {}
        self._data = {}

        self.set_specimens_file(specimens)

    def set_specimens_file(self, specimens):
        """
        Sets the path to the `Specimens.yml` file for this project. Clears all
        specimen specifications that were previously loaded.

        Parameters
        ----------
        `specimens` : `str` or `None`
            Path to the Specimens.yml file (relative to the top-level
            directory), or `None` if there is no Specimens.yml file for this
            project.

        """

        if specimens is None:
            self.specimenFile = None
        else:
            self.specimenFile = os.path.join(self._tldir, specimens)

        self.specimens = {}
        self.load_specimen_specs()

    def load_specimen_specs(self):
        """
        Loads the locations of the specimen specification files from the
        Specimens.yml file. If any specimen specifications have been loaded from
        the files defined in `Specimens.yml`, they are reloaded.
        """

        if self.specimenFile is None:
            self.specimenSpecLocs = None
            self.specimens = None
        else:
            self.specimenSpecLocs = yaml.load(
                open(self.specimenFile).read(), Loader=yaml.SafeLoader
            )
            tmp = self.specimens
            self.specimens = {}
            for s in tmp:
                if s in self.specimenSpecLocs:
                    self._load_specimen_spec(s)
        return

    def clear_file(self, filename=None):
        """
        Clears any data loaded from the file `filename`. If `filename is None`,
        all stored data is cleared.

        Returns
        -------
        `bool`
            `True` if the specified filename was present and was cleared, or
            `False` otherwise
        """

        if filename is None:
            self._data.clear()
            return True
        elif filename in self._data:
            del self._data[filename]
            return True
        else:
            return False

    def clear_test(self, specimenName, testName):
        """
        Clears any data loaded from all files required by test `testName` of
        specimen `specimenName`.

        Returns
        -------
        `bool`
            `True` if data for the specified test was loaded and was cleared,
            or `False` if there was no data present for the specified test.
        """

        try:
            reqFiles, _ = self._get_req_files(specimenName, testName)
            for f in reqFiles:
                if f in self._data:
                    del self._data[f]
            return True
        except KeyError:
            return False

    def _load_specimen_spec(self, specimenName):
        path = os.path.abspath(
            os.path.join(self._tldir, self.specimenSpecLocs[specimenName])
        )
        self.specimens[specimenName] = yaml.load(
            open(path).read(), Loader=yaml.SafeLoader
        )
        return

    def get_specimen(self, specimenName):
        """
        Gets the specification for the specimen named `specimenName`.

        Raises
        ------
        `KeyError`
            If no specimen named `specimenName` is found in the Specimens.yml
            file.
        """

        if specimenName in self.specimens:
            return self.specimens[specimenName]
        elif specimenName in self.specimenSpecLocs:
            self._load_specimen_spec(specimenName)
            return self.specimens[specimenName]
        else:
            raise KeyError("No spec found for specimen '" + str(specimenName) + "'")

    def _get_req_files(self, specimenName, testName):
        """
        Gets the files required for test `testName` of specimen `specimenName`.

        Returns
        -------
        `(reqFiles, fileTypes)`
            A list `reqFiles` of the absolute paths to all files required by
            the specified test, and a dict `fileTypes` specifying whether each
            file is for a `"quasi-static"` or a `"frequency sweep"` test.
        """

        if not specimenName in self.specimens:
            raise KeyError("Specimen '" + str(specimenName) + "' not loaded")
        elif not testName in self.specimens[specimenName]["tests"]:
            raise KeyError(
                "Test '"
                + str(testName)
                + "' does not exist for "
                + "specimen '"
                + str(specimenName)
                + "'"
            )

        testSpec = self.specimens[specimenName]["tests"][testName]

        reqFiles = []
        fileTypes = {}
        if testSpec["type"] == "quasi-static":
            for field in ("displacement data", "force data", "time data"):
                fname = testSpec[field]["file"]
                path = os.path.abspath(os.path.join(self._tldir, fname))
                reqFiles.append(path)
                fileTypes[path] = "quasi-static"
        elif testSpec["type"] == "frequency sweep":
            for field in ("displacement data", "force data"):
                try:
                    fname = testSpec[field]["file"]
                    path = os.path.abspath(os.path.join(self._tldir, fname))
                    reqFiles.append(path)
                    fileTypes[path] = "frequency sweep"
                except KeyError:
                    # Skip
                    pass

        return reqFiles, fileTypes

    def get_test_data(self, specimenName, testName):
        """
        Returns the data for a named test of a named specimen.

        Parameters
        ----------
        `specimenName` : `str`
            The name of the test specimen, as given in the `Specimens.yml` file.
        `testName` : `str`
            The name of the test for specimen `specimenName`, as given in the
            `Specimens.yml` file.

        Returns
        -------
        dict
            A mapping containing all data relevant to the requested test. For
            quasi-static tests, this will include:
                - "time" : the time that each data point was acquired
                - "disp" : the voltage from the displacement transducer
                - "force" : the voltage from the force transducer
            For frequency-sweep tests, this will include:
                - "freq"
                - "disp" : the complex-valued voltage (amplitude and phase) from
                  the displacement transducer
                - "force" : the complex-valued voltage from the force transducer

        Raises
        ------
        `SpecimenFileError`
            If a Specimens.yml file has not been specified.
        `ValueError`
            If `specimenName` and `testName` reference a test which is not
            defined in Specimens.yml.

        """

        if self.specimenFile is None:
            raise SpecimenFileError("No Specimens.yml file provided")

        # Check if a test specification exists
        specimen = self.get_specimen(specimenName)

        if not testName in specimen["tests"]:
            raise ValueError("Test spec does not exist")
        else:
            testSpec = specimen["tests"][testName]
            setupSpec = specimen["setups"][testSpec["setup"]]

        # Create a list of all files required for this test
        reqFiles, fileTypes = self._get_req_files(specimenName, testName)

        # Load all required files (if not loaded already)
        for f in set(reqFiles):
            if f not in self._data:
                if fileTypes[f] == "quasi-static":
                    self._data[f] = load_data(os.path.join(self._tldir, f), src="osc")
                elif fileTypes[f] == "frequency sweep":
                    raw_data, hdg = load_data(os.path.join(self._tldir, f), src="exp")
                    cols = Database._get_fs_col_spec(hdg)
                    self._data[f] = SweepTest(raw_data, cols, hdg)

        # Find the requested data in the database and return it
        if testSpec["type"] == "quasi-static":
            dispPath = os.path.abspath(
                os.path.join(self._tldir, testSpec["displacement data"]["file"])
            )
            forcePath = os.path.abspath(
                os.path.join(self._tldir, testSpec["force data"]["file"])
            )
            timePath = os.path.abspath(
                os.path.join(self._tldir, testSpec["time data"]["file"])
            )

            timeCol = self._data[timePath][1].index(testSpec["time data"]["column"])
            dispCol = self._data[dispPath][1].index(
                testSpec["displacement data"]["column"]
            )
            forceCol = self._data[forcePath][1].index(testSpec["force data"]["column"])
            return {
                "time": self._data[timePath][0][:, timeCol],
                "disp": self._data[dispPath][0][:, dispCol],
                "force": self._data[forcePath][0][:, forceCol],
            }
        elif testSpec["type"] == "frequency sweep":
            if "displacement data" in testSpec:
                dispPath = os.path.abspath(
                    os.path.join(self._tldir, testSpec["displacement data"]["file"])
                )
                disp = self._data[dispPath]
            else:
                disp = None

            if "force data" in testSpec:
                forcePath = os.path.abspath(
                    os.path.join(self._tldir, testSpec["force data"]["file"])
                )
                force = self._data[forcePath]
            else:
                force = None

            return {"disp": disp, "force": force}

    @staticmethod
    def _get_fs_col_spec(hdg):
        cols = []
        for h in hdg:
            if h in ("Frequency", "Amplitude", "X", "Y"):
                cols.append("x")
            elif h in ("R", "Theta", "StdDev"):
                cols.append("y")
            else:
                raise ValueError("Unknown heading: " + h)
        return ",".join(cols)

    def get_file_data(self, path, type):
        """
        Gets the data stored in the file given by `path`.

        Parameters
        ----------
        `path` : `str`
            Path to the file. If a relative path is given, it is taken relative
            to the top-level directory.
        `type`: `str`
            The type of data stored in this file: either quasi-static (`"q-s"`)
            or parametric sweep (`"sweep"`). If previously-loaded data exists
            for the file denoted by `path` and the type of that data does not
            match `type`, a `ValueError` is raised.

        Returns
        -------
        `tuple`
            A tuple `(data, hdgs)` containing the data stored in the file
            (`data`) and a list of the column headings found in the file
            (`hdgs`)

        Raises
        ------
        `ValueError`
            If data for the file denoted by `path` has been previously loaded
            and does not match the type given in `type`.
        """

        if os.path.isabs(path):
            abspath = path
        else:
            abspath = os.path.abspath(os.path.join(self._tldir, path))

        if abspath in self._data:
            existingType = (
                "sweep" if isinstance(self._data[abspath], SweepTest) else "q-s"
            )
            if existingType != type:
                raise ValueError("`type` does not match type of existing data")
            else:
                return self._data[abspath]

        # Data for this file was not previously loaded
        if type == "q-s":
            self._data[abspath] = load_data(abspath, src="osc")
        else:
            raw_data, hdg = load_data(abspath, src="exp")
            cols = Database._get_fs_col_spec(hdg)
            self._data[abspath] = SweepTest(raw_data, cols, hdg)

        return self._data[abspath]


class Sensor:
    """
    Programmatic representation of a sensor. Converts physical quantities to
    electrical representation and vice versa.
    """

    _types = {
        "force": {
            "direct": Qty(1, "V/N").dimensionality,
            "inverse": Qty(1, "N/V").dimensionality,
        },
        "displacement": {
            "direct": Qty(1, "V/m").dimensionality,
            "inverse": Qty(1, "m/V").dimensionality,
        },
        "velocity": {
            "direct": Qty(1, "V*s/m").dimensionality,
            "inverse": Qty(1, "m/s/V").dimensionality,
        },
    }

    def __init__(self, sensorData, name, cal, tare=0 * ureg.volts):
        """
        Creates a Sensor.

        Parameters
        `sensorData` : `dict`
        `name` : `str`
        `cal` : `str`
        `tare`: `pint.UnitRegistry.Quantity`
            Optional offset voltage to be subtracted from the signal before
            converting to physical units, or to be added to the signal after
            converting from physical units. Must have the dimensions of volts.
        """

        self.name = name
        self._data = sensorData[name]
        self.type = self._data["type"]
        self.model = self._data["model"]
        self.manuf = self._data["manufacturer"]
        self.serial = self._data["serial number"]
        self.cal = cal
        sens = Qty(
            self._data["calibrations"][cal]["sensitivity"]["value"],
            self._data["calibrations"][cal]["sensitivity"]["units"],
        )
        self.tare = tare

        # Validate the sensitivity. Check that the given sensitivity has the
        # correct dimensionality and is stored in the form "V/<physical unit>".
        majType = self.type.split(".")[0]
        if sens.dimensionality == Sensor._types[majType]["direct"]:
            self.sens = sens
        elif sens.dimensionality == Sensor._types[majType]["inverse"]:
            self.sens = 1 / sens
        else:
            raise ValueError("Invalid sensitivity for type " + majType)

    def toPhysical(self, signal, gain=1):
        """
        Converts the supplied voltage `signal` to physical units.

        Parameters
        ----------
        `signal` : `pint.UnitRegistry.Quantity`
            The voltage signal (with physical dimensions equal to Volts) to be
            converted.
        `gain` : `float`
            The gain (if any) that was applied to the signal post-sensor (i.e.
            by a signal conditioner)

        Returns
        -------
        `pint.UnitRegistry.Quantity`
            `signal` converted to physical units via the calibration coefficient
            (sensitivity) of this sensor: `signal / gain / self.sens`. Note that
            this may not be stated in simplified units.
        """

        if signal.dimensionality != ureg.volt.dimensionality:
            raise TypeError("`signal` must be in volts")

        return (signal / gain - self.tare) / self.sens

    def toSignal(self, stimulus, gain=1):
        """
        Converts the supplied physical stimulus (e.g. force, displacement)
        `stimulus` to voltage.

        Parameters
        ----------
        `stimulus` : `pint.UnitRegistry.Quantity`
            The physical signal (with units e.g. force or displacement) to be
            converted.
        `gain` : `float`
            The gain (if any) to be applied to the signal post-sensor (as if
            by a signal conditioner).

        Returns
        -------
        `pint.UnitRegistry.Quantity`
            `stimulus` converted to voltage via the calibration coefficient
            (sensitivity) of this sensor: `stimulus * gain * self.sens`. Note
            that this may not be stated in simplified units.
        """

        returnVal = gain * (stimulus * self.sens + self.tare)

        if returnVal.dimensionality != ureg.volt.dimensionality:
            raise TypeError("`signal` must be a compatible physical quantity")

        return returnVal


class QuasistaticTest:
    """
    Loads and organizes the data associated with a quasistatic compression test,
    including metadata (such as the sensors used) and the actual test data (e.g.
    force and displacement).

    Contains the following fields:
        specimen (`EmptyObject`) sub-fields:
            name (`str`)
            setup_name (`str`)
            length (`pint.UnitRegistry.Quantity`)
            pre_comp (`pint.UnitRegistry.Quantity`)
        test (`EmptyObject)
            name (`str`)
            details (`dict`)
            mag_field (`pint.UnitRegistry.Quantity`)
        force
        disp
        stress
        strain
    """

    def __init__(self, db, sensorSpec, specimenName, testName, loadData=True):
        self.db = db
        # A new line
        self.load_metadata(sensorSpec, specimenName, testName)

        # Load the data
        if loadData:
            self.load_data()

    def load_metadata(self, sensorSpec, specimenName, testName):
        """
        Parses the metadata contained in `specimenData` and `sensorData` for the
        given `specimenName` and `testName`. This can be used to update the
        metadata stored in a QuasistaticTest object, for example if the specimen
        details stored on disk in a YAML file were corrected and needed to be
        reloaded.

        Parameters:
        -----------
        `sensorSpec` : `dict`
            data structure describing sensors
        `specimenName` (`str`, required)
            name of a specimen contained in `specimenData`
        `testName` (`str`, required)
            name of a test performed on specimen `specimenName`

        Returns
        -------
        None
        """

        # A number of checks are needed to ensure that the metadata provided in
        # specimenData and sensorData is complete and valid. The data relevant
        # to the specified specimen and test is parsed and stored in attributes
        # on `EmptyObject`s. Some checks require the use of metadata that was
        # previously checked and stored. Since the parsing could fail at any
        # step, the parsed data is stored in temporary variables starting with
        # underscore (e.g. `_specimen`) until all checks have been completed
        # before it is stored in the attributes of this `QuasistaticTest`
        # object. This prevents the object attributes from being partially
        # updated, which would result in an incorrect test specification.

        # Check validity of specimenName
        if not isinstance(specimenName, str):
            raise TypeError("specimenName must be a string")
        else:
            # Store specimen details in a
            _specimen = EmptyObject()
            _specimen.name = specimenName
            specData = self.db.get_specimen(specimenName)

        # Parse test setup details
        if not isinstance(testName, str):
            raise TypeError("testName must be a string")
        elif testName not in specData["tests"]:
            raise ValueError(
                "No test named '" + testName + "' for specimen '" + specimenName + "'"
            )
        elif specData["tests"][testName]["type"] != "quasi-static":
            raise ValueError("Specified test is not quasi-static")
        else:
            _test = EmptyObject()
            _test.name = testName

            testDetails = specData["tests"][testName]

            _test.date = testDetails["date"]
            _test.setupName = testDetails["setup"]
            _test.magField = Qty(
                testDetails["magnetic field"]["value"],
                testDetails["magnetic field"]["units"],
            )
            if "direction" in testDetails["magnetic field"]:
                _test.magDir = testDetails["magnetic field"]["direction"]
            else:
                _test.magDir = None
            if "location" in testDetails["magnetic field"]:
                _test.magLoc = testDetails["magnetic field"]["location"]
            else:
                _test.magLoc = None
            if "loading_regions" in testDetails:
                _test.load_reg = testDetails["loading_regions"]
            else:
                _test.load_reg = None

            _force = EmptyObject()
            _force.sensor = Sensor(
                sensorSpec,
                testDetails["force sensor"]["name"],
                testDetails["force sensor"]["calibration"],
            )
            _force.gain = testDetails["force sensor"]["gain"]
            if "tare" in testDetails["force sensor"]:
                tare = Qty(
                    testDetails["force sensor"]["tare"]["value"],
                    testDetails["force sensor"]["tare"]["units"],
                )
                _force.sensor.tare = tare

            _disp = EmptyObject()
            _disp.sensor = Sensor(
                sensorSpec,
                testDetails["displacement sensor"]["name"],
                testDetails["displacement sensor"]["calibration"],
            )
            _disp.gain = testDetails["displacement sensor"]["gain"]
            if "tare" in testDetails["displacement sensor"]:
                tare = Qty(
                    testDetails["displacement sensor"]["tare"]["value"],
                    testDetails["displacement sensor"]["tare"]["units"],
                )
                _disp.sensor.tare = tare

            self.time = EmptyObject()

        # Get setup details
        if not _test.setupName in specData["setups"]:
            raise SyntaxError(
                "Setup '"
                + _test.setupName
                + "' does not exist for <"
                + specimenName
                + ">.<"
                + testName
                + ">"
            )
        else:
            setup = specData["setups"][_test.setupName]

            if specData["geometry"]["type"] == "none":
                _specimen.noSS = True
                _test.preComp = Qty(0, "in")
            elif specData["geometry"]["type"] == "cylinder":
                _specimen.length = Qty(
                    setup["initial length"]["value"], setup["initial length"]["units"]
                )
                _test.preComp = Qty(
                    setup["pre-compression"]["value"], setup["pre-compression"]["units"]
                )
                _specimen.diam = Qty(
                    specData["geometry"]["diameter"]["value"],
                    specData["geometry"]["diameter"]["units"],
                )
                _specimen.area = np.pi / 4 * _specimen.diam**2
                _specimen.noSS = False  # Stress-strain data can be calculated
            else:
                raise ValueError("Compression test geometry is non-cylindrical")

        self.specimen = _specimen
        self.test = _test
        self.force = _force
        self.disp = _disp

        self.dataLoaded = False

    def filter(self, fltr=None):

        if fltr is None:
            fltr = signal.iirdesign(
                wp=0.01, ws=0.1, gpass=1, gstop=80, ftype="cheby2", output="sos"
            )

        if not self.dataLoaded:
            self.load_data()

        self.disp.fltr = (
            signal.sosfiltfilt(fltr, self.disp.raw.magnitude) * self.disp.raw.units
        )
        self.force.fltr = (
            signal.sosfiltfilt(fltr, self.force.raw.magnitude) * self.force.raw.units
        )
        if not self.specimen.noSS:
            self.strain.fltr = self.disp.fltr / self.specimen.length
            self.stress.fltr = self.force.fltr / self.specimen.area

        self.filtered = True

    def load_data(self):
        """
        Loads the data specified by this test, overwriting any previous
        """

        data = self.db.get_test_data(self.specimen.name, self.test.name)

        self.disp.meas = self.disp.sensor.toPhysical(
            data["disp"] * ureg.volt, self.disp.gain
        )
        self.disp.raw = self.disp.meas + self.test.preComp
        self.disp.fltr = None

        self.force.raw = self.force.sensor.toPhysical(
            data["force"] * ureg.volt, self.disp.gain
        )
        self.force.fltr = None

        self.time.raw = data["time"] * ureg.second

        # Compute stress and strain
        if not self.specimen.noSS:
            self.strain = EmptyObject()
            self.strain.raw = self.disp.raw / self.specimen.length
            self.strain.fltr = None

            self.stress = EmptyObject()
            self.stress.raw = self.force.raw / self.specimen.area
            self.stress.fltr = None

        self.filtered = False
        self.dataLoaded = True

    def get_modulus(self, tTol=0.5, type="default", xvar="strain"):
        """
        Identifies the loading region of the curve and computes the elastic
        modulus as a linear fit to that portion of the curve.

        Parameters
        ----------
        `tTol` : `float`
            tolerance (in time) within which the endpoints of the loading
            portion should be identified
        `type` : `str`
            Type of data ("raw", "filtered", or "default") from which to compute
            the modulus. See `get_disp` for a description of the behavior of
            this parameter.
        `xvar` : `str`
            Parameter used to determine the edges of the loading region. Either
            `strain` or `stress`

        Returns
        -------
        (`modulus`, `units`, `idx`)
            `modulus` : fit object returned by `scipy.stats.linregress`
            `units` : `tuple` of `pint.UnitRegistry.units`
                the units of the (modulus, intercept) of `modulus`
            `idx` : `numpy.ndarray` of shape `(5,)` and `dtype=numpy.int32`
                indices of the data points that delineate the pre-loading,
                loading, unloading, and post-unloading regions of the S-S curve
        """

        if self.specimen.noSS:
            raise RuntimeError("Cannot compute stress-strain data")

        strain = self.get_strain(type)
        stress = self.get_stress(type)

        if not self.test.load_reg is None:
            moduli = []
            units = []
            idxs = []
            for i1, i2 in self.test.load_reg:
                moduli.append(
                    stats.linregress(strain[i1:i2].magnitude, stress[i1:i2].magnitude)
                )
                units.append((stress.units / strain.units, stress.units))
                idxs.append((i1, i2))
            return (moduli, units, idxs)
        else:
            return self._get_modulus_auto(strain, stress, tTol, xvar)

    def _get_modulus_auto(self, strain, stress, tTol, xvar):
        if xvar == "strain":
            var = strain
        elif xvar == "stress":
            var = stress
        else:
            raise ValueError("xvar must be 'strain' or 'stress'")

        varMin = np.amin(var.magnitude)
        varMax = np.amax(var.magnitude)

        duration = np.count_nonzero(var.magnitude > (varMin + 0.1 * (varMax - varMin)))
        div = np.ceil(duration / 4).astype(int)
        numDivs = np.ceil(var.size / div).astype(int)

        tTol = 0.1
        zeroTol = 0.1
        sameTol = 2
        edges = np.linspace(0, var.size - 1, numDivs + 1, dtype=np.int32)
        converged = False

        while not converged:
            slopes = np.zeros((edges.size - 1,), dtype=np.float64)
            for i in range(edges.size - 1):
                i1 = edges[i]
                i2 = edges[i + 1]
                slopes[i] = stats.linregress(
                    np.arange(i1, i2 + 1), var[i1 : i2 + 1].magnitude
                ).slope

            slopes /= np.amax(np.abs(slopes))

            subdivide = np.array([False] * (edges.size - 1))
            for i in range(edges.size - 1):
                if (
                    self.time.raw.magnitude[edges[i + 1]]
                    - self.time.raw.magnitude[edges[i]]
                ) < tTol:
                    pass
                elif np.abs(slopes[i]) < zeroTol:
                    if i > 0 and np.abs(slopes[i - 1]) > zeroTol:
                        subdivide[i] = True
                    if i < edges.size - 2 and np.abs(slopes[i + 1]) > zeroTol:
                        subdivide[i] = True
                else:
                    if i > 0 and (
                        np.abs(slopes[i - 1]) < zeroTol
                        or np.sign(slopes[i - 1]) != np.sign(slopes[i])
                        or np.abs(slopes[i - 1]) / np.abs(slopes[i]) > sameTol
                        or np.abs(slopes[i]) / np.abs(slopes[i - 1]) > sameTol
                    ):
                        subdivide[i] = True
                    if i < edges.size - 2 and (
                        np.abs(slopes[i + 1]) < zeroTol
                        or np.sign(slopes[i + 1]) != np.sign(slopes[i])
                        or np.abs(slopes[i + 1]) / np.abs(slopes[i]) > sameTol
                        or np.abs(slopes[i]) / np.abs(slopes[i + 1]) > sameTol
                    ):
                        subdivide[i] = True

            converged = not np.any(subdivide)

            newEdges = []
            for i in range(subdivide.size):
                newEdges.append(edges[i])
                if subdivide[i]:
                    midPt = edges[i] + (edges[i + 1] - edges[i]) // 2
                    newEdges.append(midPt)
            newEdges.append(edges[-1])
            edges = np.array(newEdges, dtype=np.int32)

        # Identify largest contiguous positive-slope and negative-slope regions
        def getType(slope):
            # Identifies a region as positive (1), zero (0), or negative (-1)
            # slope, depending on the given value of `slope`
            if slope > zeroTol:
                return 1
            elif slope < -zeroTol:
                return -1
            else:
                return 0

        regions = []
        for i in range(slopes.size):
            if i == 0:
                currReg = [edges[i], edges[i + 1]]
                currType = getType(slopes[i])
            elif getType(slopes[i]) != currType:
                regions.append({"idx": currReg, "type": currType})
                currReg = [edges[i], edges[i + 1]]
                currType = getType(slopes[i])
            else:
                currReg[1] = edges[i + 1]

            if i == slopes.size - 1:
                regions.append({"idx": currReg, "type": currType})
        largestPos = [-1, 0]
        largestNeg = [-1, 0]
        for i in range(len(regions)):
            x = regions[i]
            if x["type"] == 1 and (x["idx"][1] - x["idx"][0] > largestPos[1]):
                largestPos[0] = i
                largestPos[1] = x["idx"][1] - x["idx"][0]
            elif x["type"] == -1 and (x["idx"][1] - x["idx"][0] > largestNeg[1]):
                largestNeg[0] = i
                largestNeg[1] = x["idx"][1] - x["idx"][0]

        i1 = regions[largestPos[0]]["idx"][0]
        i3 = regions[largestNeg[0]]["idx"][1]
        if regions[largestPos[0]]["idx"][1] == regions[largestNeg[0]]["idx"][0]:
            i2 = regions[largestPos[0]]["idx"][1]
        else:
            i2 = (
                regions[largestPos[0]]["idx"][1] + regions[largestNeg[0]]["idx"][0]
            ) // 2
        edges = np.array([0, i1, i2, i3, var.size - 1])

        i1 = edges[1] + (edges[2] - edges[1]) // 10
        i2 = edges[1] + 9 * (edges[2] - edges[1]) // 10
        modulus = stats.linregress(strain[i1:i2].magnitude, stress[i1:i2].magnitude)

        return (modulus, (stress.units / strain.units, stress.units), edges)

    def get_disp(self, type="default"):
        """
        Gets the displacement data for this test. The optional argument `type`
        specifies whether to return raw or smoothed data.

        Parameters
        ----------
        `type` (`str`, optional)
            Specifies whether to return the raw (`"raw"`) or smoothed
            (`"filtered"`) data. Default value: `"default"`, which returns
            smoothed data if it is available.

        Returns
        -------
        the raw displacement data for this test if:
          - `type == "raw"`
          - `type == "default"` and smoothed data has not been created by a
            previous call to `filter()`
        the smoothed displacement data if:
          - `type == "filtered"` (note that if no previous calls to `filter()`
            have been made, this method will call `filter()` to generate the
            smoothed data)
          - `type == "default"` and smoothed data has been created by a previous
            call to `filter()`

        Raises
        ------
        `AttributeError`
            if the data for this test has not been loaded yet, or has been
            invalidated by updating the metadata with `load_metadata()`
        """

        if not self.dataLoaded:
            raise DataNotLoadedError("Data not loaded for test " + self.test.name)

        if not isinstance(type, str):
            raise TypeError("type must be a string")

        if type == "default":
            if self.filtered:
                return self.disp.fltr
            else:
                return self.disp.raw
        elif type == "raw":
            return self.disp.raw
        elif type == "filtered":
            if self.filtered:
                return self.disp.fltr
            else:
                self.filter()
                return self.disp.fltr
        else:
            raise ValueError("type must be 'raw', 'filtered', or 'default'")

    def get_force(self, type="default"):
        """
        Gets the force data for this test. The optional argument `type`
        specifies whether to return raw or smoothed data.

        Parameters
        ----------
        `type` (`str`, optional)
            Specifies whether to return the raw (`"raw"`) or smoothed
            (`"filtered"`) data. Default value: `"default"`, which returns
            smoothed data if it is available.

        Returns
        -------
        the raw force data for this test if:
          - `type == "raw"`
          - `type == "default"` and smoothed data has not been created by a
            previous call to `filter()`
        the smoothed force data if:
          - `type == "filtered"` (note that if no previous calls to `filter()`
            have been made, this method will call `filter()` to generate the
            smoothed data)
          - `type == "default"` and smoothed data has been created by a previous
            call to `filter()`

        Raises
        ------
        `AttributeError`
            if the data for this test has not been loaded yet, or has been
            invalidated by updating the metadata with `load_metadata()`
        """

        if not self.dataLoaded:
            raise DataNotLoadedError("Data not loaded for test " + self.test.name)

        if not isinstance(type, str):
            raise TypeError("type must be a string")

        if type == "default":
            if self.filtered:
                return self.force.fltr
            else:
                return self.force.raw
        elif type == "raw":
            return self.force.raw
        elif type == "filtered":
            if self.filtered:
                return self.force.fltr
            else:
                self.filter()
                return self.force.fltr
        else:
            raise ValueError("type must be 'raw', 'filtered', or 'default'")

    def get_strain(self, type="default"):
        """
        Gets the strain data for this test. The optional argument `type`
        specifies whether to return raw or smoothed data.

        Parameters
        ----------
        `type` (`str`, optional)
            Specifies whether to return the raw (`"raw"`) or smoothed
            (`"filtered"`) data. Default value: `"default"`, which returns
            smoothed data if it is available.

        Returns
        -------
        the raw strain data for this test if:
          - `type == "raw"`
          - `type == "default"` and smoothed data has not been created by a
            previous call to `filter()`
        the smoothed strain data if:
          - `type == "filtered"` (note that if no previous calls to `filter()`
            have been made, this method will call `filter()` to generate the
            smoothed data)
          - `type == "default"` and smoothed data has been created by a previous
            call to `filter()`
        `numpy.nan`
            if no geometry was specified, and the strain cannot be calculated

        Raises
        ------
        `AttributeError`
            if the data for this test has not been loaded yet, or has been
            invalidated by updating the metadata with `load_metadata()`
        """

        if self.specimen.noSS:
            return np.nan

        if not self.dataLoaded:
            raise DataNotLoadedError("Data not loaded for test " + self.test.name)

        if not isinstance(type, str):
            raise TypeError("type must be a string")

        if type == "default":
            if self.filtered:
                return self.strain.fltr
            else:
                return self.strain.raw
        elif type == "raw":
            return self.strain.raw
        elif type == "filtered":
            if not self.filtered:
                self.filter()

            return self.strain.fltr
        else:
            raise ValueError("type must be 'raw', 'filtered', or 'default'")

    def get_stress(self, type="default"):
        """
        Gets the stress data for this test. The optional argument `type`
        specifies whether to return raw or smoothed data.

        Parameters
        ----------
        `type` (`str`, optional)
            Specifies whether to return the raw (`"raw"`) or smoothed
            (`"filtered"`) data. Default value: `"default"`, which returns
            smoothed data if it is available.

        Returns
        -------
        the raw stress data for this test if:
          - `type == "raw"`
          - `type == "default"` and smoothed data has not been created by a
            previous call to `filter()`
        the smoothed stress data if:
          - `type == "filtered"` (note that if no previous calls to `filter()`
            have been made, this method will call `filter()` to generate the
            smoothed data)
          - `type == "default"` and smoothed data has been created by a previous
            call to `filter()`
        `numpy.nan`
            if no geometry was specified, and the stress cannot be calculated

        Raises
        ------
        `AttributeError`
            if the data for this test has not been loaded yet, or has been
            invalidated by updating the metadata with `load_metadata()`
        """

        if self.specimen.noSS:
            return np.nan

        if not self.dataLoaded:
            raise DataNotLoadedError("Data not loaded for test " + self.test.name)

        if not isinstance(type, str):
            raise TypeError("type must be a string")

        if type == "default":
            if self.filtered:
                return self.stress.fltr
            else:
                return self.stress.raw
        elif type == "raw":
            return self.stress.raw
        elif type == "filtered":
            if self.filtered:
                return self.stress.fltr
            else:
                self.filter()
                return self.stress.fltr
        else:
            raise ValueError("type must be 'raw', 'filtered', or 'default'")


class IndependentVariable:
    """
    Holds general information about an independent variable in a SweepTest.

    Contains the following fields, which can be accessed directly:
    --------------------------------------------------------------
    id : str
        Internal name (e.g. "x0", "x1", etc.)
    name : str
        Display name from the file (e.g. "Frequency", "Amplitude")
    axis : int
        Axis in the Y array that is parameterized by the values of this variable
    col : int
        Column number (0-based) in the file from which the values of this
        variable are extracted
    t_axis : int
        Axis in the Y array that is parameterized by different trials of this variable
    Tn : int
        The number of trials for this variable
    trial : `Trial`
        Information about repeated trials over this variable
    values : numpy.ndarray(dtype=numpy.float64)
        The parameter values of this variable

    """

    def __init__(self):
        self.id = None
        self.name = None
        self.axis = None
        self.col = None
        self.trial = None
        self.values = None

    def copy(self):
        iv = IndependentVariable()
        iv.id = self.id
        iv.axis = self.axis
        iv.col = self.col
        iv.name = self.name
        iv.trial = None
        iv.values = None if self.values is None else self.values.copy()
        return iv

    def __getattr__(self, attr):
        if attr == "t_axis":
            if self.trial is None:
                return None
            else:
                return self.trial.axis
        elif attr == "trials":
            if self.trial is None:
                return None
            else:
                return self.trial.trials
        elif attr == "Tn":
            if self.trial is None:
                return 1
            else:
                return self.trial.trials.size
        else:
            raise AttributeError(f"Attribute {attr} not found")


class Trial:
    """Wrapper class indicating a trial axes of an independent variable."""

    def __init__(self, iv: IndependentVariable):
        """Create a Trial axis linked to independent variable `iv`."""

        self.iv = iv
        self.trials = None
        self.axis = None

    def __getattr__(self, attr):
        if attr == "name":
            return f"Trial({self.iv.name})"
        elif attr == "id":
            return self.iv.id + "_t"
        elif attr in ("col",):
            return self.iv.__getattribute__(attr)
        else:
            raise AttributeError(f"Attribute {attr} not found")


class DependentVariable:
    """Data for a dependent variable in a SweepTest.

    Contains the following fields, which can be accessed directly:

    id : `str`
        Internal name (e.g. "y0", "y1", etc.)
    col : `int`
        Column number (0-based) in the file from which the values of this
        variable are extracted
    data : `pint.Quantity` or `numpy.ndarray`
        The measured data from all trials and all values of the independent variables
    name : `str`
        Display name from the file (e.g. "Frequency", "Amplitude")
    """

    def __init__(self):
        self.id = None
        self.col = None
        self.data = None
        self.idx = None
        self.name = None

    def copy(self, new_data=None):
        dv = DependentVariable()
        dv.id = self.id
        dv.col = self.col
        dv.idx = self.idx
        dv.name = self.name
        if new_data is not None:
            dv.data = new_data
        elif self.data is not None:
            dv.data = self.data.copy()
        return dv


class SweepTest:
    """Load and organize data from a general parametric sweep.

    A general parametric sweep consists of a set of tests in which one or more
    independent variables (the "parameters") are systematically varied over a grid of
    values. A parametric sweep with `N` parameters measures the response of a system
    over a discretized region of `N`-space. For each combination of the input
    parameters, the system response is characterized a scalar or vector value.

    Let `y` be a `P`-dimensional function of `N` independent variables; that is,
    let

    ```
    y = [ y_0, y_1, ..., y_p, ... y_(P-1) ]                                     (1)
    ```

    where

    ```
    y_p = f( x_0, x_1, ..., x_n, ..., x_(N-1) )                                 (2)
    ```

    with `0 <= p < P` and `0 <= n < N`. Let each of the independent variables
    `x_n` be parameterized into a discrete set of `M_n` values; that is, let

    ```
    x_n = [ x_n[0], x_n[1], ..., x_n[m_n], ..., x_n[M_n - 1] ]                  (3)
    ```

    with `0 <= m_n < M_n`. Let the domain `X` of `y` be formed by all possible
    combinations (i.e. the outer product) of the values in the vectors `x_n`, such that:

    ```
    X.shape = (M_0, M_1, M_2, ..., M_n, ..., M_(N-1))                           (4)
    X[m_0, m_1, ..., m_n, ..., m_(N-1)] = [
        x_0[m_0], x_1[m_1], ..., x_n[m_n], ..., x_(N-1)[m_(N-1)]
    ]                                                                           (5)
    ```

    for all `m_n`. Finally, let the parametric sweep be repeated `T_n` times over the
    `n`th independent variable `x_n`. Thus `y_p(x)` is
    `T_0 * T_1 * ... * T_n * ... * T_(N-1)`-valued for all `x` in `X`. The system
    response `Y` over all `X` is characterized by an array of shape

    ```
    (M_0, T_0, M_1, T_1, M_2, T_2, ..., M_n, T_n, ..., M_(N-1), T_(N-1), P)     (6)
    ```

    where

    ```
    Y[i_0, t_0, i_1, t_1, ..., i_n, t_n, ..., i_(N-1), t_(N-1), p] = y_p (
        x_0[i_0], x_1[i_1], ..., x_n[i_n], ..., x_(N-1)[i_(N-1)]
    )                                                                           (7)
    ```

    is the value of `y` for the `t_n`th iterations of the variables `x_n`,
    where `0 <= t_n < T_n`.

    # ACCESSING THE DATA IN THIS PARAMETRIC SWEEP

    The system response is stored in the `SweepTest.data` attribute, which is a `list`
    of `DependentVariable` objects, with `len(SweepTest.data) == P`. Each
    `DependentVariable` object contains a `data` attribute, which is a Numpy array or
    Pint `Quantity` of shape `(M_0, T_0, M_1, T_1, ..., M_(N-1), T_(N-1)`. The physical
    significance of each axis of `SweepTest.data` is stored in `IndependentVariable` and
    `Trial` objects. These objects can be accessed from `SweepTest` by name, ID, or
    order, from the `SweepTest.names`, `SweepTest.ids`, and `SweepTest.ind_vars`
    attributes, respectively.

    The system response can be accessed in two different ways. The first way is to
    directly use the `data` arrays of the `SweepTest` and `DependentVariable` objects
    where the data is stored. The axes in these arrays are ordered as in Equation (7)
    above, with the first axis spanning the values of the independent variable `x_0`,
    the second axis spanning the trials of `x_0`, the third axis spanning the values of
    the independent variable `x_1`, the fourth axis spanning the trials of `x_1`, and
    so on. If the variable `x_n` has only one trial, an axis is not allocated for that
    variable, but a `Trial` object will still be present in the `names`, `ids`, and
    `ind_vars` lists indicating the trial number for the variable `x_n`.

    The second way to access the system response is by indexing `SweepTest`, much like
    indexing of a Numpy array. Several indexing schemes are allowed, including indexing
    by a string, by a `DependentVariable`, by a `Trial` or `Trial`s, or by one or
    several values of an `IndependentVariable` or `IndependentVariable`s. The allowable
    indexing schemes for a `SweepTest` object `st` are as follows:

    1.  Index a single component `y_p` of the system response by name, id, order, or
        `DependentVariable` for all `x` in `X`:

        ```
        st[name: str]
        st[id: str]
        st[p: int]
        st[dv: DependentVariable]
        ```
    2.  Index by `dict`. Allowable keys are `str`; instances of `IndependentVariable` or
        `Trial`; or the `DependentVariable` class. String keys must be the name or ID of
        an independent variable or trial. Trials of the independent variable named
        `name` with ID `x0` may be indexed with the name `"Trial(name)"` or ID `"x0_t"`.
        Allowable values for independent variables are integers, slices, floats, or
        lists or `numpy.ndarrays` of integers or floats. Integers or arrays of integers
        (as determined by the `dtype` of the array) cause the independent variable to be
        indexed by position. Floats and arrays of floats cause the independent variable
        to be indexed by value. Lists will be wrapped in a `numpy.ndarray` to determine
        their type. In general, Numpy rules for basic and advanced indexing apply to the
        indexing operation. However, all lists/arrays will be flattened before the
        indexing operation, so the dimension (number of axes) of the data arrays will
        never grow. Allowable values for trials are integers, slices, or lists of
        integers. Trials are always indexed by position. Allowable values for
        `DependentVariable` are strings, instances of `DependentVariable`, or lists of
        strings and/or `DependentVariable` instances. Empty slices (index all components
        of the corresponding axis) are assumed for any independent variable, trial, or
        dependent variable that is not specified. When a scalar value is provided for
        an independent variable or a trial, the dimension of the data arrays are reduced
        by 1 and the corresponding axis is removed. Example usages are below

        ```
        st[{"Frequency": 0}]  # Index by scalar position
        st[{"Frequency": 11.5}]  # Index by scalar value
        st[{"Frequency": slice(0, 5)}]  # Index by position
        st[{"Frequency": [4, 5, 6, 7, 8]}]  # Index by position (array)
        st[{"Frequency", [1, 5.5, 10, 14.5, 19]}]  # Index by value (array)

        st[
            {"Amplitude": 0.25, "Frequency": 55.0}
        ]  # Index by value; both "Amplitude" and "Frequency" axes are removed
        st[
            {"Amplitude": 0.25, "Frequency": [55.0]}
        ]  # Index by value; "Frequency" axis is retained with size 1

        st[{DependentVariable: "Velocity"}]  # Extract only the "Velocity" output
        ```
    """

    def __init__(
        self,
        raw_data: np.ndarray = None,
        columns: str = ":,y0",
        names: typing.Union[None, list[typing.Union[str, None], ...]] = None,
        units: typing.Union[None, list[typing.Union[str, None, pint.Unit], ...]] = None,
    ):
        """Create a SweepTest instance.

        Create a `SweepTest` instance, optionally parsing the data in array
        `raw_data` with column specifiers `columns` and optional column names
        `names. If `raw_data` is `None`, `columns` and `names` are ignored.

        Parameters
        ----------
        `raw_data` : `numpy.ndarray` or `None`
            2D array containing the sweep data.
        `columns`: `str` or `None` (optional, default: ":,y0")
            Column specifier string. See `load` for details.
        `names`: `list[str or None, ...]` or `None` (optional, default: `None`)
            List of names assigned to each of the columns in `raw_data`. See
            `load` for details.
        `units`
            List of units assigned to each of the columns in `raw_data`. See
            `load` for details.
        """

        # Create collections so that the IndependentVariable and DependentVariable
        # objects can be easily accessed using different attributes.
        self.ids = {}  # For accessing variable details by internal name
        self.axes = {}  # For accessing ind. var. details by axis number
        self.data = {}  # For storing dep. var.
        self.dep_vars = []  # For storing dep. var. in order
        self.ind_vars = []  # For storing ind. var. in order
        self.names = {}  # For accessing variable details by display name

        self.num_t_axes = 0

        # Create scalars that describe the number of inputs (independent
        # variables) `N` and the number of outputs (dependent variables) `P`
        self.N = 0
        self.P = 0

        if raw_data is not None:
            self.load(raw_data, columns, names, units)

    def load(
        self,
        raw_data: np.ndarray,
        columns: str = ":,y0",
        names: typing.Union[None, list[typing.Union[str, None], ...]] = None,
        units: typing.Union[
            None, str, pint.Unit, list[typing.Union[str, None, pint.Unit], ...]
        ] = None,
    ) -> None:
        """Load data from 2D array and parse variable names.

        This method creates independent and dependent variables using the data
        given in `columns`, loads the data from `filename`, parses the sweep
        limits to determine the ranges and repetitions for each independent
        variable, and organizes the data into a multi-dimensional array with
        shape defined by the sweep structure.

        Parameters
        ----------
        `raw_data` :
            2D array of sweep data, with independent variables
        `columns` : (Optional, default: `":,y0"`)
            Comma-separated string designating each column of data as ither an
            independent ("x") or dependent ("y") variable. The order of the
            variables in the output can optionally be specified by suffixing
            each "x" or "y" with an integer. Note that the integers need not be
            zero-based, they need only be unique; the order of the variables
            will be determined by the sorted values of the integers, with the
            variable having the smallest integer being parameterized along the
            first axis of the data array `Y`. Columns for which the ordering is
            not specified will appear at the end of the data array in the order
            they are encountered. A single value starting with a colon may be
            included, which expands to the number of independent (":x") or
            dependent (":y") variables needed to match the number of columns in
            the data file. A colon with no "x" or "y" is interpreted as ":x".

            Example: `"x2,x0,x,x1,y,y0,y-1`
                This specifies a seven-column input file, with independent
                variables in the first four columns and dependent variables
                in the last three columns. In the output `Y`, the axes will
                correspond to (column numbers are 0-based):

                axis 0: parameter values from column 1
                axis 1: parameter values from column 3
                axis 2: parameter values from column 0
                axis 3: parameter values from column 2
                axis 4: the three components of `y`:
                    `y_0` (found in column 6),
                    `y_1` (found in column 5),
                    `y_2` (found in column 4),
                axis 5: trials t_0 from column 1
                axis 6: trials t_1 from column 3
                axis 7: trials t_2 from column 0
                axis 8: trials t_3 from column 2
        `names` : (Optional, default: `None`)
            List of names associated with each independent and dependent
            variable. If `names` has more entries than columns of data in
            `raw_data`, the extra entries are ignored. If `names` has fewer
            entries than columns of data in `raw_data`, it is end-padded with
            `None` to match the number of columns in `raw_data`. If `names[i]`
            is `None`, the corresponding independent or dependent variable will
            be automatically assigned the name `f"_Col{i}"`. If `names is None`,
            all variables will receive automatic names.
        `units` : (Optional, default: `None`)
            List of units associated with each independent and dependent
            variable. Specifying too many or too few entries in `units` is
            treated the same as `names`. Any entries of `None` are treated as
            dimensionless. If `None` is passed instead of a list, the data array
            will not be unit-sensitive, i.e. a pure numpy array with no units.

        Raises
        ------
        `KeyError`
            - if any of `names` are not unique
        `ValueError`
            - if `columns` is not a valid column specification
            - if `names is not None` and `len(names)` does not equal the number
              of fixed (non-expand) specifiers in `columns`
        """

        # Create variables
        self._create_vars(self._parse_col_spec(columns), raw_data, names)

        # Determine `M_n` and `T_n` for each variable
        self._calc_sweep_limits(raw_data, units)

        # Create and populate data arrays in dependent variables
        self._create_data_arrays(raw_data, units)

    @property
    def shape(self):
        shape = {}
        for i in range(len(self.Y.shape)):
            shape[i] = self.Y.shape[i]
            if self.axes[i] == DependentVariable:
                shape[DependentVariable] = len(self.data)
            else:
                shape[self.axes[i].name] = self.Y.shape[i]
        return shape

    @property
    def dim(self):
        return len(self.data[0].shape)

    def __getitem__(self, idx):
        if isinstance(idx, str):
            # Indexing a DependentVariable by name or id
            if idx in self.ids:
                if isinstance(self.ids[idx], DependentVariable):
                    return self[self.ids[idx]]
                else:
                    error_desc = f"'{idx}' is an independent variable"
            elif idx in self.names:
                if isinstance(self.names[idx], DependentVariable):
                    return self[self.names[idx]]
                else:
                    error_desc = f"'{idx}' is an independent variable"
            else:
                error_desc = f"no variable named {idx}"
            raise IndexError(f"Failed to index by string: {error_desc}")
        elif isinstance(idx, int):
            # Indexing a DependentVariable by position
            return self[self.data[idx]]
        elif isinstance(idx, DependentVariable):
            # Scalar index by a dependent variable
            if idx in self.data:
                _st = SweepTest()
                for iv in self.ind_vars:
                    iv_copy = iv.copy()
                    _st.ind_vars.append(iv_copy)
                    _st.axes[iv_copy.axis] = iv_copy
                    _st.names[iv_copy.name] = iv_copy
                    _st.ids[iv_copy.id] = iv_copy

                    trial_copy = Trial(iv_copy)
                    trial_copy.axis = iv.trial.axis
                    trial_copy.trials = iv.trial.trials
                    _st.axes[trial_copy.axis] = trial_copy
                    _st.names[trial_copy.name] = trial_copy
                    _st.ids[trial_copy.id] = trial_copy
                dv = idx.copy()
                dv.idx = 0
                _st.data.append(dv)
                _st.names[dv.name] = dv
                _st.ids[dv.id] = dv
                _st.P = 1
                _st.N = self.N
                _st.num_t_axes = self.num_t_axes
                return _st
            else:
                raise IndexError(f"Failed to index by DependentVariable: {str(idx)}")
        elif isinstance(idx, dict):
            _idx = [slice(None) for i in range(self.dim)]

            # Create list of dependent variables to include in the output
            output_dv = []

            def _extract_dvs(idx_obj):
                for idx_i in idx_obj:
                    if idx_i in self.ids:
                        output_dv.append(self.ids[idx[DependentVariable]])
                    elif idx[DependentVariable] in self.names:
                        output_dv.append(self.names[idx[DependentVariable]])
                    elif idx[DependentVariable] in self.data:
                        output_dv.append(idx[DependentVariable])
                    else:
                        raise IndexError(
                            f"Failed to index by dependent variable: {idx[DependentVariable]}"
                        )

            if DependentVariable in idx:
                if isinstance(idx[DependentVariable], str) or isinstance(
                    idx[DependentVariable], DependentVariable
                ):
                    _extract_dvs([idx[DependentVariable]])
                elif isinstance(idx[DependentVariable], list) or isinstance(
                    idx[DependentVariable], tuple
                ):
                    _extract_dvs(idx[DependentVariable])
                else:
                    raise IndexError(
                        f"Invalid type for key DependentVariable: '{type(idx[DependentVariable])}'"
                    )
            else:
                _extract_dvs(self.data)

            # Create indexing object for numpy
            indices = [None for i in self.axes]
            duplicate_err = "Duplicate index for {0:s}"
            already_err = "Variable {0:s} has already been indexed by a scalar"
            for key in idx:
                if key in self.ids:
                    axis = self.ids[key].axis
                    if axis is None:
                        raise IndexError(already_err.format(self.ids[key].name))
                    elif indices[axis] is None:
                        indices[axis] = idx[key]
                    else:
                        raise IndexError(duplicate_err.format(self.ids[key].name))
                elif key in self.names:
                    axis = self.names[key].axis
                    if axis is None:
                        raise IndexError(already_err.format(key))
                    elif indices[axis] is None:
                        indices[axis] = idx[key]
                    else:
                        raise IndexError(duplicate_err.format(key))
                elif key == DependentVariable:
                    pass
                elif key in self.ind_vars:
                    axis = key.axis
                    if axis is None:
                        raise IndexError(already_err.format(key.name))
                    elif indices[axis] is None:
                        indices[axis] = idx[key]
                    else:
                        raise IndexError(duplicate_err.format(key.name))
                else:
                    raise IndexError("No variable named " + str(key))
            for i in range(len(indices)):
                if indices[i] is None:
                    indices[i] = slice(None)

            # Perform indexing operation
            _has_adv_idx = False
            _adv_idxs = []
            _adv_idxs_contig = True
            for i, item in enumerate(indices):
                if isinstance(item, list) or isinstance(item, np.ndarray):
                    _has_adv_idx = True
                    _adv_idxs.append(i)
                    if len(_adv_idxs) > 1:
                        _adv_idxs_contig = _adv_idxs_contig and (i - _adv_idxs[-2]) == 1
            if _has_adv_idx:
                return self._do_adv_idx(
                    output_dv, tuple(indices), _adv_idxs, _adv_idxs_contig
                )
            else:
                return self._do_basic_idx(tuple(indices))
        else:
            raise IndexError(f"Indexing not supported for type '{str(type(idx))}'")

    def _do_adv_idx(
        self,
        output_dv: list[DependentVariable],
        idx: tuple[typing.Union[int, slice, list, np.ndarray], ...],
        adv_idxs: list[int, ...],
        adv_idx_contig: bool,
    ):
        """Perform advanced indexing on `self`."""

        logger.debug("_do_adv_idx: idx == " + str(idx))
        logger.debug("_do_adv_idx: adv_idxs == " + str(adv_idxs))
        logger.debug("_do_adv_idx: adv_idx_contig == " + str(adv_idx_contig))

        # Convert all advanced indexes to broadcastable shapes. Raise error if
        # a non-flat index is encountered.
        idx_copy = []
        adv_idx_count = 0
        int_axes = []
        axes_map = {}
        new_axis = 0
        new_shape = []
        for i, item in enumerate(idx):
            if i in adv_idxs:
                if isinstance(item, list):
                    item = np.array(item)

                item = item.flatten()
                L = item.size
                var = self.axes[i]
                if item.dtype in _FLOAT_TYPES:
                    # Index by value
                    if isinstance(var, Trial):
                        raise IndexError(f"Cannot index '{var.name}' by value")

                    M = var.values.size
                    matches = np.argwhere(
                        np.sum(
                            np.arange(1, M + 1).reshape(M, 1)
                            * np.isclose(var.values.reshape(M, 1), item),
                            axis=0,
                            dtype=np.int32,
                        )
                    )
                    if matches.size > 0:
                        matches = list(matches - 1)
                        idx_copy.append(
                            reshape_adv_idx(matches, len(adv_idxs), adv_idx_count)
                        )
                        new_shape.append(len(matches))
                    else:
                        raise IndexError(
                            f"Found no data matching the provided values for '{var.name}'"
                        )
                else:
                    # Index by position
                    idx_copy.append(reshape_adv_idx(item, len(adv_idxs), adv_idx_count))
                    new_shape.append(item.size)
                adv_idx_count += 1
                axes_map[i] = new_axis
                new_axis += 1
            else:
                if isinstance(item, int):
                    idx_copy.append(slice(item, item + 1, 1))
                    int_axes.append(i)
                    axes_map[i] = None
                elif isinstance(item, float):
                    if isinstance(self.axes[i], Trial):
                        raise IndexError(f"Cannot index '{self.axes[i].name}' by value")
                    _idx = np.argwhere(np.isclose(self.axes[i].values, item))
                    if _idx.size == 1:
                        idx_copy.append(slice(_idx[0], _idx[0] + 1, 1))
                        int_axes.append(i)
                        axes_map[i] = None
                    else:
                        raise IndexError(
                            f"Found no data matching the provided values for '{var.name}'"
                        )
                else:
                    idx_copy.append(item)
                    axes_map[i] = new_axis
                    new_axis += 1
                    old_shape = self.data[0].shape[i]
                    new_shape.append(len([i for i in range(*item.indices(old_shape))]))
        idx_copy = tuple(idx_copy)
        logger.debug("_do_adv_idx: idx_copy == " + str(idx_copy))
        logger.debug("_do_adv_idx: int_axes == " + str(int_axes))

        # Copy data, fix the axes reordering induced by non-contiguous advanced
        # indices, and remove all axes indexed with int
        _st = SweepTest()
        self._do_copy_vars(_st, idx_copy, axes_map)
        for _dv in output_dv:
            data_copy = _dv.data[idx_copy]
            logger.debug(f"_do_adv_idx: data_copy.shape == {str(data_copy.shape)}")
            if not adv_idx_contig:
                ax_order = [None for l in data_copy.shape]
                count_basic_idx = 0
                for i in range(len(_dv.data.shape)):
                    if i in adv_idxs:
                        ax_order[i] = adv_idxs.index(i)
                    else:
                        ax_order[i] = len(adv_idxs) + count_basic_idx
                        count_basic_idx += 1
                data_copy = data_copy.transpose(*ax_order)
            if len(int_axes) > 0:
                data_copy = data_copy.reshape(*new_shape)
            logger.debug(
                "_do_adv_idx: after reshape, data_copy.OWNDATA == "
                + str(data_copy.flags["OWNDATA"])
            )
            logger.debug("_do_adv_idx: axes_map == " + str(axes_map))
            new_dv = _dv.copy(data_copy)
            _st.data.append(new_dv)
            _st.names[new_dv.name] = new_dv
            _st.ids[new_dv.id] = new_dv
        return _st

    def _do_basic_idx(
        self,
        output_dv: list[DependentVariable],
        idx: tuple[typing.Union[int, slice], ...],
    ):
        """Perform basic indexing on `self`."""

        logger.debug("_do_adv_idx: idx == " + str(idx))

        # Convert all advanced indexes to broadcastable shapes. Raise error if
        # a non-flat index is encountered.
        axes_map = {}
        remaining_axes = 0
        for i, item in enumerate(idx):
            if isinstance(item, int):
                axes_map[i] = None
            else:
                axes_map[i] = remaining_axes
                remaining_axes += 1
        logger.debug("_do_adv_idx: axes_map == " + str(axes_map))

        _st = SweepTest()
        self._do_copy_vars(_st, idx, axes_map)
        for _dv in output_dv:
            data_copy = _dv.data[idx]
            new_dv = _dv.copy(data_copy)
            _st.data.append(new_dv)
            _st.names[new_dv.name] = new_dv
            _st.ids[new_dv.id] = new_dv
        return _st

    def _do_copy_vars(self, _st, idx, axes_map):
        # Copy all IndependentVariables:
        #   - independent variables that have previously been scalar indexed
        #     require, at most, transformations to their trial metadata.
        #   - independent variables that were previously scalar indexed or are
        #     scalar indexed in the current indexing operation are stored in all
        #     metadata collections (names, ids, ind_vars) except `axes`
        N = 0
        for iv in self.ind_vars:
            iv_copy = iv.copy()
            iv_copy.axis = axes_map[iv.axis]
            idx_obj = idx[iv.axis]
            if isinstance(idx_obj, slice) or isinstance(idx_obj, int):
                iv_copy.values = iv.values[idx_obj].copy()
            elif isinstance(idx_obj, list):
                iv_copy.values = iv.values[np.array(idx_obj).flatten()]
            else:
                raise IndexError("Unexpected type: " + str(type(idx_obj)))
            _st.names[iv_copy.name] = iv_copy
            _st.ids[iv_copy.id] = iv_copy
            _st.ind_vars.append(iv_copy)
            if iv_copy.axis is not None:
                _st.axes[iv_copy.axis] = iv_copy
                N += 1

            # Update trial metadata if present
            if iv.trial is not None:
                trial_copy = Trial(iv_copy)
                trial_copy.axis = axes_map[iv.t_axis]
                idx_obj = idx[iv.t_axis]
                if isinstance(idx_obj, slice) or isinstance(idx_obj, int):
                    trial_copy.trials = iv.trial.trials[idx_obj].copy()
                elif isinstance(idx_obj, list):
                    trial_copy.trials = iv.trial.trials[np.array(idx_obj).flatten()]
                else:
                    raise IndexError("Unexpected type: " + str(type(idx_obj)))
                iv_copy.trial = trial_copy
                if axes_map[iv.t_axis] is not None:
                    _st.axes[axes_map[iv.t_axis]] = trial_copy
                _st.names[trial_copy.name] = trial_copy
                _st.ids[trial_copy.id] = trial_copy
        _st.N = N
        return

    def _create_data_arrays(self, raw_data):
        # Create array of row indices, used for calculating the indexing array
        rows = np.arange(raw_data.shape[0], dtype=np.int32)
        M = []
        shape = []
        curr_axis = 0
        for iv in self.ind_vars:
            shape.append(iv.Mn)
            M.append((rows % (iv._blocksize)) // iv._mult)
            iv.axis = curr_axis
            self.axes[curr_axis] = iv
            curr_axis += 1

            if iv.Tn > 1:
                shape.append(iv.Tn)
                M.append((rows // iv._blocksize) % iv.Tn)
                iv.trial.axis = curr_axis
                self.axes[curr_axis] = iv.trial
                curr_axis += 1
            logger.debug(
                "Blocksize:" + str(iv._blocksize) + ", multiplicity: " + str(iv._mult)
            )

        row_idx = np.zeros(shape, dtype=np.int32)
        row_idx[tuple(M)] = rows

        for dv in self.dep_vars:
            if isinstance(units, list):
                dv.data = apply_units(raw_data[row_idx, dv.col], units[var.col])
            else:
                dv.data = apply_units(raw_data[row_idx, dv.col], units)

    def _calc_sweep_limits(self, raw_data, units):
        """Calculate the number of values `M_n` and repetitions `T_n` per
        variable."""

        n = raw_data.shape[0]
        possible_sizes = factors(n)

        block_sizes = {}
        for v in self.ids:
            var = self.ids[v]
            if isinstance(var, IndependentVariable):
                for b in possible_sizes[1:]:
                    block = raw_data[:b, var.col]
                    test_arr = raw_data[:, var.col].reshape((n // b, b))
                    if np.all(block == test_arr, axis=(0, 1)):
                        var._blocksize = b
                        block_sizes[b] = var
                        break

        prev_multiplicity = n
        for k in sorted(block_sizes.keys())[-1::-1]:
            var = block_sizes[k]
            var.trials = np.arange(prev_multiplicity // var._blocksize)
            var.Mn = np.unique(raw_data[:, var.col]).size
            var._mult = var._blocksize // var.Mn

            prev_multiplicity = var._mult

            if isinstance(units, list):
                var.values = apply_units(
                    raw_data[0 : var._blocksize : var._mult, var.col], units[var.col]
                )
            else:
                var.values = apply_units(
                    raw_data[0 : var._blocksize : var._mult, var.col], units
                )
            logger.debug(f"Var {var.name}: Tn={var.Tn}, Mn={var.Mn}")

    def _create_vars(self, col_spec, raw_data, names):
        """
        Creates the set of dependent and independent variables measured by this
        sweep test.
        """
        # Unpack col_spec
        all_spec, ind_spec, dep_spec, exp_spec = col_spec

        # Calculate number of specified columns (not including expand)
        tot_cols = len(all_spec)
        expand = not (exp_spec["idx"] is None)
        fixed_cols = tot_cols - (1 if expand else 0)

        # Check that the number of variable names matches the number of
        # specified columns
        if names is None:
            hdgs = [None for i in range(fixed_cols)]
        elif isinstance(names, list):
            if len(names) != fixed_cols:
                raise ValueError(
                    f"Wrong number of names specified: expecting {fixed_cols}, "
                    f"received {len(names)}"
                )
            else:
                hdgs = names
        else:
            raise TypeError("names must be None or a list")

        # Verify column spec has the same number of cols as the data and expand
        # the expand specifier (if present)
        if fixed_cols != raw_data.shape[1]:
            if expand:
                # Calculate the number of variables that must be added:
                expand_num = raw_data.shape[1] - fixed_cols
                # Recompute the column numbers indices for all fixed specifiers
                # following the expand specifier.
                for i in range(exp_spec["idx"] + 1, len(all_spec)):
                    all_spec[i]["idx"] += expand_num - 1
                # Next, remove the expand specifier
                del all_spec[exp_spec["idx"]]
                # Insert the appropriate number of fixed specifiers
                for i in range(expand_num):
                    fixed_spec = {
                        "type": exp_spec["type"],
                        "idx": exp_spec["idx"] + i,
                        "expand": False,
                        "numbered": False,
                        "number": None,
                    }
                    all_spec.insert(exp_spec["idx"] + i, fixed_spec)
                    hdgs.insert(exp_spec["idx"] + i, None)
                # Rebuild the lists of numbered and non-numbered fixed vars
                ind_spec["num"] = {}
                ind_spec["non"] = []
                dep_spec["num"] = {}
                dep_spec["non"] = []

                for c in all_spec:
                    if c["type"] == "dep":
                        if c["numbered"]:
                            dep_spec["num"][c["number"]] = c["idx"]
                        else:
                            dep_spec["non"].append(c["idx"])
                    else:
                        if c["numbered"]:
                            ind_spec["num"][c["number"]] = c["idx"]
                        else:
                            ind_spec["non"].append(c["idx"])
            else:
                raise ValueError("Col spec does not match # of cols in file")

        logger.debug("Expanded col_spec = " + str(all_spec))
        logger.debug("Ind. var indices = " + str(ind_spec))
        logger.debug("Dep. var indices = " + str(dep_spec))

        # Build the dictionaries of variables
        ids = {}
        axes = {}
        data = {}
        dep_vars = []
        ind_vars = []
        names = {}

        N = 0
        P = 0
        _len_dep = len(dep_spec["num"]) + len(dep_spec["non"])

        def create_var(var_type, id, col):
            logger.debug(
                "create_var: var_type = "
                + " ".join([str(var_type), "id =", id, "col =", str(col)])
            )

            # Create a variable and fill out the properties
            var = var_type()
            var.id = id
            var.col = col
            var.name = f"_Col{var.col}" if hdgs[var.col] is None else str(hdgs[var.col])

            if var.name in names:
                raise KeyError(f"Column name {var.name} is not unique")

            # Store the variable in the dictionaries
            if isinstance(var, IndependentVariable):
                ind_vars.append(var)
                var.trial = Trial(var)
                ids[var.trial.id] = var.trial
                names[var.trial.name] = var.trial
                N += 1
            else:
                var.idx = P
                data[P] = var
                dep_vars.append(var)
                P += 1
            ids[var.id] = var
            names[var.name] = var

        for i in sorted(ind_spec["num"].keys()):
            create_var(
                IndependentVariable,
                "x{0:d}".format(i),
                ind_spec["num"][i],
            )
        for i, col in enumerate(ind_spec["non"]):
            create_var(
                IndependentVariable,
                "x{0:d}".format(ind_spec["max_number"] + 1 + i),
                col,
            )
        for i in sorted(dep_spec["num"].keys()):
            create_var(
                DependentVariable,
                "y{0:d}".format(i),
                dep_spec["num"][i],
            )
        for i, col in enumerate(dep_spec["non"]):
            create_var(
                DependentVariable,
                "y{0:d}".format(dep_spec["max_number"] + 1 + i),
                col,
            )
        logger.debug("axes are " + str(axes))

        # Parsing succeeded; store the temporary variables in self
        self.N = N
        self.P = P
        self.axes = axes
        self.data = data
        self.dep_vars = dep_vars
        self.ids = ids
        self.ind_vars = ind_vars
        self.names = names
        return

    @staticmethod
    def _parse_numbered_spec(spec, spec_str):
        """
        Parses the order number in the string num_str and updates the `spec`
        accordingly.
        """

        try:
            z = int(spec_str[1:])

            spec["numbered"] = True
            spec["number"] = z

            return z
        except ValueError:
            raise ValueError("Invalid column spec: " + spec_str)

    @staticmethod
    def _parse_col_spec(columns):
        if columns is None:
            return None

        # Define a list `specs` to hold the individual column specifiers
        # encountered in the column specification string. Each entry will be a
        # dict:
        #   {
        #       "type":"dep"/"ind",
        #       "expand":`bool`
        #       "idx":`int`,
        #       "numbered":`bool`,
        #       "number":`int` or `None`
        #   }
        specs = []

        # Define dicts `dep` and `ind` to hold the indices of the numbered and
        # non-numbered dependent and independent variables, for easy access to
        # those specs.
        dep = {"num": {}, "non": [], "max_number": -np.inf}
        ind = {"num": {}, "non": [], "max_number": -np.inf}

        expand = {"idx": None, "type": None}  # location of the expand specifier

        s = columns.split(",")
        for i in range(len(s)):
            s_i = s[i].strip().lower()
            spec = {"idx": i, "expand": False, "numbered": False, "number": None}

            if s_i.startswith("x"):
                spec["type"] = "ind"

                if len(s_i) == 1:
                    ind["non"].append(i)
                else:
                    num = SweepTest._parse_numbered_spec(spec, s_i)

                    if num in ind["num"]:
                        raise ValueError(f"x{num} is repeated")

                    ind["num"][num] = i
                    ind["max_number"] = max(ind["max_number"], num)
            elif s_i.startswith("y"):
                spec["type"] = "dep"

                if len(s_i) == 1:
                    dep["non"].append(i)
                else:
                    num = SweepTest._parse_numbered_spec(spec, s_i)

                    if num in dep["num"]:
                        raise ValueError("y" + str(num) + " is repeated")

                    dep["num"][num] = i
                    dep["max_number"] = max(dep["max_number"], num)
            elif s_i.startswith(":"):
                if expand["idx"] is not None:
                    raise ValueError("Multiple expand arguments specified")
                elif len(s_i) == 1 or s_i[1:].strip() == "x":
                    spec["type"] = "ind"
                elif s_i[1:].strip() == "y":
                    spec["type"] = "dep"
                else:
                    raise ValueError("Invalid expand spec: '" + s[i] + "'")
                spec["expand"] = True
                expand["idx"] = i
                expand["type"] = spec["type"]
            else:
                raise ValueError("Invalid expand spec: '" + s[i] + "'")

            logger.debug("spec = " + str(spec))
            specs.append(spec)
        if ind["max_number"] == -np.inf:
            ind["max_number"] = -1
        if dep["max_number"] == -np.inf:
            dep["max_number"] = -1
        return specs, ind, dep, expand


class FreqSweepTest:
    """
    Loads and organizes the data associated with a frequency-sweep compression
    test, including metadata (such as the sensors used) and the actual test data
    (e.g. force and displacement).

    Contains the following fields:
        specimen (`EmptyObject`) sub-fields:
            name (`str`)
            setup_name (`str`)
            length (`pint.UnitRegistry.Quantity`)
            pre_comp (`pint.UnitRegistry.Quantity`)
        test (`EmptyObject)
            name (`str`)
            details (`dict`)
            mag_field (`pint.UnitRegistry.Quantity`)
        force
        disp
        stress
        strain
    """

    def __init__(self, db, sensorSpec, specimenName, testName, loadData=True):
        self.db = db
        self.dataLoaded = False

        self.set_metadata(sensorSpec, specimenName, testName)

        # Load the data
        if loadData:
            self.load_data()

    def set_metadata(self, sensorSpec, specimenName, testName):
        """
        Updates this `FreqSweepTest` to provide results for test `testName` of
        sample `specimenName`, using the sensors specified in `sensorSpec`.

        Parameters:
        -----------
        `sensorSpec` : `dict`
            A sensor specifications object, such as would be loaded from a
            Sensors.yml file.
        `specimenName` : `str`
            Name of the specimen
        `testName` : `str`
            Name of the test performed on specimen `specimenName`

        Returns
        -------
        None
        """

        # A number of checks are needed to ensure that the metadata provided in
        # specimenData and sensorData is complete and valid. The data relevant
        # to the specified specimen and test is parsed and stored in attributes
        # on `EmptyObject`s. Some checks require the use of metadata that was
        # previously checked and stored. Since the parsing could fail at any
        # step, the parsed data is stored in temporary variables starting with
        # underscore (e.g. `_specimen`) until all checks have been completed
        # before it is stored in the attributes of this `FreqSweepTest`
        # object. This prevents the object attributes from being partially
        # updated, which would result in an incorrect test specification.

        # Check validity of specimenName
        if not isinstance(specimenName, str):
            raise TypeError("specimenName must be a string")
        else:
            _specimen = EmptyObject()
            _specimen.name = specimenName
            specData = self.db.get_specimen(specimenName)

        # Check that the specified test is valid
        if not isinstance(testName, str):
            raise TypeError("testName must be a string")
        elif testName not in specData["tests"]:
            raise ValueError(
                "No test named '" + testName + "' for specimen '" + specimenName + "'"
            )
        elif specData["tests"][testName]["type"] != "frequency sweep":
            raise ValueError("Specified test is not a frequency sweep")

        # Parse details of the test
        _test = EmptyObject()
        _test.name = testName

        testDetails = specData["tests"][testName]

        _test.date = testDetails["date"]
        _test.setupName = testDetails["setup"]

        # Extract details about the magnetic field
        if "magnetic field" in testDetails:
            _test.magField = Qty(
                testDetails["magnetic field"]["value"],
                testDetails["magnetic field"]["units"],
            )

            if "direction" in testDetails["magnetic field"]:
                _test.magDir = testDetails["magnetic field"]["direction"]
            else:
                _test.magDir = None

            if "location" in testDetails["magnetic field"]:
                _test.magLoc = testDetails["magnetic field"]["location"]
            else:
                _test.magLoc = None
        else:
            _test.magField = Qty(0, "T")
            _test.magDir = "N/A"
            _test.magLoc = None

        # Extract details about the force sensor
        _force = EmptyObject()
        _force.sensor = Sensor(
            sensorSpec,
            testDetails["force sensor"]["name"],
            testDetails["force sensor"]["calibration"],
        )
        if "gain" in testDetails["force sensor"]:
            _force.gain = testDetails["force sensor"]["gain"]
        else:
            _force.gain = 1

        # Extract details about the displacement sensor
        _disp = EmptyObject()
        _disp.sensor = Sensor(
            sensorSpec,
            testDetails["displacement sensor"]["name"],
            testDetails["displacement sensor"]["calibration"],
        )
        if "gain" in testDetails["displacement sensor"]:
            _disp.gain = testDetails["displacement sensor"]["gain"]
        else:
            _disp.gain = 1

        # Get setup details
        if not _test.setupName in specData["setups"]:
            raise SyntaxError(
                "Setup '"
                + _test.setupName
                + "' does not exist for '"
                + specimenName
                + "'.'"
                + testName
                + "'"
            )
        else:
            setup = specData["setups"][_test.setupName]
            _specimen.batch = specData["batch"]
            if specData["geometry"]["type"] == "none":
                _specimen.noSS = True
                _test.preComp = Qty(0, "in")
            elif specData["geometry"]["type"] == "cylinder":
                _specimen.length = Qty(
                    setup["initial length"]["value"], setup["initial length"]["units"]
                )
                _test.preComp = Qty(
                    setup["pre-compression"]["value"], setup["pre-compression"]["units"]
                )
                _specimen.diam = Qty(
                    specData["geometry"]["diameter"]["value"],
                    specData["geometry"]["diameter"]["units"],
                )
                _specimen.area = np.pi / 4 * _specimen.diam**2
                _specimen.noSS = False  # Stress-strain data can be calculated
            else:
                raise ValueError("Compression test geometry is non-cylindrical")

        self.specimen = _specimen
        self.test = _test
        self.force = _force
        self.disp = _disp

        self.dataLoaded = False

    def get_pre_strain(self):
        """
        Gets the pre-strain for this test, if available.
        """

        if not self.specimen.noSS:
            return self.test.preComp / self.specimen.length
        else:
            raise DataNotAvailableError(
                "Cannot compute strain; initial length" + " not specified"
            )

    def load_data(self):
        """
        Loads the data specified for this test.
        """

        data = self.db.get_test_data(self.specimen.name, self.test.name)
        disp = data["disp"]  # type `SweepTest`
        forc = data["force"]  # type `SweepTest`

        if disp is None:
            self._noDisp = True
        else:
            R = disp["R"]
            theta = disp["Theta"]

            R.Y = self.disp.sensor.toPhysical(
                R.Y * np.exp(1j * np.pi * theta.Y / 180) * ureg.volt,
                gain=self.disp.gain,
            )
            self.disp.raw = R
            self._noDisp = False

        if forc is None:
            self._noForce = True
        else:
            R = forc["R"]
            theta = forc["Theta"]

            R.Y = self.force.sensor.toPhysical(
                R.Y * np.exp(1j * np.pi * theta.Y / 180) * ureg.volt,
                gain=self.force.gain,
            )
            self.force.raw = R
            self._noForce = False

        if not (self._noDisp or self._noForce):
            # Check for consistency between independent variables of
            # displacement and force
            self._check_data_consistency()

        # Compute stress and strain
        if not self.specimen.noSS:
            if not self._noDisp:
                self.strain = EmptyObject()
                self.strain.raw = self.disp.raw[...]
                self.strain.raw.Y /= self.specimen.length

            if not self._noForce:
                self.stress = EmptyObject()
                self.stress.raw = self.force.raw[...]
                self.stress.raw.Y /= self.specimen.area

        self.dataLoaded = True

    def _check_data_consistency(self):
        if self.disp.raw.N != self.force.raw.N:
            # Mismatched number of independent parameters
            raise DataInconsistentError(
                "Force and displacement have different number of "
                + "parameters for test "
                + self.test.name
            )
        for iv in self.disp.raw.ind_vars:
            if iv.name not in self.force.raw.names:
                raise DataInconsistentError(
                    self.test.name
                    + " has different force and "
                    + "displacement parameters"
                )
            if not np.allclose(iv.values, self.force.raw.names[iv.name].values, atol=0):
                raise DataInconsistentError(
                    "Found different parameter values for "
                    + iv.name
                    + "in test "
                    + self.test.name
                )
            if iv.Tn != self.force.raw.names[iv.name].Tn:
                raise DataInconsistentError(
                    "Found different number of repetitions for "
                    + iv.name
                    + "in test "
                    + self.test.name
                )
            if iv.axis != self.force.raw.names[iv.name].axis:
                raise DataInconsistentError(
                    "Found different axis for " + iv.name + " in test " + self.test.name
                )
            if iv.t_axis != self.force.raw.names[iv.name].t_axis:
                raise DataInconsistentError(
                    "Found different trial axis for "
                    + iv.name
                    + " in test "
                    + self.test.name
                )

    def get_disp(self, avg=False):
        """Get the displacement data for this test.

        Parameters
        ----------
        `avg` : `bool` (optional)
            Specifies whether to compute the average displacement over all
            repeated trials. Default: `False`

        Returns
        -------
        `SweepTest`
            the raw or averaged displacement data for this test

        Raises
        ------
        `DataNotLoadedError`
            if the data for this test has not been loaded yet
        `DataNotAvailableError`
            if displacement was not collected for this test
        """

        if not self.dataLoaded:
            raise DataNotLoadedError("Data not loaded for test " + self.test.name)

        if self._noDisp:
            raise DataNotAvailableError("Displ not available for this test")

        if avg:
            return self._get_avg(self.disp)
        else:
            return self.disp.raw

    def _get_avg(self, field):
        avg_ax = []

        params_to_avg = []
        for p in ("Amplitude", "Frequency"):
            if p in field.raw.names and field.raw.names[p].Tn > 1:
                avg_ax.append(field.raw.names[p].t_axis)
                params_to_avg.append(p)

        if len(avg_ax) > 0:
            data = field.raw[{s: 0 for s in params_to_avg}]
            for s in params_to_avg:
                data.names[s].trial.values = ["average"]
            data.Y = np.mean(field.raw, axis=tuple(avg_ax))
            return data
        else:
            return field.raw

    def get_freq(self):
        """Gets the frequency data for this test.

        Returns
        -------
        `IndependentVariable`
            frequency data for this test

        Raises
        ------
        KeyError
            if the test contains no frequency data
        """

        if not self.dataLoaded:
            raise DataNotLoadedError("Data not loaded for test " + self.test.name)

        if self._noForce:
            if self._noDisp:
                raise DataNotLoadedError(
                    "No frequency data available for test " + self.test.name
                )
            else:
                return self.disp.raw.names["Frequency"]
        else:
            return self.force.raw.names["Frequency"]

    def get_ampl(self):
        """Get the amplitude parameter data for this test.

        Returns
        -------
        `IndependentVariable`
            amplitude data for this test

        Raises
        ------
        KeyError
            if the test contains no amplitude data
        """

        if not self.dataLoaded:
            raise DataNotLoadedError("Data not loaded for test " + self.test.name)

        if self._noForce:
            if self._noDisp:
                raise DataNotLoadedError(
                    "No amplitude data available for test " + self.test.name
                )
            else:
                return self.disp.raw.names["Amplitude"]
        else:
            return self.force.names["Amplitude"]

    def get_force(self, avg=False):
        """
        Gets the force data for this test.

        Parameters
        ----------
        `avg` : `bool` (optional)
            Specifies whether to compute the average displacement over all
            repeated trials. Default: `False`

        Returns
        -------
        the force data for this test (if force data was collected)

        Raises
        ------
        `DataNotLoadedError`
            if the data for this test has not been loaded yet
        `DataNotAvailableError`
            if force was not collected for this test
        """

        if not self.dataLoaded:
            raise DataNotLoadedError("Data not loaded for test " + self.test.name)

        if self._noForce:
            raise DataNotAvailableError("Force not available for this test")

        if avg:
            return self._get_avg(self.force)
        else:
            return self.force.raw

    def get_strain(self, avg=False):
        """
        Gets the strain data for this test, if available.

        Parameters
        ----------
        `avg` : `bool` (optional)
            Specifies whether to compute the average displacement over all
            repeated trials. Default: `False`

        Returns
        -------
        the strain data for this test if displacement was collected and the
        original specimen length is available

        Raises
        ------
        `AttributeError`
            if the data for this test has not been loaded yet, or has been
            invalidated by updating the metadata with `load_metadata()`
        """

        if self.specimen.noSS or self._noDisp:
            raise DataNotAvailableError("Strain not available for this test")

        if avg:
            return self._get_avg(self.disp) / self.specimen.length
        else:
            return self.strain.raw

    def get_stress(self, avg=False):
        """
        Gets the stress data for this test, if available.

        Parameters
        ----------
        `avg` : `bool` (optional)
            Specifies whether to compute the average displacement over all
            repeated trials. Default: `False`

        Returns
        -------
        the stress data for this test if force was collected and the specimen
        cross-sectional area is available

        Raises
        ------
        `AttributeError`
            if the data for this test has not been loaded yet, or has been
            invalidated by updating the metadata with `load_metadata()`
        """

        if self.specimen.noSS or self._noForce:
            raise DataNotAvailableError("Stress not available for this test")

        if avg:
            return self._get_avg(self.force) / self.specimen.area
        else:
            return self.stress.raw

    def get_complex_modulus(self, avg=False):
        """
        Gets the complex modulus data for this test, if available.

        Parameters
        ----------
        `avg` : `bool` (optional)
            Specifies whether to compute the average displacement over all
            repeated trials. Default: `False`

        Returns
        -------
        the complex modulus for this test if displacement and force were
        collected, the specimen initial length was specified, and the specimen
        cross-sectional area is available
        """

        return self.get_stress(avg).Y / self.get_strain(avg).Y
