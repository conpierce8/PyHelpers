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
# Modified: 2022-09-17 22:58:10
#
# Copyright (c) 2019-2022 Connor D. Pierce
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


"""
Input/output utilities for loading data collected in experiments and exported
from simulations.
"""


### Imports ====================================================================
#
# Import all packages required to handle input, output, and storage of data and
# handle unit conversions.

import numpy as np
import os
import pint
import scipy as sp
import typing
import yaml

from helpers import ureg, Qty, EmptyObject, factors
from scipy import signal, stats

# Set up logging
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


### Function Definitions =======================================================

# Create a function so we can easily augment the input settings with
# the output settings to complete a full test definition
def augmentInput(inData, outData):
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


### Exceptions =================================================================


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


### Classes ====================================================================


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
    _name : str
        Internal name (e.g. "x0", "x1", etc.)
    name : str
        Display name from the file (e.g. "Frequency", "Amplitude")
    axis : int
        Axis in the Y array that is parameterized by the values of this variable
    col : int
        Column number (0-based) in the file from which the values of this
        variable are extracted
    t_axis : int
        Axis in the Y array that is parameterized by different trials of this
        variable
    values : numpy.ndarray(dtype=numpy.float64)
        The parameter values of this variable
    Tn : int
        The number of trials for this variable

    """

    def __init__(self):
        self._name = None
        self.name = None
        self.axis = None
        self.col = None
        self.trial = None
        self.values = None

    def copy(self):
        iv = IndependentVariable()
        iv._name = self._name
        iv.name = self.name
        iv.axis = self.axis
        iv.col = self.col
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

    def index(self, obj):
        """Index subset of values by `obj`."""

        raise NotImplementedError("TODO: not implemented yet")


class Trial:
    """Wrapper class indicating a trial axes of an independent variable."""

    def __init__(self, iv: IndependentVariable):
        """Create a Trial axis linked to independent variable `iv`."""

        self.iv = iv
        self.trials = None
        self.axis = None

    def __getattr__(self, attr):
        if attr == "name":
            return self.iv.name + " trial"
        elif attr == "_name":
            return self.iv._name + "_trial"
        elif attr in ("col",):
            return self.iv.__getattribute__(attr)
        else:
            raise AttributeError(f"Attribute {attr} not found")

    def index(self, obj):
        """Index a subset of trial indices."""

        raise NotImplementedError("TODO: not implemented")


class DependentVariable:
    """
    Holds general information about a dependent variable in a SweepTest.

    Contains the following fields, which can be accessed directly:

    _name : str
        Internal name (e.g. "y0", "y1", etc.)
    name : str
        Display name from the file (e.g. "Frequency", "Amplitude")
    axis : int
        Axis in the Y array that is parameterized by the different dependent
        variables.
    idx : int
        Index (along axis `axis` in the Y array) that holds the value of this
        independent variable.
    col : int
        Column number (0-based) in the file from which the values of this
        variable are extracted

    """

    def __init__(self):
        self._name = None
        self.name = None
        self.axis = None
        self.idx = None
        self.col = None

    def copy(self):
        dv = DependentVariable()
        dv._name = self._name
        dv.name = self.name
        dv.axis = self.axis
        dv.idx = self.idx
        dv.col = self.col
        return dv


class SweepTest:
    """
    Loads and organizes the data associated with a general parametric sweep. A
    general parametric sweep consists of a set of tests in which one or more
    independent variables (the "parameters") are systematically varied over a
    set of values. A parametric sweep with `N` parameters measures the response
    of a system over a discretized region of `N`-space. The system response may
    be a scalar or a vector

    Let `y` be a `P`-dimensional function of `N` independent variables; that is,
    let

        y = [ y_0, y_1, ..., y_p, ... y_(P-1) ]

    where

        y_p = f( x_0, x_1, ..., x_n, ..., x_(N-1) )

    with `0 <= p < P` and `0 <= n < N`. Let each of the independent variables
    `x_n` be parameterized into a discrete set of `M_n` values; that is, let

        x_n = [ x_n[0], x_n[1], ..., x_n[m_n], ..., x_n[M_n - 1] ]

    with `0 <= m_n < M_n`. Let the domain `X` of `y` be formed by all possible
    combinations (i.e. the outer product) of the values in the vectors `x_n`,
    such that:

        X[m_0,m_1,...,m_n,...,m_(N-1)] = [ x_0[m_0], x_1[m_1], ...,
            x_n[m_n], ..., x_(N-1)[m_(N-1)] ]

    for all `m_n`. Finally, let `y(X)` be evaluated `T_n` times for each
    variable `x_n`, so that `y_p(x_n)` is `T_0*T_1*...*T_n*...*T_(N-1)`-valued.

    Then the `p`th component of `y`, `y_p(x_n)`, can be expressed as an array of
    shape `(T_0, T_1, ..., T_n, ..., T_(N-1))`, where

        y_p(x_n)[t_0,t_1,...,t_n,...t_(N-1)]

    is the value of `y[p](x_n)` for the `t_n`th iteration of the variable `x_n`,
    where `0 <= t_n < T_n`.

    # ACCESSING THE DATA IN THIS PARAMETRIC SWEEP
    The data in this parametric sweep can be accessed in two different ways.
    The first way is to directly index the array `Y` where the data is stored.
    In this array, the first `N` axes correspond to the different values of the
    independent variables `x_n`, followed by an axis corresponding to the `P`
    components `y_p`, followed by `N` axes corresponding to the different trials
    `t_n` of each independent variable. Thus, the axes ordering of this array is
    as follows:

        Y[m_0, m_1, ..., m_n, ..., m_(N-1), p, t_0, t_1, ..., t_n, ..., t_(N-1)]

    and the shape of the array is:

        (M_0, M_1, ..., M_n, ..., M_(N-1), P, T_0, T_1, ..., T_n, ..., T_(N-1))

    Note that if the variable `x_n` is only repeated once, the array `Y` will
    contain an axis of length 1 for the trial index `t_n`.
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

        # Create collections so that the IndependentVariable and
        # DependentVariable objects can be easily accessed by different
        # attributes.
        self.names = {}  # For accessing variable details by display name
        self.axes = {}  # For accessing ind. var. details by axis number
        self._names = {}  # For accessing variable details by internal name
        self.dep_idx = {}  # For accessing dep. var details by index
        self.ind_vars = []  # For storing ind. var. in order
        self.dep_vars = []  # For storing dep. var. in order

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
        units: typing.Union[None, list[typing.Union[str, None, pint.Unit], ...]] = None,
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

        if units is not None:
            raise NotImplementedError("Not implemented yet.")
            # Unit-sensitive data will require refactoring self.Y into several
            # arrays, one for each dependent variable, since pint associates one
            # unit with the entire array, but dependent variables may have
            # different units

        # Create variables
        self._create_vars(self._parse_col_spec(columns), raw_data, names)

        # Determine `M_n` and `T_n` for each variable
        self._calc_sweep_limits(raw_data)

        # Assign `t` axes if necessary
        self._assign_t_axes()

        # Create and populate Y array
        self._create_y_array(raw_data)

    @property
    def shape(self):
        shape = {}
        for i in range(len(self.Y.shape)):
            shape[i] = self.Y.shape[i]
            if self.axes[i] == DependentVariable:
                shape[DependentVariable] = self.Y.shape[i]
            else:
                shape[self.axes[i].name] = self.Y.shape[i]
        return shape

    @property
    def dim(self):
        return len(self.Y.shape)

    def __getitem__(self, idx):
        """Index the SweepTest by `idx`.

        Several types of indexing are supported:

        1.  Indexing by a scalar dependent variable. If `st` is a `SweepTest`
            object containing vector-valued output data, the value of a single
            scalar component of the output can be returned by indexing `st` with
            either a string (the `name` or `_name` of the desired scalar
            component) or a `DependentVariable`.
        2.
        """

        if isinstance(idx, str):
            # Indexing a scalar component of the output
            if idx in self._names:
                if isinstance(self._names[idx], DependentVariable):
                    return self[self._names[idx]]
                else:
                    raise IndexError(
                        "Cannot index by independent variable: '" + idx + "'"
                    )
            elif idx in self.names:
                try:
                    return self[self.names[idx]._name]
                except IndexError:
                    raise IndexError("No dependent variable named " + idx)
            else:
                raise IndexError("No variable named " + idx)
        elif isinstance(idx, DependentVariable):  # Scalar index by a dependent variable
            if idx in self.dep_vars and idx._name in self._names:
                _st = SweepTest()
                for iv in self.ind_vars:
                    iv_copy = iv.copy()
                    _st.ind_vars.append(iv_copy)
                    _st.axes[iv_copy.axis] = iv_copy
                    _st.names[iv_copy.name] = iv_copy
                    _st._names[iv_copy._name] = iv_copy
                    if iv_copy.trial is not None:
                        trial_copy = Trial(iv_copy)
                        trial_copy.axis = iv.trial.axis - 1
                        trial_copy.trials = iv.trial.trials
                        _st.axes[trial_copy.axis] = trial_copy
                dv = idx.copy()
                dv.axis = None
                dv.idx = None
                _st.dep_vars.append(dv)
                slc = tuple([slice(None) for i in range(idx.axis)] + [idx.idx, ...])
                _st.P = 0
                _st.N = self.N
                _st.num_t_axes = self.num_t_axes
                _st.Y = self.Y[slc]
                return _st
            else:
                raise IndexError("No such dependent variable: " + str(idx))
        elif isinstance(idx, int) or isinstance(idx, float):
            if self.dim == 1:
                return self[(idx,)]
            else:
                raise IndexError(
                    "Scalar indexing not supported for {n:d}D array".format(n=self.dim)
                )
        elif isinstance(idx, tuple):
            if sum([1 if isinstance(i, type(Ellipsis)) else 0 for i in idx]) > 1:
                raise IndexError("An index can only have a single ellipsis.")
            if len(idx) != self.dim:
                if ... in idx:
                    # Expand the Ellipsis
                    _pos = idx.index(...)
                    return self[
                        idx[:_pos]
                        + (slice(None),) * (self.dim - len(idx) + 1)
                        + idx[_pos + 1 :]
                    ]
                elif len(idx) < self.dim:
                    # Pad the end of the index with empty slices
                    return self[idx + (slice(None),) * (self.dim - len(idx))]
                else:
                    raise IndexError(
                        "Too many indices ({0:d}) for {1:d}D array".format(
                            len(idx),
                            self.dim,
                        )
                    )
            else:
                _has_adv_idx = False
                _adv_idxs = []
                _adv_idxs_contig = True
                for i, item in enumerate(idx):
                    if isinstance(item, list) or isinstance(item, np.ndarray):
                        _has_adv_idx = True
                        _adv_idxs.append(i)
                        if len(_adv_idxs) > 1:
                            _adv_idxs_contig = (
                                _adv_idxs_contig and (i - _adv_idxs[-2]) == 1
                            )
                if _has_adv_idx:
                    return self._do_adv_idx(idx, _adv_idxs, _adv_idxs_contig)
                else:
                    return self._do_basic_idx(idx)
        elif isinstance(idx, dict):
            _idx = [slice(None) for i in range(self.dim)]
            for key in idx:
                if key in self._names:
                    _idx[self._names[key].axis] = idx[key]
                elif key in self.names:
                    _idx[self.names[key].axis] = idx[key]
                elif key == DependentVariable:
                    if self.P > 0:
                        _idx[self.N] = idx[key]
                    else:
                        raise IndexError(
                            "Cannot index by dependent variable; output is "
                            "already scalar."
                        )
                else:
                    raise IndexError("No variable named " + str(key))

            return self[tuple(_idx)]
        elif idx is Ellipsis:
            return self[tuple([slice(None) for i in self.Y.shape])]
        else:
            raise IndexError("Indexing not supported for type '" + str(type(idx)) + "'")

    @staticmethod
    def reshape_adv_idx(t: typing.Iterable, N: int, i: int, lv: int = 0):
        """Reshape advanced index to a broadcastable shape."""

        if lv == 0:
            if lv == N - 1 - i:
                return t
            else:
                return [SweepTest.reshape_adv_idx(t_i, N, i, lv + 1) for t_i in t]
        else:
            if lv == N - 1 - i:
                return [t]
            else:
                return [SweepTest.reshape_adv_idx(t, N, i, lv + 1)]

    def _do_adv_idx(
        self,
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
                    for j in item:
                        if not isinstance(j, int):
                            raise IndexError(
                                "Can only index a list of ints; found " + str(type(j))
                            )
                    idx_copy.append(
                        SweepTest.reshape_adv_idx(item, len(adv_idxs), adv_idx_count)
                    )
                    new_shape.append(len(item))
                elif isinstance(item, np.ndarray):
                    if len(item.shape) != 1:
                        raise IndexError("Can only index 1D arrays")

                    L = item.size
                    var = self.axes[i]
                    if var == DependentVariable:
                        raise IndexError("Cannot index dependent variable by value")
                    elif isinstance(var, Trial):
                        raise IndexError("Cannot index '" + var.name + "' by value")

                    M = var.values.size
                    matches = np.sum(
                        np.arange(1, M + 1).reshape(M, 1)
                        * np.isclose(var.values.reshape(M, 1), item),
                        axis=0,
                        dtype=np.int32,
                    )
                    valid = matches > 0
                    matches = list(matches[valid] - 1)
                    idx_copy.append(
                        SweepTest.reshape_adv_idx(matches, len(adv_idxs), adv_idx_count)
                    )
                    new_shape.append(len(matches))
                adv_idx_count += 1
                axes_map[i] = new_axis
                new_axis += 1
            else:
                if isinstance(item, int):
                    idx_copy.append(slice(item, item + 1, 1))
                    int_axes.append(i)
                    axes_map[i] = None
                else:
                    idx_copy.append(item)
                    axes_map[i] = new_axis
                    new_axis += 1
                    old_shape = self.Y.shape[i]
                    new_shape.append(len([i for i in range(*item.indices(old_shape))]))
        idx_copy = tuple(idx_copy)
        logger.debug("_do_adv_idx: idx_copy == " + str(idx_copy))
        logger.debug("_do_adv_idx: int_axes == " + str(int_axes))

        # Copy data, fix the axes reordering induced by non-contiguous advanced
        # indices, and remove all axes indexed with int
        data_copy = self.Y[idx_copy]
        logger.debug("_do_adv_idx: data_copy.shape == " + str(data_copy.shape))
        logger.debug(
            "_do_adv_idx: data_copy.OWNDATA == " + str(data_copy.flags["OWNDATA"])
        )
        if not adv_idx_contig:
            ax_order = [None for l in data_copy.shape]
            count_basic_idx = 0
            for i in range(len(self.Y.shape)):
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

        _st = SweepTest()
        _st.Y = data_copy
        self._do_copy_vars(_st, idx_copy, axes_map)
        return _st

    def _do_basic_idx(self, idx: tuple[typing.Union[int, slice], ...]):
        """Perform basic indexing on `self`."""

        logger.debug("_do_adv_idx: idx == " + str(idx))

        # Convert all advanced indexes to broadcastable shapes. Raise error if
        # a non-flat index is encountered.
        axes_map = {}
        new_axis = 0
        for i, item in enumerate(idx):
            if isinstance(item, int):
                axes_map[i] = None
            else:
                axes_map[i] = new_axis
                new_axis += 1
        logger.debug("_do_adv_idx: axes_map == " + str(axes_map))

        _st = SweepTest()
        _st.Y = self.Y[idx].copy()
        self._do_copy_vars(_st, idx, axes_map)
        return _st

    def _do_copy_vars(self, _st, idx, axes_map):
        # Copy all IndependentVariables:
        #   - independent variables that have previously been scalar indexed
        #     require, at most, transformations to their trial metadata.
        #   - independent variables that were previously scalar indexed or are
        #     scalar indexed in the current indexing operation are stored in all
        #     metadata collections (names, _names, ind_vars) except `axes`
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
            _st._names[iv_copy._name] = iv_copy
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
                _st._names[trial_copy._name] = trial_copy
        _st.N = N

        # Copy DependentVariables
        if self.dep_vars[0].axis is None:  # Previously indexed by a scalar
            dv_copy = self.dep_vars[0].copy()
            dv_copy.idx = None
            dv_copy.axis = None#axes_map[self.dep_vars[0].axis]
            _st.dep_vars.append(dv_copy)
            _st.P = 0
        elif axes_map[self.dep_vars[0].axis] is None:  # Currently indexed by a scalar
            _dep_idx = idx[self.dep_vars[0].axis]
            if isinstance(_dep_idx, slice):
                _dep_idx = _dep_idx.indices(len(self.dep_vars))[0]
            dv_copy = self.dep_idx[_dep_idx].copy()
            dv_copy.axis = None
            dv_copy.idx = None
            _st.dep_vars.append(dv_copy)
            _st.names[dv_copy.name] = dv_copy
            _st._names[dv_copy._name] = dv_copy
            _st.P = 0
        else:
            old_axis = self.dep_vars[0].axis
            new_axis = axes_map[old_axis]
            if isinstance(idx[old_axis], slice):
                _dep_idx = [i for i in range(*idx[old_axis].indices(self.P))]
                _st.P = len(_dep_idx)
            else:
                _dep_idx = np.array(idx[old_axis]).flatten()
                _st.P = _dep_idx.size
            for i, _dep_idx_i in enumerate(_dep_idx):
                dv_copy = self.dep_idx[_dep_idx_i].copy()
                dv_copy.axis = new_axis
                dv_copy.idx = i
                _st.dep_vars.append(dv_copy)
                _st.names[dv_copy.name] = dv_copy
                _st._names[dv_copy._name] = dv_copy
                _st.dep_idx[i] = dv_copy
            _st.axes[new_axis] = DependentVariable
        return

    def _create_y_array(self, raw_data):
        # Create array of row indices, used for calculating the indexing array
        rows = np.arange(raw_data.shape[0], dtype=np.int32)
        M = []
        T = []

        shape = []
        idx_shape = []
        for iv in self.ind_vars:
            shape.append(iv.values.size)
            idx_shape.append(iv.values.size)
            logger.debug(
                "Blocksize:" + str(iv._blocksize) + ", multiplicity: " + str(iv._mult)
            )
            M.append((rows % (iv._blocksize)) // iv._mult)

        shape.append(len(self.dep_idx))
        idx_shape.append(1)

        for iv in self.ind_vars:
            if iv._Tn > 1:
                shape.append(iv._Tn)
                idx_shape.append(iv._Tn)
                T.append((rows // (iv._blocksize)) % iv._Tn)

        row_idx = np.zeros(idx_shape, dtype=np.int32)
        row_idx[(*M, 0, *T)] = rows

        col_shape = (self.P, *np.ones(len(T), dtype=np.int32))
        cols = []
        for d in sorted(self.dep_idx.keys()):
            dv = self.dep_idx[d]
            cols.append(dv.col)
        col_idx = np.array(cols).reshape(col_shape)

        self.Y = raw_data[row_idx, col_idx]

    def _assign_t_axes(self):
        """Assign trial axis numbers for all independent variables."""

        ax_count = self.N + 1
        for iv in self.ind_vars:
            if iv._Tn > 1:
                trial = Trial(iv)
                trial.axis = ax_count
                trial.trials = np.arange(iv._Tn)
                self.axes[ax_count] = trial
                self.names[trial.name] = trial
                self._names[trial._name] = trial
                iv.trial = trial
                ax_count += 1
                self.num_t_axes += 1

    def _calc_sweep_limits(self, raw_data):
        """Calculate the number of values `M_n` and repetitions `T_n` per
        variable."""

        n = raw_data.shape[0]
        possible_sizes = factors(n)

        block_sizes = {}
        for v in self._names:
            var = self._names[v]
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
            Tn = prev_multiplicity // var._blocksize
            Mn = np.unique(raw_data[:, var.col]).size

            prev_multiplicity = var._blocksize // Mn

            var._Tn = Tn
            var.values = raw_data[0 : var._blocksize : prev_multiplicity, var.col]
            var._Mn = Mn
            var._mult = prev_multiplicity
            logger.debug("Var " + var.name + ": Tn=" + str(Tn) + ", Mn=" + str(Mn))

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
        # the expand specified (if present)
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
        axes = {}
        names = {}
        _names = {}
        ind_vars = []
        dep_vars = []
        dep_idx = {}

        N = 0
        P = 0
        _len_dep = len(dep_spec["num"]) + len(dep_spec["non"])

        def create_var(var_type, _name, col):
            logger.debug(
                "create_var: var_type = "
                + " ".join([str(var_type), "_name =", _name, "col =", str(col)])
            )

            # Create a variable and fill out the properties
            var = var_type()
            var._name = _name
            var.col = col
            var.name = f"_Col{var.col}" if hdgs[var.col] is None else str(hdgs[var.col])

            if var.name in names:
                raise KeyError(f"Column name {var.name} is not unique")

            # Store the variable in the dictionaries
            if isinstance(var, IndependentVariable):
                var.axis = N
                ind_vars.append(var)
                axes[var.axis] = var
            else:
                if _len_dep > 1:
                    logger.debug("_len_dep > 1; N = " + str(N))
                    var.axis = N
                    var.idx = P
                    dep_idx[var.idx] = var
                else:
                    var.axis = None
                    var.idx = None
                dep_vars.append(var)
            names[var.name] = var
            _names[var._name] = var

        for i in sorted(ind_spec["num"].keys()):
            create_var(
                IndependentVariable,
                "x{0:d}".format(i),
                ind_spec["num"][i],
            )
            N += 1
        for i, col in enumerate(ind_spec["non"]):
            create_var(
                IndependentVariable,
                "x{0:d}".format(ind_spec["max_number"] + 1 + i),
                col,
            )
            N += 1
        for i in sorted(dep_spec["num"].keys()):
            create_var(
                DependentVariable,
                "y{0:d}".format(i),
                dep_spec["num"][i],
            )
            P += 1
        for i, col in enumerate(dep_spec["non"]):
            create_var(
                DependentVariable,
                "y{0:d}".format(dep_spec["max_number"] + 1 + i),
                col,
            )
            P += 1
        logger.debug("axes are " + str(axes))

        # Parsing succeeded; store the temporary variables in self
        self.N = N
        if _len_dep > 1:
            # Allocate an axis for the dependent variable components
            axes[self.N] = DependentVariable
            self.P = P
        self.names = names
        self.axes = axes
        self._names = _names
        self.ind_vars = ind_vars
        self.dep_vars = dep_vars
        self.dep_idx = dep_idx
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
                    "Found different axis for "
                    + iv.name
                    + " in test "
                    + self.test.name
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
            data = field.raw[{s:0 for s in params_to_avg}]
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
