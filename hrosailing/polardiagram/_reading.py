"""
Methods for reading polar diagram information from files.
"""

import csv
from ast import literal_eval

import numpy as np

from hrosailing.core.exceptions import FileReadingException
from hrosailing.polardiagram._polardiagramtable import (
    PolarDiagram,
    PolarDiagramTable,
)
from hrosailing.polardiagram._incompletedatahandler import (
    interpolate_missing_values,
)


def from_csv(csv_path, fmt="hro", interpolate_missing=False, interpolator=None):
    """Reads a .csv file and returns the `PolarDiagram`
    instance contained in it.

    Parameters
    ----------
    csv_path : path-like
        Path to a .csv file.

    fmt : str
        The format of the .csv file.

        - `"hro"` : format created by the `to_csv`-method of the `PolarDiagram` class,
        - `"orc"` : format found at [ORC](https://\
            jieter.github.io/orc-data/site/),
        - `"opencpn"` : format created by the [OpenCPN Polar Plugin](https://\
            opencpn.org/OpenCPN/plugins/polar.html),
        - `"array"` : tab-separated polar diagram in form of a table, also
            see the example files for a better look at the format.

    interpolate_missing : bool, optional
        If True, interpolate missing values in incomplete polar diagrams.
        Works with orc and opencpn formats (which support missing values).
        Array and hro formats typically don't contain missing values. Default is False.

    interpolator : Interpolator, optional
        Custom interpolator to use for missing value interpolation.
        If None, uses ArithmeticMeanInterpolator(params=(50,)) as default.

    Returns
    -------
    out : PolarDiagram
        `PolarDiagram` instance contained in the .csv file.

    Raises
    ------
    FileReadingException
        If, in the format `hro`, the first row does not match any
        `PolarDiagram` subclass.

    OSError
        If file does not exist or no read permission for that file is given.

    Examples
    --------
    (For the following and more files also see
    [examples](https://github.com/hrosailing/hrosailing/tree/main/examples))

        >>> from polardiagram._reading import from_csv
        >>> pd = from_csv("table_hro_format_example.csv", fmt="hro")
        >>> print(pd)
        
        >>> # Load incomplete ORC file with automatic interpolation
        >>> pd_incomplete = from_csv("incomplete.orc.csv", fmt="orc", interpolate_missing=True)
        >>> print(f"Interpolation performed: {pd_incomplete._interpolation_performed}")
        
        >>> # Use custom interpolator for missing values
        >>> from hrosailing.processing.interpolator import ShepardInterpolator
        >>> custom_interp = ShepardInterpolator()
        >>> pd_custom = from_csv("sparse.csv", fmt="orc", interpolate_missing=True, interpolator=custom_interp)
          TWA / TWS    6.0    8.0    10.0    12.0    14.0    16.0    20.0
        -----------  -----  -----  ------  ------  ------  ------  ------
        52.0          3.74   4.48    4.96    5.27    5.47    5.66    5.81
        60.0          3.98   4.73    5.18    5.44    5.67    5.94    6.17
        75.0          4.16   4.93    5.35    5.66    5.95    6.27    6.86
        90.0          4.35   5.19    5.64    6.09    6.49    6.70    7.35
        110.0         4.39   5.22    5.68    6.19    6.79    7.48    8.76
        120.0         4.23   5.11    5.58    6.06    6.62    7.32    9.74
        135.0         3.72   4.64    5.33    5.74    6.22    6.77    8.34
        150.0         3.21   4.10    4.87    5.40    5.78    6.22    7.32
    """
    if fmt not in {"array", "hro", "opencpn", "orc"}:
        raise ValueError("`fmt` unknown")

    with open(csv_path, "r", newline="", encoding="utf-8") as file:
        if fmt == "hro":
            return _read_intern_format(file)

        return _read_extern_format(file, fmt, interpolate_missing, interpolator)


def _read_intern_format(file):
    subclasses = {cls.__name__: cls for cls in PolarDiagram.__subclasses__()}

    first_row = file.readline().rstrip()
    if first_row not in subclasses:
        raise FileReadingException(
            f"no polar diagram format with the name {first_row} exists"
        )

    pd = subclasses[first_row]
    return pd.__from_csv__(file)


def _read_extern_format(file, fmt, interpolate_missing=False, interpolator=None):
    if fmt == "array":
        ws_res, wa_res, bsps = _read_from_array(file)
    elif fmt == "orc":
        ws_res, wa_res, bsps = _read_orc_format(file)
    else:
        ws_res, wa_res, bsps = _read_opencpn_format(file)

    # Apply interpolation to any format if requested
    interpolation_performed = False
    if interpolate_missing:
        bsps, interpolation_performed = interpolate_missing_values(ws_res, wa_res, bsps, interpolator)

    pd = PolarDiagramTable(ws_res, wa_res, bsps)
    
    # Store interpolation flag for later access
    pd._interpolation_performed = interpolation_performed
    
    return pd


def _read_from_array(file):
    file_data = np.genfromtxt(file)
    return file_data[0, 1:], file_data[1:, 0], file_data[1:, 1:]


def _read_orc_format(file):
    csv_reader = csv.reader(file, delimiter=";")

    ws_res = _read_wind_speeds(csv_reader)

    # skip line of zeros
    next(csv_reader)

    wa_res, bsps = _read_wind_angles_and_boat_speeds(csv_reader)

    return ws_res, wa_res, bsps


def _read_wind_speeds(csv_reader):
    return [literal_eval(ws) for ws in next(csv_reader)[1:]]


def _read_wind_angles_and_boat_speeds(csv_reader):
    wa_res = []
    bsps = []

    for row in csv_reader:
        if row:  # Skip empty rows
            wa_res.append(literal_eval(row[0].replace("°", "")))
            bsps.append([literal_eval(bsp) if bsp != "" else np.nan for bsp in row[1:]])

    return wa_res, bsps


def _read_opencpn_format(file):
    csv_reader = csv.reader(file, delimiter=",")

    ws_res = _read_wind_speeds(csv_reader)
    wa_res, bsps = _read_wind_angles_and_boat_speeds(csv_reader)

    return ws_res, wa_res, bsps

