"""
Functions to convert wind from apparent to true and vice versa
"""

# Author: Valentin F. Dannenberg / Ente


from collections.abc import Iterable
import warnings

import numpy as np


class WindException(Exception):
    """Custom exception for errors that may appear during
    wind conversion or setting wind resolutions"""

    pass


def apparent_wind_to_true(wind):
    """Converts apparent wind to true wind

    Parameters
    ----------
    wind : array_like of shape (n, 3)
        Wind data given as a sequence of points consisting of wind speed,
        wind angle and boat speed, where the wind speed and wind angle are
        measured as apparent wind

    Returns
    -------
    out : numpy.ndarray of shape (n, 3)
        Array containing the same data as wind_arr, but the wind speed
        and wind angle now measured as true wind

    Raises a WindException if wind contains NaNs or infinite values
    """
    return convert_wind(wind, -1, tw=False, _check_finite=True)


def true_wind_to_apparent(wind):
    """Converts true wind to apparent wind

    Parameters
    ----------
    wind : array_like of shape (n, 3)
        Wind data given as a sequence of points consisting of wind speed,
        wind angle and boat speed, where the wind speed and wind angle are
        measured as true wind

    Returns
    -------
    out : numpy.ndarray of shape (n, 3)
        Array containing the same data as wind_arr, but the wind speed
        and wind angle now measured as apparent wind

    Raises a WindException if wind contains NaNs or infinite values
    """
    return convert_wind(wind, 1, tw=False, _check_finite=True)


def convert_wind(wind, sign, tw, _check_finite=True):
    # Only check for NaNs and infinite values, if wanted
    if _check_finite:
        # NaNs and infinite values can't be handled
        wind = np.asarray_chkfinite(wind)
    else:
        wind = np.asarray(wind)

    if wind.dtype is object:
        raise WindException("`wind` is not array_like")

    if wind.shape[1] != 3 or wind.ndim != 2:
        raise WindException("`wind` has incorrect shape")

    if tw:
        return wind

    ws, wa, bsp = np.hsplit(wind, 3)
    wa_above_180 = wa > 180
    wa = np.deg2rad(wa)

    cws = np.sqrt(
        np.square(ws) + np.square(bsp) + sign * 2 * ws * bsp * np.cos(wa)
    )

    # account for computer error
    temp = (ws * np.cos(wa) + sign * bsp) / cws
    temp[temp > 1] = 1
    temp[temp < -1] = -1

    cwa = np.arccos(temp)

    # standardize angles to [0, 360) inverval after conversion
    cwa[wa_above_180] = 360 - np.rad2deg(cwa[wa_above_180])
    cwa[np.invert(wa_above_180)] = np.rad2deg(cwa[np.invert(wa_above_180)])

    return np.column_stack((cws, cwa, bsp))


def set_resolution(res, speed_or_angle):
    b = speed_or_angle == "speed"

    if res is None:
        return np.arange(2, 42, 2) if b else np.arange(0, 360, 5)

    if isinstance(res, Iterable):
        # NaN's and infinite values can't be handled
        res = np.asarray_chkfinite(res)

        if res.dtype is object:
            raise WindException("`res` is not array_like")

        if not res.size or res.ndim != 1:
            raise WindException("`res` has incorrect shape")

        if len(set(res)) != len(res):
            warnings.warn(
                "`res` contains duplicate data. "
                "This may lead to unwanted behaviour"
            )

        return res

    if res <= 0:
        raise WindException("`res` is nonpositive")

    return np.arange(res, 40, res) if b else np.arange(res, 360, res)
