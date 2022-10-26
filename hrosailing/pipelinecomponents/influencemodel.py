"""
Defines the `InfluenceModel` abstract base class that can be used
to create custom influence models.

Subclasses of `InfluenceModel` can be used with

- the `PolarPipeline` class in the `hrosailing.pipeline` module,
- various functions in the `hrosailing.cruising` module.
"""

from ._utils import data_dict_to_numpy
from hrosailing.wind import convert_apparent_wind_to_true

from abc import ABC, abstractmethod

import numpy as np


class InfluenceException(Exception):
    """Raised when removing or adding influence does not work."""


class InfluenceModel(ABC):
    """Base class for all influence model classes.


    Abstract Methods
    ----------------
    remove_influence(data)

    add_influence(pd, influence_data)

    fit(training_data)
    """

    @abstractmethod
    def remove_influence(self, data):
        """This method should be used to create a `numpy.ndarray`
        output from a given `Data` object where the columns correspond to wind speed, wind angle and
        boat speed respectively.

        Parameters
        ----------
        data : Data
            Should contain at least keys for wind speed, wind angle
            and either speed over ground, speed over water or boat speed.

        Returns
        -------
        out : numpy.ndarray

        statistics : dict
            Dictionary containing relevant statistics.
        """

    @abstractmethod
    def add_influence(self, pd, influence_data):
        """This method should be used, given a polar diagram and a
        dictionary, to obtain a modified boat speed of that given
        in the polar diagram, based on the influences presented in
        the given dictionary, such as wave height, underlying currents etc.

        Parameters
        ----------
        pd : PolarDiagram

        influence_data : Data or dict
            Further influences to be considered.
        """

    @abstractmethod
    def fit(self, training_data):
        """
        This method should be used to fit parameters of the influence
        model to the given training data.

        Parameters
        ----------
        training_data : Data
        """


class IdentityInfluenceModel(InfluenceModel):
    """An influence model which ignores most influences and just calculates
    the true wind if only the apparent wind is given.
    IF 'BSP' is not provided by the data, 'SOG' is used instead."""

    def remove_influence(self, data: dict):
        """
        Ignores most influences as described above.

        Parameters
        ----------
        data : dict
            Data dictionary, must either provide 'BSP' or 'SOG' key as well as
            either the keys 'TWS', 'TWA' or 'AWS', 'AWA'.

        Returns
        -------
        (n,3)-array, {}

        See also
        --------
        `InfluenceModel.remove_influence`
        """
        return _get_true_wind_data(data), {}

    def add_influence(self, pd, influence_data: dict):
        """
        Ignores most influences and strictly calculates the boat speed using the
        polar diagram.

        Parameters
        ----------
        pd : PolarDiagram

        influence_data : dict
            Either a dictionary of lists or a dictionary of values containing
            one or more sets of influence data.
            At least the keys 'TWS' and 'TWA' should be provided.

        Returns
        -------
        speeds : float or list of floats
            The boat speed if `influence_data` contained values,
            a list of respective boat speeds if `influence_data` contained
            lists.

        See also
        --------
        `InfluenceModel.add_influence`
        """
        if isinstance(influence_data["TWS"], list):
            wind = zip(influence_data["TWS"], influence_data["TWA"])
            speed = [pd(ws, wa) for ws, wa in wind]
        else:
            ws, wa = influence_data["TWS"], influence_data["TWA"]
            speed = pd(ws, wa)
        return speed

    def fit(self, training_data: dict):
        """
        Does nothing, returns an empty dictionary.

        See also
        --------
        `InfluenceModel.fit`
        """
        return {}


class WindAngleCorrectingInfluenceModel(InfluenceModel):
    """An influence model which corrects a structural measurement error in 'TWA'.

    Parameter
    ----------
    wa_shift: int or float, optional
        Difference between real wind angle and measured wind angle (correction value).

        Defaults to 0.

    interval_size: int or float, optional
        The size in degrees of the interval that is used to evaluate the points.

        Defaults to 30.
    """

    def __init__(self, interval_size = 30, wa_shift = 0):
        self._wa_shift = wa_shift
        self._interval_size = interval_size

    def remove_influence(self, data):
        """
        Removes the correction value to the data.

        See also
        -------
        `InfluenceModel.remove_influence`
        """
        wind_data = _get_true_wind_data(data)
        wa = wind_data[:, 1]
        wind_data[:, 1] = (wa - self._wa_shift)%360
        return wind_data, {}

    def add_influence(self, pd, influence_data):
        """
        Adds the correction value from the data.

        See also
        -------
        `InfluenceModel.remove_influence`

        """
        if isinstance(influence_data["TWS"], (list, np.ndarray)):
            wind = zip(influence_data["TWS"], influence_data["TWA"])
            speed = [pd(ws, (wa + self._wa_shift)%360) for ws, wa in wind]
        else:
            ws, wa = influence_data["TWS"], influence_data["TWA"]
            speed = pd(ws, wa)
        return speed

    def fit(self, training_data):
        """
        Assumes, that the wind angle with the fewest data points in a region is the actual zero angle.

        See also
        --------
        `InfluenceModel.fit`
        """
        wind_angles = training_data["TWA"]
        sample = np.linspace(0, 359, 10*360)
        counts = [
            sum([np.exp(-((wa - other_wa)/self._interval_size)**2) for other_wa in wind_angles])
            for wa in sample
        ]
        min_angle, _ = min(zip(sample, counts), key=lambda x: x[1])
        self._wa_shift = min_angle
        return {}


def _get_true_wind_data(data: dict):
    speed = "BSP" if "BSP" in data else "SOG"
    if "AWA" in data and "AWS" in data:
        apparent_data = data_dict_to_numpy(data, ["AWS", "AWA", speed])
        return convert_apparent_wind_to_true(apparent_data)
    elif "TWA" in data and "TWS" in data:
        if isinstance(data["TWS"], list):
            return data_dict_to_numpy(data, ["TWS", "TWA", speed])
        else:
            return np.array([data["TWS"], data["TWA"], data[speed]])
    else:
        raise InfluenceException(
            "No sufficient wind data is given in order to apply influence"
            "model. Either give 'AWA' and 'AWS' or 'TWA' and 'TWS'"
        )
