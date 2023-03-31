from unittest import TestCase

import numpy as np

import hrosailing.core.data as dt
import hrosailing.processing.weigher as wgh


class TestAllOneWeigher(TestCase):
    def setUp(self) -> None:
        self.data = dt.Data().from_dict(
            {"TWS": [13, 14, 15], "TWA": [35, 37, 36]}
        )
        self.np_arr = np.array([[13, 35], [14, 37], [15, 36]])

    def test_weigh_data(self):
        """
        Input/Output-Test.
        """

        result = wgh.AllOneWeigher().weigh(self.data)
        expected_result = np.ones(3)
        np.testing.assert_array_equal(
            result,
            expected_result,
            f"Expected {expected_result} but got {result}!",
        )

    def test_weigh_array(self):
        """
        Input/Output-Test.
        """

        result = wgh.AllOneWeigher().weigh(self.np_arr)
        expected_result = np.ones(3)
        np.testing.assert_array_equal(
            result,
            expected_result,
            f"Expected {expected_result} but got {result}!",
        )

    def test_weigh_edge_empty_array(self):
        """
        EdgeCase: Empty np.array.
        """
        result = wgh.AllOneWeigher().weigh(np.array([]))
        expected_result = []
        np.testing.assert_array_equal(
            result,
            expected_result,
            f"Expected {expected_result} but got {result}!",
        )
