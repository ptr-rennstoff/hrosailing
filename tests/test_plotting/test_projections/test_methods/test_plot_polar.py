import matplotlib.pyplot as plt
import numpy as np

from tests.test_plotting.image_testcase import ImageTestcase

from hrosailing.plotting.projections import plot_polar
from hrosailing.polardiagram import PolarDiagramTable

class TestPlotPolar(ImageTestcase):
    def test_regular_plot(self):
        # Input/Output
        pd = PolarDiagramTable(
            [1, 2, 3],
            [0, 90, 180],
            [[0, 1, 2], [1, 2, 3], [2, 3, 4]]
        )
        plot_polar(pd)
        self.set_result_plot()

        ax = plt.subplot(projection="polar")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction("clockwise")
        ax.plot([0, np.pi/2, np.pi], [0, 1, 2], color=(0, 1, 0))
        ax.plot([0, np.pi/2, np.pi], [1, 2, 3], color=(0.5, 0.5, 0))
        ax.plot([0, np.pi/2, np.pi], [2, 3, 4], color=(1, 0, 0))
        self.set_expected_plot()

        self.assertPlotsEqual()

    def test_with_keywords(self):
        # Input/Output with keywords

        keywords = {
            "marker": "H",
            "linestyle": "--"
        }
        pd = PolarDiagramTable(
            [1, 2, 3],
            [0, 90, 180],
            [[0, 1, 2], [1, 2, 3], [2, 3, 4]]
        )
        plot_polar(pd, **keywords)
        self.set_result_plot()

        ax = plt.subplot(projection="polar")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction("clockwise")
        ax.plot([0, np.pi / 2, np.pi], [0, 1, 2], color=(0, 1, 0), **keywords)
        ax.plot([0, np.pi / 2, np.pi], [1, 2, 3], color=(0.5, 0.5, 0), **keywords)
        ax.plot([0, np.pi / 2, np.pi], [2, 3, 4], color=(1, 0, 0), **keywords)
        self.set_expected_plot()

        self.assertPlotsEqual()