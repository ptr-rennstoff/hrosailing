"""Contains various helper functions for the `plot_*`-methods()."""

# pylint: disable=missing-function-docstring


import itertools

import matplotlib.pyplot as plt
from matplotlib.projections.polar import PolarAxes
from matplotlib.axes import Axes
from matplotlib.projections import register_projection
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import (
    LinearSegmentedColormap,
    Normalize,
    is_color_like,
    to_rgb,
)
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull

from hrosailing.polardiagram import PolarDiagram


class HROPolar(PolarAxes):
    name = "hro polar"

    def plot(self,
             *args,
             ws=None,
             colors=("green", "red"),
             show_legend=False,
             legend_kw=None,
             **kwargs
             ):
        if not isinstance(args[0], PolarDiagram):
            super().plot(*args, **kwargs)
            return

        pd = args[0]
        self._plot_polar(pd, ws, colors, show_legend, legend_kw, **kwargs)

    def _plot_polar(
            self, pd, ws, colors, show_legend, legend_kw, **kwargs
    ):
        labels, slices, info = pd.get_slices(ws, full_info=True)
        _set_polar_axis(self)
        _configure_axes(self, labels, colors, show_legend, legend_kw, **kwargs)

        if info is None:
            self._plot_without_info(slices, **kwargs)
            return
        self._plot_with_info(slices, info, **kwargs)


    def _plot_without_info(self, slices, **kwargs):
        for slice in slices:
            ws, wa, bsp = slice
            wa = np.deg2rad(wa)
            super().plot(wa, bsp, **kwargs)


    def _plot_with_info(self, slices, info, **kwargs):
        intervals = self._get_info_intervals(info)
        for slice, info_ in zip(slices, info):
            ws, wa, bsp = slice
            wa = np.deg2rad(wa)
            for interval in intervals:
                super().plot(wa[interval], bsp[interval], **kwargs)

    def _get_info_intervals(self, info):
        intervals = {}
        for i, info_ in enumerate(info):
            for j, entry in enumerate(info_):
                if entry not in intervals:
                    intervals[entry] = [[] for _ in range(i)]
                intervals[entry][-1].append(j)
        return intervals

class HROFlat(Axes):
    name = "hro flat"

    def plot(self,
             *args,
             ws=None,
             colors=("green", "red"),
             show_legend=False,
             legend_kw=None,
             **kwargs
             ):
        if not isinstance(args[0], PolarDiagram):
            super().plot(*args, **kwargs)
            return

        pd = args[0]
        labels, slices = pd.get_slices(ws)
        self._plot_flat(labels, slices, colors, show_legend, legend_kw, **kwargs)

    def _plot_flat(
            self, labels, slices, colors, show_legend, legend_kw, **kwargs
    ):
        _configure_axes(self, labels, colors, show_legend, legend_kw, **kwargs)

        for slice in slices:
            ws, wa, bsp = slice
            super().plot(wa, bsp, **kwargs)


class HROColorGradient(Axes):
    name = "hro color gradient"

    def plot(self,
             *args,
             ws=None,
             colors=("green", "red"),
             show_legend=False,
             legend_kw=None,
             **kwargs
             ):
        if not isinstance(args[0], PolarDiagram):
            super().plot(*args, **kwargs)
            return

        pd = args[0]
        ws, wa, bsp = pd.get_slices(ws)
        wa = np.rad2deg(wa)

        self._plot_color_gradient(ws, wa, bsp, colors, show_legend, **kwargs)


    def _plot_color_gradient(
            self, ws, wa, bsp, colors, show_legend, **legend_kw
    ):
        if show_legend:
            _show_legend(self, bsp, colors, "Boat Speed", legend_kw)

        color_gradient = _determine_color_gradient(colors, bsp.ravel())

        if wa.ndim == 1:
            ws, wa = np.meshgrid(ws, wa)
            ws, wa = ws.T, wa.T

        if ws.ndim == 1:
            ws = np.array([[ws_ for _ in range(len(wa_))] for ws_, wa_ in zip(ws, wa)])

        self.scatter(ws.ravel(), wa.ravel(), c=color_gradient, **legend_kw)


class HRO3D(Axes3D):
    name = "hro 3d"

    def plot(self, *args, colors=("green", "red"), **kwargs):
        if not isinstance(args[0], PolarDiagram):
            super().plot(*args, **kwargs)
            return

        pd = args[0]
        ws, wa, bsp = pd.get_slices()
        lines_ = _check_for_lines(wa)
        if wa.ndim == 1 and ws.ndim == 1:
            wa, ws = np.meshgrid(np.flip(wa), ws)
            bsp = np.flip(bsp, axis=1)
        elif ws.ndim == 1:
            ws = np.array(
                [[ws_ for _ in range(len(wa_))] for ws_, wa_ in
                 zip(ws, wa)]
            )

        y, z = np.cos(np.pi/2-wa) * bsp, np.sin(np.pi/2-wa) * bsp

        if lines_:
            self._plot_surf(ws, y, z, colors, **kwargs)
            return

        self._plot3d(ws, y, z, colors, **kwargs)

    def _plot3d(self, ws, wa, bsp, colors, **plot_kw):
        _set_3d_axis_labels(self)
        _remove_3d_tick_labels_for_polar_coordinates(self)

        color_map = _create_color_map(colors)

        super().scatter(ws, wa, bsp, c=ws, cmap=color_map, **plot_kw)

    def _plot_surf(self, ws, wa, bsp, colors, **plot_kw):
        _set_3d_axis_labels(self)
        _remove_3d_tick_labels_for_polar_coordinates(self)

        color_map = _create_color_map(colors)
        face_colors = _determine_face_colors(color_map, ws)

        super().plot_surface(
            ws, wa, bsp, facecolors=face_colors
        )



register_projection(HROPolar)
register_projection(HROFlat)
register_projection(HROColorGradient)
register_projection(HRO3D)

def _check_for_lines(wa):
    return wa.ndim == 1

def _get_new_axis(kind):
    return plt.axes(projection=kind)

def _configure_axes(ax, labels, colors, show_legend, legend_kw, **kwargs):
    _configure_colors(ax, labels, colors)
    _check_plot_kw(kwargs, True)
    if show_legend:
        _show_legend(ax, labels, colors, "True Wind Speed", legend_kw)

def _set_polar_axis(ax):
    ax.set_theta_zero_location("N")
    ax.set_theta_direction("clockwise")


def _check_plot_kw(plot_kw, lines=True):
    ls = plot_kw.pop("linestyle", None) or plot_kw.pop("ls", None)
    if ls is None:
        plot_kw["ls"] = "-" if lines else ""
    else:
        plot_kw["ls"] = ls

    if plot_kw.get("marker", None) is None and not lines:
        plot_kw["marker"] = "o"


def _plot(ws, wa, bsp, ax, colors, show_legend, legend_kw, **plot_kw):
    _configure_colors(ax, ws, colors)

    if _only_one_color(colors):
        colors = [colors] * len(ws)

    if show_legend:
        _show_legend(ax, ws, colors, "True Wind Speed", legend_kw)

    # if wa.ndim == 1:
    #     for y in bsp:
    #         ax.plot(wa, y, **plot_kw)
    #     return

    for x, y in zip(wa, bsp):
        ax.plot(x, y, **plot_kw)


def _configure_colors(ax, ws, colors):
    if _only_one_color(colors):
        ax.set_prop_cycle("color", [colors])
        return

    if _more_colors_than_plots(ws, colors) or _no_color_gradient(colors):
        _set_color_cycle(ax, ws, colors)
        return

    _set_color_gradient(ax, ws, colors)


def _only_one_color(colors):
    return is_color_like(colors)


def _more_colors_than_plots(ws, colors):
    return len(ws) <= len(colors)


def _no_color_gradient(colors):
    all_color_format = all(_has_color_format(c) for c in colors)
    return len(colors) != 2 or not all_color_format


def _has_color_format(obj):
    if isinstance(obj, str):
        return True
    if len(obj) in [3, 4]:
        return True
    return False


def _set_color_cycle(ax, ws, colors):
    color_cycle = ["blue"] * len(ws)
    _configure_color_cycle(color_cycle, colors, ws)

    ax.set_prop_cycle("color", color_cycle)


def _configure_color_cycle(color_cycle, colors, ws):
    if isinstance(colors[0], tuple):
        for w, color in colors:
            i = list(ws).index(w)
            color_cycle[i] = color

        return

    colors = itertools.islice(colors, len(color_cycle))

    for i, color in enumerate(colors):
        color_cycle[i] = color


def _set_color_gradient(ax, ws, colors):
    color_gradient = _determine_color_gradient(colors, ws)
    ax.set_prop_cycle("color", color_gradient)


def _determine_color_gradient(colors, gradient):
    gradient_coeffs = _get_gradient_coefficients(gradient)
    color_gradient = _determine_colors_from_coefficients(
        gradient_coeffs, colors
    )
    return color_gradient


def _get_gradient_coefficients(gradient):
    min_gradient = gradient.min()
    max_gradient = gradient.max()

    return [
        (grad - min_gradient) / (max_gradient - min_gradient)
        for grad in gradient
    ]


def _determine_colors_from_coefficients(coefficients, colors):
    min_color = np.array(to_rgb(colors[0]))
    max_color = np.array(to_rgb(colors[1]))

    return [
        (1 - coeff) * min_color + coeff * max_color for coeff in coefficients
    ]


def _show_legend(ax, ws, colors, label, legend_kw):
    if legend_kw is None:
        legend_kw = {}

    _configure_legend(ax, ws, colors, label, **legend_kw)


def _configure_legend(ax, ws, colors, label, **legend_kw):
    if _plot_with_color_gradient(ws, colors):
        _set_colormap(ws, colors, ax, label, **legend_kw)
        return

    if isinstance(colors[0], tuple) and not is_color_like(colors[0]):
        _set_legend_without_wind_speeds(ax, colors, legend_kw)
        return

    _set_legend_with_wind_speeds(ax, colors, ws, legend_kw)


def _plot_with_color_gradient(ws, colors):
    return not _no_color_gradient(colors) and len(ws) > len(colors) == 2


def _set_colormap(ws, colors, ax, label, **legend_kw):
    color_map = _create_color_map(colors)

    label_kw, legend_kw = _extract_possible_text_kw(legend_kw)
    plt.colorbar(
        ScalarMappable(
            norm=Normalize(vmin=min(ws), vmax=max(ws)), cmap=color_map
        ),
        ax=ax,
        **legend_kw,
    ).set_label(label, **label_kw)


def _extract_possible_text_kw(legend_kw):
    return {}, legend_kw


def _set_legend_without_wind_speeds(ax, colors, legend_kw):
    ax.legend(
        handles=[
            Line2D([0], [0], color=color, lw=1, label=f"TWS {ws}")
            for (ws, color) in colors
        ],
        **legend_kw,
    )


def _set_legend_with_wind_speeds(ax, colors, ws, legend_kw):
    slices = zip(ws, colors)

    ax.legend(
        handles=[
            Line2D([0], [0], color=color, lw=1, label=f"TWS {ws}")
            for (ws, color) in slices
        ],
        **legend_kw,
    )


def _set_3d_axis_labels(ax):
    ax.set_xlabel("TWS")
    ax.set_ylabel("Polar plane: TWA / BSP ")


def _remove_3d_tick_labels_for_polar_coordinates(ax):
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])


def _create_color_map(colors):
    return LinearSegmentedColormap.from_list("cmap", list(colors))


def _determine_face_colors(color_map, ws):
    return color_map((ws - ws.min()) / float((ws - ws.min()).max()))


def plot_convex_hull(
        ws, wa, bsp, ax, colors, show_legend, legend_kw, _lines, **plot_kw
):
    if ax is None:
        ax = _get_new_axis("polar")
    _set_polar_axis(ax)

    _check_plot_kw(plot_kw, _lines)

    wa, bsp = _convex_hull(zip(wa, bsp))

    _plot(ws, wa, bsp, ax, colors, show_legend, legend_kw, **plot_kw)


def _convex_hull(slices):
    xs, ys = [], []
    for wa, bsp in slices:
        wa = np.asarray(wa)
        bsp = np.asarray(bsp)

        # convex hull is line between the two points
        # or is equal to one point
        if len(wa) < 3:
            xs.append(wa)
            ys.append(bsp)
            continue

        conv = _convex_hull_in_polar_coordinates(wa, bsp)
        vert = conv.vertices
        x, y = zip(
            *([(wa[i], bsp[i]) for i in vert] + [(wa[vert[0]], bsp[vert[0]])])
        )
        xs.append(list(x))
        ys.append(list(y))

    return xs, ys


def _convex_hull_in_polar_coordinates(wa, bsp):
    polar_points = np.column_stack((bsp * np.cos(wa), bsp * np.sin(wa)))
    return ConvexHull(polar_points)


def plot_convex_hull_multisails(
        ws, wa, bsp, members, ax, colors, show_legend, legend_kw, **plot_kw
):
    if ax is None:
        ax = _get_new_axis("polar")

    _set_polar_axis(ax)

    _check_plot_kw(plot_kw)

    xs, ys, members = _get_convex_hull_multisails(ws, wa, bsp, members)

    if colors is None:
        colors = plot_kw.pop("color", None) or plot_kw.pop("c", None) or []
    colors = dict(colors)
    _set_colors_multisails(ax, members, colors)

    if legend_kw is None:
        legend_kw = {}
    if show_legend:
        _set_legend_multisails(ax, colors, **legend_kw)

    for x, y in zip(list(xs), list(ys)):
        ax.plot(x, y, **plot_kw)


def _get_convex_hull_multisails(ws, wa, bsp, members):
    xs = []
    ys = []
    membs = []
    for s, w, b in zip(ws, wa, bsp):
        w = np.asarray(w)
        b = np.asarray(b)
        conv = _convex_hull_in_polar_coordinates(w, b)
        vert = sorted(conv.vertices)

        x, y, memb = zip(
            *(
                    [(w[i], b[i], members[i]) for i in vert]
                    + [(w[vert[0]], b[vert[0]], members[vert[0]])]
            )
        )
        x = list(x)
        y = list(y)
        memb = list(memb)

        for i in range(len(vert)):
            xs.append(x[i: i + 2])
            ys.append(y[i: i + 2])
            membs.append(memb[i: i + 2] + [s])

    return xs, ys, membs


def _set_colors_multisails(ax, members, colors):
    colorlist = []

    for member in members:
        # check if edge belongs to one or two sails
        # If it belongs to one sail, color it in that sails color
        # else color it in neutral color
        if len(set(member[:2])) == 1:
            color = colors.get(member[0], "blue")
        else:
            color = colors.get("neutral", "gray")

        if is_color_like(color):
            colorlist.append(color)
            continue

        color = dict(color)
        colorlist.append(color.get(member[2], "blue"))

    ax.set_prop_cycle("color", colorlist)


def _set_legend_multisails(ax, colors, **legend_kw):
    handles = []
    for key in colors:
        color = colors.get(key, "blue")

        if is_color_like(color):
            legend = Line2D([0], [0], color=color, lw=1, label=key)
            handles.append(legend)
            continue

        color = dict(color)
        legends = [
            Line2D(
                [0],
                [0],
                color=color.get(ws, "blue"),
                lw=1,
                label=f"{key} at TWS {ws}",
            )
            for ws in color
        ]
        handles.extend(legends)

    if "neutral" not in colors:
        legend = Line2D([0], [0], color="gray", lw=1, label="neutral")
        handles.append(legend)

    ax.legend(handles=handles, **legend_kw)
