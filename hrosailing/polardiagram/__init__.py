"""
`PolarDiagram` classes to work with and represent polar diagrams in various
forms.
"""

from ._reading import from_csv
from ._basepolardiagram import PolarDiagram
from ._polardiagramcurve import PolarDiagramCurve
from ._polardiagrammultisails import PolarDiagramMultiSails
from ._polardiagrampointcloud import PolarDiagramPointcloud
from ._polardiagramtable import PolarDiagramTable

__all__ = [
    "PolarDiagram",
    "PolarDiagramCurve",
    "PolarDiagramMultiSails",
    "PolarDiagramPointcloud",
    "PolarDiagramTable",
    "from_csv"
]


