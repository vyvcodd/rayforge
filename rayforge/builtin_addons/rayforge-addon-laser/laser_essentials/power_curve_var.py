"""The raster power-curve (tone-mapping) VarSet variable."""

from collections.abc import Callable
from gettext import gettext as _
from typing import Any

from rayforge.core.varset import Var


class PowerCurveVar(Var):
    """A Var for a piecewise-linear brightness -> power curve.

    Hints the UI to render a :class:`CurveWidget`: draggable control
    points mapping (already levels-stretched) brightness to power,
    applied after the black/white points. The default two-point
    ``[(0, 0), (1, 1)]`` curve is the identity -- a straight line, no
    correction -- with as many additional points as the user adds.
    """

    display_name = None  # not offered in the generic var-editor UI

    def __init__(
        self,
        key: str = "curve_points",
        label: str = _("Power Curve"),
        description: str | None = None,
        default: list[tuple[float, float]] | None = None,
        value: list[tuple[float, float]] | None = None,
        *,
        visible_when: "Callable[[dict[str, Any]], bool] | None" = None,
    ):
        super().__init__(
            key=key,
            label=label,
            var_type=list,
            description=description,
            default=default if default is not None else [(0.0, 0.0), (1.0, 1.0)],
            value=value,
            visible_when=visible_when,
        )
