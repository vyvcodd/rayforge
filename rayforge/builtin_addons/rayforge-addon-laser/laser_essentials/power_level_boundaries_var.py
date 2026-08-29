"""The raster power-level brightness-boundaries VarSet variable."""

from collections.abc import Callable
from gettext import gettext as _
from typing import Any

from rayforge.core.varset import Var


class PowerLevelBoundariesVar(Var):
    """A Var for the brightness boundaries between discrete power bands.

    Hints the UI to render a :class:`PowerBandsWidget`: draggable
    dividers partitioning the 0..255 brightness axis into
    ``num_power_levels`` bands, each mapped to an evenly-spaced power
    value between the step's min/max power -- only where each band
    *starts and ends* is adjustable, not the power values themselves.

    ``None`` (the default) means "evenly spaced", matching the
    quantizer's prior unconditional behavior exactly -- this is purely
    additive, existing recipes see no change until a boundary is
    dragged. A non-``None`` value is a list of ``num_power_levels - 1``
    floats in 0..255, strictly increasing.
    """

    display_name = None  # not offered in the generic var-editor UI

    def __init__(
        self,
        key: str = "power_level_boundaries",
        label: str = _("Power Level Boundaries"),
        description: str | None = None,
        default: list[float] | None = None,
        value: list[float] | None = None,
        *,
        visible_when: "Callable[[dict[str, Any]], bool] | None" = None,
    ):
        super().__init__(
            key=key,
            label=label,
            var_type=list,
            description=description,
            default=default,
            value=value,
            visible_when=visible_when,
        )
