"""Adapter for the raster power-curve (tone-mapping) row."""

from gettext import gettext as _
from typing import Any

from gi.repository import Adw, Gtk

from rayforge.core.varset import Var
from rayforge.ui_gtk.shared.curve_widget import CurveWidget
from rayforge.ui_gtk.varset.adapter import (
    RowAdapter,
    escape_title,
    register_adapter,
)

from ..power_curve_var import PowerCurveVar


@register_adapter(PowerCurveVar)
class PowerCurveAdapter(RowAdapter):
    """A row for the brightness -> power curve.

    Renders an :class:`Adw.ActionRow` whose suffix carries a
    :class:`CurveWidget` for dragging the curve's control points.
    Like :class:`~.levels_adapter.LevelsAdapter`, the host page can
    pull the curve widget out into a wider side panel via
    :meth:`detach_curve` -- the row stays registered with the varset
    widget purely to keep ``curve_points`` flowing through the normal
    adapter machinery, and is expected to be hidden once detached.
    """

    def __init__(self, row: Adw.ActionRow, curve: CurveWidget) -> None:
        super().__init__()
        self._row = row
        self.curve = curve
        self._subtitle_label: Gtk.Label | None = None
        curve.points_changed.connect(
            lambda s, **kw: self.changed.send(self),
            weak=False,
        )

    @classmethod
    def create(
        cls, var: Var, target_property: str
    ) -> tuple[Adw.PreferencesRow, "PowerCurveAdapter"]:
        assert isinstance(var, PowerCurveVar)
        row = Adw.ActionRow(title=escape_title(var.label))
        if var.description:
            row.set_subtitle(escape_title(var.description))

        curve = CurveWidget()
        if var.value:
            curve.set_points(var.value)
        row.add_suffix(curve)

        return row, cls(row, curve)

    def detach_curve(self) -> CurveWidget:
        """Remove the curve widget from its row's suffix and return
        it, so a host page can mount it elsewhere (e.g. a wide side
        panel) instead of inline in the settings list."""
        self._row.remove(self.curve)
        return self.curve

    def set_subtitle_label(self, label: Gtk.Label | None) -> None:
        """Mirror a hint onto an external label, for the same reason
        :meth:`~.levels_adapter.LevelsAdapter.set_subtitle_label`
        does: once the curve moves out of the row, its subtitle has
        nowhere left to show."""
        self._subtitle_label = label
        if label is not None:
            label.set_label(
                _(
                    "Drag to add/move points, right-click a point to "
                    "remove it"
                )
            )

    def get_value(self) -> Any | None:
        return self.curve.get_points()

    def set_value(self, value: Any) -> None:
        if value:
            self.curve.set_points(value)

    def update_from_var(self, var: Var):
        assert isinstance(var, PowerCurveVar)
        if var.label:
            self._row.set_title(escape_title(var.label))
        if var.description:
            self._row.set_subtitle(escape_title(var.description))
