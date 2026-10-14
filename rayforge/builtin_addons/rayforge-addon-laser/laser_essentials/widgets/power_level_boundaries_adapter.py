"""Adapter for the non-uniform power-level boundaries row."""

from gettext import gettext as _
from typing import Any

import numpy as np
from gi.repository import Adw, Gtk
from raygeo.image.grayscale import compute_auto_levels, normalize_grayscale

from rayforge.core.varset import Var
from rayforge.image.util import get_visible_grayscale_values
from rayforge.ui_gtk.shared.power_bands_widget import PowerBandsWidget
from rayforge.ui_gtk.varset.adapter import (
    RowAdapter,
    escape_title,
    register_adapter,
)

from ..power_level_boundaries_var import PowerLevelBoundariesVar

_MAX_PREVIEW_PX = 256


def _curve_lut(points: list[tuple[float, float]]) -> np.ndarray:
    """Mirrors CurveWidget.to_lut()/assembler_helpers._curve_to_lut --
    duplicated rather than imported since those live behind a GTK
    widget instance and a pipeline-internal name respectively, for
    what's only ever an approximate preview histogram here."""
    xs = np.array([p[0] for p in points]) * 255.0
    ys = np.array([p[1] for p in points]) * 255.0
    lut = np.interp(np.arange(256), xs, ys)
    return np.clip(np.round(lut), 0, 255).astype(np.uint8)


@register_adapter(PowerLevelBoundariesVar)
class PowerLevelBoundariesAdapter(RowAdapter):
    """A row for the non-uniform power-level brightness boundaries.

    Renders an :class:`Adw.ActionRow` whose suffix carries a
    :class:`PowerBandsWidget`. Like
    :class:`~.levels_adapter.LevelsAdapter`, the host page pulls the
    widget out into the side panel via :meth:`detach_widget`.

    The histogram it shows is the *final* brightness distribution --
    after image edits, auto/manual levels, and the power curve -- since
    that's the actual input the boundaries partition (matching
    ``_boundaries_to_lut``'s place in the pipeline, right before the
    raster assembler's own quantization). It therefore needs the
    sibling auto_levels/black_point/white_point/curve_points/
    num_power_levels/min_power_level/max_power_level values, supplied
    via :meth:`update_from_values`, to recompute both the band count
    and the histogram whenever any of those change -- not just its own
    value.
    """

    def __init__(
        self,
        row: Adw.ActionRow,
        widget: PowerBandsWidget,
    ) -> None:
        super().__init__()
        self._row = row
        self.widget = widget
        self._step: Any = None
        self._raw_gray: np.ndarray | None = None
        self._last_context: tuple | None = None
        self._subtitle_label: Gtk.Label | None = None
        widget.boundaries_changed.connect(
            lambda s, **kw: self.changed.send(self),
            weak=False,
        )

    @classmethod
    def create(
        cls, var: Var, target_property: str
    ) -> tuple[Adw.PreferencesRow, "PowerLevelBoundariesAdapter"]:
        assert isinstance(var, PowerLevelBoundariesVar)
        row = Adw.ActionRow(title=escape_title(var.label))
        if var.description:
            row.set_subtitle(escape_title(var.description))

        widget = PowerBandsWidget()
        if var.value:
            widget.set_boundaries(var.value)
        row.add_suffix(widget)

        return row, cls(row, widget)

    def detach_widget(self) -> PowerBandsWidget:
        """Remove the bands widget from its row's suffix and return
        it, so a host page can mount it elsewhere (e.g. a wide side
        panel), mirroring detach_preview()/detach_curve()."""
        self._row.remove(self.widget)
        return self.widget

    def set_subtitle_label(self, label: Gtk.Label | None) -> None:
        self._subtitle_label = label
        if label is not None:
            label.set_label(
                _("Drag to move where each power level starts/ends")
            )

    def set_histogram_source(self, step: Any) -> None:
        """Point the adapter at the step whose layer supplies the raw
        grayscale samples. The dialog calls this once at build time,
        mirroring LevelsAdapter.set_histogram_source."""
        self._step = step

    def compute_raw_samples(self) -> None:
        """(Re)sample the step's layer into raw grayscale values.
        No-op without a histogram source. Separate from the
        levels/curve-processed histogram pushed to the widget, since
        those two only need to be redone at different times: raw
        samples only when the image actually changes, the
        processed histogram whenever any tone setting changes too.
        """
        step = self._step
        if step is None:
            return
        layer = step.layer
        if not layer or not layer.all_workpieces:
            self._raw_gray = None
            self._push_histogram()
            return

        pixels_per_mm = step.pixels_per_mm
        all_gray_values = []
        for workpiece in layer.all_workpieces:
            size = workpiece.size
            if not size or size[0] <= 0 or size[1] <= 0:
                continue
            width_px = int(size[0] * pixels_per_mm[0])
            height_px = int(size[1] * pixels_per_mm[1])
            if width_px <= 0 or height_px <= 0:
                continue
            if width_px > _MAX_PREVIEW_PX or height_px > _MAX_PREVIEW_PX:
                scale = min(
                    _MAX_PREVIEW_PX / width_px, _MAX_PREVIEW_PX / height_px
                )
                width_px = max(int(width_px * scale), 1)
                height_px = max(int(height_px * scale), 1)
            surface = workpiece.render_to_pixels(width_px, height_px)
            if not surface:
                continue
            gray_values = get_visible_grayscale_values(surface)
            if gray_values.size > 0:
                all_gray_values.append(gray_values)

        self._raw_gray = (
            np.concatenate(all_gray_values) if all_gray_values else None
        )
        self._push_histogram()

    def _push_histogram(self) -> None:
        if self._raw_gray is None or self._raw_gray.size == 0:
            self.widget.update_histogram(None)
            return
        ctx = self._last_context or {}
        gray = self._raw_gray

        auto_levels = bool(ctx.get("auto_levels", True))
        if auto_levels:
            bp, wp = compute_auto_levels(gray)
        else:
            bp = int(ctx.get("black_point", 0))
            wp = int(ctx.get("white_point", 255))
        if bp > 0 or wp < 255:
            gray = normalize_grayscale(gray, bp, wp)

        curve_points = ctx.get("curve_points")
        if curve_points and curve_points != [(0.0, 0.0), (1.0, 1.0)]:
            gray = _curve_lut(curve_points)[gray]

        histogram, _edges = np.histogram(gray, bins=64, range=(0, 255))
        self.widget.update_histogram(histogram)

    def get_value(self) -> Any | None:
        return self.widget.get_boundaries()

    def set_value(self, value: Any) -> None:
        if value:
            self.widget.set_boundaries(value)

    def update_from_var(self, var: Var):
        assert isinstance(var, PowerLevelBoundariesVar)
        if var.label:
            self._row.set_title(escape_title(var.label))
        if var.description:
            self._row.set_subtitle(escape_title(var.description))

    def update_from_values(self, values: dict[str, Any]) -> None:
        num_levels = values.get("num_power_levels")
        if num_levels:
            self.widget.set_num_bands(int(num_levels))

        min_p = values.get("min_power_level")
        max_p = values.get("max_power_level")
        if min_p is not None and max_p is not None:
            self.widget.set_power_range(float(min_p), float(max_p))

        context = (
            bool(values.get("auto_levels", True)),
            values.get("black_point"),
            values.get("white_point"),
            tuple(values.get("curve_points") or ()),
        )
        if context != self._last_context_key():
            self._last_context = {
                "auto_levels": context[0],
                "black_point": context[1],
                "white_point": context[2],
                "curve_points": values.get("curve_points"),
            }
            self._push_histogram()

    def _last_context_key(self):
        if self._last_context is None:
            return None
        c = self._last_context
        return (
            c["auto_levels"],
            c["black_point"],
            c["white_point"],
            tuple(c["curve_points"] or ()),
        )
