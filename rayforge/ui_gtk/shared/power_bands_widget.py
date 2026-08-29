from gettext import gettext as _

import cairo
import numpy as np
from blinker import Signal
from gi.repository import Gdk, Gtk

#: Minimum gap enforced between adjacent boundaries (and against the
#: 0/255 edges), in raw 0..255 units -- mirrors CurveWidget's
#: _MIN_X_GAP, so a drag can't collapse a band to zero width or push
#: boundaries past each other.
_MIN_GAP = 4

#: Pixel radius within which a click/drag targets a boundary handle.
_HANDLE_RADIUS_PX = 10


class PowerBandsWidget(Gtk.DrawingArea):
    """A step-histogram editor for non-uniform power-level bands.

    Shows a histogram of the final (post-levels, post-curve) grayscale
    values that feed the raster assembler's power quantization, with
    draggable vertical dividers partitioning the 0..255 axis into
    ``num_power_levels`` bands. Each band is shaded and labeled with
    the power percentage it maps to -- evenly spaced between the
    step's min/max power, only the *brightness* boundaries between
    bands move.

    Unlike :class:`HistogramPreview` (two fixed "black"/"white"
    handles, click-empty-space semantics don't apply), the number of
    handles here is fixed by ``num_power_levels`` and is set
    externally via :meth:`set_num_bands` -- dragging only ever moves
    an existing boundary.
    """

    WIDTH = 200
    HEIGHT = 100
    MARGIN = 5

    def __init__(self):
        super().__init__()
        self.histogram: np.ndarray | None = None
        self._boundaries: list[float] = [255.0 / 2]
        self._min_power: float = 0.0
        self._max_power: float = 1.0
        self._dragging: int | None = None
        self._hovering: int | None = None
        self._drag_changed: bool = False

        self.boundaries_changed = Signal()

        self.set_content_width(self.WIDTH)
        self.set_content_height(self.HEIGHT)
        self.set_draw_func(self._draw_func)

        click = Gtk.GestureClick.new()
        click.connect("pressed", self._on_pressed)
        click.connect("released", self._on_released)
        self.add_controller(click)

        motion = Gtk.EventControllerMotion.new()
        motion.connect("motion", self._on_motion)
        motion.connect("leave", self._on_leave)
        self.add_controller(motion)

    def update_histogram(self, histogram: np.ndarray | None):
        self.histogram = histogram
        self.queue_draw()

    def set_power_range(self, min_power: float, max_power: float):
        if (min_power, max_power) != (self._min_power, self._max_power):
            self._min_power, self._max_power = min_power, max_power
            self.queue_draw()

    def get_boundaries(self) -> list[float]:
        return list(self._boundaries)

    def set_boundaries(self, boundaries: list[float]):
        self._boundaries = self._clamp_all(sorted(boundaries))
        self.queue_draw()

    def set_num_bands(self, num_bands: int):
        """Resize to ``num_bands`` bands (``num_bands - 1`` handles).

        Only rebuilds evenly-spaced boundaries when the count actually
        changes -- called on every settings sync, so a no-op change
        must not clobber a boundary the user just dragged.
        """
        num_bands = max(1, num_bands)
        n_handles = num_bands - 1
        if n_handles == len(self._boundaries):
            return
        self._boundaries = [
            255.0 * (i + 1) / num_bands for i in range(n_handles)
        ]
        self.queue_draw()

    def _clamp_all(self, boundaries: list[float]) -> list[float]:
        n = len(boundaries)
        result = list(boundaries)
        for i in range(n):
            lo = result[i - 1] + _MIN_GAP if i > 0 else _MIN_GAP
            hi = (
                result[i + 1] - _MIN_GAP
                if i + 1 < n
                else 255.0 - _MIN_GAP
            )
            result[i] = max(lo, min(hi, result[i]))
        return result

    def _value_to_x(self, value: float, width: int) -> float:
        draw_width = width - 2 * self.MARGIN
        return self.MARGIN + (value / 255.0) * draw_width

    def _x_to_value(self, x: float, width: int) -> float:
        draw_width = width - 2 * self.MARGIN
        ratio = (x - self.MARGIN) / draw_width if draw_width else 0.0
        return max(0.0, min(255.0, ratio * 255.0))

    def _handle_at(self, x: float, width: int) -> int | None:
        for i, value in enumerate(self._boundaries):
            hx = self._value_to_x(value, width)
            if abs(x - hx) <= _HANDLE_RADIUS_PX:
                return i
        return None

    def _on_pressed(self, gesture, n_press, x, y):
        width = self.get_width()
        handle = self._handle_at(x, width)
        if handle is None:
            return
        self._dragging = handle
        self._drag_changed = False
        gesture.set_state(Gtk.EventSequenceState.CLAIMED)

    def _on_released(self, gesture, n_press, x, y):
        if self._dragging is not None and self._drag_changed:
            self.boundaries_changed.send(
                self, boundaries=self.get_boundaries()
            )
        self._dragging = None
        self._drag_changed = False
        self.queue_draw()

    def _on_motion(self, controller, x, y):
        width = self.get_width()
        if self._dragging is not None:
            i = self._dragging
            value = self._x_to_value(x, width)
            lo = (
                self._boundaries[i - 1] + _MIN_GAP
                if i > 0
                else _MIN_GAP
            )
            hi = (
                self._boundaries[i + 1] - _MIN_GAP
                if i + 1 < len(self._boundaries)
                else 255.0 - _MIN_GAP
            )
            value = max(lo, min(hi, value))
            if value != self._boundaries[i]:
                self._boundaries[i] = value
                self._drag_changed = True
                self.queue_draw()
        else:
            handle = self._handle_at(x, width)
            if handle != self._hovering:
                self._hovering = handle
                self.set_cursor(
                    Gdk.Cursor.new_from_name("ew-resize")
                    if handle is not None
                    else None
                )
                self.queue_draw()

    def _on_leave(self, controller):
        if self._hovering is not None:
            self._hovering = None
            self.set_cursor(None)
            self.queue_draw()

    def _band_power(self, band_index: int, num_bands: int) -> float:
        """Power for the band at ``band_index`` (0 = leftmost/darkest
        original brightness). Darker input gets *more* power (see
        scan.rs's ``compute_power_values``: ``fraction = min_power +
        (1 - gray/255) * power_range``), so band 0 is max_power and
        the last band is min_power -- the reverse of brightness order.
        """
        if num_bands <= 1:
            return self._max_power
        frac = band_index / (num_bands - 1)
        return self._max_power - frac * (self._max_power - self._min_power)

    def _draw_func(self, area, ctx: cairo.Context, width: int, height: int):
        ctx.set_source_rgba(0, 0, 0, 0)
        ctx.set_operator(cairo.OPERATOR_SOURCE)
        ctx.paint()
        ctx.set_operator(cairo.OPERATOR_OVER)

        draw_width = width - 2 * self.MARGIN
        draw_height = height - 2 * self.MARGIN
        num_bands = len(self._boundaries) + 1
        edges = [0.0, *self._boundaries, 255.0]

        # Band backgrounds shaded to match actual output darkness --
        # band 0 (darkest original brightness) gets max_power (see
        # _band_power's docstring), so it's shaded darkest; the last
        # band (lightest original brightness) gets min_power and is
        # shaded lightest.
        for band in range(num_bands):
            x0 = self._value_to_x(edges[band], width)
            x1 = self._value_to_x(edges[band + 1], width)
            shade = 0.35 + 0.5 * (band / max(1, num_bands - 1))
            ctx.set_source_rgba(shade, shade, shade, 0.35)
            ctx.rectangle(x0, self.MARGIN, x1 - x0, draw_height)
            ctx.fill()

        if self.histogram is not None:
            log_counts = np.log1p(self.histogram)
            log_max = np.max(log_counts) if np.max(log_counts) > 0 else 1
            bar_width = draw_width / len(self.histogram)
            color = self.get_color()
            ctx.set_source_rgba(
                color.red, color.green, color.blue, color.alpha
            )
            for i, count in enumerate(log_counts):
                x = self.MARGIN + i * bar_width
                bar_height = (count / log_max) * draw_height
                ctx.rectangle(
                    x,
                    height - self.MARGIN - bar_height,
                    bar_width,
                    bar_height,
                )
            ctx.fill()
        else:
            ctx.set_source_rgba(0.5, 0.5, 0.5, 1.0)
            ctx.set_font_size(12)
            ctx.move_to(width // 2 - 40, height // 2)
            ctx.show_text(_("No image"))

        # Power-percentage label per band.
        ctx.set_font_size(10)
        for band in range(num_bands):
            x0 = self._value_to_x(edges[band], width)
            x1 = self._value_to_x(edges[band + 1], width)
            power = self._band_power(band, num_bands)
            label = f"{round(power * 100)}%"
            extents = ctx.text_extents(label)
            if extents.width + 4 > x1 - x0:
                continue
            color = self.get_color()
            ctx.set_source_rgba(
                color.red, color.green, color.blue, 0.8
            )
            ctx.move_to(
                (x0 + x1) / 2 - extents.width / 2,
                self.MARGIN + extents.height + 2,
            )
            ctx.show_text(label)

        # Boundary handles.
        for i, value in enumerate(self._boundaries):
            hx = self._value_to_x(value, width)
            ctx.set_source_rgba(0.2, 0.8, 0.3, 0.9)
            ctx.set_line_width(
                3 if i in (self._hovering, self._dragging) else 2
            )
            ctx.move_to(hx, self.MARGIN)
            ctx.line_to(hx, height - self.MARGIN)
            ctx.stroke()
