import cairo
import numpy as np
from blinker import Signal
from gi.repository import Gdk, Gtk

#: Minimum x-gap enforced between adjacent points (normalized 0..1),
#: so a click can't create a degenerate near-vertical LUT segment.
_MIN_X_GAP = 0.02

#: Pixel radius within which a click/drag targets an existing point
#: rather than inserting a new one.
_HANDLE_RADIUS_PX = 10


class CurveWidget(Gtk.DrawingArea):
    """An interactive piecewise-linear tone curve editor.

    Draws an input (x, 0..1) -> output (y, 0..1) curve through a list
    of draggable control points, with a faint diagonal identity line
    for reference. Click empty space to add a point (and start
    dragging it immediately); drag an existing point to move it;
    right-click a non-endpoint point to remove it. The two endpoints
    (x=0 and x=1) can't be removed and can only move vertically -- the
    curve always covers the full input range.

    Mirrors the interaction pattern of :class:`HistogramPreview`
    (press/motion/release, commit-on-release rather than on every
    motion event) so dragging a point doesn't flood the pipeline with
    recompute requests until the drag settles.
    """

    WIDTH = 200
    HEIGHT = 140
    MARGIN = 8

    def __init__(self):
        super().__init__()
        self._points: list[tuple[float, float]] = [(0.0, 0.0), (1.0, 1.0)]
        self._dragging: int | None = None
        self._hovering: int | None = None
        self._drag_changed: bool = False

        self.points_changed = Signal()

        self.set_content_width(self.WIDTH)
        self.set_content_height(self.HEIGHT)
        self.set_draw_func(self._draw_func)

        click = Gtk.GestureClick.new()
        click.connect("pressed", self._on_pressed)
        click.connect("released", self._on_released)
        self.add_controller(click)

        remove_click = Gtk.GestureClick.new()
        remove_click.set_button(Gdk.BUTTON_SECONDARY)
        remove_click.connect("pressed", self._on_secondary_pressed)
        self.add_controller(remove_click)

        motion = Gtk.EventControllerMotion.new()
        motion.connect("motion", self._on_motion)
        motion.connect("leave", self._on_leave)
        self.add_controller(motion)

    def get_points(self) -> list[tuple[float, float]]:
        return list(self._points)

    def set_points(self, points: list[tuple[float, float]]):
        if not points:
            points = [(0.0, 0.0), (1.0, 1.0)]
        self._points = sorted(
            (
                (max(0.0, min(1.0, x)), max(0.0, min(1.0, y)))
                for x, y in points
            ),
            key=lambda p: p[0],
        )
        self.queue_draw()

    def to_lut(self) -> np.ndarray:
        """The curve as a 256-entry uint8 lookup table, piecewise-
        linearly interpolated between control points -- matching
        exactly what's drawn, so there's no surprise between the
        curve shown and the mapping applied."""
        xs = np.array([p[0] for p in self._points]) * 255.0
        ys = np.array([p[1] for p in self._points]) * 255.0
        lut = np.interp(np.arange(256), xs, ys)
        return np.clip(np.round(lut), 0, 255).astype(np.uint8)

    def _to_px(self, x: float, y: float, width: int, height: int):
        draw_w = width - 2 * self.MARGIN
        draw_h = height - 2 * self.MARGIN
        return (
            self.MARGIN + x * draw_w,
            height - self.MARGIN - y * draw_h,
        )

    def _to_point(self, px: float, py: float, width: int, height: int):
        draw_w = width - 2 * self.MARGIN
        draw_h = height - 2 * self.MARGIN
        x = (px - self.MARGIN) / draw_w if draw_w else 0.0
        y = 1.0 - (py - self.MARGIN) / draw_h if draw_h else 0.0
        return max(0.0, min(1.0, x)), max(0.0, min(1.0, y))

    def _handle_at(self, px: float, py: float, width: int, height: int):
        for i, (x, y) in enumerate(self._points):
            hx, hy = self._to_px(x, y, width, height)
            if (hx - px) ** 2 + (hy - py) ** 2 <= _HANDLE_RADIUS_PX**2:
                return i
        return None

    def _clamp_x_for_index(self, index: int, x: float) -> float:
        if index == 0:
            return 0.0
        if index == len(self._points) - 1:
            return 1.0
        lo = self._points[index - 1][0] + _MIN_X_GAP
        hi = self._points[index + 1][0] - _MIN_X_GAP
        if lo > hi:
            return self._points[index][0]
        return max(lo, min(hi, x))

    def _on_pressed(self, gesture, n_press, x, y):
        width, height = self.get_width(), self.get_height()
        handle = self._handle_at(x, y, width, height)
        if handle is None:
            px, py = self._to_point(x, y, width, height)
            px = max(_MIN_X_GAP, min(1.0 - _MIN_X_GAP, px))
            insert_at = len(self._points)
            for i, (ex, _ey) in enumerate(self._points):
                if px < ex:
                    insert_at = i
                    break
            self._points.insert(insert_at, (px, py))
            handle = insert_at
        self._dragging = handle
        self._drag_changed = False
        gesture.set_state(Gtk.EventSequenceState.CLAIMED)
        self.queue_draw()

    def _on_released(self, gesture, n_press, x, y):
        if self._dragging is not None and self._drag_changed:
            self.points_changed.send(self, points=self.get_points())
        self._dragging = None
        self._drag_changed = False
        self.queue_draw()

    def _on_secondary_pressed(self, gesture, n_press, x, y):
        width, height = self.get_width(), self.get_height()
        handle = self._handle_at(x, y, width, height)
        if handle is None or handle in (0, len(self._points) - 1):
            return
        del self._points[handle]
        self._dragging = None
        self._hovering = None
        self.points_changed.send(self, points=self.get_points())
        self.queue_draw()

    def _on_motion(self, controller, x, y):
        width, height = self.get_width(), self.get_height()
        if self._dragging is not None:
            px, py = self._to_point(x, y, width, height)
            px = self._clamp_x_for_index(self._dragging, px)
            new_point = (px, py)
            if new_point != self._points[self._dragging]:
                self._points[self._dragging] = new_point
                self._drag_changed = True
                self.queue_draw()
        else:
            handle = self._handle_at(x, y, width, height)
            if handle != self._hovering:
                self._hovering = handle
                self.set_cursor(
                    Gdk.Cursor.new_from_name("pointer")
                    if handle is not None
                    else None
                )
                self.queue_draw()

    def _on_leave(self, controller):
        if self._hovering is not None:
            self._hovering = None
            self.set_cursor(None)
            self.queue_draw()

    def _draw_func(self, area, ctx: cairo.Context, width: int, height: int):
        ctx.set_source_rgba(0, 0, 0, 0)
        ctx.set_operator(cairo.OPERATOR_SOURCE)
        ctx.paint()
        ctx.set_operator(cairo.OPERATOR_OVER)

        color = self.get_color()

        # Faint identity reference line.
        ctx.set_source_rgba(color.red, color.green, color.blue, 0.25)
        ctx.set_dash([3, 3])
        ctx.set_line_width(1)
        x0, y0 = self._to_px(0.0, 0.0, width, height)
        x1, y1 = self._to_px(1.0, 1.0, width, height)
        ctx.move_to(x0, y0)
        ctx.line_to(x1, y1)
        ctx.stroke()
        ctx.set_dash([])

        # The curve itself.
        ctx.set_source_rgba(0.2, 0.6, 1.0, 0.9)
        ctx.set_line_width(2)
        for i, (x, y) in enumerate(self._points):
            px, py = self._to_px(x, y, width, height)
            if i == 0:
                ctx.move_to(px, py)
            else:
                ctx.line_to(px, py)
        ctx.stroke()

        # Control points.
        for i, (x, y) in enumerate(self._points):
            px, py = self._to_px(x, y, width, height)
            radius = 5 if i in (self._hovering, self._dragging) else 3.5
            ctx.set_source_rgba(0.2, 0.6, 1.0, 1.0)
            ctx.arc(px, py, radius, 0, 2 * 3.14159265)
            ctx.fill()
