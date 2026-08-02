from gettext import gettext as _
from typing import Optional

import cairo
from gi.repository import Gtk

# Generous, mostly-arbitrary bounds -- past _MAX_ZOOM you're just
# magnifying/interpolating already-rendered pixels rather than seeing
# new detail (the surface's actual resolution is fixed once rendered;
# see ImageEngraveWidget._recompute_burn_preview), but that's normal,
# expected behavior for a zoomable preview, not something to prevent.
_MAX_ZOOM = 50.0
_MIN_ZOOM = 0.25
_ZOOM_STEP = 1.2


class BurnPreviewWidget(Gtk.DrawingArea):
    """Read-only display of a pre-rendered "burn preview" bitmap: an
    approximation of what a set of Ops-derived segments will actually
    burn (line darkness from commanded power/feed, line width from
    laser spot size). See rayforge/image/burn_preview.py and
    PLAN_burn_preview.md.

    The widget itself only blits a finished cairo.ImageSurface -- the
    actual stroke-heavy rendering (potentially tens of thousands of
    segments for a real image) happens elsewhere, off the GTK main
    thread, via render_to_surface(). Doing that work inside this
    widget's own draw func was the main-thread blocking bug: GTK
    requires draw funcs to run on the main thread, so any amount of
    segment-stroking done there freezes the UI regardless of whether
    the segments themselves were computed in the background.

    Scroll to zoom (centered on the widget, not the cursor -- a
    simpler first pass), click-drag to pan. Zoom/pan persist across
    update_surface() calls (a settings tweak shouldn't reset your
    view); call reset_view() to clear them explicitly.
    """

    WIDTH = 400
    HEIGHT = 320

    def __init__(self):
        super().__init__()
        self.surface: Optional[cairo.ImageSurface] = None
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._drag_start_pan = (0.0, 0.0)

        self.set_content_width(self.WIDTH)
        self.set_content_height(self.HEIGHT)
        self.set_hexpand(True)
        self.set_vexpand(True)
        self.set_draw_func(self._draw_func)

        scroll = Gtk.EventControllerScroll.new(
            Gtk.EventControllerScrollFlags.VERTICAL
        )
        scroll.connect("scroll", self._on_scroll)
        self.add_controller(scroll)

        drag = Gtk.GestureDrag.new()
        drag.connect("drag-begin", self._on_drag_begin)
        drag.connect("drag-update", self._on_drag_update)
        self.add_controller(drag)

    def update_surface(self, surface: Optional[cairo.ImageSurface]):
        self.surface = surface
        self.queue_draw()

    def reset_view(self):
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self.queue_draw()

    def _on_scroll(self, controller, dx, dy) -> bool:
        if self.surface is None:
            return False
        factor = _ZOOM_STEP if dy < 0 else (1.0 / _ZOOM_STEP)
        self._zoom = max(_MIN_ZOOM, min(_MAX_ZOOM, self._zoom * factor))
        self.queue_draw()
        return True

    def _on_drag_begin(self, gesture, start_x, start_y):
        self._drag_start_pan = (self._pan_x, self._pan_y)

    def _on_drag_update(self, gesture, offset_x, offset_y):
        self._pan_x = self._drag_start_pan[0] + offset_x
        self._pan_y = self._drag_start_pan[1] + offset_y
        self.queue_draw()

    def _draw_func(self, area, ctx: cairo.Context, width: int, height: int):
        ctx.set_source_rgb(1, 1, 1)
        ctx.paint()

        if self.surface is None:
            ctx.set_source_rgba(0.5, 0.5, 0.5, 1.0)
            ctx.set_font_size(12)
            text = _("No preview")
            extents = ctx.text_extents(text)
            ctx.move_to(
                width / 2 - extents.width / 2, height / 2 + extents.height / 2
            )
            ctx.show_text(text)
            return

        sw, sh = self.surface.get_width(), self.surface.get_height()
        base_scale = min(width / sw, height / sh) if sw and sh else 1.0
        scale = base_scale * self._zoom

        ctx.save()
        ctx.translate(width / 2 + self._pan_x, height / 2 + self._pan_y)
        ctx.scale(scale, scale)
        ctx.translate(-sw / 2, -sh / 2)
        ctx.set_source_surface(self.surface, 0, 0)
        ctx.get_source().set_filter(cairo.FILTER_GOOD)
        ctx.paint()
        ctx.restore()
