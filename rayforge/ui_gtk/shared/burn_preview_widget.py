from gettext import gettext as _
from typing import Optional

import cairo
import numpy as np
from gi.repository import GLib, Gtk

# Generous, mostly-arbitrary bounds -- past _MAX_ZOOM you're just
# magnifying/interpolating already-rendered pixels rather than seeing
# new detail (the surface's actual resolution is fixed once rendered;
# see ImageEngraveWidget._recompute_burn_preview), but that's normal,
# expected behavior for a zoomable preview, not something to prevent.
_MAX_ZOOM = 50.0
_MIN_ZOOM = 0.25
_ZOOM_STEP = 1.2

# The exact box-average downsample (see _box_average_downsample) can
# take up to ~1s on a large surface. Continuous scrolling changes the
# target size on every tick, which would otherwise trigger that cost
# on every single tick -- debounce it so only the settled zoom level
# after scrolling pauses gets the precise recompute; ticks in between
# reuse a cheap cairo-scaled approximation of whatever's already cached.
_DOWNSAMPLE_DEBOUNCE_MS = 150


def _box_average_downsample(
    surface: cairo.ImageSurface, dst_w: int, dst_h: int
) -> cairo.ImageSurface:
    """Area-average `surface` down to `dst_w`x`dst_h`.

    Neither of cairo's own scaling filters does true area-averaging
    when minifying: FILTER_GOOD blurs but still aliases, and
    FILTER_NEAREST point-samples, which for a periodic fine pattern
    like scan lines skips or double-hits source rows depending on how
    the line pitch happens to land on the destination grid -- visible
    as some lines rendering thicker than others, worst at the
    in-between zoom levels rather than fully zoomed in or out. A
    summed-area table makes an exact box-filter average for any
    (non-integer-ratio included) destination size cheap: every
    destination pixel's value is the mean of the exact source
    rectangle it covers, via 4 lookups instead of iterating that
    rectangle.
    """
    src_w, src_h = surface.get_width(), surface.get_height()
    stride = surface.get_stride()
    buf = np.frombuffer(surface.get_data(), dtype=np.uint8)
    arr = buf.reshape(src_h, stride // 4, 4)[:, :src_w, :].astype(np.float64)

    integral = np.zeros((src_h + 1, src_w + 1, 4), dtype=np.float64)
    integral[1:, 1:] = np.cumsum(np.cumsum(arr, axis=0), axis=1)

    y_idx = np.clip(
        np.round(np.linspace(0, src_h, dst_h + 1)).astype(np.int64), 0, src_h
    )
    x_idx = np.clip(
        np.round(np.linspace(0, src_w, dst_w + 1)).astype(np.int64), 0, src_w
    )

    a = integral[y_idx[1:]][:, x_idx[1:]]
    b = integral[y_idx[:-1]][:, x_idx[1:]]
    c = integral[y_idx[1:]][:, x_idx[:-1]]
    d = integral[y_idx[:-1]][:, x_idx[:-1]]
    box_sum = a - b - c + d

    area = (
        (y_idx[1:] - y_idx[:-1]).astype(np.float64)[:, None, None]
        * (x_idx[1:] - x_idx[:-1]).astype(np.float64)[None, :, None]
    )
    area = np.maximum(area, 1.0)
    result = np.clip(box_sum / area, 0, 255).astype(np.uint8)

    dst = cairo.ImageSurface(cairo.FORMAT_ARGB32, dst_w, dst_h)
    dst_stride = dst.get_stride()
    dst_buf = np.frombuffer(dst.get_data(), dtype=np.uint8).reshape(
        dst_h, dst_stride // 4, 4
    )
    dst_buf[:, :dst_w, :] = result
    dst.mark_dirty()
    return dst


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
        # FILTER_GOOD smooths the blit whenever the surface is scaled
        # to fit the widget (i.e. almost always) -- fine for a normal
        # preview, but it reintroduces blending/moire on top of an
        # already-rendered surface no matter how crisp that surface
        # itself is (see render_segments's blend_multiply/cap_darkness
        # toggles). set_smooth_scaling(False) switches to a literal,
        # nearest-neighbor pixel view.
        self._smooth_scaling = True
        # Cache for the area-averaged surface used when minifying (see
        # _draw_func) -- keyed by a direct reference to the source
        # surface (not id(), which the allocator can reuse once a
        # surface is freed) plus the target size, so pan frames at an
        # unchanged zoom level reuse it instead of recomputing.
        self._downsampled: Optional[cairo.ImageSurface] = None
        self._downsampled_source: Optional[cairo.ImageSurface] = None
        self._downsampled_size: Optional[tuple] = None
        self._downsample_timeout_id = 0
        self._pending_downsample_target: Optional[tuple] = None

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

    def cleanup(self):
        """Cancel any pending debounced downsample. Call this from the
        owning page's own cleanup when the dialog closes, or the
        one-shot GLib timeout can fire after the widget is gone."""
        if self._downsample_timeout_id:
            GLib.source_remove(self._downsample_timeout_id)
            self._downsample_timeout_id = 0

    def _schedule_downsample(self, dst_w: int, dst_h: int):
        self._pending_downsample_target = (dst_w, dst_h)
        if self._downsample_timeout_id:
            GLib.source_remove(self._downsample_timeout_id)
        self._downsample_timeout_id = GLib.timeout_add(
            _DOWNSAMPLE_DEBOUNCE_MS, self._run_downsample
        )

    def _run_downsample(self) -> bool:
        self._downsample_timeout_id = 0
        target = self._pending_downsample_target
        if self.surface is None or target is None:
            return False
        self._downsampled = _box_average_downsample(self.surface, *target)
        self._downsampled_source = self.surface
        self._downsampled_size = target
        self.queue_draw()
        return False  # one-shot

    def reset_view(self):
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self.queue_draw()

    def get_zoom(self) -> float:
        return self._zoom

    def set_smooth_scaling(self, smooth: bool):
        if self._smooth_scaling == smooth:
            return
        self._smooth_scaling = smooth
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

        if scale < 1.0:
            # Minifying: cairo's own filters alias periodic fine
            # content (see _box_average_downsample's docstring), so
            # pre-downsample with a proper box filter and blit that at
            # ~1:1 instead of letting cairo scale the native surface
            # down directly. That recompute is too slow to run on
            # every scroll tick (see _DOWNSAMPLE_DEBOUNCE_MS), so this
            # frame draws whatever's already cached -- cheaply
            # cairo-scaled to the exact target size, which can itself
            # alias but only transiently, while scrolling is still in
            # motion -- and schedules the precise recompute to replace
            # it once scrolling settles.
            dst_w = max(1, round(sw * scale))
            dst_h = max(1, round(sh * scale))
            stale = self._downsampled_source is not self.surface
            if stale or self._downsampled_size != (dst_w, dst_h):
                self._schedule_downsample(dst_w, dst_h)
            approx = (
                self._downsampled
                if self._downsampled is not None and not stale
                else self.surface
            )
            approx_scale_x = dst_w / approx.get_width()
            approx_scale_y = dst_h / approx.get_height()
            ctx.scale(approx_scale_x, approx_scale_y)
            ctx.translate(-approx.get_width() / 2, -approx.get_height() / 2)
            ctx.set_source_surface(approx, 0, 0)
            ctx.get_source().set_filter(cairo.FILTER_NEAREST)
            ctx.paint()
        else:
            ctx.scale(scale, scale)
            ctx.translate(-sw / 2, -sh / 2)
            ctx.set_source_surface(self.surface, 0, 0)
            filt = (
                cairo.FILTER_GOOD
                if self._smooth_scaling
                else cairo.FILTER_NEAREST
            )
            ctx.get_source().set_filter(filt)
            ctx.paint()
        ctx.restore()
