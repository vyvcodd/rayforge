"""Shared "burn preview" rendering core.

Renders a laser toolpath as an approximation of what it will actually
burn: line darkness driven by commanded power and feed rate, line width
by laser spot size. Used both by scripts/gcode_burn_preview.py (parses
real G-code text, for catching G-code-encoder bugs) and by
BurnPreviewWidget (walks a raygeo Ops object directly, for live
settings-popup feedback).
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cairo

# A single pass at full commanded power renders at this darkness, not
# 1.0 (pure black) -- see render_segments for why. 0.85 leaves a 0.15
# gray floor per pass, so two overlapping full-power passes multiply
# down to 0.15*0.15 ~= 0.02 (near-black), clearly distinguishable from
# one pass, while three or more get darker still.
_MAX_SINGLE_PASS_DARKNESS = 0.85


@dataclass
class Segment:
    x0: float
    y0: float
    x1: float
    y1: float
    power_fraction: float  # 0.0-1.0, already normalized. 0 if travel/off.
    feed: Optional[float]  # mm/min, None if never set
    is_travel: bool


def render_segments(
    ctx: cairo.Context,
    segments: List[Segment],
    spot_size_mm: Tuple[float, float],
    reference_speed: float,
    px_per_mm: float,
    show_travel: bool = False,
) -> None:
    """Stroke `segments` into `ctx` using MULTIPLY blending so
    overlapping passes darken realistically. Caller owns the surface,
    coordinate transform, and background fill.

    Segments are drawn as strokes running along the scan direction, so
    the stroke's thickness (perpendicular to that direction) has to be
    the laser's cross-scan spot dimension -- spot_size_mm[1] ("Spot
    Size Y" in the laser's settings) -- not spot_size_mm[0] (along the
    scan direction) or an average of the two. That's specifically the
    dimension line_interval_mm is compared against elsewhere (e.g.
    resolve_machine_defaults), and it's what determines whether
    adjacent rows overlap or leave a visible gap. This doesn't account
    for a nonzero scan angle rotating which axis is "cross-scan" --
    accurate for the common angle=0 case, an approximation otherwise.
    """
    line_width_mm = max(spot_size_mm[1], 1.0 / px_per_mm)

    ctx.set_operator(cairo.OPERATOR_MULTIPLY)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_width(line_width_mm)

    for seg in segments:
        if seg.is_travel:
            continue
        feed = seg.feed if seg.feed and seg.feed > 0 else reference_speed
        darkness = seg.power_fraction * (reference_speed / feed)
        # Capped below 1.0 so a single pass at full power renders as a
        # dark gray, not pure black -- leaving headroom for the
        # MULTIPLY blend below to show overlapping passes (e.g. from
        # too-tight line spacing or a wrong bidir offset) as visibly
        # darker still. Full black would otherwise be a dead end: black
        # multiplied by anything stays black, so a second overlapping
        # pass would be invisible.
        darkness = max(0.0, min(_MAX_SINGLE_PASS_DARKNESS, darkness))
        if darkness <= 0.0:
            continue
        gray = 1.0 - darkness
        ctx.set_source_rgb(gray, gray, gray)
        ctx.move_to(seg.x0, seg.y0)
        ctx.line_to(seg.x1, seg.y1)
        ctx.stroke()

    if show_travel:
        ctx.set_operator(cairo.OPERATOR_OVER)
        ctx.set_line_width(0.5 / px_per_mm * 4)
        ctx.set_source_rgba(0.1, 0.4, 1.0, 0.4)
        for seg in segments:
            if not seg.is_travel:
                continue
            ctx.move_to(seg.x0, seg.y0)
            ctx.line_to(seg.x1, seg.y1)
            ctx.stroke()


Bounds = Tuple[float, float, float, float]  # (min_x, min_y, max_x, max_y)


def render_to_surface(
    segments: List[Segment],
    bounds: Bounds,
    spot_size_mm: Tuple[float, float],
    reference_speed: float,
    width_px: int,
    height_px: int,
    margin_px: float = 4.0,
    show_travel: bool = False,
) -> cairo.ImageSurface:
    """Render `segments` fit-to-size into a standalone ARGB32 surface.

    Building a `cairo.ImageSurface` and painting into it touches no GTK
    widget state, so unlike a `Gtk.DrawingArea` draw func (which must
    run on the main thread), this is safe to call from a background
    thread -- letting a caller do the expensive stroke-heavy rendering
    of a large segment list off the GTK main thread, then hand the
    finished bitmap back for a cheap blit.
    """
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width_px, height_px)
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(1, 1, 1)
    ctx.paint()

    min_x, min_y, max_x, max_y = bounds
    content_w = max(max_x - min_x, 1e-6)
    content_h = max(max_y - min_y, 1e-6)
    avail_w = max(1.0, width_px - 2 * margin_px)
    avail_h = max(1.0, height_px - 2 * margin_px)
    scale = min(avail_w / content_w, avail_h / content_h)

    offset_x = margin_px + (avail_w - content_w * scale) / 2
    offset_y = margin_px + (avail_h - content_h * scale) / 2

    ctx.save()
    # Geometry is Cartesian (Y-up); surface is top-left origin (Y-down).
    ctx.translate(offset_x - min_x * scale, offset_y + max_y * scale)
    ctx.scale(scale, -scale)
    render_segments(
        ctx, segments, spot_size_mm, reference_speed, scale, show_travel
    )
    ctx.restore()
    return surface