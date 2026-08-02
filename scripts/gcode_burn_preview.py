#!/usr/bin/env python3
"""Render a G-code file to a "burn preview" PNG.

Parses the *actual* G-code text (not rayforge's internal Ops model), so
bugs introduced by GcodeEncoder itself -- wrong power scaling, dropped
modal state, bad dialect templates -- show up in the image, not just
bugs in the operation-generation pipeline upstream of encoding.

Targets GRBL-style dialects such as
rayforge/resources/devices/neje-master-3-max/dialect.yaml: G0/G1/G2/G3
moves carrying inline S (power) and modal F (feed), M3/M4/M5 laser
on/off. Unrecognized lines (homing, WCS setup, jog commands, comments)
are skipped.

Darkness per segment is modeled as commanded_power * (reference_speed /
feed_rate), clamped to [0, 1] -- i.e. slower moves and higher power both
burn darker. This is a relative visualization to sanity-check *commanded*
values against expectations, not a calibrated physical simulation: it
ignores material response, acceleration/deceleration, and (for
M4/"dynamic power" dialects like the NEJE) GRBL's own real-time power
scaling near corners.

Usage:
    python gcode_burn_preview.py job.gcode -o preview.png \\
        --max-power 1000 --spot-size 0.08 0.06 --max-cut-speed 3000
"""

import argparse
import importlib.util
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cairo


def _load_burn_preview_module():
    """Load rayforge/image/burn_preview.py directly by file path, not
    via `from rayforge.image.burn_preview import ...`.

    burn_preview.py itself only needs cairo (stdlib + pycairo, nothing
    else) -- but a normal package import would run rayforge/__init__.py
    and rayforge/image/__init__.py first, and the latter pulls in
    raygeo, PIL, pymupdf, vtracer and the rest of rayforge's
    importer/exporter machinery just to reach this one file. Loading it
    directly by path sidesteps that entirely, so this script only ever
    needs pycairo installed -- no pixi env, no rayforge runtime deps.
    """
    path = Path(__file__).parent.parent / "rayforge" / "image" / "burn_preview.py"
    spec = importlib.util.spec_from_file_location("burn_preview", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_burn_preview = _load_burn_preview_module()
Segment = _burn_preview.Segment
render_segments = _burn_preview.render_segments

_WORD_RE = re.compile(r"([A-Za-z])\s*(-?\d+\.?\d*)")


@dataclass
class ModalState:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    feed: Optional[float] = None
    power: float = 0.0
    laser_mode: Optional[str] = None  # 'M3' or 'M4', informational only
    laser_on: bool = False


def _parse_words(line: str) -> List[Tuple[str, float]]:
    line = line.split(";", 1)[0]  # strip GRBL-style comments
    line = re.sub(r"\(.*?\)", "", line)  # strip (...) comments
    return [(letter.upper(), float(val)) for letter, val in _WORD_RE.findall(line)]


def _arc_points(
    start: Tuple[float, float],
    end: Tuple[float, float],
    i: float,
    j: float,
    cw: bool,
    max_segment_len: float = 0.5,
) -> List[Tuple[float, float]]:
    cx, cy = start[0] + i, start[1] + j
    r = math.hypot(start[0] - cx, start[1] - cy)
    if r < 1e-9:
        return [end]

    a0 = math.atan2(start[1] - cy, start[0] - cx)
    a1 = math.atan2(end[1] - cy, end[0] - cx)

    if cw:
        while a1 >= a0:
            a1 -= 2 * math.pi
    else:
        while a1 <= a0:
            a1 += 2 * math.pi

    arc_len = abs(a1 - a0) * r
    n = max(1, int(arc_len / max_segment_len))
    points = []
    for k in range(1, n + 1):
        a = a0 + (a1 - a0) * (k / n)
        points.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    points[-1] = end  # avoid drift from float error
    return points


def parse_gcode(text: str, max_power: float) -> List[Segment]:
    state = ModalState()
    segments: List[Segment] = []

    def power_fraction() -> float:
        if not state.laser_on or not max_power:
            return 0.0
        return max(0.0, min(1.0, state.power / max_power))

    for raw_line in text.splitlines():
        words = _parse_words(raw_line)
        if not words:
            continue

        letters = {w[0]: w[1] for w in words}
        gcodes = [v for l, v in words if l == "G"]

        if "M" in letters:
            m = int(letters["M"])
            if m in (3, 4):
                state.laser_mode = f"M{m}"
                state.laser_on = True
            elif m == 5:
                state.laser_on = False

        if "F" in letters:
            state.feed = letters["F"]
        if "S" in letters:
            state.power = letters["S"]

        for g in gcodes:
            start = (state.x, state.y)
            new_x = letters.get("X", state.x)
            new_y = letters.get("Y", state.y)
            new_z = letters.get("Z", state.z)

            if g == 0:  # rapid / travel
                segments.append(
                    Segment(
                        start[0], start[1], new_x, new_y,
                        power_fraction=0.0, feed=state.feed, is_travel=True,
                    )
                )
                state.x, state.y, state.z = new_x, new_y, new_z
            elif g == 1:  # linear cut
                segments.append(
                    Segment(
                        start[0], start[1], new_x, new_y,
                        power_fraction=power_fraction(),
                        feed=state.feed, is_travel=False,
                    )
                )
                state.x, state.y, state.z = new_x, new_y, new_z
            elif g in (2, 3):  # arc cut, cw / ccw
                i = letters.get("I", 0.0)
                j = letters.get("J", 0.0)
                end = (new_x, new_y)
                pts = _arc_points(start, end, i, j, cw=(g == 2))
                prev = start
                for pt in pts:
                    segments.append(
                        Segment(
                            prev[0], prev[1], pt[0], pt[1],
                            power_fraction=power_fraction(),
                            feed=state.feed,
                            is_travel=False,
                        )
                    )
                    prev = pt
                state.x, state.y, state.z = new_x, new_y, new_z
            # G4 (dwell), G20/21/90/91, G10, G38.2 etc. are intentionally
            # ignored -- they don't affect the burn preview.

    return segments


def render_preview(
    segments: List[Segment],
    output_path: Path,
    spot_size_mm: Tuple[float, float],
    reference_speed: float,
    px_per_mm: float,
    show_travel: bool,
) -> None:
    burn_segments = [s for s in segments if not s.is_travel]
    if not burn_segments:
        print("No cutting/engraving moves found -- nothing to render.", file=sys.stderr)
        return

    # Bound the canvas to actual burn content, not travel moves -- the
    # first move in a job is always a travel from the machine's last
    # position (nominally (0,0) here), which would otherwise drag the
    # frame out to include wherever the job happens to start from.
    xs = [c for s in burn_segments for c in (s.x0, s.x1)]
    ys = [c for s in burn_segments for c in (s.y0, s.y1)]
    margin_mm = max(spot_size_mm) * 4 + 1.0
    min_x, max_x = min(xs) - margin_mm, max(xs) + margin_mm
    min_y, max_y = min(ys) - margin_mm, max(ys) + margin_mm

    width_px = max(1, int((max_x - min_x) * px_per_mm))
    height_px = max(1, int((max_y - min_y) * px_per_mm))

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width_px, height_px)
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(1, 1, 1)  # white "material"
    ctx.paint()

    # Geometry is Cartesian (Y-up); surface is top-left origin (Y-down).
    ctx.translate(-min_x * px_per_mm, max_y * px_per_mm)
    ctx.scale(px_per_mm, -px_per_mm)

    render_segments(
        ctx, segments, spot_size_mm, reference_speed, px_per_mm, show_travel
    )

    surface.write_to_png(str(output_path))
    print(f"Wrote {output_path} ({width_px}x{height_px}px)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("gcode_file", type=Path)
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output PNG path (default: <gcode_file>.png)",
    )
    parser.add_argument(
        "--max-power", type=float, default=1000.0,
        help="Absolute S value corresponding to 100%% power "
             "(machine.heads[].max_power, e.g. 1000 for the NEJE Master 3 Max)",
    )
    parser.add_argument(
        "--spot-size", type=float, nargs=2, default=[0.08, 0.06],
        metavar=("X_MM", "Y_MM"),
        help="Laser spot size in mm (machine.heads[].spot_size_mm)",
    )
    parser.add_argument(
        "--reference-speed", type=float, default=None,
        help="Feed rate (mm/min) treated as the 'lightest' burn intensity "
             "at full power; slower moves render darker, faster render "
             "lighter. Default: the fastest cut feed actually used in the "
             "file, so darkness is relative to this job. Pass "
             "machine.max_cut_speed instead if you want absolute, "
             "cross-job comparability.",
    )
    parser.add_argument(
        "--px-per-mm", type=float, default=40.0,
        help="Render resolution. Should be high enough to resolve spot "
             "size (>=4-8px across the spot recommended).",
    )
    parser.add_argument(
        "--show-travel", action="store_true",
        help="Overlay rapid/travel moves as thin blue lines for reference.",
    )
    args = parser.parse_args()

    output = args.output or args.gcode_file.with_suffix(".png")
    text = args.gcode_file.read_text()
    segments = parse_gcode(text, max_power=args.max_power)

    reference_speed = args.reference_speed
    if reference_speed is None:
        cut_feeds = [s.feed for s in segments if not s.is_travel and s.feed]
        reference_speed = max(cut_feeds) if cut_feeds else 1.0
        print(f"Auto reference speed (fastest cut feed in file): "
              f"{reference_speed:g} mm/min", file=sys.stderr)

    render_preview(
        segments,
        output,
        spot_size_mm=tuple(args.spot_size),
        reference_speed=reference_speed,
        px_per_mm=args.px_per_mm,
        show_travel=args.show_travel,
    )


if __name__ == "__main__":
    main()
