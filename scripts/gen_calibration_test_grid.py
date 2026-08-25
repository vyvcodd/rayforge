#!/usr/bin/env python3
"""Generate a one-off G-code test grid for calibrating line interval and
dot width correction.

Row 1 sweeps line_interval_mm (dot_width_correction_mm fixed at 0) --
inspect visually for gaps/banding between scan lines; the largest
interval with no visible gaps is your effective spot diameter.

Row 2 sweeps dot_width_correction_mm (line_interval_mm fixed) -- measure
each square's burned width with calipers against the design width
(SQUARE_MM); the value that matches design width is your correction.

Usage:
    python scripts/gen_calibration_test_grid.py [-o OUTPUT.gcode]
"""

import argparse
from pathlib import Path

import numpy as np
from raygeo.ops import Ops
from raygeo.ops.assembly.raster import raster
from raygeo.ops.convert import GcodeDialectSpec
from raygeo.ops.part import Part

SQUARE_MM = 15.0
GAP_MM = 5.0
ROW_GAP_MM = 20.0
PIXELS_PER_MM = 10.0
FEED_MM_MIN = 3000.0
STEP_POWER = 0.6
MAX_POWER = 1000.0  # matches gcode_burn_preview.py's --max-power default

LINE_INTERVAL_VALUES_MM = [0.05, 0.07, 0.09, 0.11, 0.13, 0.15]
DOT_WIDTH_LINE_INTERVAL_MM = 0.08
# mask_scan quantizes dot_width_correction_mm to whole image pixels (its
# trim granularity is 1/pixels_per_mm, not raster()'s sample_interval_mm
# arg, which mask_scan ignores) -- use a fine image resolution here so
# each 0.01mm step below actually resolves to a distinct trim amount.
DOT_WIDTH_PIXELS_PER_MM = 100.0
DOT_WIDTH_VALUES_MM = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]


def _square_ops(
    x0,
    y0,
    line_interval_mm,
    dot_width_correction_mm,
    pixels_per_mm=PIXELS_PER_MM,
):
    # raster()'s offset_x_mm/offset_y_mm don't translate output geometry
    # (verified: identical rect() regardless of offset), so position via
    # Ops.translate() instead.
    part = Part(
        size_mm=(SQUARE_MM, SQUARE_MM),
        pixels_per_mm=(pixels_per_mm, pixels_per_mm),
    )
    w = int(SQUARE_MM * pixels_per_mm)
    h = int(SQUARE_MM * pixels_per_mm)
    part.image = np.full((h, w), 255, dtype=np.uint8)
    result = raster(
        part,
        mode="mask_scan",
        line_interval_mm=line_interval_mm,
        step_power=STEP_POWER,
        dot_width_correction_mm=dot_width_correction_mm,
    )
    ops = result.ops
    ops.translate(x0, y0)
    return ops


def build_grid():
    combined = Ops()
    combined.job_start()
    combined.set_feed_rate(FEED_MM_MIN)

    legend = []
    x = 0.0
    for v in LINE_INTERVAL_VALUES_MM:
        combined.extend(_square_ops(x, 0.0, v, 0.0))
        legend.append(f"Row 1, x={x:5.1f}mm: line_interval_mm={v}")
        x += SQUARE_MM + GAP_MM

    row2_y = SQUARE_MM + ROW_GAP_MM
    x = 0.0
    for v in DOT_WIDTH_VALUES_MM:
        combined.extend(
            _square_ops(
                x,
                row2_y,
                DOT_WIDTH_LINE_INTERVAL_MM,
                v,
                pixels_per_mm=DOT_WIDTH_PIXELS_PER_MM,
            )
        )
        legend.append(
            f"Row 2, x={x:5.1f}mm: dot_width_correction_mm={v} "
            f"(line_interval_mm={DOT_WIDTH_LINE_INTERVAL_MM})"
        )
        x += SQUARE_MM + GAP_MM

    combined.job_end()
    return combined, legend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("calibration_test_grid.gcode"),
    )
    args = parser.parse_args()

    ops, legend = build_grid()
    dialect = GcodeDialectSpec()
    context = dict(
        gcode_precision=3,
        max_travel_speed=6000.0,
        default_head_uid="default",
        heads=[{"uid": "default", "tool_number": 0, "max_power": MAX_POWER}],
        active_wcs="",
        layer_wcs={},
        macros={},
        path_vars={},
        layer_path_vars={},
        workpiece_path_vars={},
    )
    result = ops.to_gcode(dialect, context)

    args.output.write_text(result["text"])
    legend_path = args.output.with_suffix(".legend.txt")
    legend_path.write_text("\n".join(legend) + "\n")

    print(f"Wrote {args.output}")
    print(f"Wrote {legend_path}")
    print(
        f"Square size: {SQUARE_MM}mm, step_power={STEP_POWER} "
        f"(S{int(STEP_POWER * MAX_POWER)} @ max_power={MAX_POWER:g})"
    )
    print(
        "Adjust STEP_POWER/FEED_MM_MIN in the script for your material "
        "before burning."
    )


if __name__ == "__main__":
    main()
