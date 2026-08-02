"""Convert a raygeo Ops sequence directly into burn_preview Segments.

Companion to scripts/gcode_burn_preview.py's G-code text parser: this
walks the already-computed Ops object instead, which is the data source
BurnPreviewWidget uses for live settings-popup feedback (see
PLAN_burn_preview.md). Follows the same modal-walk precedent as
MachineState.apply_command (rayforge/simulator/machine_state.py) and the
Ruida encoder's linearize-then-walk pattern
(rayforge/machine/driver/ruida/ruida_encoder.py).

Deliberately does not decimate/skip rows or samples: this exists to
show exactly what the laser will actually do, including effects (e.g.
dot-width correction trimming a row's ends, bidirectional scan offset
shifting only reverse-direction rows) that only show up in specific
rows or specific samples within a row. Skipping any of that by stride
can silently delete the exact evidence a setting change is meant to
show.
"""

from typing import List, Optional, Tuple

from raygeo.ops import Ops
from raygeo.ops.types import CommandCategory, CommandType

from .burn_preview import Segment

_CURVE_TYPES = (
    CommandType.ARC_TO,
    CommandType.BEZIER_TO,
    CommandType.QUADRATIC_BEZIER_TO,
    CommandType.SCAN_LINE,
)


def segments_from_ops(ops: Ops) -> List[Segment]:
    segments: List[Segment] = []
    power = 0.0
    feed: Optional[float] = None
    pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    for i in range(ops.len()):
        ct = ops.command_type(i)
        cat = ops.category(i)

        if cat == CommandCategory.STATE:
            if ct == CommandType.SET_POWER:
                power = ops.power(i)
            elif ct == CommandType.SET_FEED_RATE:
                feed = ops.rate(i)
            continue

        if cat != CommandCategory.MOVING:
            continue

        end = ops.endpoint(i)

        if ct == CommandType.MOVE_TO:
            segments.append(
                Segment(
                    pos[0], pos[1], end[0], end[1],
                    power_fraction=0.0, feed=feed, is_travel=True,
                )
            )
            pos = end
        elif ct == CommandType.LINE_TO:
            segments.append(
                Segment(
                    pos[0], pos[1], end[0], end[1],
                    power_fraction=power, feed=feed, is_travel=False,
                )
            )
            pos = end
        elif ct in _CURVE_TYPES:
            sub_ops = ops.linearize(i, pos)
            sub_power = power
            for j in range(sub_ops.len()):
                sub_ct = sub_ops.command_type(j)
                if sub_ct == CommandType.SET_POWER:
                    sub_power = sub_ops.power(j)
                elif sub_ct == CommandType.LINE_TO:
                    sub_end = sub_ops.endpoint(j)
                    segments.append(
                        Segment(
                            pos[0], pos[1], sub_end[0], sub_end[1],
                            power_fraction=sub_power, feed=feed,
                            is_travel=False,
                        )
                    )
                    pos = sub_end
            pos = end  # avoid drift from float error, mirrors _arc_points

    return segments
