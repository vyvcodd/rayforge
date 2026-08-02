# Live burn preview in the Raster/Engrave settings popup

## Context

We built `scripts/gcode_burn_preview.py`, a standalone tool that parses actual
G-code text and renders a "what will this actually burn" PNG (line darkness
driven by commanded power and feed rate, line width by laser spot size). It's
useful for catching G-code-encoder bugs specifically.

The user now wants the same kind of visual feedback live inside the
"Engraving settings" popup (`RasterSettingsWidget`), so that dragging the
threshold/power/dither sliders shows an immediate burn preview instead of
requiring an export-and-run-the-script round trip. For this use case, the
goal is different: sanity-checking how the raster→power mapping (threshold,
black/white points, dithering, min/max power) turns into a toolpath, not
verifying G-code-encoder fidelity. The user confirmed rendering from the
already-computed `Ops` object (rather than round-tripping through G-code
text) is the right data source for this — it's the same object the popup's
settings already produce internally, and it's re-derivable cheaply at low
resolution. The script stays as-is for encoder-level debugging.

## Design

### 1. Extract a shared, source-agnostic render core

Pull the `Segment` dataclass and cairo-drawing logic out of
`scripts/gcode_burn_preview.py` into `rayforge/image/burn_preview.py`:

- `Segment(x0, y0, x1, y1, power_fraction: float, feed: Optional[float], is_travel: bool)`
  — note the field becomes `power_fraction` (always 0.0–1.0), not an absolute
  S value. `raygeo`'s `ops.power(idx)` is already normalized 0–1, so this
  keeps the contract uniform between the two producers (script's G-code
  parser divides by `max_power` itself before constructing a `Segment`; the
  new `Ops` walker doesn't need to).
- `render_segments(ctx: cairo.Context, segments, spot_size_mm, reference_speed, px_per_mm, show_travel=False)`
  — the existing MULTIPLY-blend stroke logic from `render_preview()`, but
  taking a cairo context instead of owning the surface/file-write. Callers
  (the script, and the new widget) own their own surface and coordinate
  transform.
- `scripts/gcode_burn_preview.py` keeps its own `parse_gcode()` (G-code text →
  `Segment`s, dividing S by `--max-power`) and its PNG-writing `main()`, but
  calls into the shared `render_segments()` instead of duplicating it.

### 2. `Ops` → `Segment` extraction

New function in the same module (or `rayforge/image/ops_burn_preview.py` if
`rayforge/image/` shouldn't depend on GTK-adjacent code — it won't, this is
pure computation): `segments_from_ops(ops: "Ops") -> List[Segment]`.

Follow the existing modal-walk precedent in
`rayforge/simulator/machine_state.py:43-75` (`MachineState.apply_command`)
and `rayforge/pipeline/encoder/gcode.py` — iterate `for i in range(ops.len())`,
track modal `power` (via `CommandType.SET_POWER` → `ops.power(idx)`, already
0–1) and `feed` (via `SET_FEED_RATE`/`SET_RAPID_RATE` → `ops.rate(idx)`), and
emit a `Segment` per `MOVE_TO` (travel), `LINE_TO` (cut), and
`ARC_TO`/`BEZIER_TO`/`QUADRATIC_BEZIER_TO`/`SCAN_LINE` (cut, flattened via
`ops.linearize(idx, current_pos)` — same pattern as `gcode.py:453,705`).

### 3. New `BurnPreviewWidget`

New file `rayforge/ui_gtk/shared/burn_preview_widget.py`, modeled directly on
`rayforge/ui_gtk/shared/histogram_preview.py`:

- `Gtk.DrawingArea` subclass, fixed `set_content_width/height` (propose
  240x180 — wider than the 200x100 histogram since burn images benefit from
  more area, but keep it compact for a popup row).
- `update_segments(segments: Optional[List[Segment]], bounds, spot_size_mm, reference_speed)`
  stores the data and calls `queue_draw()`; `segments=None` draws a "No
  preview" placeholder (same fallback pattern as `HistogramPreview`).
- `_draw_func` computes a fit-to-widget scale from `bounds` (the combined
  `(min_x,min_y,max_x,max_y)` across all rendered workpieces), applies the
  same Y-flip transform used in `OpsRenderer._render_to_cairo_surface`
  (`rayforge/image/ops_renderer.py:112-124`), fills white, then calls the
  shared `render_segments()`.
- No interactivity needed (unlike `HistogramPreview`'s drag handles) — it's
  read-only.

### 4. Wire into `RasterSettingsWidget`

File: `rayforge/builtin_addons/rayforge-addon-laser/laser_essentials/widgets/raster_widget.py`.

- Instantiate `self.burn_preview = BurnPreviewWidget()`, add it as an
  `Adw.ActionRow("Burn Preview")` suffix, placed near the histogram row
  (after it, in the same visual area — not replacing it; histogram shows raw
  tonal distribution, burn preview shows post-assembly output, they answer
  different questions).
- New method `_recompute_burn_preview(self)`, closely mirroring
  `_compute_and_update_histogram` (lines 400-450):
  - `layer = self.step.layer`; bail to `update_segments(None, ...)` if no
    layer/workpieces (same early-return shape as the histogram method).
  - `machine = self.editor.context.machine`; `laser = self.step.get_selected_laser(machine)`
    (this method already exists — `rayforge/core/step.py:480-493` — reuse it
    rather than reinventing laser lookup).
  - For each workpiece: build the same ≤256px downsampled surface the
    histogram already builds (`workpiece.render_to_pixels(width_px, height_px)`,
    same downscale-to-max_px=256 logic, lines 419-431) — reuse, don't
    duplicate; consider factoring that downsample-size computation into a
    small shared helper if it's easy, otherwise just copy the ~10 lines.
  - `settings = self.step.to_dict()`; mirror the machine-derived keys
    production code adds in `workpiece_stage.py:106-132`
    (`prepare_task_settings`) — at minimum `settings["machine_supports_arcs"] = machine.supports_arcs`.
    Exact parity with every key `prepare_task_settings` sets isn't critical
    here (arcs/curves support only affects path segmentation shape, which
    barely matters for a coarse power/darkness preview); note this as an
    accepted simplification.
  - `machine_defaults = resolve_machine_defaults(laser, settings)`
    (`rayforge/pipeline/stage/assembler_helpers.py:131`).
  - `artifact = self.step.assemble_on_surface(workpiece, laser, generation_id=0, surface=surface, pixels_per_mm=pixels_per_mm, machine_defaults=machine_defaults)`
    — wrap in `try/except (AssertionError, KeyError, ValueError)` since it
    can throw on partial/invalid settings mid-edit (per
    `raster_step.py:193-304`); on failure, treat like "no data" for that
    workpiece rather than crashing the popup.
  - `segments_from_ops(artifact.ops)`, offset already baked in (Ops
    coordinates are absolute — `assemble_on_surface` passes
    `offset_x_mm=workpiece.bbox[0]`, so segments from multiple workpieces
    naturally composite into one shared coordinate space, same as the real
    job).
  - Concatenate segments across workpieces, compute combined bounds via
    min/max over all segment endpoints (mirroring how the script computes
    `min_x/max_x/min_y/max_y`), auto-pick `reference_speed` as the max feed
    among cut segments (same auto-reference logic as the script's `main()`).
  - Call `self.burn_preview.update_segments(segments, bounds, laser.spot_size_mm, reference_speed)`.
- **Threading**: run this synchronously on the GTK main thread, same as the
  histogram method (which already does a non-trivial `render_to_pixels` vips
  call synchronously today). Production code always runs
  `assemble_on_surface` in a subprocess (`workpiece_compute.py`,
  `task_manager.run_process`), but that's for full-resolution artifacts; at
  the same ≤256px cap the histogram already uses, cost should be small and
  bounded, and the existing 150ms debounce already prevents recomputation on
  every intermediate drag tick. If manual testing shows UI jank on complex
  images, the fallback is routing through `task_manager.run_process`
  following the `WorkPiecePipelineStage` pattern — flagging this as the one
  design choice to revisit if it doesn't hold up, not blocking it.
- **Triggering**: `DebounceMixin._debounce` holds a single shared timer (not
  per-key — a second `_debounce()` call replaces the first callback
  entirely, see `rayforge/shared/util/glib.py:31-42`), so the preview
  recompute can't be scheduled independently alongside each parameter's own
  debounced `_on_param_changed` call without racing it. Instead, hook it at
  the tail end of `_on_param_changed` itself (`raster_widget.py:611-612`) —
  by the time that method runs, whatever debounce triggered it has already
  fired and reset (`_debounce_timer = 0`), so starting a *new*
  `self._debounce(self._recompute_burn_preview)` there is safe and covers
  every call site that funnels through it (which is nearly all of them:
  mode, threshold, dither algorithm, black/white point, auto levels,
  cross-hatch, scan mode, invert, line/sample interval, bidir offset,
  power-levels, depth-levels, z-step, angle-increment).
  One path bypasses `_on_param_changed`: `_commit_power_range_change` (min/max
  power sliders, lines 452-467) calls `set_step_property` directly — add the
  same `self._debounce(self._recompute_burn_preview)` call there too.
- Also call `self._recompute_burn_preview()` once at init via
  `GLib.idle_add`, alongside the existing histogram idle_add (line 253).

## Files touched

- `rayforge/image/burn_preview.py` — new; shared `Segment` + `render_segments()`.
- `rayforge/image/ops_burn_preview.py` — new; `segments_from_ops()`.
- `rayforge/ui_gtk/shared/burn_preview_widget.py` — new; `BurnPreviewWidget`.
- `rayforge/builtin_addons/rayforge-addon-laser/laser_essentials/widgets/raster_widget.py` — add widget + `_recompute_burn_preview` + wiring.
- `scripts/gcode_burn_preview.py` — refactor to import `Segment`/`render_segments` from the new shared module instead of owning them.

## Verification

1. Unit-level sanity: run the refactored `scripts/gcode_burn_preview.py`
   against the existing test G-code snippet used earlier in this session and
   confirm pixel-level output is unchanged (same darkness/gray values at the
   same coordinates as before the refactor).
2. Launch the app (`pixi run rayforge` or equivalent — check for an existing
   `run` skill/script first), open a document with a raster/image workpiece,
   add an Engrave step, open its settings popup, and confirm:
   - The burn preview renders on dialog open.
   - Dragging threshold, black/white points, min/max power, and switching
     depth mode / dither algorithm all update the preview after the 150ms
     debounce settles, without visible UI freeze.
   - The preview clears gracefully (no crash) for a layer with no workpieces
     or an invalid partial setting combination during a drag.
   - Compare the live preview's darkness pattern against a matching
     `scripts/gcode_burn_preview.py` run on the exported G-code for the same
     step/settings — they should agree qualitatively (same relative
     light/dark regions), confirming the `Ops`-based and G-code-text-based
     paths aren't diverging in an unexpected way.
