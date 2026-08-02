"""Settings page for the ImageEngraveStep.

PoC copy of RasterSettingsPage (see image_engrave_step.py's
docstring), plus a live burn preview (PLAN_burn_preview.md). The
preview is driven by the real pipeline: it listens for
``editor.pipeline.workpiece_artifact_ready`` and re-reads whatever Ops
are currently computed for this step via ``pipeline.get_artifact``,
rather than triggering its own compute -- the document pipeline
already recomputes on a debounce whenever settings change, so this
page only needs to react, not drive.
"""

import logging
from dataclasses import replace
from gettext import gettext as _
from typing import TYPE_CHECKING, Any

import cairo
from gi.repository import GLib, Gtk

from rayforge.core.varset import VarSet
from rayforge.image.burn_preview import Segment, render_to_surface
from rayforge.image.ops_burn_preview import segments_from_ops
from rayforge.machine.models.laser import LaserHead
from rayforge.ui_gtk.shared.burn_preview_widget import BurnPreviewWidget

from .laser_step_page import LaserStepSettingsPage
from .levels_adapter import LevelsAdapter
from .raster_power_widget import RasterPowerWidget

logger = logging.getLogger(__name__)

#: Engrave-section keys (mode, geometry, multi-pass) vs. Power-section
#: keys (histogram, brightness range). The recipe varset keeps them in
#: one group; the dialog splits them into two sections.
_ENGRAVE_KEYS = {
    "depth_mode",
    "threshold",
    "dither_algorithm",
    "scan_angle",
    "cross_hatch",
    "scan_mode",
    "line_interval_mm",
    "sample_interval_mm",
    "dot_width_correction_mm",
    "bidir_x_offset_mm",
    "invert",
    "num_depth_levels",
    "z_step_down",
    "angle_increment",
}

_POWER_KEYS = {
    "auto_levels",
    "black_point",
    "white_point",
    "min_power_level",
    "max_power_level",
    "num_power_levels",
}

# Target pixel resolution for the burn preview's offscreen render (see
# _recompute_burn_preview): pixels across the smallest physically
# relevant feature -- the spot, the row-to-row line interval, or the
# gap between rows, whichever is smallest -- and the largest allowed
# surface dimension regardless of job size.
_DOTS_PER_SPOT = 8
_MAX_SURFACE_DIM = 4000

if TYPE_CHECKING:
    from rayforge.doceditor.editor import DocEditor


class ImageEngraveStepSettingsPage(LaserStepSettingsPage):
    """UI for configuring the ImageEngraveStep."""

    cut_speed_title = _("Engrave Speed")

    def __init__(
        self,
        editor: "DocEditor",
        step: Any,
    ):
        super().__init__(editor, step)
        groups = self.step.recipe_varset_groups()
        step_vars = groups[-1][1] if len(groups) > 1 else None
        if step_vars is None:
            return
        engrave_vs = VarSet(
            vars=[v for v in step_vars if v.key in _ENGRAVE_KEYS]
        )
        power_vs = VarSet(vars=[v for v in step_vars if v.key in _POWER_KEYS])
        self.engrave_widget = self.add_varset_section(
            _("Engrave"),
            engrave_vs,
            description=_("Raster the image onto the material."),
        )
        self.power_widget = self.add_varset_section(
            _("Power"),
            power_vs,
            description=_("Power modulation and brightness range."),
            widget_cls=RasterPowerWidget,
        )

        # The Power section's visible_when predicates and histogram
        # depend on depth_mode and invert, whose rows live in the
        # Engrave section. Feed them in as context; the adapters fire
        # changed synchronously on user interaction and on resync.
        self._sync_power_context()
        for key in ("depth_mode", "invert"):
            adapter = self.engrave_widget.adapter_for(key)
            if adapter is not None:
                adapter.changed.connect(
                    lambda sender, k=key: self._sync_power_context(),
                    weak=False,
                )

        levels = self.power_widget.adapter_for("black_point")
        if isinstance(levels, LevelsAdapter):
            levels.set_histogram_source(step)
            GLib.idle_add(levels.compute_histogram)

        self._push_head_defaults()

        # --- Burn Preview (handed to the dialog's side panel, not
        # added as a settings row -- see StepSettingsDialog.set_side_panel) ---
        self.burn_preview = BurnPreviewWidget()
        self.editor.pipeline.workpiece_artifact_ready.connect(
            self._on_workpiece_artifact_ready
        )
        GLib.idle_add(self._attach_side_panel)
        GLib.idle_add(self._recompute_burn_preview)

    def _push_head_defaults(self):
        """Show head-derived values for the auto (None) interval rows.

        The step keeps ``None`` (auto) until the user edits the row;
        the old dialog displayed the laser spot-size default instead.
        """
        head = self.get_selected_head()
        if not isinstance(head, LaserHead):
            return
        spot_x, spot_y = head.spot_size_mm
        values = {}
        if self.step.line_interval_mm is None:
            values["line_interval_mm"] = spot_y
        if self.step.sample_interval_mm is None:
            values["sample_interval_mm"] = spot_x / 2.0
        if self.step.dot_width_correction_mm is None:
            values["dot_width_correction_mm"] = spot_x / 2.0
        if values:
            self.engrave_widget.set_values(values)

    def _sync_power_context(self):
        """Feed the current depth_mode/invert into the Power section."""
        values = self.engrave_widget.get_values()
        self.power_widget.set_context_values(
            {
                "depth_mode": values.get("depth_mode"),
                "invert": values.get("invert"),
            }
        )

    def _on_varset_data_changed(self, widget, key):
        if key in ("min_power_level", "max_power_level"):
            self._commit_power_range(widget, key)
            return
        super()._on_varset_data_changed(widget, key)

    def _commit_power_range(self, widget, moved_key):
        """Commit min/max power together, keeping min <= max.

        When one slider is dragged past the other, the other follows
        (max follows min up, min follows max down), mirroring the old
        dialog behavior.
        """
        min_p = widget.adapter_for("min_power_level").get_value()
        max_p = widget.adapter_for("max_power_level").get_value()
        if min_p is None or max_p is None:
            return
        if moved_key == "min_power_level" and min_p > max_p:
            max_p = min_p
        elif moved_key == "max_power_level" and max_p < min_p:
            min_p = max_p
        with self.history_manager.transaction(_("Change Power Range")):
            self.set_step_property("min_power_level", min_p)
            self.set_step_property("max_power_level", max_p)

    def _attach_side_panel(self):
        root = self.get_root()
        set_side_panel = getattr(root, "set_side_panel", None)
        if callable(set_side_panel):
            panel = Gtk.Box(
                orientation=Gtk.Orientation.VERTICAL,
                margin_start=12,
                margin_end=12,
                margin_top=12,
                margin_bottom=12,
            )
            header_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
            header_row.append(
                Gtk.Label(
                    label=_("Burn Preview"),
                    halign=Gtk.Align.START,
                    hexpand=True,
                    css_classes=["title-4"],
                )
            )
            reset_view_button = Gtk.Button.new_from_icon_name(
                "zoom-original-symbolic"
            )
            reset_view_button.set_tooltip_text(_("Reset zoom and pan"))
            reset_view_button.connect(
                "clicked", lambda b: self.burn_preview.reset_view()
            )
            header_row.append(reset_view_button)
            panel.append(header_row)
            panel.append(
                Gtk.Label(
                    label=_("Scroll to zoom, drag to pan"),
                    halign=Gtk.Align.START,
                    css_classes=["dim-label", "caption"],
                )
            )
            panel.append(self.burn_preview)
            set_side_panel(panel)
        if isinstance(root, Gtk.Window):
            root.set_default_size(950, 700)
        return False

    def _on_workpiece_artifact_ready(
        self, sender, *, step, workpiece, handle, **kwargs
    ):
        if step.uid != self.step.uid:
            return
        self._recompute_burn_preview()

    def _recompute_burn_preview(self):
        """Pull whatever Ops the real pipeline currently has computed
        for this step's workpieces and render them, rather than
        triggering a compute ourselves -- the document pipeline
        already recomputes (debounced) whenever settings change and
        fires workpiece_artifact_ready, which is what re-triggers
        this.

        Both segments_from_ops() (walking every command in the Ops
        sequence one Python/FFI call at a time) and the actual cairo
        stroking of those segments are slow enough, for a real raster
        image with potentially tens of thousands of scan-line samples,
        to visibly block the GTK main thread -- and unlike the Ops
        walk, the stroking can't just be moved into a Gtk.DrawingArea
        draw func on a background thread, since GTK requires draw
        funcs to run on the main thread. So both steps run in the
        background via render_to_surface(), producing a plain
        cairo.ImageSurface (which doesn't touch GTK widget state,
        hence is safe off-thread); the main thread only ever does the
        cheap blit in BurnPreviewWidget's own draw func.
        """
        layer = self.step.layer
        if not layer or not layer.all_workpieces:
            self.burn_preview.update_surface(None)
            return False

        pipeline = self.editor.pipeline
        ops_list = []
        for workpiece in layer.all_workpieces:
            artifact = pipeline.get_artifact(self.step, workpiece)
            if artifact is not None and artifact.ops is not None:
                ops_list.append(artifact.ops)

        machine = self.editor.context.machine
        laser = self.step.get_selected_laser(machine) if machine else None
        spot_size_mm = laser.spot_size_mm if laser else (0.1, 0.1)
        line_interval_mm = (
            self.step.line_interval_mm
            if self.step.line_interval_mm is not None
            else spot_size_mm[1]
        )
        # Commanded power (what Ops.power()/power_fraction carries) is
        # max_power_setting * step.power -- the general "Power" slider
        # scales everything down before it ever reaches the laser. Divide
        # it back out so the preview always shows the full 0..Max Power
        # contrast range, regardless of what the general Power happens
        # to be dialed to right now (e.g. turned down for safety while
        # iterating on dot/line settings) -- otherwise a low general
        # Power would wash the whole preview out towards gray, which
        # says nothing about how the dot/overlap settings actually
        # behave. Holds for every depth mode: power-modulated pixels are
        # step_power*max_power_setting per-pixel; mask/dither "on"
        # pixels are exactly step_power (no separate max-power scaling)
        # so this correctly maps them to 1.0 either way.
        step_power = max(self.step.power, 1e-6)

        if not ops_list:
            self.burn_preview.update_surface(None)
            return False

        display_width_px = (
            self.burn_preview.get_width() or BurnPreviewWidget.WIDTH
        )
        display_height_px = (
            self.burn_preview.get_height() or BurnPreviewWidget.HEIGHT
        )

        def _compute(ops_list: list) -> cairo.ImageSurface | None:
            # Deliberately no decimation: this has to show exactly what
            # the laser will actually do, including effects (dot-width
            # correction trimming a row's ends, bidirectional scan
            # offset shifting only reverse-direction rows) that live in
            # specific rows/samples a stride-based skip could delete
            # outright. See segments_from_ops's docstring.
            all_segments: list[Segment] = []
            for ops in ops_list:
                for seg in segments_from_ops(ops):
                    if seg.power_fraction:
                        seg = replace(
                            seg,
                            power_fraction=min(
                                1.0, seg.power_fraction / step_power
                            ),
                        )
                    all_segments.append(seg)

            burn_only = [s for s in all_segments if not s.is_travel]
            if not burn_only:
                return None

            xs = [c for s in burn_only for c in (s.x0, s.x1)]
            ys = [c for s in burn_only for c in (s.y0, s.y1)]
            bounds = (min(xs), min(ys), max(xs), max(ys))
            content_w_mm = max(bounds[2] - bounds[0], 1e-6)
            content_h_mm = max(bounds[3] - bounds[1], 1e-6)

            cut_feeds = [s.feed for s in burn_only if s.feed]
            reference_speed = max(cut_feeds) if cut_feeds else 1.0

            # Render resolution is scaled to the smallest physically
            # relevant feature, not just the widget's display size, so
            # zooming in resolves it (~_DOTS_PER_SPOT pixels across)
            # rather than just magnifying already-blurred pixels. That
            # feature is whichever of these is smallest:
            #
            # - line_interval_mm itself: the row-to-row pitch. Sizing
            #   resolution off the spot alone (the old behavior) is
            #   fine when spacing roughly matches the spot, but once
            #   rows overlap tightly for a smoother continuous-tone
            #   raster, the pitch can be a fraction of the spot size --
            #   ignoring it means each row's center lands at whatever
            #   sub-pixel offset the coarse grid happens to give it,
            #   which is exactly what produces uneven-looking row
            #   thickness (banding) baked into the surface itself.
            # - the spot, spot_avg_mm.
            # - once line spacing exceeds the spot, the real unburned
            #   gap between passes (spacing - spot), which is smaller
            #   than the spot. Sizing resolution off the spot alone in
            #   that case renders the gap at only a few px, which
            #   anti-aliasing/multiply-blending across many overlapping
            #   strokes turns into noisy gray banding instead of a
            #   clean visible gap.
            #
            # Capped to _MAX_SURFACE_DIM (memory/time; cairo itself
            # refuses surfaces much larger than this anyway) and
            # floored to the widget's own display size (never render
            # *below* what a non-zoomed view needs).
            spot_avg_mm = sum(spot_size_mm) / 2
            smallest_feature_mm = min(spot_avg_mm, line_interval_mm)
            if line_interval_mm > spot_avg_mm:
                smallest_feature_mm = min(
                    smallest_feature_mm, line_interval_mm - spot_avg_mm
                )
            px_per_mm = _DOTS_PER_SPOT / max(smallest_feature_mm, 1e-6)
            width_px = content_w_mm * px_per_mm
            height_px = content_h_mm * px_per_mm

            largest = max(width_px, height_px)
            if largest > _MAX_SURFACE_DIM:
                shrink = _MAX_SURFACE_DIM / largest
                width_px *= shrink
                height_px *= shrink

            width_px = max(int(width_px), display_width_px)
            height_px = max(int(height_px), display_height_px)

            return render_to_surface(
                all_segments,
                bounds,
                spot_size_mm,
                reference_speed,
                width_px,
                height_px,
            )

        def _on_done(task):
            if task.is_cancelled():
                # Superseded by a newer recompute; let that one's
                # completion be the one that updates the widget.
                return
            try:
                surface = task.result()
            except Exception:
                logger.exception("Burn preview rendering failed")
                return
            self.burn_preview.update_surface(surface)

        self.editor.task_manager.run_thread(
            _compute,
            ops_list,
            key=f"image-engrave-burn-preview:{self.step.uid}",
            when_done=_on_done,
        )
        return False

    def _cleanup(self):
        super()._cleanup()
        self.editor.pipeline.workpiece_artifact_ready.disconnect(
            self._on_workpiece_artifact_ready
        )
