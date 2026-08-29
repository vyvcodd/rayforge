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
from gi.repository import Adw, GLib, Gtk

from rayforge.core.varset import VarSet
from rayforge.image.burn_preview import Segment, render_to_surface
from rayforge.image.ops_burn_preview import segments_from_ops
from rayforge.machine.models.laser import LaserHead
from rayforge.ui_gtk.shared.burn_preview_widget import BurnPreviewWidget

from ..levels_range_var import LevelsRangeVar
from ..power_curve_var import PowerCurveVar
from ..power_level_boundaries_var import PowerLevelBoundariesVar
from .laser_step_page import LaserStepSettingsPage
from .levels_adapter import LevelsAdapter
from .power_curve_adapter import PowerCurveAdapter
from .power_level_boundaries_adapter import PowerLevelBoundariesAdapter
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
    "curve_points",
    "min_power_level",
    "max_power_level",
    "num_power_levels",
    "power_level_boundaries",
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
        # The histogram/levels and power-curve controls are rendered
        # in the burn-preview side panel instead (see
        # _attach_side_panel), wider and next to the preview they
        # describe -- swap in copies of these vars with visible_when
        # forced to False rather than dropping them, so their step
        # attributes keep flowing through the normal varset get/set/
        # adapter machinery. visible_when has no setter (Var exposes
        # it read-only), so this rebuilds the vars instead of
        # mutating the originals.
        def _hidden(v):
            if v.key == "black_point":
                return LevelsRangeVar(visible_when=lambda values: False)
            if v.key == "curve_points":
                return PowerCurveVar(visible_when=lambda values: False)
            if v.key == "power_level_boundaries":
                return PowerLevelBoundariesVar(
                    visible_when=lambda values: False
                )
            return v

        power_vs = VarSet(
            vars=[
                _hidden(v) for v in step_vars if v.key in _POWER_KEYS
            ]
        )
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
        self.levels_adapter = (
            levels if isinstance(levels, LevelsAdapter) else None
        )
        if self.levels_adapter is not None:
            self.levels_adapter.set_histogram_source(step)
            GLib.idle_add(self.levels_adapter.compute_histogram)

        curve = self.power_widget.adapter_for("curve_points")
        self.curve_adapter = (
            curve if isinstance(curve, PowerCurveAdapter) else None
        )

        bands = self.power_widget.adapter_for("power_level_boundaries")
        self.bands_adapter = (
            bands if isinstance(bands, PowerLevelBoundariesAdapter) else None
        )
        if self.bands_adapter is not None:
            self.bands_adapter.set_histogram_source(step)
            GLib.idle_add(self.bands_adapter.compute_raw_samples)

        self._push_head_defaults()

        # --- Burn Preview (handed to the dialog's side panel, not
        # added as a settings row -- see StepSettingsDialog.set_side_panel) ---
        # Debug toggles for the two effects that make a single-sweep
        # preview harder to read than the job actually is -- see
        # render_segments's docstring. Default on (prior behavior);
        # the panel switches let both be checked independently.
        self._preview_cap_darkness = True
        self._preview_blend_multiply = True
        self._preview_subdivide_dot = False
        # Manual override for the render resolution's zoom factor (see
        # _on_rerender_clicked): 1.0 = the normal fixed _DOTS_PER_SPOT
        # resolution. Deliberately not automatic/tied to scroll events
        # -- re-rendering is the expensive background compute, and
        # doing that on every zoom tick is exactly the kind of
        # unbounded-cost-on-a-continuous-gesture mistake the debounced
        # downsample fix above had to walk back. A button lets this be
        # validated cheaply before committing to that.
        self._preview_render_zoom_boost = 1.0
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
            preview_box = Gtk.Box(
                orientation=Gtk.Orientation.VERTICAL,
                margin_start=12,
                margin_end=12,
                margin_top=12,
                margin_bottom=12,
            )
            rest_box = Gtk.Box(
                orientation=Gtk.Orientation.VERTICAL,
                margin_start=12,
                margin_end=12,
                margin_top=12,
                margin_bottom=12,
            )
            panel = preview_box
            header_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
            header_row.append(
                Gtk.Label(
                    label=_("Burn Preview"),
                    halign=Gtk.Align.START,
                    hexpand=True,
                    css_classes=["title-4"],
                )
            )
            rerender_button = Gtk.Button.new_from_icon_name(
                "view-refresh-symbolic"
            )
            rerender_button.set_tooltip_text(
                _(
                    "Re-render at the current zoom level -- the "
                    "preview's native resolution is fixed at render "
                    "time (just enough to show gaps/overlap, not to "
                    "resolve individual dots), so scrolling in past "
                    "1:1 only magnifies those same pixels rather than "
                    "showing real detail. This re-renders from scratch "
                    "at a resolution matched to your current zoom."
                )
            )
            rerender_button.connect("clicked", self._on_rerender_clicked)
            header_row.append(rerender_button)
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
            panel = rest_box

            if self.levels_adapter is not None:
                panel.append(
                    Gtk.Separator(margin_top=6, margin_bottom=6)
                )
                panel.append(
                    Gtk.Label(
                        label=_("Levels"),
                        halign=Gtk.Align.START,
                        css_classes=["title-4"],
                    )
                )
                levels_subtitle = Gtk.Label(
                    halign=Gtk.Align.START,
                    css_classes=["dim-label", "caption"],
                    wrap=True,
                )
                panel.append(levels_subtitle)
                levels_preview = self.levels_adapter.detach_preview()
                levels_preview.set_hexpand(True)
                levels_preview.set_content_height(110)
                panel.append(levels_preview)
                self.levels_adapter.set_subtitle_label(levels_subtitle)

            if self.curve_adapter is not None:
                panel.append(
                    Gtk.Separator(margin_top=6, margin_bottom=6)
                )
                panel.append(
                    Gtk.Label(
                        label=_("Power Curve"),
                        halign=Gtk.Align.START,
                        css_classes=["title-4"],
                    )
                )
                curve_subtitle = Gtk.Label(
                    halign=Gtk.Align.START,
                    css_classes=["dim-label", "caption"],
                    wrap=True,
                )
                panel.append(curve_subtitle)
                curve_widget = self.curve_adapter.detach_curve()
                curve_widget.set_hexpand(True)
                curve_widget.set_content_height(140)
                panel.append(curve_widget)
                self.curve_adapter.set_subtitle_label(curve_subtitle)

            if self.bands_adapter is not None:
                panel.append(
                    Gtk.Separator(margin_top=6, margin_bottom=6)
                )
                panel.append(
                    Gtk.Label(
                        label=_("Power Levels"),
                        halign=Gtk.Align.START,
                        css_classes=["title-4"],
                    )
                )
                bands_subtitle = Gtk.Label(
                    halign=Gtk.Align.START,
                    css_classes=["dim-label", "caption"],
                    wrap=True,
                )
                panel.append(bands_subtitle)
                bands_widget = self.bands_adapter.detach_widget()
                bands_widget.set_hexpand(True)
                bands_widget.set_content_height(110)
                panel.append(bands_widget)
                self.bands_adapter.set_subtitle_label(bands_subtitle)

            # Debug toggles for the preview's own darkening effects
            # (see render_segments's docstring) -- off by default
            # would misrepresent the job as flatter than it is; on
            # lets you check whether either effect is actually
            # relevant to what you're looking at, by turning it off
            # and re-rendering.
            debug_list = Gtk.ListBox(
                selection_mode=Gtk.SelectionMode.NONE,
                css_classes=["boxed-list"],
                margin_top=6,
            )
            cap_row = Adw.SwitchRow(
                title=_("Cap Single-Pass Darkness"),
                subtitle=_(
                    "Limit one pass to dark gray, leaving room to "
                    "show overlapping passes as darker still"
                ),
                active=self._preview_cap_darkness,
            )
            cap_row.connect(
                "notify::active", self._on_cap_darkness_toggled
            )
            debug_list.append(cap_row)
            blend_row = Adw.SwitchRow(
                title=_("Darken Overlapping Passes"),
                subtitle=_(
                    "Multiply overlapping strokes together and "
                    "antialias edges, instead of a crisp, un-blended "
                    "paint-over"
                ),
                active=self._preview_blend_multiply,
            )
            blend_row.connect(
                "notify::active", self._on_blend_multiply_toggled
            )
            debug_list.append(blend_row)
            subdivide_row = Adw.SwitchRow(
                title=_("Subdivide Dot Width"),
                subtitle=_(
                    "Draw each row as several thin hairlines tiling "
                    "the true dot width, instead of one solid stroke "
                    "-- shows partial row-to-row overlap as a gradient "
                    "rather than an all-or-nothing band"
                ),
                active=self._preview_subdivide_dot,
            )
            subdivide_row.connect(
                "notify::active", self._on_subdivide_dot_toggled
            )
            debug_list.append(subdivide_row)
            panel.append(debug_list)

            # rest_box (Levels/Power Curve/Power Levels/debug toggles)
            # can run long -- wrap it in its own scroller so dragging
            # the preview larger doesn't just clip it.
            rest_scroller = Gtk.ScrolledWindow(
                hscrollbar_policy=Gtk.PolicyType.NEVER,
                vexpand=True,
            )
            rest_scroller.set_child(rest_box)

            # A vertical Paned between the preview and everything below
            # it, so the preview's height is a plain drag instead of a
            # fixed size -- useful when you need to see fine dot/line
            # detail that the default height crops off.
            vpaned = Gtk.Paned(orientation=Gtk.Orientation.VERTICAL)
            vpaned.set_start_child(preview_box)
            vpaned.set_resize_start_child(True)
            vpaned.set_shrink_start_child(True)
            vpaned.set_end_child(rest_scroller)
            vpaned.set_resize_end_child(True)
            vpaned.set_shrink_end_child(True)
            vpaned.set_position(420)

            set_side_panel(vpaned)
        if isinstance(root, Gtk.Window):
            root.set_default_size(950, 700)
        return False

    def _on_workpiece_artifact_ready(
        self, sender, *, step, workpiece, handle, **kwargs
    ):
        if step.uid != self.step.uid:
            return
        self._recompute_burn_preview()

    def _on_cap_darkness_toggled(self, row, pspec):
        self._preview_cap_darkness = row.get_active()
        self._recompute_burn_preview()

    def _on_blend_multiply_toggled(self, row, pspec):
        self._preview_blend_multiply = row.get_active()
        # The widget's own blit also smooths/interpolates whenever the
        # rendered surface is scaled to fit -- almost always -- which
        # reintroduces blending no matter how crisp render_to_surface
        # made the surface itself. Keep both in lockstep so the toggle
        # actually gives a literal, un-blended pixel view.
        self.burn_preview.set_smooth_scaling(self._preview_blend_multiply)
        self._recompute_burn_preview()

    def _on_subdivide_dot_toggled(self, row, pspec):
        self._preview_subdivide_dot = row.get_active()
        self._recompute_burn_preview()

    def _on_rerender_clicked(self, button):
        self._preview_render_zoom_boost = max(1.0, self.burn_preview.get_zoom())
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
        # Snapshot the debug toggles for this render: _compute runs on
        # a background thread, so it must not read self._preview_* live
        # while the user might be flipping them for the next one.
        cap_darkness = self._preview_cap_darkness
        blend_multiply = self._preview_blend_multiply
        subdivide_dot = self._preview_subdivide_dot
        render_zoom_boost = self._preview_render_zoom_boost

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
            # render_zoom_boost > 1.0 only when the user explicitly hit
            # "re-render at current zoom" -- see _on_rerender_clicked.
            # Not tied to scroll events automatically: re-rendering is
            # the expensive background compute, and doing that on
            # every zoom tick would reintroduce the same
            # continuous-gesture cost problem the downsample debounce
            # above had to fix.
            px_per_mm = (
                _DOTS_PER_SPOT / max(smallest_feature_mm, 1e-6)
            ) * render_zoom_boost
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
                cap_darkness=cap_darkness,
                blend_multiply=blend_multiply,
                subdivide_dot=subdivide_dot,
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
        self.burn_preview.cleanup()
