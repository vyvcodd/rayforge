"""
Helper functions for the assembler-based pipeline.

These functions absorb the Part-construction, image-preprocessing,
and result-wrapping logic that currently lives inside each producer's
``run()`` method, so that the stage can call raygeo assemblers
directly without needing producer class instances.
"""

from __future__ import annotations

import logging
from enum import Enum, auto
from gettext import gettext as _
from typing import (
    TYPE_CHECKING,
)

import numpy as np
import pyvips
from raygeo.geo import Geometry, Matrix
from raygeo.image.grayscale import (
    compute_auto_levels,
    normalize_grayscale,
)
from raygeo.ops.part import Part
from raygeo.ops.types import RasterMode

from ...core.vectorization_spec import TraceSpec
from ...image.dither import (
    DEFAULT_HALFTONE_ANGLE,
    DEFAULT_HALFTONE_CELL_MM,
    DitherAlgorithm,
    grayscale_to_dithered_array,
)
from ...image.tracing import trace_surface
from ...image.util.grayscale import surface_to_binary, surface_to_grayscale
from ...image.util.vips import normalize_to_rgba

if TYPE_CHECKING:
    import cairo

    from ...core.workpiece import WorkPiece

logger = logging.getLogger(__name__)


class DepthMode(Enum):
    """Rasterisation depth mode.

    Each mode controls how pixel intensity maps to laser output:

    * ``POWER_MODULATION`` — variable power proportional to darkness.
    * ``CONSTANT_POWER`` — binary mask, constant-power scan lines.
    * ``DITHER`` — error-diffusion, ordered or halftone dither to
      binary.
    * ``MULTI_PASS`` — repeated Z-stepped passes through the depth.
    """

    POWER_MODULATION = auto()
    CONSTANT_POWER = auto()
    DITHER = auto()
    MULTI_PASS = auto()

    @property
    def display_name(self) -> str:
        names = {
            DepthMode.POWER_MODULATION: _("Variable Power"),
            DepthMode.CONSTANT_POWER: _("Constant Power"),
            DepthMode.DITHER: _("Dither"),
            DepthMode.MULTI_PASS: _("Multiple Depths"),
        }
        return names[self]

    @property
    def short_name(self) -> str:
        names = {
            DepthMode.POWER_MODULATION: _("Variable"),
            DepthMode.CONSTANT_POWER: _("Constant"),
            DepthMode.DITHER: _("Dither"),
            DepthMode.MULTI_PASS: _("Multi-Pass"),
        }
        return names[self]

    @property
    def raygeo_name(self) -> str:
        """Return the string expected by the raygeo ``raster()`` call."""
        names = {
            DepthMode.POWER_MODULATION: "power_modulated",
            DepthMode.CONSTANT_POWER: "mask_scan",
            DepthMode.DITHER: "dither",
            DepthMode.MULTI_PASS: "multi_pass",
        }
        return names[self]

    @property
    def raster_mode(self) -> RasterMode:
        """Return the :class:`RasterMode` for this depth mode."""
        _raster_mode_map = {
            DepthMode.POWER_MODULATION: RasterMode.VARIABLE_POWER,
            DepthMode.CONSTANT_POWER: RasterMode.CONSTANT_POWER,
            DepthMode.DITHER: RasterMode.CONSTANT_POWER,
            DepthMode.MULTI_PASS: RasterMode.DEPTH_MAP,
        }
        return _raster_mode_map[self]


def _trace_surface_to_mm_geometry(
    surface: cairo.ImageSurface,
    workpiece: WorkPiece,
    threshold: float = 0.5,
    auto_threshold: bool = True,
    invert: bool = False,
) -> Geometry | None:
    """Trace a rendered surface into a single Geometry in mm-space.

    The traced contours come back in pixel space (Y-down, origin
    top-left).  They are transformed to mm-space (Y-up, origin
    bottom-left) at the workpiece's physical size.

    Returns ``None`` if tracing yields no contours.
    """
    spec = TraceSpec(
        threshold=threshold,
        auto_threshold=auto_threshold,
        invert=invert,
    )
    traced = trace_surface(surface, vectorization_spec=spec)
    if not traced:
        return None

    width_mm, height_mm = workpiece.size
    px_w = surface.get_width()
    px_h = surface.get_height()
    if px_w <= 0 or px_h <= 0:
        return None

    scale_x = width_mm / px_w
    scale_y = height_mm / px_h
    transform = Matrix.translation(0, height_mm) @ Matrix.scale(
        scale_x, -scale_y
    )

    merged = Geometry()
    for geo in traced:
        geo.transform(transform)
        merged.extend(geo)
    return merged


def build_part_vector(
    workpiece: WorkPiece,
    surface: cairo.ImageSurface | None = None,
    *,
    override_threshold: bool = False,
    threshold: float = 0.5,
    normalize_windings: bool = False,
) -> Part | None:
    """Build a ``Part`` carrying vector geometry for an assembler.

    This absorbs the Part-construction logic shared by
    ``ContourProducer``, ``FrameProducer``, ``ShrinkWrapProducer``,
    and ``WavefrontProducer``.

    Resolution order:

    1. **Vector source** — if the workpiece has boundaries and
       ``override_threshold`` is False, use ``workpiece.to_part()``.
       When ``normalize_windings`` is True (e.g. WavefrontStep),
       the geometry is re-scaled manually so that
       ``normalize_winding_orders()`` can be applied before
       constructing the Part.

    2. **Raster fallback** — if a ``surface`` is available (either
       because there are no boundaries or because
       ``override_threshold`` is True), trace the surface into
       mm-space geometry and build a Part from that.

    3. Returns ``None`` if no geometry could be obtained.

    Args:
        workpiece: The WorkPiece to derive geometry from.
        surface: Optional rendered Cairo surface for raster tracing.
        override_threshold: If True, ignore vector boundaries and
            trace the surface instead.
        threshold: Brightness threshold (0–1) for raster tracing.
        normalize_windings: If True, normalize winding orders
            (outer CCW, inner CW) on the scaled geometry.  Required
            by wavefront/pocketing assemblers.
    """
    boundaries = workpiece.boundaries
    has_vector_source = boundaries is not None and not boundaries.is_empty()

    # 1. Vector source — preferred path.
    if has_vector_source and not override_threshold:
        if normalize_windings:
            assert boundaries is not None
            scaled = boundaries.copy()
            w, h = workpiece.size
            if w > 0 and h > 0:
                scaled.transform(Matrix.scale(w, h))
            scaled.normalize_winding_orders()
            return Part.from_geometry_multi_face(
                geometry=scaled, size_mm=(w, h)
            )
        return workpiece.to_part()

    # 2. Raster fallback — trace the surface.
    if surface is not None:
        geo = _trace_surface_to_mm_geometry(
            surface,
            workpiece,
            threshold=threshold,
            auto_threshold=not override_threshold,
        )
        if geo is not None and not geo.is_empty():
            if normalize_windings:
                geo.normalize_winding_orders()
                return Part.from_geometry_multi_face(
                    geometry=geo, size_mm=workpiece.size
                )
            return Part(geometry=geo, size_mm=workpiece.size)

    return None


MAX_VECTOR_TRACE_PIXELS = 16 * 1024 * 1024

# Dithering's minimum-feature floor must clear apply_dot_width_trim's
# deletion threshold (2 * dot_width_correction_mm) with room to spare --
# sizing it at exactly that threshold means many runs (it's a hard
# floor, so lots land right on it) sit exactly on trim's "delete
# outright" boundary instead of surviving with something left to
# gracefully shrink. This margin (in samples, not mm) is that room.
_DOT_WIDTH_TRIM_MARGIN_PX = 2


def build_part_vector_with_raster_fallback(
    workpiece: WorkPiece,
    pixels_per_mm: tuple[float, float],
    *,
    override_threshold: bool = False,
    threshold: float = 0.5,
    normalize_windings: bool = False,
) -> Part:
    """Build a vector :class:`Part`, rendering the workpiece source to a
    raster surface and tracing it when no vector boundaries are
    available.

    This mirrors the old ``_execute_vector`` pipeline path that
    fell back to render-and-trace when a workpiece had no boundaries
    (e.g. an SVG whose ``pristine_geometry`` is empty).

    Returns a :class:`Part` with at least ``size_mm`` set.  The
    geometry may be empty (``None``) if neither the vector source
    nor the raster trace yields any contours.
    """
    boundaries = workpiece.boundaries
    has_vector = boundaries is not None and not boundaries.is_empty()
    if has_vector and not override_threshold:
        part = build_part_vector(
            workpiece,
            surface=None,
            override_threshold=False,
            normalize_windings=normalize_windings,
        )
        if part is not None and part.has_geometry():
            return part

    size_mm = workpiece.size
    if not size_mm or size_mm[0] <= 0 or size_mm[1] <= 0:
        return Part(size_mm=size_mm)

    target_w = int(size_mm[0] * pixels_per_mm[0])
    target_h = int(size_mm[1] * pixels_per_mm[1])
    num_pixels = target_w * target_h
    if num_pixels > MAX_VECTOR_TRACE_PIXELS:
        scale = (MAX_VECTOR_TRACE_PIXELS / num_pixels) ** 0.5
        target_w = int(target_w * scale)
        target_h = int(target_h * scale)

    if target_w <= 0 or target_h <= 0:
        return Part(size_mm=size_mm)

    surface = workpiece.render_to_pixels(target_w, target_h)
    if surface is None:
        return Part(size_mm=size_mm)

    part = build_part_vector(
        workpiece,
        surface=surface,
        override_threshold=True,
        threshold=threshold,
        normalize_windings=normalize_windings,
    )
    if part is not None:
        return part
    return Part(size_mm=size_mm)


def preprocess_raster_image(
    surface: cairo.ImageSurface,
    *,
    mode: DepthMode,
    auto_levels: bool = True,
    computed_auto_levels: tuple[int, int] | None = None,
    black_point: int = 0,
    white_point: int = 255,
    curve_points: list[tuple[float, float]] | None = None,
    threshold: int = 128,
    dither_algorithm: DitherAlgorithm | None = None,
    laser_spot_x_mm: float = 0.1,
    pixels_per_mm_x: float = 1.0,
    pixels_per_mm_y: float | None = None,
    dither_serpentine: bool = False,
    halftone_cell_mm: float = DEFAULT_HALFTONE_CELL_MM,
    halftone_angle: float = DEFAULT_HALFTONE_ANGLE,
    dot_width_correction_mm: float | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Convert a Cairo surface into an image array for a raster assembler.

    Handles depth-mode preprocessing: grayscale with optional levels,
    dithering, or binary thresholding. Levels (auto or manual
    black/white points) apply to the grayscale modes and to dithering,
    where they shape the input brightness before the dither.

    Args:
        surface: The rendered Cairo ARGB32 surface.
        mode: The rasterisation depth mode.
        auto_levels: If True, auto-compute black/white points from
            the image histogram.
        computed_auto_levels: Pre-computed ``(black, white)`` points
            from a low-resolution preview (e.g. from
            ``compute_raster_auto_levels``).  When provided, skips
            per-chunk auto-level computation.
        black_point: Manual black-point (0–255) when
            ``auto_levels`` is False.
        white_point: Manual white-point (0–255) when
            ``auto_levels`` is False.
        curve_points: Piecewise-linear (input, output) control points
            in 0..1, applied as a midtone curve after the black/white
            levels stretch without moving the black or white points.
            ``None``/the two-point identity ``[(0, 0), (1, 1)]``
            means no correction.
        threshold: Brightness threshold (0–255) for binary
            ``CONSTANT_POWER`` mode.
        dither_algorithm: Dithering algorithm for ``DITHER`` mode.
        laser_spot_x_mm: Laser spot X diameter in mm, used to
            compute minimum feature size for dithering.
        pixels_per_mm_x: Rendered pixels-per-mm in X, used with
            ``laser_spot_x_mm`` for dither minimum feature size.
        pixels_per_mm_y: Rendered pixels-per-mm in Y; defaults to
            ``pixels_per_mm_x``. Screens use both to keep their cells
            square in millimetres.
        dither_serpentine: Alternate the scan direction per row for
            the error-diffusion algorithms.
        halftone_cell_mm: Dot spacing of the halftone screen in mm.
        halftone_angle: Halftone screen angle in degrees.
        dot_width_correction_mm: The dot-width correction the raster
            assembler will apply downstream (Rust-side
            ``apply_dot_width_trim``, which deletes an entire dithered
            run outright once its length drops to ``2 *
            dot_width_correction_mm`` or below, rather than shrinking
            it gracefully). Folded into the dither minimum-feature-size
            floor below, with margin (see ``_DOT_WIDTH_TRIM_MARGIN_PX``),
            so every run dithering produces clears trim's deletion
            boundary and only ever gets its edges shrunk, never deleted
            outright. ``None``/``0`` leaves the floor at just the
            physical spot size, as before.

    Returns:
        ``(image, alpha)`` where *image* is a 2-D ``uint8`` array
        (grayscale or binary depending on mode) and *alpha* is a
        ``float32`` alpha array (grayscale modes) or ``None``
        (dither / constant-power modes).
    """
    if mode in (DepthMode.MULTI_PASS, DepthMode.POWER_MODULATION):
        gray_image, alpha = surface_to_grayscale(surface)
        gray_image = _apply_raster_levels(
            gray_image,
            alpha,
            auto_levels=auto_levels,
            computed_auto_levels=computed_auto_levels,
            black_point=black_point,
            white_point=white_point,
            curve_points=curve_points,
        )
        return gray_image, alpha

    if mode == DepthMode.DITHER:
        # A dithered run only survives apply_dot_width_trim if it's
        # longer than 2 * dot_width_correction_mm (shorter runs are
        # deleted outright, not shrunk -- see raygeo's
        # apply_dot_width_trim). Fold that into the minimum feature
        # size dithering itself respects, with margin (see
        # _DOT_WIDTH_TRIM_MARGIN_PX), so a run that clears dithering's
        # floor also clears trim's deletion boundary -- trim then only
        # ever shrinks a surviving run's edges, never deletes it
        # outright.
        spot_min_px = int(round(laser_spot_x_mm * pixels_per_mm_x))
        dot_width_min_px = 0
        if dot_width_correction_mm:
            dot_width_min_px = (
                int(round(2 * dot_width_correction_mm * pixels_per_mm_x))
                + _DOT_WIDTH_TRIM_MARGIN_PX
            )
        min_feature_px = max(1, spot_min_px, dot_width_min_px)
        algo = dither_algorithm or DitherAlgorithm.FLOYD_STEINBERG
        gray_image, alpha = surface_to_grayscale(surface)
        gray_image = _apply_raster_levels(
            gray_image,
            alpha,
            auto_levels=auto_levels,
            computed_auto_levels=computed_auto_levels,
            black_point=black_point,
            white_point=white_point,
            curve_points=curve_points,
        )
        image = grayscale_to_dithered_array(
            gray_image,
            algo,
            min_feature_px=min_feature_px,
            serpentine=dither_serpentine,
            halftone_cell_mm=halftone_cell_mm,
            halftone_angle=halftone_angle,
            pixels_per_mm=(
                pixels_per_mm_x,
                pixels_per_mm_y or pixels_per_mm_x,
            ),
        )
        return image, None

    if mode == DepthMode.CONSTANT_POWER:
        image = surface_to_binary(surface, threshold=threshold)
        return image, None

    return None, None


def preprocess_raster_image_vips(
    vips_image: "pyvips.Image",
    *,
    mode: DepthMode,
    invert: bool = False,
    auto_levels: bool = True,
    computed_auto_levels: tuple[int, int] | None = None,
    black_point: int = 0,
    white_point: int = 255,
    curve_points: list[tuple[float, float]] | None = None,
    threshold: int = 128,
) -> tuple["pyvips.Image", "pyvips.Image | None"]:
    """Streaming equivalent of :func:`preprocess_raster_image` for the
    modes that carry no cross-pixel state (everything except
    ``DITHER`` -- Floyd-Steinberg's error diffusion has no chunked
    implementation available, see :func:`_build_raster_part`'s
    docstring).

    Builds the result as a graph of lazy vips operations instead of a
    fully-materialized numpy array, so it scales to a workpiece of any
    size without :data:`MAX_RASTER_RENDER_PIXELS`: the caller wraps
    the returned image in a ``VipsChunkSource``, which raygeo reads
    back in horizontal slabs on demand rather than requiring one
    giant buffer up front.

    ``auto_levels=True`` requires ``computed_auto_levels`` -- unlike
    :func:`preprocess_raster_image`, this has no fallback that
    computes levels from the (potentially huge) full image; callers
    always have a small-preview auto-levels result available (see
    ``compute_raster_auto_levels``) since that's already how the
    existing capped path keeps levels consistent across chunks.

    Returns ``(image, alpha)`` where *image* is a single-band uchar
    vips image (grayscale for MULTI_PASS/POWER_MODULATION, binary
    0/1 for CONSTANT_POWER) and *alpha* is a single-band uchar vips
    image, or ``None`` when the source has no real transparency --
    skipping alpha avoids materializing a same-size second buffer for
    the common case of a fully-opaque workpiece.
    """
    rgba = normalize_to_rgba(vips_image)
    if rgba is None:
        raise ValueError("Failed to normalize vips image to RGBA")

    alpha_band = rgba[3]
    # A global stat, not a materialize: vips computes this by streaming
    # through the pipeline without ever holding the full image at once.
    has_transparency = bool(alpha_band.min() < 255)
    alpha = alpha_band if has_transparency else None

    gray = rgba[0:3].colourspace("b-w")

    if mode == DepthMode.CONSTANT_POWER:
        # Matches raygeo's rgba_to_binary: burn (1) when strictly past
        # threshold on the appropriate side; the threshold value itself
        # never burns, in either direction. vips relational ops yield
        # 0/255, not 0/1 -- ifthenelse maps that to the 0/1 convention
        # the assembler expects (a bare cast would keep 0/255).
        mask = gray > threshold if invert else gray < threshold
        image = mask.ifthenelse(1, 0).cast("uchar")
        return image, alpha

    # MULTI_PASS / POWER_MODULATION.
    if invert:
        gray = 255 - gray

    if auto_levels:
        if computed_auto_levels is None:
            raise ValueError(
                "preprocess_raster_image_vips requires "
                "computed_auto_levels when auto_levels=True"
            )
        bp, wp = computed_auto_levels
    else:
        bp, wp = black_point, white_point

    if bp > 0 or wp < 255:
        # round-then-cast, not a bare truncating cast, to track
        # normalize_grayscale's rounding (still within +/-1 of it at
        # a handful of exact half-integer boundaries -- an
        # imperceptible ~0.4% power difference, not worth chasing
        # bit-exactness against an undocumented internal rounding
        # mode in a separate compiled dependency).
        scale = 255.0 / max(wp - bp, 1)
        offset = -bp * scale
        gray = gray.linear(scale, offset).round("rint").cast("uchar")

    if curve_points and curve_points != [(0.0, 0.0), (1.0, 1.0)]:
        # Midtone curve after the levels stretch, as a LUT applied via
        # vips's own maplut op -- matches the numpy path's
        # _curve_to_lut()/fancy-indexing exactly, just via vips's
        # native LUT machinery instead of a Python array index.
        lut_arr = _curve_to_lut(curve_points)
        lut_img = pyvips.Image.new_from_array([list(lut_arr)])
        gray = gray.maplut(lut_img).cast("uchar")

    return gray, alpha


def _apply_raster_levels(
    gray_image: np.ndarray,
    alpha: np.ndarray,
    *,
    auto_levels: bool = True,
    computed_auto_levels: tuple[int, int] | None = None,
    black_point: int = 0,
    white_point: int = 255,
    curve_points: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    """Apply auto-levels or manual black/white-point normalization,
    then a midtone curve.

    This is the standalone equivalent of ``Rasterizer._apply_levels``.
    """
    if auto_levels:
        if computed_auto_levels is not None:
            bp, wp = computed_auto_levels
        else:
            bp, wp = compute_auto_levels(gray_image[alpha > 0])
    else:
        bp, wp = black_point, white_point

    if bp > 0 or wp < 255:
        gray_image = normalize_grayscale(gray_image, bp, wp)
    if curve_points and curve_points != [(0.0, 0.0), (1.0, 1.0)]:
        gray_image = _curve_to_lut(curve_points)[gray_image]
    return gray_image


def _curve_to_lut(curve_points: list[tuple[float, float]]) -> np.ndarray:
    """A 256-entry uint8 lookup table from piecewise-linear (input,
    output) control points in 0..1, applied after the black/white
    levels stretch so it reshapes the already-normalized 0..255 range
    without moving the black or white points themselves. Mirrors
    :meth:`CurveWidget.to_lut` exactly, so what's drawn in the UI is
    what actually gets applied.
    """
    points = sorted(curve_points, key=lambda p: p[0])
    xs = np.array([p[0] for p in points]) * 255.0
    ys = np.array([p[1] for p in points]) * 255.0
    lut = np.interp(np.arange(256), xs, ys)
    return np.clip(np.round(lut), 0, 255).astype(np.uint8)


def compute_raster_auto_levels(
    workpiece: WorkPiece,
    pixels_per_mm: tuple[float, float],
    *,
    max_preview_pixels: int = 512,
) -> tuple[int, int] | None:
    """Compute auto-levels from a low-resolution preview render.

    Renders a small preview of the workpiece, converts it to
    grayscale, and returns the ``(black_point, white_point)`` tuple
    suitable for passing to ``preprocess_raster_image`` as
    ``computed_auto_levels``.

    This ensures consistent black/white points across all chunks
    when processing large images.

    Args:
        workpiece: The WorkPiece to render a preview of.
        pixels_per_mm: The ``(x, y)`` resolution of the full render.
        max_preview_pixels: Maximum preview dimension in pixels.

    Returns:
        ``(black_point, white_point)`` tuple, or ``None`` if the
        preview render failed or auto-levels are not applicable.
    """
    px_per_mm_x, px_per_mm_y = pixels_per_mm
    size = workpiece.size

    scale = min(
        1.0,
        max_preview_pixels / (size[0] * px_per_mm_x),
        max_preview_pixels / (size[1] * px_per_mm_y),
    )

    preview_width = max(1, int(size[0] * px_per_mm_x * scale))
    preview_height = max(1, int(size[1] * px_per_mm_y * scale))

    surface = workpiece.render_to_pixels(preview_width, preview_height)
    if not surface:
        return None

    gray_image, alpha = surface_to_grayscale(surface)

    surface.flush()

    return compute_auto_levels(gray_image[alpha > 0])
