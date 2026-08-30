from __future__ import annotations

import base64
import logging
import re
import warnings
from gettext import gettext as _
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips

from raygeo.geo import Geometry, Matrix
from raygeo.svg import (
    svg_string_to_geometries,
    svg_string_to_geometries_by_color,
    svg_string_to_geometry_by_color,
    svg_string_to_geometry_by_layer,
    svg_string_to_images_by_layer,
)
from raygeo.svg.color import ColorAttr

from ...core.layer import Layer
from ...core.source_asset import SourceAsset
from ...core.source_asset_segment import SourceAssetSegment
from ...core.vectorization_spec import (
    LayerImportMode,
    LayerSource,
    PassthroughSpec,
    TraceSpec,
    VectorizationSpec,
)
from ...core.workpiece import WorkPiece
from .. import util
from ..base_importer import ImporterFeature
from ..engine import NormalizationEngine
from ..jpg.renderer import JPG_RENDERER
from ..png.renderer import PNG_RENDERER
from ..structures import (
    ImportPayload,
    LayerGeometry,
    ParsingResult,
    VectorizationResult,
)
from .svg_base import SvgImporterBase
from .svgutil import NO_COLOR_KEY, dedupe_layer_ids, extract_layer_manifest

if TYPE_CHECKING:
    from ...core.item import DocItem

logger = logging.getLogger(__name__)

# RFC 2397 data URI, e.g. "data:image/png;base64,iVBORw0K...". Only
# base64-encoded data URIs are handled -- an external href (a real
# file path/URL) or a #fragment reference to a <defs> element isn't,
# since embedding is by far the common case (e.g. Inkscape's "embed
# image") and everything else would need network/filesystem access
# this importer has no business doing.
_DATA_URI_RE = re.compile(
    r"^data:(?P<mime>[^;,]+)?;base64,(?P<data>.*)$", re.DOTALL
)

# Embedded-image formats this importer knows how to decode into a
# real raster WorkPiece. Anything else (SVG's own <image> format
# support is broader, e.g. GIF) is skipped with a warning rather than
# guessed at.
_RENDERER_BY_MIME = {
    "image/png": PNG_RENDERER,
    "image/jpeg": JPG_RENDERER,
    "image/jpg": JPG_RENDERER,
}


def _decode_data_uri(href: str) -> tuple[str, bytes] | None:
    """``(mime_type, raw_bytes)`` from a base64 data URI, or ``None``
    if `href` isn't one (see ``_DATA_URI_RE``'s docstring)."""
    m = _DATA_URI_RE.match(href.strip())
    if not m:
        return None
    mime = (m.group("mime") or "").strip().lower()
    try:
        raw = base64.b64decode(m.group("data"))
    except (ValueError, base64.binascii.Error):
        return None
    return mime, raw


def _native_point_to_world_mm(
    point: tuple[float, float], parse_result: ParsingResult
) -> tuple[float, float]:
    """Converts one Native-space point (e.g. a corner of an embedded
    <image>) to World mm, Y-up.

    This is the per-point generalization of the per-bounds-rect formula
    NormalizationEngine.calculate_layout_item uses for pos_y_mm (which
    only ever handles the bottom of a bounding box, not an arbitrary
    point) -- needed here because an embedded image's placement can be
    rotated, so its four corners don't reduce to a single axis-aligned
    bounding box the way calculate_layout_item expects.
    """
    x_native, y_native = point
    unit_to_mm = parse_result.native_unit_to_mm
    x_mm = x_native * unit_to_mm
    ref_bounds = (
        parse_result.untrimmed_document_bounds
        or parse_result.document_bounds
    )
    _ref_x, ref_y, _ref_w, ref_h = ref_bounds
    if parse_result.is_y_down:
        y_mm = ((ref_y + ref_h) - y_native) * unit_to_mm
    else:
        y_mm = y_native * unit_to_mm
    return (x_mm, y_mm)


def _dedupe_layer_ids(
    layers_raw: list[tuple[str, Geometry]],
) -> list[tuple[str, Geometry]]:
    """Disambiguate duplicate layer ids from raygeo's per-layer output.

    ``_layer_geometries_by_svg`` builds a *list* entry per (layer_id,
    geo) pair -- duplicates included -- but ``vectorize()`` keys its
    geometry lookup by layer_id in a plain dict, so a repeated id would
    otherwise make every occurrence after the first silently overwrite
    the previous one's geometry: multiple doc items still get created
    (one per list entry, each with its own position), but they'd all
    render whichever occurrence came last.

    Delegates the actual disambiguation to
    ``svgutil.dedupe_layer_ids`` -- see its docstring -- so this and
    ``extract_layer_manifest``'s manifest (built independently, via a
    separate Python XML scan rather than raygeo) agree on ids for the
    same document.
    """
    raw_ids = [raw_id for raw_id, _ in layers_raw]
    deduped_ids = dedupe_layer_ids(raw_ids)
    return [
        (deduped_id, geo)
        for deduped_id, (_, geo) in zip(deduped_ids, layers_raw)
    ]


class SvgVectorImporter(SvgImporterBase):
    """
    Imports SVG files by parsing vector paths directly.
    """

    label = "SVG (Vector Strategy)"
    mime_types = ()
    extensions = ()
    features: ClassVar[set[ImporterFeature]] = {
        ImporterFeature.DIRECT_VECTOR,
        ImporterFeature.LAYER_SELECTION,
        ImporterFeature.COLOR_LAYERS,
    }

    def __init__(self, data: bytes, source_file: Path | None = None):
        super().__init__(data, source_file)
        # Populated by parse(), consumed by _post_process_payload() --
        # bridges data between the two since _post_process_payload only
        # receives the assembled payload, not parse()'s own result.
        # Mirrors SvgTraceImporter.traced_artefacts.
        self._embedded_images: list[dict[str, Any]] = []
        self._image_parse_result: ParsingResult | None = None

    def parse(self) -> ParsingResult | None:
        # 1. Use base class to get dimensions, units, and the parsed SVG object
        basics = self._calculate_parsing_basics()
        if not basics:
            return None

        # Unpack. Both trimmed and untrimmed bounds are now available.
        (
            document_bounds,
            unit_to_mm,
            untrimmed_document_bounds,
            world_frame,
        ) = basics

        # 2. Extract layer geometry from trimmed data.
        assert self.trimmed_data is not None
        svg_str = self.trimmed_data.decode("utf-8")
        spec = self._vectorization_spec
        if (
            isinstance(spec, PassthroughSpec)
            and spec.layer_source == LayerSource.COLORS
        ):
            use_color_layers = True
            color_attr = spec.color_attr
        else:
            use_color_layers = False
            color_attr = ColorAttr.ANY

        # 3. Build layer geometries with names from manifest.
        if use_color_layers:
            layer_geometries = self._layer_geometries_by_color(
                svg_str, color_attr
            )
            self._embedded_images = []
        else:
            layer_geometries = self._layer_geometries_by_svg(svg_str)
            # Embedded <image> elements have no "color" -- only
            # meaningful to extract in the by-layer path.
            self._embedded_images = self._extract_embedded_images(svg_str)

        # Create temporary result to calculate background transform
        temp_result = ParsingResult(
            document_bounds=document_bounds,
            native_unit_to_mm=unit_to_mm,
            is_y_down=True,
            layers=[],
            untrimmed_document_bounds=untrimmed_document_bounds,
            world_frame_of_reference=world_frame,
            background_world_transform=None,  # type: ignore
        )

        bg_item = NormalizationEngine.calculate_layout_item(
            document_bounds, temp_result
        )

        result = ParsingResult(
            document_bounds=document_bounds,
            native_unit_to_mm=unit_to_mm,
            is_y_down=True,
            layers=layer_geometries,
            untrimmed_document_bounds=untrimmed_document_bounds,
            geometry_is_relative_to_bounds=False,
            is_cropped_to_content=True,
            world_frame_of_reference=world_frame,
            background_world_transform=bg_item.world_matrix,
        )
        # _post_process_payload only sees the assembled payload, not
        # this result -- stash it so per-image coordinate conversion
        # (native -> world mm) can use the exact same reference frame
        # (untrimmed_document_bounds, native_unit_to_mm) the rest of
        # this import used.
        self._image_parse_result = result
        return result

    def vectorize(
        self,
        parse_result: ParsingResult,
        spec: VectorizationSpec,
    ) -> VectorizationResult:
        """
        Extracts vector geometry from SVG for direct import.
        """
        if not isinstance(spec, PassthroughSpec):
            spec = PassthroughSpec()

        assert self.trimmed_data is not None

        svg_str = self.trimmed_data.decode("utf-8")
        all_layer_ids = [layer.layer_id for layer in parse_result.layers]

        target_layer_ids = (
            spec.active_layer_ids
            if spec.active_layer_ids is not None
            else all_layer_ids
        )

        geometries_by_layer: dict[str | None, Geometry] = {}
        # Whether raygeo found *any* layer-structured geometry anywhere
        # in the document, before filtering down to target_layer_ids --
        # distinct from geometries_by_layer ending up empty, which can
        # also happen when every *selected* layer is legitimately
        # geometry-less (e.g. a layer holding only a raster <image>,
        # which this vector-only strategy can't extract paths from).
        # Falling back to "parse the whole SVG" on that second
        # condition instead of the first ignores the selection
        # entirely and silently substitutes other layers' content.
        document_has_layer_geometry = False
        if spec.layer_source == LayerSource.COLORS:
            # Extract one merged geometry per resolved color.
            buckets_raw = svg_string_to_geometry_by_color(
                svg_str, 1.0, 1.0, spec.color_attr
            )
            document_has_layer_geometry = any(
                not geo.is_empty() for _, geo in buckets_raw
            )
            for color_key, geo in buckets_raw:
                if color_key in target_layer_ids and not geo.is_empty():
                    geometries_by_layer[color_key] = geo
        else:
            # Extract per-layer geometries via raygeo (already in user space).
            layers_raw = _dedupe_layer_ids(
                svg_string_to_geometry_by_layer(svg_str, 1.0, 1.0)
            )
            document_has_layer_geometry = any(
                not geo.is_empty() for _, geo in layers_raw
            )
            for layer_id, geo in layers_raw:
                if layer_id in target_layer_ids and not geo.is_empty():
                    geometries_by_layer[layer_id] = geo

        # Only fall back to "parse the whole SVG" when the document has
        # no layer structure at all (e.g. a flat SVG with no <g id>
        # groups). If it does have layers but none of the *selected*
        # ones produced geometry, leave geometries_by_layer empty
        # rather than substituting unrelated layers' content.
        if not geometries_by_layer and not document_has_layer_geometry:
            logger.debug(
                "No layer-specific geometry found, parsing entire SVG as "
                "default."
            )
            geos = svg_string_to_geometries(svg_str, 1.0, 1.0)
            if geos:
                geo = Geometry()
                for g in geos:
                    geo.extend(g)
                if not geo.is_empty():
                    geometries_by_layer[None] = geo

        return VectorizationResult(
            geometries_by_layer=geometries_by_layer,
            source_parse_result=parse_result,
        )

    def _layer_geometries_by_svg(self, svg_str: str) -> list[LayerGeometry]:
        """Builds LayerGeometry entries from top-level SVG layer groups."""
        layers_raw = _dedupe_layer_ids(
            svg_string_to_geometry_by_layer(svg_str, 1.0, 1.0)
        )
        assert self.trimmed_data is not None
        layer_manifest = extract_layer_manifest(self.trimmed_data)
        layer_names_by_id = {
            layer["id"]: layer["name"] for layer in layer_manifest
        }

        layer_geometries: list[LayerGeometry] = []
        for layer_id, geo in layers_raw:
            if not geo.is_empty():
                # layer_names_by_id is keyed by the raw (pre-dedupe) id
                # from the manifest scan, so look up by that -- then
                # carry the disambiguating suffix, if any, into the
                # display name too instead of showing it the raw
                # "id#2" form.
                base_id, _, suffix = layer_id.partition("#")
                layer_name = layer_names_by_id.get(base_id, base_id)
                if suffix:
                    layer_name = f"{layer_name} ({suffix})"
                layer_geometries.append(
                    self._layer_geometry(layer_id, layer_name, geo)
                )
        return layer_geometries

    def _extract_embedded_images(self, svg_str: str) -> list[dict[str, Any]]:
        """Builds embedded-<image> entries (layer id/name + placement),
        keyed to the same layer ids _layer_geometries_by_svg uses.

        Note: this dedupes duplicate raw layer ids independently of
        _layer_geometries_by_svg's own dedupe pass, since a layer can
        have vector geometry, embedded images, both, or neither --
        the two raygeo calls don't necessarily see the same set of
        layer ids. A layer id duplicated *and* containing images in
        more than one of those duplicate occurrences is a rare enough
        edge case that the two passes' "#2"/"#3" numbering isn't
        guaranteed to agree in that specific case.
        """
        raw_layers = svg_string_to_images_by_layer(svg_str)
        if not raw_layers:
            return []

        deduped_ids = dedupe_layer_ids([lid for lid, _ in raw_layers])

        assert self.trimmed_data is not None
        layer_manifest = extract_layer_manifest(self.trimmed_data)
        layer_names_by_id = {
            layer["id"]: layer["name"] for layer in layer_manifest
        }

        images: list[dict[str, Any]] = []
        for layer_id, (raw_layer_id, image_refs) in zip(
            deduped_ids, raw_layers
        ):
            layer_name = layer_names_by_id.get(raw_layer_id, raw_layer_id)
            for (
                href,
                width,
                height,
                top_left,
                top_right,
                bottom_left,
            ) in image_refs:
                images.append(
                    {
                        "layer_id": layer_id,
                        "layer_name": layer_name,
                        "href": href,
                        "width": width,
                        "height": height,
                        "top_left": top_left,
                        "top_right": top_right,
                        "bottom_left": bottom_left,
                    }
                )
        return images

    def _layer_geometries_by_color(
        self, svg_str: str, color_attr: ColorAttr
    ) -> list[LayerGeometry]:
        """Builds LayerGeometry entries from resolved SVG colors."""
        buckets_raw = svg_string_to_geometries_by_color(
            svg_str, 1.0, 1.0, color_attr
        )
        layer_geometries: list[LayerGeometry] = []
        for color_key, geos in buckets_raw:
            geo = Geometry()
            for g in geos:
                geo.extend(g)
            if not geo.is_empty():
                layer_geometries.append(
                    self._layer_geometry(
                        color_key,
                        f"Color {color_key}",
                        geo,
                        color=(
                            None if color_key == NO_COLOR_KEY else color_key
                        ),
                    )
                )
        return layer_geometries

    def _layer_geometry(
        self,
        layer_id: str,
        name: str,
        geo: Geometry,
        color: str | None = None,
    ) -> LayerGeometry:
        min_x, min_y, max_x, max_y = geo.rect()
        w = max_x - min_x
        h = max_y - min_y
        return LayerGeometry(
            layer_id=layer_id,
            name=name,
            content_bounds=(min_x, min_y, w, h),
            color=color,
        )

    def _post_process_payload(self, payload: ImportPayload) -> ImportPayload:
        """Injects a raster WorkPiece for every embedded <image> covered
        by the active layer selection.

        Embedded images have no vector geometry, so they never flow
        through Phases 3-5 (Vectorize/Layout/Assemble) the way
        path-based layer content does -- VectorizationResult has no
        slot for "this is raster bytes, not Geometry". This hook is
        where they're spliced in instead, once the normal vector
        pipeline has already run (see base_importer._run_pipeline,
        which routes both its empty-plan and populated-plan exits
        through this method for exactly this reason).
        """
        if not self._embedded_images or self._image_parse_result is None:
            return payload

        spec = self._vectorization_spec
        if not isinstance(spec, PassthroughSpec):
            # A TraceSpec means bitmap tracing was requested, which
            # already flattens embedded images into its one rendered
            # bitmap (see SvgTraceImporter) -- extracting them again
            # here would just duplicate that content.
            return payload

        target_layer_ids = (
            set(spec.active_layer_ids)
            if spec.active_layer_ids is not None
            else None
        )
        wrap_in_layer = spec.layer_import_mode != LayerImportMode.FLATTEN

        for image in self._embedded_images:
            if (
                target_layer_ids is not None
                and image["layer_id"] not in target_layer_ids
            ):
                continue
            built = self._build_image_item(image, spec, wrap_in_layer)
            if built is None:
                continue
            source_asset, item = built
            payload.assets.append(source_asset)
            payload.items.append(item)

        return payload

    def _build_image_item(
        self,
        image: dict[str, Any],
        spec: PassthroughSpec,
        wrap_in_layer: bool,
    ) -> tuple[SourceAsset, "DocItem"] | None:
        decoded = _decode_data_uri(image["href"])
        if decoded is None:
            self.add_warning(
                _(
                    "Skipped an embedded image in layer '{layer}': only "
                    "embedded (base64) images are supported."
                ).format(layer=image["layer_name"])
            )
            return None

        mime, raw_bytes = decoded
        renderer = _RENDERER_BY_MIME.get(mime)
        if renderer is None:
            self.add_warning(
                _(
                    "Skipped an embedded image in layer '{layer}': "
                    "unsupported image format '{mime}'."
                ).format(layer=image["layer_name"], mime=mime or _("unknown"))
            )
            return None

        try:
            vips_image = pyvips.Image.new_from_buffer(raw_bytes, "")
        except pyvips.Error:
            self.add_warning(
                _(
                    "Skipped an embedded image in layer '{layer}': "
                    "could not decode image data."
                ).format(layer=image["layer_name"])
            )
            return None

        width_px = vips_image.width
        height_px = vips_image.height
        if width_px <= 0 or height_px <= 0:
            return None

        parse_result = self._image_parse_result
        assert parse_result is not None

        world_top_left = _native_point_to_world_mm(
            image["top_left"], parse_result
        )
        world_top_right = _native_point_to_world_mm(
            image["top_right"], parse_result
        )
        world_bottom_left = _native_point_to_world_mm(
            image["bottom_left"], parse_result
        )

        x_axis = (
            world_top_right[0] - world_top_left[0],
            world_top_right[1] - world_top_left[1],
        )
        y_axis = (
            world_top_left[0] - world_bottom_left[0],
            world_top_left[1] - world_bottom_left[1],
        )
        world_matrix = Matrix.from_list(
            [
                [x_axis[0], y_axis[0], world_bottom_left[0]],
                [x_axis[1], y_axis[1], world_bottom_left[1]],
                [0.0, 0.0, 1.0],
            ]
        )
        width_mm = (x_axis[0] ** 2 + x_axis[1] ** 2) ** 0.5
        height_mm = (y_axis[0] ** 2 + y_axis[1] ** 2) ** 0.5

        # normalization_matrix only depends on the image's own pixel
        # space (Native px, Y-down -> Unit square 0-1, Y-up), never on
        # where it ends up in the world -- reuse calculate_layout_item
        # for that half and discard its world_matrix (axis-aligned
        # only), since world_matrix above is rotation-capable and it
        # isn't.
        pixel_bounds = (0.0, 0.0, float(width_px), float(height_px))
        temp_result = ParsingResult(
            document_bounds=pixel_bounds,
            native_unit_to_mm=1.0,
            is_y_down=True,
            layers=[],
            world_frame_of_reference=pixel_bounds,
            background_world_transform=Matrix.identity(),
        )
        normalization_matrix = NormalizationEngine.calculate_layout_item(
            pixel_bounds, temp_result
        ).normalization_matrix

        metadata = util.extract_vips_metadata(vips_image)
        metadata["image_format"] = "PNG" if mime == "image/png" else "JPEG"
        natural_width_mm, natural_height_mm = util.get_physical_size_mm(
            vips_image
        )

        source_asset = SourceAsset(
            source_file=self.source_file,
            original_data=raw_bytes,
            renderer=renderer,
            metadata=metadata,
            thumbnail_data=self._render_thumbnail_from_vips(vips_image),
            width_px=width_px,
            height_px=height_px,
            width_mm=natural_width_mm,
            height_mm=natural_height_mm,
        )

        segment = SourceAssetSegment(
            source_asset_uid=source_asset.uid,
            vectorization_spec=spec,
            layer_id=image["layer_id"],
            normalization_matrix=normalization_matrix,
            crop_window_px=(0, 0, width_px, height_px),
        )
        segment.cropped_width_mm = width_mm
        segment.cropped_height_mm = height_mm

        wp = WorkPiece(name=image["layer_name"], source_segment=segment)
        wp.matrix = world_matrix
        wp.natural_width_mm = width_mm
        wp.natural_height_mm = height_mm

        item: "DocItem" = wp
        if wrap_in_layer and image["layer_id"] not in (None, "__default__"):
            layer = Layer(name=image["layer_name"])
            layer.add_child(wp)
            item = layer

        return source_asset, item
