from __future__ import annotations

import logging
from typing import ClassVar

from raygeo.geo import Geometry
from raygeo.svg import (
    svg_string_to_geometries,
    svg_string_to_geometries_by_color,
    svg_string_to_geometry_by_color,
    svg_string_to_geometry_by_layer,
)
from raygeo.svg.color import ColorAttr

from ...core.vectorization_spec import (
    LayerSource,
    PassthroughSpec,
    VectorizationSpec,
)
from ..base_importer import ImporterFeature
from ..engine import NormalizationEngine
from ..structures import (
    LayerGeometry,
    ParsingResult,
    VectorizationResult,
)
from .svg_base import SvgImporterBase
from .svgutil import NO_COLOR_KEY, dedupe_layer_ids, extract_layer_manifest

logger = logging.getLogger(__name__)


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
        else:
            layer_geometries = self._layer_geometries_by_svg(svg_str)

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

        return ParsingResult(
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
