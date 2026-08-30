import logging
import warnings
from typing import Any
from xml.etree import ElementTree as ET

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips

from raygeo.geo.types import Rect
from raygeo.svg import svg_string_to_geometries_by_color
from raygeo.svg.color import ColorAttr
from raygeo.svg.length import parse_svg_length, svg_length_to_mm
from raygeo.svg.metadata import extract_svg_metadata

from .svg_fallback import (
    SVG_LOAD_AVAILABLE,
    cairo_surface_to_vips,
    render_svg_to_cairo,
)

logger = logging.getLogger(__name__)

# A standard fallback conversion factor for pixel units. Corresponds to 96 DPI.
PPI: float = 96.0
"""Standard Pixels Per Inch, used for fallback conversions."""

MM_PER_PX: float = 25.4 / PPI
"""Conversion factor for pixels to millimeters, based on 96 PPI."""

# Raygeo bucket key for shapes whose chosen color attribute is `none` or
# unset. These shapes have no color but still contain real geometry.
NO_COLOR_KEY: str = "_no_color"

INKSCAPE_NS = "http://www.inkscape.org/namespaces/inkscape"
SVG_NS = "http://www.w3.org/2000/svg"

# Tags that represent vector geometry
SHAPE_TAGS = {
    "path",
    "rect",
    "circle",
    "ellipse",
    "line",
    "polyline",
    "polygon",
    "text",
    "image",
}


# Register namespaces to prevent ElementTree from mangling them (ns0:tags)
try:
    ET.register_namespace("", SVG_NS)
    ET.register_namespace("inkscape", INKSCAPE_NS)
    ET.register_namespace(
        "sodipodi", "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
    )
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
except ValueError:
    pass  # Best effort registration


def _get_margins_from_data(
    data: bytes,
) -> Rect:
    """
    Calculates content margins as ratios from raw SVG data using pyvips.
    Returns (left, top, right, bottom) margins as fractions of total size.
    """
    if not data:
        return 0.0, 0.0, 0.0, 0.0

    try:
        root = ET.fromstring(data)

        # 1. Get original dimensions to determine aspect ratio.
        w_str = root.get("width")
        h_str = root.get("height")
        if not w_str or not h_str:
            return 0.0, 0.0, 0.0, 0.0  # Cannot determine aspect ratio.

        orig_w, _w_unit = parse_svg_length(w_str)
        orig_h, _h_unit = parse_svg_length(h_str)
        # Normalise to numeric values (unit suffix is handled separately)
        orig_w = float(orig_w)
        orig_h = float(orig_h)

        if orig_w <= 0 or orig_h <= 0:
            return 0.0, 0.0, 0.0, 0.0

        aspect_ratio = orig_w / orig_h

        # 2. Calculate proportional dimensions for rendering.
        measurement_size = 4096.0  # Use a larger size for better precision
        if aspect_ratio > 1:  # Wider than tall
            render_w = measurement_size
            render_h = measurement_size / aspect_ratio
        else:  # Taller than wide or square
            render_h = measurement_size
            render_w = measurement_size * aspect_ratio

        # 3. Modify SVG for a large, proportional render.
        root.set("width", f"{render_w}px")
        root.set("height", f"{render_h}px")
        root.set("preserveAspectRatio", "none")

        # Create viewBox if it's missing, which is crucial for the renderer
        # to have a coordinate system.
        if not root.get("viewBox"):
            root.set("viewBox", f"0 0 {orig_w} {orig_h}")

        # Add overflow:visible to ensure all geometry, including parts
        # defined by control points outside the viewBox, is rendered for
        # accurate margin calculation.
        root.set("style", "overflow: visible")

        if SVG_LOAD_AVAILABLE:
            img = pyvips.Image.svgload_buffer(ET.tostring(root))
        else:
            surface = render_svg_to_cairo(
                ET.tostring(root), int(render_w), int(render_h)
            )
            if not surface:
                return 0.0, 0.0, 0.0, 0.0
            img = cairo_surface_to_vips(surface)
            if not img:
                return 0.0, 0.0, 0.0, 0.0

        if img.bands < 4:
            img = img.bandjoin(255)  # Ensure alpha channel for trimming

        # Create a sharp, binary mask from the alpha channel.
        alpha = img[3]
        mask = alpha > 0
        # Explicitly tell find_trim that the background color is 0.
        left, top, w, h = mask.find_trim(background=0)

        if w == 0 or h == 0:
            # No content found, so no margins
            return 0.0, 0.0, 0.0, 0.0

        # 4. Calculate margins as a ratio of the PROPORTIONAL render size.
        return (
            left / render_w,
            top / render_h,
            (render_w - (left + w)) / render_w,
            (render_h - (top + h)) / render_h,
        )
    except (pyvips.Error, ET.ParseError, ValueError):
        # Return zero margins if SVG is invalid or processing fails
        return 0.0, 0.0, 0.0, 0.0


def trim_svg(data: bytes) -> bytes:
    """
    Crops an SVG to its content by adjusting the viewBox attribute.

    This function renders the SVG to a bitmap, finds the bounding box of the
    non-transparent content, and calculates new viewBox and dimensions to
    effectively trim the empty space. The original aspect ratio of the
    content is preserved.

    Args:
        data: The raw SVG data in bytes.

    Returns:
        The raw bytes of the modified, trimmed SVG. Returns original data if
        no trimming is necessary or if the SVG is invalid.
    """
    margins = _get_margins_from_data(data)
    # If there's nothing to trim (within a small tolerance), return the
    # original data.
    if all(m < 1e-5 for m in margins):
        return data

    left, top, right, bottom = margins

    try:
        root = ET.fromstring(data)

        w_str = root.get("width")
        h_str = root.get("height")
        if not w_str or not h_str:
            return data  # Cannot proceed without dimensions

        w_val, w_unit = parse_svg_length(w_str)
        h_val, h_unit = parse_svg_length(h_str)

        vb_str = root.get("viewBox")
        if vb_str:
            vb_x, vb_y, vb_w, vb_h = map(float, vb_str.split())
        else:
            # If no viewBox, it's implicitly '0 0 width height'
            vb_x, vb_y, vb_w, vb_h = 0, 0, w_val, h_val

        # Calculate new viewBox based on margins
        new_vb_x = vb_x + (left * vb_w)
        new_vb_y = vb_y + (top * vb_h)
        new_vb_w = vb_w * (1 - left - right)
        new_vb_h = vb_h * (1 - top - bottom)

        if new_vb_w <= 0 or new_vb_h <= 0:
            return data  # Avoid creating an invalid SVG

        root.set("viewBox", f"{new_vb_x} {new_vb_y} {new_vb_w} {new_vb_h}")

        # Update width and height to reflect the trimmed size
        new_w_val = w_val * (1 - left - right)
        new_h_val = h_val * (1 - top - bottom)
        root.set("width", f"{new_w_val}{w_unit or 'px'}")
        root.set("height", f"{new_h_val}{h_unit or 'px'}")

        # This attribute forces non-proportional scaling and causes issues
        # when rendering filtered layers. It's safer to rely on librsvg's
        # default proportional scaling.
        if "preserveAspectRatio" in root.attrib:
            del root.attrib["preserveAspectRatio"]

        return ET.tostring(root)

    except (ET.ParseError, ValueError):
        return data


def is_unitless_svg(data: bytes) -> bool:
    """
    Returns True if the SVG's dimensions have no physical unit, meaning the
    physical size depends on a DPI assumption. This is the case when:

    - width/height attributes are missing (viewBox-only), or
    - width/height use no unit (bare number) or the "px" unit.
    """
    if not data:
        return False
    try:
        meta = extract_svg_metadata(data.decode("utf-8"))
        if meta.width is None and meta.height is None:
            return True
        return meta.width_unit in ("", "px") and meta.height_unit in ("", "px")
    except (ValueError, ET.ParseError):
        return False


def get_natural_size(
    data: bytes, ppi: float = PPI
) -> tuple[float, float] | None:
    """
    Analyzes raw SVG data to extract its natural, untrimmed dimensions in mm.

    Args:
        data: The raw SVG data in bytes.
        ppi: Pixels per inch used for converting unitless/px values to mm.

    Returns:
        A tuple of (width_mm, height_mm), or None if dimensions cannot be
        determined.
    """
    if not data:
        return None

    try:
        meta = extract_svg_metadata(data.decode("utf-8"))
        if meta.width is None or meta.height is None:
            # Unitless SVG (viewBox only, no width/height): derive the natural
            # size from the viewBox, treating its user units as pixels at the
            # requested ppi. Otherwise such files report no size and are not
            # picked up correctly.
            if meta.viewbox is None:
                return None
            _vb_x, _vb_y, vb_w, vb_h = meta.viewbox
            if vb_w <= 0 or vb_h <= 0:
                return None
            mm_per_px = 25.4 / ppi
            return vb_w * mm_per_px, vb_h * mm_per_px

        width_mm = svg_length_to_mm(f"{meta.width}{meta.width_unit}", dpi=ppi)
        height_mm = svg_length_to_mm(
            f"{meta.height}{meta.height_unit}", dpi=ppi
        )
        return width_mm, height_mm

    except (ValueError, ET.ParseError):
        return None


def _get_local_tag_name(element: ET.Element) -> str:
    """Robustly gets the local tag name, ignoring any namespace."""
    return element.tag.rsplit("}", 1)[-1]


def hex_color_to_rgb(color: str) -> tuple[float, float, float]:
    """
    Converts a "#rrggbb" hex color string to an RGB tuple in the 0-1 range.
    """
    value = color.lstrip("#")
    r = int(value[0:2], 16) / 255.0
    g = int(value[2:4], 16) / 255.0
    b = int(value[4:6], 16) / 255.0
    return r, g, b


def extract_color_manifest(
    data: bytes, color_attr: ColorAttr = ColorAttr.ANY
) -> list[dict[str, Any]]:
    """
    Parses the SVG to find distinct colors, treating each as a layer.

    Buckets shapes by their resolved color attribute (`fill`, `stroke`,
    `fill_else_stroke`, or `any`) using raygeo, and counts the number of
    geometric elements per color. In `any` mode a shape whose fill differs
    from its stroke lands in two buckets, one per color.

    Args:
        data: Raw SVG data in bytes.
        color_attr: The color attribute to bucket by.

    Returns:
        A list of layer dicts with keys "id", "name", "count" and "color".
    """
    if not data:
        return []

    try:
        buckets = svg_string_to_geometries_by_color(
            data.decode("utf-8"), 1.0, 1.0, color_attr
        )
    except (ValueError, ET.ParseError):
        return []

    layers = []
    for color_key, geos in buckets:
        count = len(geos)
        color = None if color_key == NO_COLOR_KEY else color_key
        layers.append(
            {
                "id": color_key,
                "name": f"Color {color_key}",
                "count": count,
                "color": color,
            }
        )
        logger.debug(f"Found color layer: ID='{color_key}', Count={count}")
    return layers


def _is_id_bearing_group(element: ET.Element) -> bool:
    return _get_local_tag_name(element) == "g" and bool(element.get("id"))


def _has_nested_layer_group(element: ET.Element) -> bool:
    """True if `element` has other id-bearing `<g>` elements directly
    inside it -- see `_collect_layer_elements`'s docstring."""
    return any(_is_id_bearing_group(child) for child in element)


def _collect_layer_elements(root: ET.Element) -> list[ET.Element]:
    """Recursively find the `<g id>` elements that should each become
    their own layer, treating an id-bearing group as a pass-through
    wrapper -- recursing into it instead of stopping there -- whenever
    it directly contains other id-bearing `<g>` children (re-grouped
    layers, a viewport/clip wrapper some export tools add per layer,
    etc.), rather than merging its whole subtree under the wrapper's
    own id.

    Mirrors raygeo's `collect_layer_nodes` (src/svg/mod.rs) exactly, so
    this manifest's ids -- used to build `active_layer_ids` for
    filtering -- line up 1:1 with what the actual geometry extraction
    (raygeo, layer-aware) finds. A mismatch here would silently filter
    out layers that raygeo does find, since layer_id membership is
    tested by exact string match.
    """
    result: list[ET.Element] = []
    for child in root:
        if not _is_id_bearing_group(child):
            continue
        if _has_nested_layer_group(child):
            result.extend(_collect_layer_elements(child))
        else:
            result.append(child)
    return result


def dedupe_layer_ids(raw_ids: list[str]) -> list[str]:
    """Disambiguate duplicate ids by appending a "#2", "#3", ...
    suffix to every occurrence after the first.

    SVG doesn't guarantee unique ids across a document -- hand-edited
    files, or files produced by merging/re-exporting, routinely repeat
    ids like "layer1". Downstream code (both this manifest and
    ``svg_vector._dedupe_layer_ids``) uses layer_id as a dict key, so a
    repeated id would otherwise make every occurrence after the first
    silently collide with/overwrite the previous one. Shared between
    the manifest (this module) and the actual geometry extraction
    (``svg_vector.py``) so a duplicate id resolves to the same
    disambiguated key in both -- they call this on the same ids in the
    same document order, since both derive from the same parsed SVG.
    """
    seen: dict[str, int] = {}
    result = []
    for raw_id in raw_ids:
        seen[raw_id] = seen.get(raw_id, 0) + 1
        n = seen[raw_id]
        result.append(raw_id if n == 1 else f"{raw_id}#{n}")
    return result


def extract_layer_manifest(data: bytes) -> list[dict[str, Any]]:
    """
    Parses the SVG to find top-level groups with IDs, treating them as layers.
    Also counts the number of geometric elements in each layer.
    """
    if not data:
        return []

    layers = []
    logger.debug("--- Starting SVG Layer Extraction ---")
    try:
        root = ET.fromstring(data)
        layer_elements = _collect_layer_elements(root)
        deduped_ids = dedupe_layer_ids(
            [el.get("id", "") for el in layer_elements]
        )
        for element, layer_id in zip(layer_elements, deduped_ids):
            raw_id = element.get("id")
            label = element.get(f"{{{INKSCAPE_NS}}}label") or raw_id
            if layer_id != raw_id:
                # Disambiguated (duplicate raw id) -- carry the "#2"
                # suffix into the displayed name too, so the import
                # dialog's layer list doesn't show two identical-
                # looking, unexplained entries.
                suffix = layer_id.rsplit("#", 1)[-1]
                label = f"{label} ({suffix})"

            # Count visual elements recursively to detect empty layers
            feature_count = 0
            for elem in element.iter():
                if _get_local_tag_name(elem) in SHAPE_TAGS:
                    feature_count += 1

            layers.append(
                {
                    "id": layer_id,
                    "name": label,
                    "count": feature_count,
                }
            )
            logger.debug(
                f"Found layer: ID='{layer_id}', "
                f"Name='{label}', Count={feature_count}"
            )
    except ET.ParseError as e:
        logger.error(f"Failed to parse SVG for layer extraction: {e}")
        return []

    return layers


def filter_svg_layers(data: bytes, visible_layer_ids: list[str]) -> bytes:
    """
    Returns a modified SVG with only specified top-level groups visible.
    """
    if not data:
        return b""

    try:
        root = ET.fromstring(data)
        elements_to_remove = []

        for child in root:
            tag = _get_local_tag_name(child)
            if tag == "g":
                layer_id = child.get("id")
                # If ID exists AND it is NOT in the visible list, remove it.
                if layer_id and layer_id not in visible_layer_ids:
                    elements_to_remove.append(child)

        for elem in elements_to_remove:
            root.remove(elem)

        # Registering namespaces at module level helps, but ET.tostring
        # needs to know we want to preserve the environment.
        return ET.tostring(root, encoding="utf-8")
    except ET.ParseError:
        return data
