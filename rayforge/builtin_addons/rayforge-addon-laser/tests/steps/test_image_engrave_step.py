from unittest.mock import MagicMock

import pytest
from laser_essentials.steps import ImageEngraveStep

from rayforge.core.doc import Doc
from rayforge.core.image_edit import ImageEditStep
from rayforge.core.workpiece import WorkPiece


@pytest.fixture
def mock_context():
    context = MagicMock()
    machine = MagicMock()
    machine.max_cut_speed = 5000
    machine.max_travel_speed = 10000
    machine.acceleration = 3000
    default_head = MagicMock()
    default_head.uid = "test-laser-uid"
    default_head.spot_size_mm = (0.1, 0.1)
    machine.get_default_laser_head.return_value = default_head
    context.machine = machine
    return context


class TestImageEngraveStep:
    def test_default_optimize_disallows_flip(self, mock_context):
        """Optimize always runs before BidirScanOffsetTransformer
        (Phase::GeometryRefinement precedes Phase::PostProcessing,
        regardless of list order). If Optimize is allowed to flip an
        individual run's own recorded direction, a row split into
        disjoint runs by a hole/gap in the content can end up with its
        two halves independently reversed for travel efficiency --
        decoupling them from the row's one true mechanical scan
        direction. BidirScanOffsetTransformer's classification, made
        afterward, then corrects one half and not the other for what
        is physically a single pass. allow_flip=False keeps a run's
        own recorded direction faithful to the original scan, so
        travel reordering can't desync it from bidir's classification.
        """
        step = ImageEngraveStep.create(mock_context, name="Created")
        for t_dict in step.per_workpiece_transformers_dicts:
            if t_dict.get("name") == "Optimize":
                assert t_dict.get("allow_flip") is False
                break
        else:
            pytest.fail("Optimize transformer not found")
        for t_dict in step.per_step_transformers_dicts:
            if t_dict.get("name") == "Optimize":
                assert t_dict.get("allow_flip") is False
                break
        else:
            pytest.fail("Optimize transformer not found")

    def test_shortened_sweep_scan_mode_roundtrips(self):
        step = ImageEngraveStep(name="Test")
        step.scan_mode = "SHORTENED_SWEEP"  # type: ignore[assignment]
        data = step.to_dict()
        assert data["scan_mode"] == "SHORTENED_SWEEP"
        restored = ImageEngraveStep.from_dict(data)
        assert restored.scan_mode == "SHORTENED_SWEEP"

    def test_get_assembler_kwargs_shortened_sweep_scan_mode(self, machine):
        step = ImageEngraveStep(name="Test")
        step.scan_mode = "SHORTENED_SWEEP"  # type: ignore[assignment]
        workpiece = MagicMock(spec=["size"])
        workpiece.size = (100, 100)
        kwargs = step.get_assembler_kwargs(machine, workpiece)
        assert kwargs["scan_mode"] == "shortened_sweep"

    def test_cache_params_include_bidir_x_offset_mm(self):
        """Regression test: bidir_x_offset_mm reaches
        BidirScanOffsetTransformer purely through the step-settings
        dict built at transformer-spec time, not through
        get_assembler_kwargs() or the transformer's own to_dict() --
        so it has to be in get_cache_params() or IntentBuilder's
        compute token never changes when the user edits it, and the
        pipeline keeps serving a stale, unoffset (or wrongly offset)
        cached Ops artifact until some unrelated param happens to
        bump the token.
        """
        step = ImageEngraveStep(name="Test")
        step.bidir_x_offset_mm = 0.0
        baseline = step.get_cache_params()

        step.bidir_x_offset_mm = 3.0
        changed = step.get_cache_params()

        assert baseline != changed
        assert changed["bidir_x_offset_mm"] == 3.0


LEGACY_INVERT = ImageEditStep("invert")


def _image_engrave_step_dicts(data) -> list[dict]:
    found = []
    if isinstance(data, dict):
        if data.get("step_type") == "ImageEngraveStep":
            found.append(data)
        for value in data.values():
            found += _image_engrave_step_dicts(value)
    elif isinstance(data, list):
        for value in data:
            found += _image_engrave_step_dicts(value)
    return found


class TestLegacyInvert:
    """Invert moved from the step to the workpieces' image edits."""

    def _saved(self, invert: bool) -> dict:
        data = ImageEngraveStep(name="Image Engrave").to_dict()
        data["invert"] = invert
        return data

    def test_invert_is_no_longer_a_step_setting(self):
        step = ImageEngraveStep()
        assert "invert" not in step.to_dict()
        assert "invert" not in step.get_cache_params()
        assert not hasattr(step, "invert")
        keys = {var.key for var in ImageEngraveStep.recipe_varset()}
        assert "invert" not in keys

    def test_saved_invert_is_handed_over_once(self):
        step = ImageEngraveStep.from_dict(self._saved(True))
        assert "invert" not in step.extra
        assert step.pop_legacy_image_edits() == [LEGACY_INVERT]
        assert step.pop_legacy_image_edits() == []

    def test_saved_invert_false_hands_over_nothing(self):
        step = ImageEngraveStep.from_dict(self._saved(False))
        assert step.pop_legacy_image_edits() == []

    def test_saved_invert_is_not_written_back(self):
        step = ImageEngraveStep.from_dict(self._saved(True))
        assert "invert" not in step.to_dict()

    def test_loading_a_document_moves_invert_to_the_workpieces(self):
        doc = Doc()
        wp = WorkPiece("photo")
        doc.active_layer.add_child(wp)
        doc.active_layer.workflow.add_step(
            ImageEngraveStep(name="Image Engrave")
        )
        data = doc.to_dict()
        _image_engrave_step_dicts(data)[0]["invert"] = True

        restored = Doc.from_dict(data)

        (restored_wp,) = restored.all_workpieces
        assert restored_wp.image_edits == (LEGACY_INVERT,)
        again = Doc.from_dict(restored.to_dict())
        assert again.all_workpieces[0].image_edits == (LEGACY_INVERT,)
