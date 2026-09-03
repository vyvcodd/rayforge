from unittest.mock import MagicMock

import pytest
from laser_essentials.steps import ImageEngraveStep


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
