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
