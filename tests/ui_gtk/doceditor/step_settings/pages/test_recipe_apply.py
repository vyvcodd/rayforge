# flake8: noqa: E402
"""Tests for applying a recipe and syncing the settings page widgets."""

import gi
import pytest

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from rayforge.core.recipe import Recipe
from rayforge.core.varset import FloatVar, IntVar, VarSet
from rayforge.ui_gtk.doceditor.step_settings.pages import StepSettingsPage
from rayforge.ui_gtk.varset.varsetwidget import VarSetWidget


def _make_page(editor, step) -> tuple[StepSettingsPage, VarSetWidget]:
    page = StepSettingsPage(editor, step)
    var_set = VarSet(
        vars=[
            IntVar(key="count", label="Count", default=3, min_val=1),
            FloatVar(
                key="power",
                label="Power",
                default=0.5,
                min_val=0.0,
                max_val=1.0,
                digits=2,
            ),
        ]
    )
    widget = page.add_varset_section("Params", var_set)
    return page, widget


@pytest.mark.ui
def test_applying_recipe_updates_page_widgets(editor, step):
    page, widget = _make_page(editor, step)

    recipe = Recipe(
        name="Test Recipe",
        setting_dicts=[
            {"name": "count", "value": 9, "recipe_apply": True},
            {"name": "power", "value": 0.8, "recipe_apply": True},
        ],
    )

    updated = []
    step.updated.connect(lambda *_: updated.append(1), weak=False)
    page.recipe_control._apply_recipe(recipe)

    assert step.applied_recipe_uid == recipe.uid
    assert step.count == 9
    assert step.power == pytest.approx(0.8)
    assert updated, "applying a recipe must emit step.updated"
    assert widget.get_values()["count"] == 9
    assert widget.get_values()["power"] == pytest.approx(0.8)


@pytest.mark.ui
def test_applying_recipe_uses_setters(editor, step):
    page, _widget = _make_page(editor, step)
    recipe = Recipe(
        name="Setter Recipe",
        setting_dicts=[{"name": "count", "value": 5, "recipe_apply": True}],
    )

    page.recipe_control._apply_recipe(recipe)

    assert step.count == 5


@pytest.mark.ui
def test_resync_overrides_pending_edit(editor, step):
    """Applying a recipe overrides a pending (debounced) user edit."""
    page, widget = _make_page(editor, step)

    count_adapter = widget.adapter_for("count")
    assert count_adapter is not None
    count_adapter.set_value(9)
    widget._on_data_changed("count")
    assert widget._pending_keys

    recipe = Recipe(
        name="Override Recipe",
        setting_dicts=[{"name": "count", "value": 4, "recipe_apply": True}],
    )
    page.recipe_control._apply_recipe(recipe)

    assert widget.get_values()["count"] == 4
    assert not widget._pending_keys


def test_sync_from_model_preserves_pending_edit(editor, step):
    """A model resync skips keys with pending (debounced) edits."""
    _page, widget = _make_page(editor, step)

    count_adapter = widget.adapter_for("count")
    assert count_adapter is not None
    count_adapter.set_value(9)
    widget._on_data_changed("count")

    widget.sync_from_model({"count": 4, "power": 0.5})

    assert widget.get_values()["count"] == 9
    assert widget.get_values()["power"] == pytest.approx(0.5)


def test_config_changed_commits_pending_edit_instead_of_discarding_it(
    editor, step
):
    """An unrelated config change must not silently discard an edit
    still sitting in its debounce window.

    ``config.changed`` fires for far more than machine switches --
    Config wires the active machine's own ``changed`` signal straight
    through to it, and several view preferences (grid/tabs/camera
    visibility, ...) send it directly too. ``_on_config_changed``
    rebuilds every varset section on each such event so
    machine/unit-dependent rows stay current; that rebuild must flush
    a pending edit into the model first, not silently revert it, since
    ``VarSetWidget.set_values`` (unlike ``sync_from_model``) does not
    check for pending edits.
    """
    page = StepSettingsPage(editor, step)
    widget = page.add_varset_section("Params", step.recipe_varset())

    speed_adapter = widget.adapter_for("cut_speed")
    assert speed_adapter is not None
    speed_adapter.set_value(750)
    widget._on_data_changed("cut_speed")
    assert widget._pending_keys

    # Simulate an unrelated config change (e.g. a canvas-view toggle),
    # not a real machine switch or recipe apply.
    editor.context.config.changed.send(editor.context.config)

    assert step.cut_speed == 750
    assert widget.get_values()["cut_speed"] == 750
    assert not widget._pending_keys


def test_cleanup_commits_pending_edit_instead_of_discarding_it(editor, step):
    """Closing the settings dialog must not discard an edit still
    sitting in its debounce window.

    Typing a value, hitting Enter, and closing the dialog right away
    is a completely normal, fast sequence -- well within the 300ms
    varset debounce. ``_cleanup`` previously called
    ``widget.cancel_pending()``, which drops such an edit outright
    instead of committing it, so it silently never reached the step.
    """
    page, widget = _make_page(editor, step)

    count_adapter = widget.adapter_for("count")
    assert count_adapter is not None
    count_adapter.set_value(9)
    widget._on_data_changed("count")
    assert widget._pending_keys

    page._cleanup()

    assert step.count == 9
    assert widget.get_values()["count"] == 9
    assert not widget._pending_keys
