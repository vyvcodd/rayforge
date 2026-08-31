from gi.repository import GLib

from rayforge.shared.util.glib import DebounceMixin, falsify


class TestFalsify:
    def test_calls_function_with_args(self):
        result = []

        def callback(a, b):
            result.append((a, b))

        falsify(callback, 1, 2)

        assert result == [(1, 2)]

    def test_calls_function_with_kwargs(self):
        result = []

        def callback(a, b=None):
            result.append((a, b))

        falsify(callback, 1, b=2)

        assert result == [(1, 2)]

    def test_returns_false(self):
        def callback():
            pass

        result = falsify(callback)

        assert result is False

    def test_returns_false_regardless_of_callback_return(self):
        def callback():
            return "something"

        result = falsify(callback)

        assert result is False


def _run_main_loop_until(predicate, timeout_ms=2000):
    """Pumps the GLib main loop until ``predicate()`` is true."""
    context = GLib.MainContext.default()
    deadline = GLib.get_monotonic_time() + timeout_ms * 1000
    while not predicate():
        if GLib.get_monotonic_time() > deadline:
            raise TimeoutError("predicate never became true")
        context.iteration(True)


class TestDebounceMixin:
    def test_init_sets_defaults(self):
        class MyClass(DebounceMixin):
            pass

        obj = MyClass()

        assert obj._debounce_timers == {}
        assert obj._debounce_pending_args == {}

    def test_can_be_combined_with_other_init(self):
        class MyClass(DebounceMixin):
            def __init__(self, value):
                super().__init__()
                self.value = value

        obj = MyClass(42)

        assert obj.value == 42
        assert obj._debounce_timers == {}

    def test_repeated_calls_to_same_callback_coalesce(self):
        obj = DebounceMixin()
        calls = []

        def callback(value):
            calls.append(value)

        obj._debounce(callback, "first")
        obj._debounce(callback, "second")
        obj._debounce(callback, "third")

        _run_main_loop_until(lambda: len(calls) > 0)

        assert calls == ["third"]

    def test_different_callbacks_do_not_clobber_each_other(self):
        """Regression test: debouncing field B must not silently drop
        a pending debounced commit for field A. Previously all
        debounced calls on an instance shared one timer/callback slot,
        so scheduling a second field's commit cancelled the first
        field's pending commit outright.
        """
        obj = DebounceMixin()
        calls_a = []
        calls_b = []

        def callback_a(value):
            calls_a.append(value)

        def callback_b(value):
            calls_b.append(value)

        obj._debounce(callback_a, "passes=1")
        obj._debounce(callback_b, "z_step_down=0.5")

        _run_main_loop_until(lambda: calls_a and calls_b)

        assert calls_a == ["passes=1"]
        assert calls_b == ["z_step_down=0.5"]

    def test_cancel_all_debounced_drops_pending_calls(self):
        obj = DebounceMixin()
        calls = []

        obj._debounce(lambda: calls.append(1))
        obj._cancel_all_debounced()

        assert obj._debounce_timers == {}
        assert obj._debounce_pending_args == {}

        # Nothing should fire even after the original delay elapses.
        waited = []
        GLib.timeout_add(200, lambda: (waited.append(1), False)[1])
        _run_main_loop_until(lambda: waited)
        assert calls == []
