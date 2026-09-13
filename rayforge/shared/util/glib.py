from collections.abc import Callable
from typing import Any

from gi.repository import GLib


def falsify(func, *args, **kwargs):
    """
    Wrapper for GLib.idle_add, as function must return False, otherwise it
    is automatically rescheduled into the event loop.
    """
    func(*args, **kwargs)
    return False


def idle_add(func, *args, **kwargs):
    """
    Wrapper for GLib.idle_add to support multiple args and kwargs.
    """
    GLib.idle_add(lambda: falsify(func, *args, **kwargs))


class DebounceMixin:
    """A mixin to add debouncing capabilities to a class.

    Each distinct callback gets its own pending-timer slot, keyed by
    the callback itself. A class that debounces more than one field
    (e.g. a settings group with both a "passes" row and a "z step
    down" row, each calling ``self._debounce(...)`` with its own
    handler) previously shared a single slot: debouncing field B
    within the delay window silently cancelled field A's pending
    commit, so field A's change never reached the model. Keying by
    callback keeps unrelated fields' debounces independent while still
    coalescing rapid repeated edits to the *same* field, which is what
    debouncing is for.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._debounce_timers: dict[Callable, int] = {}
        self._debounce_pending_args: dict[Callable, tuple] = {}

    def _debounce(self, callback: Callable, *args: Any):
        """
        Schedules ``callback`` to be called after a short delay,
        cancelling any previously scheduled call to that *same*
        callback (other callbacks debounced on this instance are
        unaffected).
        """
        existing = self._debounce_timers.get(callback)
        if existing is not None:
            GLib.source_remove(existing)
        self._debounce_pending_args[callback] = args
        self._debounce_timers[callback] = GLib.timeout_add(
            150, self._commit_debounced_change, callback
        )

    def _commit_debounced_change(self, callback: Callable) -> bool:
        """Executes ``callback`` with its pending args and clears its
        slot."""
        args = self._debounce_pending_args.pop(callback, ())
        self._debounce_timers.pop(callback, None)
        callback(*args)
        return GLib.SOURCE_REMOVE

    def _cancel_all_debounced(self):
        """Cancels every pending debounced call without executing it."""
        for timer_id in self._debounce_timers.values():
            GLib.source_remove(timer_id)
        self._debounce_timers.clear()
        self._debounce_pending_args.clear()

    def _flush_all_debounced(self):
        """Immediately executes every pending debounced call, in place
        of letting its timer fire later.

        Use this instead of :meth:`_cancel_all_debounced` when the
        owning object is about to go away (e.g. a dialog closing): a
        cancelled call is simply dropped, discarding whatever edit is
        still sitting in its debounce window, while a flush commits it
        immediately.
        """
        callbacks = list(self._debounce_timers)
        for callback in callbacks:
            GLib.source_remove(self._debounce_timers.pop(callback))
            args = self._debounce_pending_args.pop(callback, ())
            callback(*args)
