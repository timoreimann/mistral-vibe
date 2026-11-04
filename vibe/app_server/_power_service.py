"""Process-wide sleep prevention, held while any session's turn runs.

One service per OS process (ADR 0019): every session's turn start acquires a
share and every turn completion releases it, so the machine stays awake exactly
while some agent is working. The assertion strength is the strongest mode any
active turn asked for, mirroring oh-my-pi's cumulative ``power.sleepPrevention``
levels.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
import logging

from vibe.core.power import (
    PowerAssertion,
    SleepPreventionMode,
    power_assertion_options,
    start_power_assertion,
)

logger = logging.getLogger(__name__)


def _default_start(mode: SleepPreventionMode) -> PowerAssertion | None:
    # Unsupported platforms return a no-op handle; real acquisition failures
    # propagate to the caller, which logs them without disabling the service.
    return start_power_assertion(power_assertion_options(mode))


class SleepPreventionService:
    """Refcounts active turns and holds the platform power assertion between them."""

    def __init__(
        self,
        start: Callable[[SleepPreventionMode], PowerAssertion | None] = _default_start,
    ) -> None:
        self._start = start
        self._active: Counter[SleepPreventionMode] = Counter()
        self._held: PowerAssertion | None = None
        self._held_mode: SleepPreventionMode | None = None
        self._backend_unavailable = False

    @property
    def active_turns(self) -> int:
        return sum(self._active.values())

    def turn_started(self, mode: SleepPreventionMode) -> None:
        if mode is SleepPreventionMode.OFF:
            return
        self._active[mode] += 1
        self._sync()

    def turn_finished(self, mode: SleepPreventionMode) -> None:
        if mode is SleepPreventionMode.OFF:
            return
        if self._active[mode] > 0:
            self._active[mode] -= 1
        # Unary plus drops zeroed entries, so an empty Counter really is empty.
        self._active = +self._active
        self._sync()

    def close(self) -> None:
        """Release everything, whatever the counts; called at process shutdown."""
        self._active.clear()
        self._release()

    def _sync(self) -> None:
        desired = (
            max(self._active, key=lambda mode: mode.rank) if self._active else None
        )
        if desired is None:
            self._release()
            return
        if (
            self._held is not None
            and self._held_mode is not None
            and desired.rank <= self._held_mode.rank
        ):
            return
        if self._backend_unavailable:
            return
        options = power_assertion_options(desired)
        if options is None:
            return
        try:
            held = self._start(desired)
        except Exception:
            logger.warning("Failed to acquire power assertion", exc_info=True)
            return
        if held is None:
            # Unsupported platform: the backend is not going to appear
            # mid-process, so stop retrying on every turn.
            self._backend_unavailable = True
            return
        self._release()
        self._held = held
        self._held_mode = desired

    def _release(self) -> None:
        held, self._held = self._held, None
        self._held_mode = None
        if held is not None:
            try:
                held.stop()
            except Exception:
                logger.warning("Failed to release power assertion", exc_info=True)


__all__ = ["SleepPreventionService"]
