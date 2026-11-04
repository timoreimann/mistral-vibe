"""Cross-platform power assertions that keep the machine awake while a turn runs.

On macOS this drives IOKit's ``IOPMAssertionCreateWithName`` through ctypes —
the same primitive ``caffeinate(8)`` uses — so no native build step is needed.
Other platforms get a no-op handle, keeping one call path for every host.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from enum import StrEnum, auto
import logging
import sys
from typing import Final

logger = logging.getLogger(__name__)


class SleepPreventionMode(StrEnum):
    """How strongly a running turn may hold the system awake.

    Levels are cumulative: each level adds the assertions of the lower ones,
    mirroring ``caffeinate -i`` / ``-i -d`` / ``-i -d -s -u``.
    """

    OFF = auto()
    IDLE = auto()
    DISPLAY = auto()
    SYSTEM = auto()

    @property
    def rank(self) -> int:
        return _MODE_RANK[self]


_MODE_RANK: Final[dict[SleepPreventionMode, int]] = {
    SleepPreventionMode.OFF: 0,
    SleepPreventionMode.IDLE: 1,
    SleepPreventionMode.DISPLAY: 2,
    SleepPreventionMode.SYSTEM: 3,
}


@dataclass(frozen=True, slots=True)
class PowerAssertionOptions:
    """Which sleep-inhibition primitives to acquire.

    Each flag maps to a ``caffeinate(8)`` flag: ``idle`` is ``-i``,
    ``display`` is ``-d``, ``system`` is ``-s`` (AC power only) and ``user``
    is ``-u`` (declares the user active).
    """

    reason: str = "vibe agent session"
    idle: bool = False
    system: bool = False
    user: bool = False
    display: bool = False


def power_assertion_options(mode: SleepPreventionMode) -> PowerAssertionOptions | None:
    """Translate a sleep-prevention mode into assertion options, or ``None`` for off."""
    if mode is SleepPreventionMode.OFF:
        return None
    return PowerAssertionOptions(
        reason="vibe agent session",
        idle=True,
        display=mode in {SleepPreventionMode.DISPLAY, SleepPreventionMode.SYSTEM},
        system=mode is SleepPreventionMode.SYSTEM,
        user=mode is SleepPreventionMode.SYSTEM,
    )


class PowerAssertion:
    """A held sleep-inhibition assertion; released by ``stop`` or on GC."""

    def __init__(self, options: PowerAssertionOptions | None = None) -> None:
        self._inner: _AssertionInner | None = None
        if options is not None:
            self._inner = _start(options)

    def stop(self) -> None:
        inner, self._inner = self._inner, None
        if inner is not None:
            inner.stop()

    def __del__(self) -> None:
        try:
            self.stop()
        except Exception:
            pass


def start_power_assertion(
    options: PowerAssertionOptions | None,
) -> PowerAssertion | None:
    """Acquire a power assertion; ``None`` options or an unsupported platform give ``None``."""
    if options is None:
        return None
    return PowerAssertion(options)


if sys.platform == "darwin":

    class _AssertionInner:
        """One or more IOKit power assertions, released together."""

        _CF_STRING_ENCODING_UTF8: Final = 0x0800_0100
        _ASSERTION_LEVEL_ON: Final = 255
        _ASSERTION_ID_NONE: Final = 0

        _PREVENT_USER_IDLE_SYSTEM_SLEEP: Final = "PreventUserIdleSystemSleep"
        _PREVENT_SYSTEM_SLEEP: Final = "PreventSystemSleep"
        _PREVENT_USER_IDLE_DISPLAY_SLEEP: Final = "PreventUserIdleDisplaySleep"
        _USER_IS_ACTIVE: Final = "UserIsActive"

        def __init__(self, kinds: list[str], reason: str) -> None:
            cf = ctypes.CDLL(
                "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
            )
            iokit = ctypes.CDLL("/System/Library/Frameworks/IOKit.framework/IOKit")
            cf.CFStringCreateWithCString.restype = ctypes.c_void_p
            cf.CFStringCreateWithCString.argtypes = [
                ctypes.c_void_p,
                ctypes.c_char_p,
                ctypes.c_uint32,
            ]
            cf.CFRelease.argtypes = [ctypes.c_void_p]
            iokit.IOPMAssertionCreateWithName.restype = ctypes.c_int32
            iokit.IOPMAssertionCreateWithName.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint32,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint32),
            ]
            iokit.IOPMAssertionRelease.restype = ctypes.c_int32
            iokit.IOPMAssertionRelease.argtypes = [ctypes.c_uint32]

            def cf_string(value: str) -> int:
                encoded = value.encode("utf-8")
                ref = cf.CFStringCreateWithCString(
                    None, encoded, self._CF_STRING_ENCODING_UTF8
                )
                if not ref:
                    raise RuntimeError(
                        "Failed to allocate CoreFoundation string for power assertion"
                    )
                return ref

            self._cf = cf
            self._iokit = iokit
            self._assertion_ids: list[int] = []
            try:
                reason_ref = cf_string(reason)
                try:
                    for kind in kinds:
                        kind_ref = cf_string(kind)
                        try:
                            assertion_id = ctypes.c_uint32(self._ASSERTION_ID_NONE)
                            status = iokit.IOPMAssertionCreateWithName(
                                kind_ref,
                                self._ASSERTION_LEVEL_ON,
                                reason_ref,
                                ctypes.byref(assertion_id),
                            )
                        finally:
                            cf.CFRelease(kind_ref)
                        if status != 0:
                            raise RuntimeError(
                                f"Failed to acquire macOS power assertion {kind} (IOReturn={status})"
                            )
                        self._assertion_ids.append(assertion_id.value)
                finally:
                    cf.CFRelease(reason_ref)
            except BaseException:
                self.stop()
                raise

        def stop(self) -> None:
            ids, self._assertion_ids = self._assertion_ids, []
            for assertion_id in ids:
                if assertion_id != self._ASSERTION_ID_NONE:
                    self._iokit.IOPMAssertionRelease(assertion_id)

    def _start(options: PowerAssertionOptions) -> _AssertionInner:
        kinds: list[str] = []
        if options.idle:
            kinds.append(_AssertionInner._PREVENT_USER_IDLE_SYSTEM_SLEEP)
        if options.system:
            kinds.append(_AssertionInner._PREVENT_SYSTEM_SLEEP)
        if options.user:
            kinds.append(_AssertionInner._USER_IS_ACTIVE)
        if options.display:
            kinds.append(_AssertionInner._PREVENT_USER_IDLE_DISPLAY_SLEEP)
        if not kinds:
            # Preserve the historical default: an empty option set behaves as
            # ``caffeinate -i``.
            kinds.append(_AssertionInner._PREVENT_USER_IDLE_SYSTEM_SLEEP)
        return _AssertionInner(kinds, options.reason)

else:

    class _AssertionInner:
        def __init__(self, reason: str) -> None:
            logger.debug(
                "Sleep prevention is not supported on %s; continuing without it",
                sys.platform,
            )

        def stop(self) -> None:
            return None

    def _start(options: PowerAssertionOptions) -> _AssertionInner:
        return _AssertionInner(options.reason)


__all__ = [
    "PowerAssertion",
    "PowerAssertionOptions",
    "SleepPreventionMode",
    "power_assertion_options",
    "start_power_assertion",
]
