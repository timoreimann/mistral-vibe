"""Tests for sleep-prevention power assertions and the process-wide service."""

from __future__ import annotations

from typing import Any, cast

import pytest

from vibe.app_server._power_service import SleepPreventionService
from vibe.app_server.models import PublicTurn
from vibe.core.config.vibe_schema import VibeConfigSchema
from vibe.core.power import (
    PowerAssertion,
    SleepPreventionMode,
    power_assertion_options,
    start_power_assertion,
)


class _FakeAssertion:
    def __init__(self, mode: SleepPreventionMode) -> None:
        self.mode = mode
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


def _service() -> tuple[SleepPreventionService, list[PowerAssertion]]:
    held: list[PowerAssertion] = []

    def start(mode: SleepPreventionMode) -> PowerAssertion | None:
        assertion = _FakeAssertion(mode)
        held.append(assertion)  # type: ignore[arg-type]
        return assertion  # type: ignore[return-value]

    return SleepPreventionService(start=start), held  # type: ignore[arg-type]


class TestPowerAssertionOptions:
    def test_off_maps_to_none(self) -> None:
        assert power_assertion_options(SleepPreventionMode.OFF) is None

    def test_idle_prevents_idle_sleep_only(self) -> None:
        options = power_assertion_options(SleepPreventionMode.IDLE)
        assert options is not None
        assert (
            options.idle
            and not options.display
            and not options.system
            and not options.user
        )

    def test_display_adds_display_sleep(self) -> None:
        options = power_assertion_options(SleepPreventionMode.DISPLAY)
        assert options is not None
        assert options.idle and options.display
        assert not options.system and not options.user

    def test_system_adds_system_sleep_and_user_active(self) -> None:
        options = power_assertion_options(SleepPreventionMode.SYSTEM)
        assert options is not None
        assert options.idle and options.display and options.system and options.user

    def test_levels_are_cumulative(self) -> None:
        ranks = [
            SleepPreventionMode.OFF,
            SleepPreventionMode.IDLE,
            SleepPreventionMode.DISPLAY,
            SleepPreventionMode.SYSTEM,
        ]
        assert [mode.rank for mode in ranks] == sorted(mode.rank for mode in ranks)


class TestSleepPreventionService:
    def test_no_assertion_when_idle(self) -> None:
        service, held = _service()
        assert service.active_turns == 0
        assert held == []

    def test_turn_start_acquires_and_finish_releases(self) -> None:
        service, held = _service()
        service.turn_started(SleepPreventionMode.IDLE)
        assert service.active_turns == 1
        assert len(held) == 1
        service.turn_finished(SleepPreventionMode.IDLE)
        assert service.active_turns == 0
        assert held[0].stopped  # type: ignore[attr-defined]

    def test_off_mode_never_acquires(self) -> None:
        service, held = _service()
        service.turn_started(SleepPreventionMode.OFF)
        service.turn_finished(SleepPreventionMode.OFF)
        assert held == []

    def test_overlapping_turns_hold_until_the_last_finishes(self) -> None:
        service, held = _service()
        service.turn_started(SleepPreventionMode.IDLE)
        service.turn_started(SleepPreventionMode.IDLE)
        assert len(held) == 1
        service.turn_finished(SleepPreventionMode.IDLE)
        assert not held[0].stopped  # type: ignore[attr-defined]
        service.turn_finished(SleepPreventionMode.IDLE)
        assert held[0].stopped  # type: ignore[attr-defined]

    def test_stronger_mode_escalates_mid_turn(self) -> None:
        service, held = _service()
        service.turn_started(SleepPreventionMode.IDLE)
        service.turn_started(SleepPreventionMode.SYSTEM)
        assert len(held) == 2
        assert held[0].stopped  # type: ignore[attr-defined]
        assert held[1].mode is SleepPreventionMode.SYSTEM  # type: ignore[attr-defined]
        service.turn_finished(SleepPreventionMode.SYSTEM)
        # The weaker turn is still active: the assertion must stay held.
        assert not held[1].stopped  # type: ignore[attr-defined]
        service.turn_finished(SleepPreventionMode.IDLE)
        assert held[1].stopped  # type: ignore[attr-defined]

    def test_close_releases_regardless_of_counts(self) -> None:
        service, held = _service()
        service.turn_started(SleepPreventionMode.DISPLAY)
        service.close()
        assert service.active_turns == 0
        assert held[0].stopped  # type: ignore[attr-defined]

    def test_unavailable_backend_is_not_retried(self) -> None:
        calls: list[SleepPreventionMode] = []

        def start(mode: SleepPreventionMode) -> PowerAssertion | None:
            calls.append(mode)
            return None

        service = SleepPreventionService(start=start)
        service.turn_started(SleepPreventionMode.IDLE)
        service.turn_finished(SleepPreventionMode.IDLE)
        service.turn_started(SleepPreventionMode.IDLE)
        assert calls == [SleepPreventionMode.IDLE]


class TestConfigSetting:
    def test_default_is_idle(self) -> None:
        assert VibeConfigSchema().power_sleep_prevention is SleepPreventionMode.IDLE

    @pytest.mark.parametrize("value", ["off", "idle", "display", "system"])
    def test_accepts_all_modes(self, value: str) -> None:
        config = VibeConfigSchema.model_validate({"power_sleep_prevention": value})
        assert config.power_sleep_prevention == SleepPreventionMode(value)

    def test_rejects_unknown_mode(self) -> None:
        with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError
            VibeConfigSchema.model_validate({"power_sleep_prevention": "turbo"})


class TestAdapterTurnPowerTracking:
    """The backend adapter's TurnStarted/TurnCompleted hook drives the service."""

    def _adapter(self) -> tuple[Any, SleepPreventionService]:
        from vibe.app_server._unified_harness_backend_adapter import (
            UnifiedHarnessBackendAdapter,
        )

        service, held = _service()

        class _Orchestrator:
            @property
            def config(self) -> VibeConfigSchema:
                return VibeConfigSchema(
                    power_sleep_prevention=SleepPreventionMode.DISPLAY
                )

        class _Context:
            config_orchestrator = _Orchestrator()
            sleep_prevention = service

        adapter = cast(
            Any, UnifiedHarnessBackendAdapter.__new__(UnifiedHarnessBackendAdapter)
        )
        adapter._context = _Context()
        adapter._power_held_turns = {}
        return adapter, service

    @staticmethod
    def _turn(turn_id: str) -> PublicTurn:
        from vibe.app_server.models import PublicTurn, PublicTurnStatus

        return PublicTurn(
            id=turn_id,
            session_id="s",
            status=PublicTurnStatus.IN_PROGRESS,
            started_at=0,
        )

    def test_turn_started_acquires_and_completed_releases(self) -> None:
        from vibe.app_server.events import TurnCompleted, TurnStarted

        adapter, service = self._adapter()
        adapter._track_turn_power(TurnStarted(turn=self._turn("t1")))
        assert service.active_turns == 1
        adapter._track_turn_power(
            TurnStarted(turn=self._turn("t1"))
        )  # repeats while running
        assert service.active_turns == 1
        adapter._track_turn_power(TurnCompleted(turn=self._turn("t1")))
        assert service.active_turns == 0
        # A completion with no matching start is a no-op.
        adapter._track_turn_power(TurnCompleted(turn=self._turn("t2")))
        assert service.active_turns == 0

    def test_shutdown_releases_held_turns(self) -> None:
        from vibe.app_server.events import TurnStarted

        adapter, service = self._adapter()
        adapter._track_turn_power(TurnStarted(turn=self._turn("t1")))
        adapter._release_turn_power()
        assert service.active_turns == 0


class TestStartPowerAssertion:
    def test_none_options_give_no_handle(self) -> None:
        assert start_power_assertion(None) is None

    def test_options_give_a_stoppable_handle(self) -> None:
        options = power_assertion_options(SleepPreventionMode.IDLE)
        assert options is not None
        assertion = start_power_assertion(options)
        assert assertion is not None
        assertion.stop()
        assertion.stop()  # idempotent
