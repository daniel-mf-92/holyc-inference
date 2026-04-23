#!/usr/bin/env python3
"""Harness for IQ-1251 runtime profile state (`secure-local` default)."""

from __future__ import annotations

from pathlib import Path

INFERENCE_PROFILE_OK = 0
INFERENCE_PROFILE_ERR_NULL_PTR = 1
INFERENCE_PROFILE_ERR_BAD_PARAM = 2

INFERENCE_PROFILE_SECURE_LOCAL = 1
INFERENCE_PROFILE_DEV_LOCAL = 2


class RuntimeProfileState:
    def __init__(self) -> None:
        self.mode = INFERENCE_PROFILE_SECURE_LOCAL

    def name_from_id(self, profile_id: int) -> str:
        if profile_id == INFERENCE_PROFILE_SECURE_LOCAL:
            return "secure-local"
        if profile_id == INFERENCE_PROFILE_DEV_LOCAL:
            return "dev-local"
        return "unknown"

    def try_set_checked(self, requested_profile_id: int) -> int:
        if requested_profile_id not in (
            INFERENCE_PROFILE_SECURE_LOCAL,
            INFERENCE_PROFILE_DEV_LOCAL,
        ):
            return INFERENCE_PROFILE_ERR_BAD_PARAM
        self.mode = requested_profile_id
        return INFERENCE_PROFILE_OK

    def set_dev_local_checked(self) -> int:
        return self.try_set_checked(INFERENCE_PROFILE_DEV_LOCAL)

    def set_secure_local_checked(self) -> int:
        return self.try_set_checked(INFERENCE_PROFILE_SECURE_LOCAL)

    def status_checked(self) -> tuple[int, int, str, int]:
        if self.mode not in (INFERENCE_PROFILE_SECURE_LOCAL, INFERENCE_PROFILE_DEV_LOCAL):
            return INFERENCE_PROFILE_ERR_BAD_PARAM, 0, "", 0

        mode = self.mode
        return (
            INFERENCE_PROFILE_OK,
            mode,
            self.name_from_id(mode),
            1 if mode == INFERENCE_PROFILE_SECURE_LOCAL else 0,
        )


def test_source_contains_iq1251_symbols() -> None:
    src = Path("src/runtime/profile.HC").read_text(encoding="utf-8")

    assert "INFERENCE_PROFILE_SECURE_LOCAL" in src
    assert "INFERENCE_PROFILE_DEV_LOCAL" in src
    assert 'return (U8 *)"secure-local";' in src
    assert 'return (U8 *)"dev-local";' in src

    assert "I32 InferenceProfileTrySetChecked(I64 requested_profile_id)" in src
    assert "I32 InferenceProfileStatusChecked(I64 *out_profile_id," in src
    assert "I64 InferenceProfileStatus()" in src


def test_default_profile_is_secure_local() -> None:
    state = RuntimeProfileState()
    status, profile_id, profile_name, is_secure_default = state.status_checked()

    assert status == INFERENCE_PROFILE_OK
    assert profile_id == INFERENCE_PROFILE_SECURE_LOCAL
    assert profile_name == "secure-local"
    assert is_secure_default == 1


def test_dev_local_requires_explicit_set_call() -> None:
    state = RuntimeProfileState()

    status, profile_id, profile_name, is_secure_default = state.status_checked()
    assert status == INFERENCE_PROFILE_OK
    assert profile_id == INFERENCE_PROFILE_SECURE_LOCAL
    assert profile_name == "secure-local"
    assert is_secure_default == 1

    assert state.set_dev_local_checked() == INFERENCE_PROFILE_OK
    status, profile_id, profile_name, is_secure_default = state.status_checked()
    assert status == INFERENCE_PROFILE_OK
    assert profile_id == INFERENCE_PROFILE_DEV_LOCAL
    assert profile_name == "dev-local"
    assert is_secure_default == 0


def test_invalid_profile_rejected_without_state_change() -> None:
    state = RuntimeProfileState()

    assert state.set_dev_local_checked() == INFERENCE_PROFILE_OK
    assert state.try_set_checked(0) == INFERENCE_PROFILE_ERR_BAD_PARAM
    assert state.try_set_checked(-1) == INFERENCE_PROFILE_ERR_BAD_PARAM
    assert state.try_set_checked(99) == INFERENCE_PROFILE_ERR_BAD_PARAM

    status, profile_id, profile_name, is_secure_default = state.status_checked()
    assert status == INFERENCE_PROFILE_OK
    assert profile_id == INFERENCE_PROFILE_DEV_LOCAL
    assert profile_name == "dev-local"
    assert is_secure_default == 0


def test_round_trip_back_to_secure_local() -> None:
    state = RuntimeProfileState()

    assert state.set_dev_local_checked() == INFERENCE_PROFILE_OK
    assert state.set_secure_local_checked() == INFERENCE_PROFILE_OK

    status, profile_id, profile_name, is_secure_default = state.status_checked()
    assert status == INFERENCE_PROFILE_OK
    assert profile_id == INFERENCE_PROFILE_SECURE_LOCAL
    assert profile_name == "secure-local"
    assert is_secure_default == 1


def test_unknown_mode_yields_bad_param_status() -> None:
    state = RuntimeProfileState()
    state.mode = 777

    status, profile_id, profile_name, is_secure_default = state.status_checked()
    assert status == INFERENCE_PROFILE_ERR_BAD_PARAM
    assert profile_id == 0
    assert profile_name == ""
    assert is_secure_default == 0


if __name__ == "__main__":
    test_source_contains_iq1251_symbols()
    test_default_profile_is_secure_local()
    test_dev_local_requires_explicit_set_call()
    test_invalid_profile_rejected_without_state_change()
    test_round_trip_back_to_secure_local()
    test_unknown_mode_yields_bad_param_status()
    print("ok")
