"""Harness for IQ-1318 secure-local + audit hook benchmark gate."""

from pathlib import Path

SECURE = 1
DEV = 2

GATE_OK = 0
GATE_ERR_NULL_PTR = 1
GATE_ERR_BAD_PARAM = 2
GATE_ERR_PROFILE_NOT_SECURE = 3
GATE_ERR_AUDIT_HOOKS_INACTIVE = 4


def gate_checked(profile_id: int, audit_hooks_active: int):
    if profile_id not in (SECURE, DEV):
        return GATE_ERR_BAD_PARAM, None, None
    if audit_hooks_active not in (0, 1):
        return GATE_ERR_BAD_PARAM, None, None

    gate_passed = 0
    policy_error = GATE_OK

    if profile_id != SECURE:
        policy_error = GATE_ERR_PROFILE_NOT_SECURE
    elif not audit_hooks_active:
        policy_error = GATE_ERR_AUDIT_HOOKS_INACTIVE
    else:
        gate_passed = 1
        policy_error = GATE_OK

    return GATE_OK, gate_passed, policy_error


def test_secure_with_audit_hooks_passes():
    status, gate_passed, policy_error = gate_checked(SECURE, 1)
    assert status == GATE_OK
    assert gate_passed == 1
    assert policy_error == GATE_OK


def test_secure_without_audit_hooks_fails_closed():
    status, gate_passed, policy_error = gate_checked(SECURE, 0)
    assert status == GATE_OK
    assert gate_passed == 0
    assert policy_error == GATE_ERR_AUDIT_HOOKS_INACTIVE


def test_dev_local_rejected_even_with_audit_hooks():
    status, gate_passed, policy_error = gate_checked(DEV, 1)
    assert status == GATE_OK
    assert gate_passed == 0
    assert policy_error == GATE_ERR_PROFILE_NOT_SECURE


def test_bad_inputs_rejected():
    status, _, _ = gate_checked(0, 1)
    assert status == GATE_ERR_BAD_PARAM
    status, _, _ = gate_checked(SECURE, -1)
    assert status == GATE_ERR_BAD_PARAM


def test_source_contains_iq1318_symbols_and_constants():
    src = Path("src/runtime/profile.HC").read_text(encoding="utf-8")
    assert "Q8_0DotBenchRunDefaultSuiteSecureLocalAuditGate" in src
    assert "Q8_0_DOT_BENCH_SECURE_GATE_ERR_PROFILE_NOT_SECURE" in src
    assert "Q8_0_DOT_BENCH_SECURE_GATE_ERR_AUDIT_HOOKS_INACTIVE" in src
    assert "INFERENCE_PROFILE_SECURE_LOCAL" in src

