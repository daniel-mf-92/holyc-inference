#!/usr/bin/env bash
set -euo pipefail

unset CDPATH
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT_DEFAULT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd -P)"
REPO_ROOT="${INFERENCE_GATE_ROOT:-${REPO_ROOT_DEFAULT}}"
GATE_NAME="trinity-policy-sync"

INFERENCE_DOC="${TRINITY_INFERENCE_DOC:-${REPO_ROOT}/LOOP_PROMPT.md}"
TEMPLE_DOC="${TRINITY_TEMPLE_DOC:-${HOME}/Documents/local-codebases/TempleOS/MODERNIZATION/MASTER_TASKS.md}"
SANHEDRIN_DOC="${TRINITY_SANHEDRIN_DOC:-${HOME}/Documents/local-codebases/temple-sanhedrin/LOOP_PROMPT.md}"

passed=0
failed=0

secure_default_failed=0
dev_local_guard_failed=0
quarantine_hash_failed=0
gpu_iommu_bot_failed=0
attestation_digest_failed=0
trinity_drift_guard_failed=0

emit_check() {
  local check_id="$1"
  local status="$2"
  local scope="$3"
  local invariant="$4"
  local detail="$5"
  local evidence="$6"
  printf '{"type":"check","gate":"%s","id":"%s","status":"%s","scope":"%s","invariant":"%s","detail":"%s","evidence":"%s"}\n' \
    "${GATE_NAME}" "${check_id}" "${status}" "${scope}" "${invariant}" "${detail}" "${evidence}"
}

emit_summary() {
  local status="$1"
  local drift="$2"
  printf '{"type":"summary","gate":"%s","status":"%s","drift":"%s","passed":%d,"failed":%d,"inference_doc":"%s","temple_doc":"%s","sanhedrin_doc":"%s"}\n' \
    "${GATE_NAME}" "${status}" "${drift}" "${passed}" "${failed}" "${INFERENCE_DOC}" "${TEMPLE_DOC}" "${SANHEDRIN_DOC}"
}

mark_invariant_failure() {
  local invariant="$1"
  case "${invariant}" in
    secure-default) secure_default_failed=$((secure_default_failed + 1)) ;;
    dev-local-guard) dev_local_guard_failed=$((dev_local_guard_failed + 1)) ;;
    quarantine-hash) quarantine_hash_failed=$((quarantine_hash_failed + 1)) ;;
    gpu-iommu-bot) gpu_iommu_bot_failed=$((gpu_iommu_bot_failed + 1)) ;;
    attestation-digest) attestation_digest_failed=$((attestation_digest_failed + 1)) ;;
    trinity-drift-guard) trinity_drift_guard_failed=$((trinity_drift_guard_failed + 1)) ;;
  esac
}

check_file() {
  local check_id="$1"
  local scope="$2"
  local file="$3"
  local detail="$4"

  if [[ -f "${file}" ]]; then
    passed=$((passed + 1))
    emit_check "${check_id}" "pass" "${scope}" "doc-presence" "${detail}" "${file}"
  else
    failed=$((failed + 1))
    emit_check "${check_id}" "fail" "${scope}" "doc-presence" "${detail}" "missing:${file}"
  fi
}

check_pattern() {
  local check_id="$1"
  local scope="$2"
  local invariant="$3"
  local file="$4"
  local regex="$5"
  local detail="$6"
  local evidence_line=""

  if [[ ! -f "${file}" ]]; then
    failed=$((failed + 1))
    mark_invariant_failure "${invariant}"
    emit_check "${check_id}" "fail" "${scope}" "${invariant}" "${detail}" "missing:${file}"
    return
  fi

  evidence_line="$(grep -En -m1 "${regex}" "${file}" || true)"
  if [[ -n "${evidence_line}" ]]; then
    passed=$((passed + 1))
    emit_check "${check_id}" "pass" "${scope}" "${invariant}" "${detail}" "${file}:${evidence_line%%:*}"
  else
    failed=$((failed + 1))
    mark_invariant_failure "${invariant}"
    emit_check "${check_id}" "fail" "${scope}" "${invariant}" "${detail}" "missing-pattern:${file}"
  fi
}

check_file "TRI-DOC-01" "inference" "${INFERENCE_DOC}" "inference control doc exists"
check_file "TRI-DOC-02" "templeos" "${TEMPLE_DOC}" "templeos control doc exists"
check_file "TRI-DOC-03" "sanhedrin" "${SANHEDRIN_DOC}" "sanhedrin control doc exists"

check_pattern "TRI-SEC-01-INF" "inference" "secure-default" "${INFERENCE_DOC}" "secure-local.{0,80}default" "secure-local remains default in inference policy"
check_pattern "TRI-SEC-01-TEM" "templeos" "secure-default" "${TEMPLE_DOC}" "secure-local.{0,80}default" "secure-local remains default in TempleOS policy"
check_pattern "TRI-SEC-01-SAN" "sanhedrin" "secure-default" "${SANHEDRIN_DOC}" "default profile is not .{0,20}secure-local" "Sanhedrin audits secure-local default invariant"

check_pattern "TRI-SEC-02-INF" "inference" "dev-local-guard" "${INFERENCE_DOC}" "dev-local.{0,120}(explicit|opt-in).{0,120}(air-gap|air-gapped|Book of Truth)" "dev-local guardrails preserve air-gap + Book of Truth in inference policy"
check_pattern "TRI-SEC-02-TEM" "templeos" "dev-local-guard" "${TEMPLE_DOC}" "dev-local.{0,140}(explicit|opt-in).{0,140}(air-gapped|Book of Truth)" "dev-local guardrails preserve air-gap + Book of Truth in TempleOS policy"
check_pattern "TRI-SEC-02-SAN" "sanhedrin" "dev-local-guard" "${SANHEDRIN_DOC}" "secure-local\|dev-local\|quarantine\|Book of Truth\|IOMMU\|GPU" "Sanhedrin parity scan explicitly covers dev-local signature"

check_pattern "TRI-SEC-03-INF" "inference" "quarantine-hash" "${INFERENCE_DOC}" "(quarantine.{0,80}hash-manifest|quarantine/hash verification)" "inference trusted-load path requires quarantine/hash verification"
check_pattern "TRI-SEC-03-TEM" "templeos" "quarantine-hash" "${TEMPLE_DOC}" "quarantine.{0,80}hash verification" "TempleOS trusted-load path requires quarantine/hash verification"
check_pattern "TRI-SEC-03-SAN" "sanhedrin" "quarantine-hash" "${SANHEDRIN_DOC}" "trusted model load path can bypass quarantine/hash verification" "Sanhedrin audits quarantine/hash bypass as CRITICAL"

check_pattern "TRI-SEC-04-INF" "inference" "gpu-iommu-bot" "${INFERENCE_DOC}" "IOMMU.{0,120}Book[- ]of[- ]Truth.{0,120}(dispatch|hooks|logging)" "inference GPU dispatch requires IOMMU + Book-of-Truth hooks"
check_pattern "TRI-SEC-04-TEM" "templeos" "gpu-iommu-bot" "${TEMPLE_DOC}" "GPU.{0,120}IOMMU.{0,120}Book[- ]of[- ]Truth" "TempleOS GPU policy requires IOMMU + Book-of-Truth"
check_pattern "TRI-SEC-04-SAN" "sanhedrin" "gpu-iommu-bot" "${SANHEDRIN_DOC}" "GPU tasks bypass IOMMU or Book-of-Truth audit hooks" "Sanhedrin audits GPU IOMMU/Book-of-Truth bypass as CRITICAL"

check_pattern "TRI-SEC-05-INF" "inference" "attestation-digest" "${INFERENCE_DOC}" "attestation/policy-digest handshake" "inference policy enforces attestation/policy-digest handshake"
check_pattern "TRI-SEC-05-TEM" "templeos" "attestation-digest" "${TEMPLE_DOC}" "attestation evidence \+ policy digest match" "TempleOS trusted-load path requires attestation + policy digest parity"
check_pattern "TRI-SEC-05-SAN" "sanhedrin" "attestation-digest" "${SANHEDRIN_DOC}" "attestation \+ policy digest parity" "Sanhedrin treats missing attestation/policy-digest parity as CRITICAL"

check_pattern "TRI-SEC-06-INF" "inference" "trinity-drift-guard" "${INFERENCE_DOC}" "Trinity drift|policy changes that create Trinity drift" "inference policy blocks Trinity drift"
check_pattern "TRI-SEC-06-TEM" "templeos" "trinity-drift-guard" "${TEMPLE_DOC}" "policy drift.{0,80}release blocker" "TempleOS policy treats Trinity drift as release blocker"
check_pattern "TRI-SEC-06-SAN" "sanhedrin" "trinity-drift-guard" "${SANHEDRIN_DOC}" "Trinity policy parity mismatches as CRITICAL" "Sanhedrin treats Trinity parity drift as CRITICAL"

drift_detected=0
if (( secure_default_failed > 0 || dev_local_guard_failed > 0 || quarantine_hash_failed > 0 || gpu_iommu_bot_failed > 0 || attestation_digest_failed > 0 || trinity_drift_guard_failed > 0 )); then
  drift_detected=1
fi

if (( failed > 0 || drift_detected == 1 )); then
  emit_summary "fail" "true"
  exit 1
fi

emit_summary "pass" "false"
exit 0
