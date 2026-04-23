#!/usr/bin/env bash
set -euo pipefail

unset CDPATH
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT_DEFAULT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd -P)"
REPO_ROOT="${INFERENCE_GATE_ROOT:-${REPO_ROOT_DEFAULT}}"
GATE_NAME="inference-secure-local-release"

passed=0
failed=0

emit_check() {
  local ws_id="$1"
  local status="$2"
  local detail="$3"
  local evidence="$4"
  printf '{"type":"check","gate":"%s","id":"%s","status":"%s","detail":"%s","evidence":"%s"}\n' \
    "${GATE_NAME}" "${ws_id}" "${status}" "${detail}" "${evidence}"
}

emit_summary() {
  local status="$1"
  printf '{"type":"summary","gate":"%s","status":"%s","passed":%d,"failed":%d,"root":"%s"}\n' \
    "${GATE_NAME}" "${status}" "${passed}" "${failed}" "${REPO_ROOT}"
}

check_contains() {
  local ws_id="$1"
  local rel_file="$2"
  local needle="$3"
  local detail="$4"
  local abs_file="${REPO_ROOT}/${rel_file}"

  if [[ -f "${abs_file}" ]] && grep -Fq "${needle}" "${abs_file}"; then
    passed=$((passed + 1))
    emit_check "${ws_id}" "pass" "${detail}" "${rel_file}:${needle}"
  else
    failed=$((failed + 1))
    emit_check "${ws_id}" "fail" "${detail}" "missing:${rel_file}:${needle}"
  fi
}

check_file() {
  local ws_id="$1"
  local rel_file="$2"
  local detail="$3"
  local abs_file="${REPO_ROOT}/${rel_file}"

  if [[ -f "${abs_file}" ]]; then
    passed=$((passed + 1))
    emit_check "${ws_id}" "pass" "${detail}" "${rel_file}"
  else
    failed=$((failed + 1))
    emit_check "${ws_id}" "fail" "${detail}" "missing:${rel_file}"
  fi
}

check_contains "WS16-03" "src/model/trust_manifest.HC" "ModelTrustManifestVerifySHA256Checked" "trusted model manifest SHA256 verifier is present"
check_contains "WS16-04" "src/model/eval_gate.HC" "ModelEvalPromotionGateChecked" "deterministic promotion parity gate is present"
check_contains "WS16-05" "src/gguf/hardening_gate.HC" "GGUFParserHardeningGateChecked" "parser hardening gate is present"

check_contains "WS9-02" "src/gpu/policy.HC" "GPU_POLICY_ERR_IOMMU_GUARD" "IOMMU hard guard is wired into dispatch policy"
check_contains "WS9-08" "src/gpu/book_of_truth_bridge.HC" "BOTGPUBridgeRecordMMIOWrite" "Book-of-Truth DMA/MMIO/dispatch hooks exist"
check_contains "WS9-17" "src/gpu/book_of_truth_bridge.HC" "BOT_GPU_DMA_UNMAP" "IOMMU map/update/unmap lifecycle audit hooks exist"
check_contains "WS9-18" "src/gpu/command_verify.HC" "GPUCommandVerifyDescriptorChecked" "GPU command-stream verifier exists"
check_contains "WS9-22" "src/gpu/policy.HC" "GPUPolicyAllowDispatchChecked" "secure profile gate rejects unsafe GPU dispatch"

check_file "WS16-08" "automation/inference-secure-gate.sh" "secure-local release gate script exists"

if (( failed > 0 )); then
  emit_summary "fail"
  exit 1
fi

emit_summary "pass"
exit 0
