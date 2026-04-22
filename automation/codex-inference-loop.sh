#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/Documents/local-codebases/holyc-inference}"
PROMPT_FILE="${PROMPT_FILE:-$REPO_DIR/LOOP_PROMPT.md}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/automation/logs}"
SLEEP_SECONDS="${SLEEP_SECONDS:-60}"
BRANCH_NAME="${BRANCH_NAME:-main}"
CODEX_MODEL="${CODEX_MODEL:-gpt-5.3-codex}"
CODEX_TIMEOUT_SECONDS="${CODEX_TIMEOUT_SECONDS:-1200}"
CODEX_KILL_AFTER_SECONDS="${CODEX_KILL_AFTER_SECONDS:-30}"
CODEX_MAX_RETRIES="${CODEX_MAX_RETRIES:-10}"
CODEX_RETRY_DELAY_SECONDS="${CODEX_RETRY_DELAY_SECONDS:-20}"
MAX_ITERATIONS="${MAX_ITERATIONS:-0}"
LOCK_DIR="${LOCK_DIR:-$REPO_DIR/automation/.codex-loop.lock}"
LOCK_FILE="${LOCK_FILE:-$LOCK_DIR/.flock}"
LOCK_PID_FILE="${LOCK_PID_FILE:-$LOCK_DIR/pid}"
LOCK_FD=""
HEARTBEAT_FILE="${HEARTBEAT_FILE:-$LOG_DIR/loop.heartbeat}"
LAST_RESULT_FILE="${LAST_RESULT_FILE:-$LOG_DIR/last-result.txt}"
LAST_ITERATION_FILE="${LAST_ITERATION_FILE:-$LOG_DIR/last-iteration.txt}"
AIRGAP_GUARD_SCRIPT="${AIRGAP_GUARD_SCRIPT:-/usr/bin/true}"
CREDENTIALS_FILE="${CREDENTIALS_FILE:-$HOME/.mcp-credentials.env}"
CODEX_USE_CO_WRAPPER="${CODEX_USE_CO_WRAPPER:-0}"
CODEX_CO_WRAPPER="${CODEX_CO_WRAPPER:-$HOME/bin/co}"
LB_ENDPOINT_FILES="${LB_ENDPOINT_FILES:-}"
LB_STATE_FILE="${LB_STATE_FILE:-$HOME/.co-codex53-lb-state-inference}"
CODEX_FORCE_SKIP_GIT_CHECK="${CODEX_FORCE_SKIP_GIT_CHECK:-1}"
CODEX_PYTHON_SAFEPATH="${CODEX_PYTHON_SAFEPATH:-1}"
NO_NETWORK_GUARD=$'Hard safety requirement: keep TempleOS guest fully air-gapped at all times.\n- Do not add or enable networking stack, NIC drivers, sockets, TCP/IP, UDP, TLS, DHCP, DNS, HTTP, or similar networking features.\n- Do not execute WS8 networking tasks; record them as out-of-scope due to air-gap policy.\n- Any QEMU or VM command must explicitly disable networking (use `-nic none`; legacy fallback: `-net none`).'
HOLYC_ONLY_GUARD=$'Hard language requirement: core TempleOS modernization is HolyC-only.\n- For core subsystems, do not introduce C/C++/Rust/Go/Python/JS/TS implementation code.\n- Non-HolyC is allowed only for host-side automation/tooling around the repo.\n- Reject tasks requiring network-dependent package ecosystems or remote runtime services.'

DEFAULT_LB_ENDPOINT_FILES=(
  "$HOME/.codex/codex53-endpoints.json"
  "$HOME/.codex/codex53-2-endpoints.json"
  "$HOME/.codex/codex53-3-endpoints.json"
  "$HOME/.codex/codex53-4-endpoints.json"
)

LB_SELECTED_NAME=""
LB_SELECTED_BASE_URL=""
LB_SELECTED_API_KEY=""
LB_SELECTED_MODEL=""
LB_SELECTED_ACTIVE="0"
LB_MIN_ACTIVE="0"
LB_ENDPOINT_FILE_LIST=()

mkdir -p "$LOG_DIR"

if [[ ! -f "$PROMPT_FILE" ]]; then
  echo "Prompt file not found: $PROMPT_FILE"
  exit 1
fi

if [[ ! -x "$AIRGAP_GUARD_SCRIPT" ]]; then
  echo "Air-gap guard script missing/executable bit not set: $AIRGAP_GUARD_SCRIPT"
  exit 1
fi

if [[ -f "$CREDENTIALS_FILE" ]]; then
  set +u
  # shellcheck source=/dev/null
  source "$CREDENTIALS_FILE"
  set -u
fi

init_codex_runner() {
  if [[ "$CODEX_USE_CO_WRAPPER" == "1" && -x "$CODEX_CO_WRAPPER" ]]; then
    CODEX_RUNNER_LABEL="co"
    CODEX_RUNNER=("$CODEX_CO_WRAPPER" "--interactive")
  else
    CODEX_RUNNER_LABEL="codex"
    CODEX_RUNNER=("codex")
  fi
}

build_endpoint_file_list() {
  LB_ENDPOINT_FILE_LIST=()
  local candidates=()

  if [[ -n "$LB_ENDPOINT_FILES" ]]; then
    IFS=':' read -r -a candidates <<< "$LB_ENDPOINT_FILES"
  else
    candidates=("${DEFAULT_LB_ENDPOINT_FILES[@]}")
  fi

  local candidate
  for candidate in "${candidates[@]}"; do
    [[ -f "$candidate" ]] && LB_ENDPOINT_FILE_LIST+=("$candidate")
  done
}

select_lb_endpoint() {
  if ! command -v jq >/dev/null 2>&1; then
    echo "jq not found; cannot select load-balanced endpoint"
    return 1
  fi

  local -a endpoint_lines=()
  local endpoint_file line

  for endpoint_file in "${LB_ENDPOINT_FILE_LIST[@]}"; do
    while IFS= read -r line; do
      endpoint_lines+=("$line")
    done < <(
      jq -r '
        if type=="array" then .[] else . end
        | select(type=="object" and (.name|type=="string") and (.base_url|type=="string") and (.api_key|type=="string"))
        | [.name, .base_url, .api_key, ((.model // "gpt-53-codex")|tostring)]
        | @tsv
      ' "$endpoint_file" 2>/dev/null || true
    )
  done

  if (( ${#endpoint_lines[@]} == 0 )); then
    echo "No valid load-balanced endpoint entries found"
    return 1
  fi

  local ps_output
  ps_output="$(ps -axo command 2>/dev/null || true)"

  local -a names=() bases=() keys=() models=() counts=()
  local min_count=999999999

  for line in "${endpoint_lines[@]}"; do
    local name base key model count
    IFS=$'\t' read -r name base key model <<< "$line"
    [[ -n "$name" && -n "$base" && -n "$key" ]] || continue

    count="$(printf '%s\n' "$ps_output" | awk -v b="$base" 'index($0,b) && ($0 ~ /(codex|codex-autogo-v2.py)/) { c++ } END { print c+0 }')"

    names+=("$name")
    bases+=("$base")
    keys+=("$key")
    models+=("${model:-gpt-53-codex}")
    counts+=("$count")

    (( count < min_count )) && min_count="$count"
  done

  local total="${#names[@]}"
  if (( total == 0 )); then
    echo "No usable endpoint entries after parsing"
    return 1
  fi

  local last_idx=-1
  if [[ -f "$LB_STATE_FILE" ]]; then
    last_idx="$(cat "$LB_STATE_FILE" 2>/dev/null || echo -1)"
  fi
  [[ "$last_idx" =~ ^-?[0-9]+$ ]] || last_idx=-1

  local selected_idx=-1 offset idx
  for (( offset=1; offset<=total; offset++ )); do
    idx=$(( (last_idx + offset) % total ))
    if (( counts[idx] == min_count )); then
      selected_idx="$idx"
      break
    fi
  done

  (( selected_idx >= 0 )) || selected_idx=0

  printf '%s\n' "$selected_idx" > "$LB_STATE_FILE" 2>/dev/null || true

  LB_SELECTED_NAME="${names[selected_idx]}"
  LB_SELECTED_BASE_URL="${bases[selected_idx]}"
  LB_SELECTED_API_KEY="${keys[selected_idx]}"
  LB_SELECTED_MODEL="${models[selected_idx]}"
  LB_SELECTED_ACTIVE="${counts[selected_idx]}"
  LB_MIN_ACTIVE="$min_count"

  return 0
}

init_codex_runner
build_endpoint_file_list

acquire_lock() {
  if command -v flock >/dev/null 2>&1; then
    mkdir -p "$LOCK_DIR"
    exec {LOCK_FD}>"$LOCK_FILE"

    if ! flock -n "$LOCK_FD"; then
      local existing_pid=""
      if [[ -f "$LOCK_PID_FILE" ]]; then
        existing_pid="$(cat "$LOCK_PID_FILE" 2>/dev/null || true)"
      fi

      if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
        echo "Loop already running (PID $existing_pid; lock: $LOCK_FILE)"
      else
        echo "Loop already running (lock busy: $LOCK_FILE)"
      fi
      exit 0
    fi

    echo "$$" > "$LOCK_PID_FILE"
    return
  fi

  if mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "$$" > "$LOCK_PID_FILE"
    return
  fi

  local waited=0
  while [[ ! -f "$LOCK_PID_FILE" && "$waited" -lt 5 ]]; do
    sleep 1
    waited=$((waited + 1))
  done

  local existing_pid=""
  if [[ -f "$LOCK_PID_FILE" ]]; then
    existing_pid="$(cat "$LOCK_PID_FILE" 2>/dev/null || true)"
  fi

  if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "Loop already running (PID $existing_pid; lock exists: $LOCK_DIR)"
    exit 0
  fi

  echo "Stale lock detected at $LOCK_DIR; recovering."
  rm -rf "$LOCK_DIR"
  mkdir "$LOCK_DIR"
  echo "$$" > "$LOCK_PID_FILE"
}

active_codex_pid=""
cleanup_loop() {
  if [[ -n "$active_codex_pid" ]] && kill -0 "$active_codex_pid" 2>/dev/null; then
    kill -TERM "$active_codex_pid" 2>/dev/null || true
    sleep 2
    kill -KILL "$active_codex_pid" 2>/dev/null || true
  fi
  rm -f "$LOCK_PID_FILE" 2>/dev/null || true
  if [[ -n "$LOCK_FD" ]]; then
    eval "exec ${LOCK_FD}>&-"
    LOCK_FD=""
  fi
}
trap cleanup_loop EXIT INT TERM

touch_heartbeat() {
  date '+%Y-%m-%d %H:%M:%S %Z' > "$HEARTBEAT_FILE"
}

run_codex_with_watchdog() {
  local final_message="$1"
  local prompt_content="$2"
  local start_epoch now elapsed term_epoch=0 sent_term=0

  set +e
  if [[ "$CODEX_RUNNER_LABEL" == "co" ]]; then
    local -a co_args=(exec --json --cd "$REPO_DIR")
    [[ "$CODEX_FORCE_SKIP_GIT_CHECK" == "1" ]] && co_args+=(--skip-git-repo-check)
    co_args+=(--output-last-message "$final_message" "$prompt_content")

    PYTHONSAFEPATH="$CODEX_PYTHON_SAFEPATH" \
      "${CODEX_RUNNER[@]}" "${co_args[@]}" &
  else
    if ! select_lb_endpoint; then
      set -e
      return 1
    fi

    echo "LB profile: $LB_SELECTED_NAME ($LB_SELECTED_BASE_URL) model: $LB_SELECTED_MODEL active_sessions=$LB_SELECTED_ACTIVE min_active=$LB_MIN_ACTIVE"

    local -a codex_args=(
      -a never
      -s workspace-write
      exec
      --json
      --model "$LB_SELECTED_MODEL"
      -c "model_providers.azure.base_url=$LB_SELECTED_BASE_URL"
      -c "model_providers.azure.wire_api=responses"
      -c "model_providers.azure.timeout=7200"
      -c "model_providers.azure.stream_idle_timeout_ms=3600000"
      -c "model_providers.azure.request_max_retries=10"
      -c "model_providers.azure.stream_max_retries=8"
      --cd "$REPO_DIR"
    )

    [[ "$CODEX_FORCE_SKIP_GIT_CHECK" == "1" ]] && codex_args+=(--skip-git-repo-check)
    codex_args+=(--output-last-message "$final_message" "$prompt_content")

    PYTHONSAFEPATH="$CODEX_PYTHON_SAFEPATH" \
      OPENAI_API_KEY="${OPENAI_API_KEY:-sk-placeholder-key-for-model-refresh}" \
      AZURE_OPENAI_API_KEY="$LB_SELECTED_API_KEY" \
      "${CODEX_RUNNER[@]}" "${codex_args[@]}" &
  fi

  active_codex_pid="$!"
  local codex_pid="$active_codex_pid"
  start_epoch="$(date +%s)"

  echo "Codex PID: $codex_pid (runner=$CODEX_RUNNER_LABEL timeout=${CODEX_TIMEOUT_SECONDS}s, kill-after=${CODEX_KILL_AFTER_SECONDS}s)"

  while kill -0 "$codex_pid" 2>/dev/null; do
    touch_heartbeat
    now="$(date +%s)"
    elapsed=$((now - start_epoch))

    if (( elapsed >= CODEX_TIMEOUT_SECONDS && sent_term == 0 )); then
      echo "Codex watchdog timeout after ${elapsed}s; sending TERM to PID $codex_pid"
      kill -TERM "$codex_pid" 2>/dev/null || true
      sent_term=1
      term_epoch="$(date +%s)"
    elif (( sent_term == 1 )); then
      if (( now - term_epoch >= CODEX_KILL_AFTER_SECONDS )); then
        if kill -0 "$codex_pid" 2>/dev/null; then
          echo "Codex still alive after TERM; sending KILL to PID $codex_pid"
          kill -KILL "$codex_pid" 2>/dev/null || true
        fi
      fi
    fi

    sleep 5
  done

  wait "$codex_pid"
  local exit_code=$?
  active_codex_pid=""

  if (( sent_term == 1 )); then
    return 124
  fi

  return "$exit_code"
}

acquire_lock
cd "$REPO_DIR"
touch_heartbeat

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Not a git repo: $REPO_DIR"
  exit 1
fi

if git show-ref --verify --quiet "refs/heads/$BRANCH_NAME"; then
  git checkout "$BRANCH_NAME"
else
  git checkout -b "$BRANCH_NAME"
fi

iteration=0

while true; do
  iteration=$((iteration + 1))
  timestamp="$(date +%Y%m%d-%H%M%S)"
  run_log="$LOG_DIR/$timestamp.log"
  final_message="$LOG_DIR/$timestamp.final.txt"
  attempt=0
  codex_exit=1
  iteration_outcome="ok"

  {
    echo "=== Codex iteration $iteration @ $timestamp ==="
    echo "Repo: $REPO_DIR"
    echo "Branch: $(git branch --show-current)"
    echo "Runner: $CODEX_RUNNER_LABEL"

    if [[ -n "$(git status --porcelain)" ]]; then
      git add -A
      git commit -m "chore(codex): checkpoint before iteration $timestamp" || true
    fi

    prompt_content="$(cat "$PROMPT_FILE")"$'\n\n'"$NO_NETWORK_GUARD"$'\n\n'"$HOLYC_ONLY_GUARD"

    while (( attempt <= CODEX_MAX_RETRIES )); do
      attempt=$((attempt + 1))
      echo "--- Codex attempt $attempt/$((CODEX_MAX_RETRIES + 1)) ---"
      set +e
      run_codex_with_watchdog "$final_message" "$prompt_content"
      codex_exit="$?"
      set -e

      if [[ "$codex_exit" -eq 0 ]]; then
        break
      fi

      echo "Codex exit code: $codex_exit"

      if (( attempt <= CODEX_MAX_RETRIES )); then
        echo "Retrying in ${CODEX_RETRY_DELAY_SECONDS}s..."
        sleep "$CODEX_RETRY_DELAY_SECONDS"
      fi
    done

    echo "Codex final exit code: $codex_exit"

    if ! "$AIRGAP_GUARD_SCRIPT"; then
      iteration_outcome="guard-failed"
      echo "Air-gap guard failed. Discarding uncommitted iteration changes."
      git reset --hard HEAD
      git clean -fd
    elif [[ "$codex_exit" -ne 0 ]]; then
      iteration_outcome="codex-failed"
      if [[ -n "$(git status --porcelain)" ]]; then
        echo "Codex did not complete cleanly. Discarding uncommitted iteration changes."
        git reset --hard HEAD
        git clean -fd
      fi
    elif [[ -n "$(git status --porcelain)" ]]; then
      git add -A
      git commit -m "feat(inference): codex iteration $timestamp" || true
      git push origin HEAD 2>/dev/null || true
    else
      echo "No changes produced in this iteration."
    fi

    echo "=== End iteration $iteration (outcome=$iteration_outcome) ==="
  } >> "$run_log" 2>&1

  printf 'timestamp=%s\niteration=%s\noutcome=%s\nexit_code=%s\nrun_log=%s\nfinal_message=%s\n' \
    "$timestamp" "$iteration" "$iteration_outcome" "$codex_exit" "$run_log" "$final_message" > "$LAST_RESULT_FILE"
  printf '%s|%s|%s|%s\n' "$timestamp" "$iteration" "$run_log" "$final_message" > "$LAST_ITERATION_FILE"
  touch_heartbeat

  if [[ "$MAX_ITERATIONS" != "0" && "$iteration" -ge "$MAX_ITERATIONS" ]]; then
    echo "Reached MAX_ITERATIONS=$MAX_ITERATIONS" >> "$run_log"
    break
  fi

  sleep "$SLEEP_SECONDS"
done
