#!/usr/bin/env bash
# Start everything for local development and open the UI.
#
#   ./dev.sh            agent server (+ web dashboard once web/ exists), opens the browser
#   ./dev.sh --no-open  same, without opening a browser
#
# Starts:
#   1. the LangGraph Agent Server on :2024 (the analyst graph plus the dashboard
#      routes /meta, /charts, /geo, /feedback);
#   2. the Nuxt dashboard on :3000, when web/ exists (Plan W2).
# Until the dashboard exists it opens LangGraph Studio on the local server instead.
# Ctrl+C stops everything it started. Logs go to .dev/.

set -euo pipefail
cd "$(dirname "$0")"

AGENT_PORT=2024
WEB_PORT=3000
LOG_DIR=.dev
OPEN_BROWSER=1
[[ "${1:-}" == "--no-open" ]] && OPEN_BROWSER=0

say() { printf '\033[1;34m[dev]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[dev]\033[0m %s\n' "$*" >&2; exit 1; }

open_url() {
  [[ $OPEN_BROWSER == 1 ]] || { say "open $1"; return; }
  if command -v xdg-open >/dev/null; then xdg-open "$1" >/dev/null 2>&1 &
  elif command -v open >/dev/null; then open "$1"
  else say "open $1"; fi
}

wait_for() {  # url, name, seconds
  local url=$1 name=$2 budget=$3
  local deadline=$((SECONDS + budget))
  while ((SECONDS < deadline)); do
    curl -fsS --max-time 2 -o /dev/null "$url" 2>/dev/null && return 0
    for pid in "${PIDS[@]}"; do
      kill -0 "$pid" 2>/dev/null || die "$name exited during startup; see $LOG_DIR/"
    done
    sleep 1
  done
  die "$name did not come up at $url within ${budget}s; see $LOG_DIR/"
}

PIDS=()
cleanup() {
  local status=$?
  trap - INT TERM EXIT
  [[ ${#PIDS[@]} -gt 0 ]] || exit "$status"
  say "stopping…"
  for pid in "${PIDS[@]}"; do
    kill -INT "$pid" 2>/dev/null || true
  done
  for pid in "${PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM EXIT

# --- preflight ---------------------------------------------------------------
command -v uv >/dev/null || die "uv is not installed (https://docs.astral.sh/uv/)"
[[ -f .env ]] || die "no .env — copy .env.example to .env and fill in the keys"
compgen -G "data/raw/*.parquet" >/dev/null \
  || say "warning: no data in data/raw — run scripts/download_data.sh"
if command -v docker >/dev/null && docker info >/dev/null 2>&1; then
  docker image inspect analyst-sandbox:1 >/dev/null 2>&1 \
    || { say "building the run_python sandbox image…"; make sandbox-image; }
else
  say "warning: Docker is not running — run_python will be unavailable"
fi
if curl -fsS --max-time 2 -o /dev/null "http://127.0.0.1:$AGENT_PORT/ok" 2>/dev/null; then
  die "something is already serving on :$AGENT_PORT (another ./dev.sh or make dev?)"
fi
mkdir -p "$LOG_DIR"

# --- agent server ------------------------------------------------------------
say "starting the agent server on :$AGENT_PORT (log: $LOG_DIR/agent.log)"
uv run langgraph dev --no-browser --port "$AGENT_PORT" --n-jobs-per-worker 10 \
  >"$LOG_DIR/agent.log" 2>&1 &
PIDS+=($!)
wait_for "http://127.0.0.1:$AGENT_PORT/ok" "agent server" 120
say "agent server ready: http://127.0.0.1:$AGENT_PORT  (dashboard data: /meta)"

# --- web dashboard (Plan W2) ---------------------------------------------------
if [[ -f web/package.json ]]; then
  command -v pnpm >/dev/null || die "pnpm is not installed (corepack enable pnpm)"
  [[ -d web/node_modules ]] || { say "installing web dependencies…"; (cd web && pnpm install); }
  say "starting the dashboard on :$WEB_PORT (log: $LOG_DIR/web.log)"
  (cd web && exec pnpm dev --port "$WEB_PORT") >"$LOG_DIR/web.log" 2>&1 &
  PIDS+=($!)
  wait_for "http://localhost:$WEB_PORT" "dashboard" 120
  say "dashboard ready: http://localhost:$WEB_PORT"
  open_url "http://localhost:$WEB_PORT"
else
  STUDIO="https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:$AGENT_PORT"
  say "no web/ yet (the Nuxt dashboard is Plan W2) — opening LangGraph Studio instead"
  open_url "$STUDIO"
fi

say "running — Ctrl+C to stop"
wait -n "${PIDS[@]}" || true
say "a process exited; see $LOG_DIR/"
