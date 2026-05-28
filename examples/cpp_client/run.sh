#!/usr/bin/env bash
# Spawn the tokenizer-server (if not already running on TOKENIZER_PORT) and
# drive every RPC from the C++ client. Exits non-zero on any failure.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PORT="${TOKENIZER_PORT:-50051}"
SERVER_BIN="$HERE/../../target/release/tokenizer-server"
CLIENT_BIN="$HERE/build/cpp_client"

if [ ! -x "$SERVER_BIN" ]; then
  echo "ERROR: server binary not found: $SERVER_BIN" >&2
  echo "Run \`cargo build --release\` from the workspace root first." >&2
  exit 1
fi
if [ ! -x "$CLIENT_BIN" ]; then
  echo "ERROR: client binary not found: $CLIENT_BIN" >&2
  echo "Run \`$HERE/build.sh\` first." >&2
  exit 1
fi

# Start the server unless something is already listening on the chosen port.
SERVER_PID=""
if (echo >/dev/tcp/127.0.0.1/"$PORT") 2>/dev/null; then
  echo "[run.sh] server already running on port $PORT"
else
  LISTEN_ADDR="127.0.0.1:$PORT" "$SERVER_BIN" >/tmp/tokenizer_server.log 2>&1 &
  SERVER_PID=$!
  trap 'if [ -n "$SERVER_PID" ]; then kill "$SERVER_PID" 2>/dev/null || true; fi' EXIT
  for _ in $(seq 1 40); do
    if (echo >/dev/tcp/127.0.0.1/"$PORT") 2>/dev/null; then break; fi
    sleep 0.25
  done
  if ! (echo >/dev/tcp/127.0.0.1/"$PORT") 2>/dev/null; then
    echo "ERROR: server failed to come up on port $PORT" >&2
    echo "--- server log ---" >&2
    cat /tmp/tokenizer_server.log >&2 || true
    exit 1
  fi
fi

"$CLIENT_BIN" "127.0.0.1:$PORT"
