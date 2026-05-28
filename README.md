# tokenizer-service

A Rust gRPC microservice that exposes HuggingFace / tiktoken tokenizers — chat-template
rendering, encode, decode, and server-streaming incremental detokenize.

Primary target models: `deepseek-ai/DeepSeek-V3.1`, `moonshotai/Kimi-K2-Instruct`
and `zai-org/GLM-4.7` — all three ship as offline archives under `models/` with
full chat-template (Jinja2) rendering, encode, decode, and tool-call support.
tiktoken-based OpenAI models (`gpt-4`, `gpt-3.5-turbo`, …) are also supported
for raw encode/decode.

## Layout

```
tokenizer-service/
├── proto/tokenizer.proto              # gRPC IDL (package tokenizer.v1)
├── crates/
│   ├── tokenizer-core/                # adapted from xds-router; HF/tiktoken/mock tokenizers
│   ├── tokenizer-proto/               # tonic-prost-build output (lib only)
│   ├── tokenizer-server/              # gRPC server binary
│   └── tokenizer-tests/               # Rust client + integration tests
└── examples/cpp_client/               # C++ gRPC reference client
```

## Prerequisites

- Rust 1.80+ (workspace uses edition 2021).
- protoc: `apt-get install -y protobuf-compiler` (tonic-prost-build invokes it during the build script).

## Build

```bash
cd tokenizer-service
cargo build --release
```

Binary lands at `target/release/tokenizer-server`.

## Run

```bash
# default 0.0.0.0:50051
./target/release/tokenizer-server

# bind elsewhere
LISTEN_ADDR=127.0.0.1:50051 ./target/release/tokenizer-server
```

### Environment variables

| Variable        | Purpose                                                                 |
|-----------------|-------------------------------------------------------------------------|
| `LISTEN_ADDR`   | Server bind address. Default `0.0.0.0:50051`.                           |
| `RUST_LOG`      | tracing-subscriber env filter. Default `info`.                          |
| `HF_TOKEN`      | HuggingFace API token, required for gated repos.                        |
| `HF_HOME`       | Override the hf-hub cache directory (defaults to `~/.cache/huggingface`).|
| `TOKENIZER_MODELS_DIR` | Path to scan for offline model archives. Default: `<workspace>/models`. |

## Model URIs

The `model` field of every request accepts the following forms:

- `hf://deepseek-ai/DeepSeek-V3.1` — explicit HuggingFace repo, downloads tokenizer.json on first use.
- `deepseek-ai/DeepSeek-V3.1` — bare repo id, treated as HF.
- `file:///abs/path/to/tokenizer.json` — local tokenizer file (chat template loaded from sibling `tokenizer_config.json` if present).
- `gpt-4`, `gpt-3.5-turbo`, etc. — tiktoken (no chat template support, messages concatenated trivially).
- `mock` (alias `test`) — built-in 8-token MockTokenizer, useful for offline testing.
- `kimi://moonshotai/Kimi-K2-Instruct` — Moonshot Kimi tokenizer (downloads `tiktoken.model` + config from HF on first use).
- `model://<friendly-name>` — fully offline; resolved from the on-disk archive under `models/<friendly-name>/`. See "Offline model archive" below.

Loaded tokenizers are cached in-process for the lifetime of the server.

## Offline model archive

Three production tokenizers are shipped as **offline archives** so the server
can serve them with zero network IO at request time:

| URI                              | Backend         | Vocab   | HF source                       |
|----------------------------------|-----------------|---------|---------------------------------|
| `model://deepseek-v3.1`          | huggingface     | 128 000 | `deepseek-ai/DeepSeek-V3.1`     |
| `model://kimi-k2-instruct`       | kimi-tiktoken   | 163 840 | `moonshotai/Kimi-K2-Instruct`   |
| `model://glm-4.7`                | huggingface     | 151 329 | `zai-org/GLM-4.7`               |

Each archive lives at `models/<friendly-name>/` and contains the tokenizer
files (no model weights) plus a `manifest.toml` recording per-file sha256
hashes, the vocab size, and a verify text + expected token-ids round-trip.
Total disk footprint: ~30 MB.

### Producing an archive

```bash
cargo build --release -p archive-model

./target/release/archive-model --name kimi-k2-instruct \
    --repo moonshotai/Kimi-K2-Instruct --backend kimi
./target/release/archive-model --name deepseek-v3.1 \
    --repo deepseek-ai/DeepSeek-V3.1 --backend hf
./target/release/archive-model --name glm-4.7 \
    --repo zai-org/GLM-4.7 --backend hf
```

`archive-model` reuses the standard `~/.cache/huggingface/hub` cache, so a
second run is effectively a copy + sha256.

### Scan & verify on server startup

`Registry::new()` auto-scans (in order) `$TOKENIZER_MODELS_DIR`, the
`models/` directory next to the workspace root, or `./models`. Each archive
discovered is registered as a `model://<name>` URI. Use
`Registry::warmup_and_verify_archived()` to eagerly load every archive and
re-run its verify-text round-trip — any mismatch is logged as an error so an
operator can fail-fast at startup.

```text
INFO archive scan: registered model://kimi-k2-instruct (KimiTiktoken, vocab=163840)
INFO archive scan: registered model://deepseek-v3.1 (Huggingface, vocab=128000)
INFO archive scan: registered model://glm-4.7 (Huggingface, vocab=151329)
INFO archive scan: 3 archived model(s) registered from /…/tokenizer-service/models
INFO verify OK for model://kimi-k2-instruct: 7 tokens (vocab=163840)
```

## gRPC API

See `proto/tokenizer.proto`. Service `tokenizer.v1.Tokenizer` exposes:

- `Tokenize` — chat messages → rendered prompt + token ids (or `raw_text` to skip the jinja step).
- `Detokenize` — token ids → text.
- `DetokenizeStream` — server-streaming incremental decode, one `DetokenizeChunk` per token (UTF-8-safe; multi-byte chars are held until complete).
- `GetModelInfo` — vocab size, BOS/EOS, and whether a chat template is available.

## Testing

Both reference clients exercise the full RPC surface against two offline
backends — `gpt-3.5-turbo` (tiktoken, vocab embedded in the binary, no
network) and the built-in `mock` tokenizer.

### Rust client + integration tests

Crate: `crates/tokenizer-tests/`. The integration test spawns
`target/release/tokenizer-server` as a child process on `127.0.0.1:50061`,
waits for the port, runs every RPC, and tears the server down on drop.

```bash
cargo build --release                       # build the server first
cargo test  --release -p tokenizer-tests -- --nocapture --test-threads=1
```

### Chat-template branch coverage

`tests/template_branches.rs` (35 tests) exercises every major branch of the
three offline templates against a single shared server on `127.0.0.1:50081`.
Coverage per model:

- **DeepSeek V3.1** — multi-system concatenation, multi-turn dialogue, BOS/EOS
  framing, `thinking=true/false` kwarg, `prefix=true` prefilled turns,
  single & multiple tool calls in one assistant turn, full agent loop with
  `<｜tool▁output▁begin｜>` / `<｜tool▁output▁end｜>`, inline `<think>…</think>`
  stripping, CJK + Markdown code blocks with byte-exact roundtrip.
- **Kimi K2** — default-system injection, explicit-system override, tool
  declaration (`<|im_system|>tool_declare<|im_middle|>…`), assistant `tool_calls`
  section markers, `## Return of {id}` tool replies, named `user` and named
  `assistant`, single image and mixed text/image/text multimodal content,
  full agent loop, long multi-turn CJK + Python dialogue.
- **GLM 4.7** — `[gMASK]<sop>` prefix, system/user/assistant turn markers,
  `<tools>…</tools>` block, `reasoning_content` last-turn render &
  old-turn drop, `clear_thinking` / `enable_thinking` kwargs, inline
  `<think>…</think>` parsing branch, `visible_text` macro with list-of-dicts
  content, string vs. structured arg-value branches, consecutive tool replies
  sharing one `<|observation|>`, end-to-end realistic scenario.

```bash
cargo build --release -p tokenizer-server   # the test launches this binary
cargo test  --release --test template_branches -- --test-threads=1
```

There is also a stand-alone demo client. Start a server on `:50051`
(the default), then:

```bash
./target/release/rust_client
# or override the endpoint
TOKENIZER_ADDR=http://127.0.0.1:50052 ./target/release/rust_client
```

For `"Hello, world!"` under `gpt-3.5-turbo` it prints `token_ids=[9906, 11, 1917, 0]`.

### C++ client

Files: `examples/cpp_client/{CMakeLists.txt,main.cc,build.sh,run.sh}`.

CMake prefers the system gRPC (`apt-get install -y cmake libgrpc++-dev
protobuf-compiler-grpc`) so the build finishes in seconds; if you have a
gRPC `CONFIG` install on `CMAKE_PREFIX_PATH` it will be picked up instead.
protoc-generated stubs land under `build/generated/`.

```bash
bash examples/cpp_client/build.sh           # installs deps if missing, compiles
bash examples/cpp_client/run.sh             # spawns a server on :50051, runs the client
```

`run.sh` boots `target/release/tokenizer-server` only if nothing is already
listening, then drives all four RPCs and prints `C++ client tests passed`
on success. Token ids match the Rust client byte-for-byte
(`[9906, 11, 1917, 0]` for `"Hello, world!"`).
