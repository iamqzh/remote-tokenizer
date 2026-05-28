use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use tokenizer_proto::tokenizer_client::TokenizerClient;
use tokenizer_proto::{
    ChatMessage, DetokenizeRequest, DetokenizeStreamRequest, GetModelInfoRequest, TokenizeRequest,
};
use tokio_stream::StreamExt;

const TEST_PORT: u16 = 50061;

fn server_binary() -> PathBuf {
    // Tests live in crates/tokenizer-tests; binary lives in workspace target/release.
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let mut p = PathBuf::from(manifest_dir);
    p.pop(); // crates/
    p.pop(); // workspace root
    p.push("target");
    p.push("release");
    p.push("tokenizer-server");
    p
}

struct ServerGuard {
    child: Child,
}

impl Drop for ServerGuard {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn workspace_root() -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let mut p = PathBuf::from(manifest_dir);
    p.pop(); // crates/
    p.pop(); // workspace root
    p
}

fn spawn_server(port: u16) -> ServerGuard {
    let bin = server_binary();
    assert!(
        bin.exists(),
        "server binary not found at {:?}; run `cargo build --release` first",
        bin
    );
    let models_dir = workspace_root().join("models");
    // RUST_LOG can be overridden by the caller to see the archive scan output.
    let rust_log = std::env::var("TOKENIZER_TEST_LOG").unwrap_or_else(|_| "error".to_string());
    let child = Command::new(&bin)
        .env("LISTEN_ADDR", format!("127.0.0.1:{}", port))
        .env("RUST_LOG", rust_log)
        .env("TOKENIZER_MODELS_DIR", &models_dir)
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("failed to spawn tokenizer-server");
    ServerGuard { child }
}

fn wait_ready(port: u16) {
    let deadline = Instant::now() + Duration::from_secs(15);
    while Instant::now() < deadline {
        if TcpStream::connect(("127.0.0.1", port)).is_ok() {
            // give the gRPC stack one extra moment to be fully wired
            std::thread::sleep(Duration::from_millis(150));
            return;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    panic!("server did not become ready on port {}", port);
}

async fn client(port: u16) -> TokenizerClient<tonic::transport::Channel> {
    let endpoint = format!("http://127.0.0.1:{}", port);
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match TokenizerClient::connect(endpoint.clone()).await {
            Ok(c) => return c,
            Err(e) => {
                if Instant::now() > deadline {
                    panic!("could not connect to server: {e}");
                }
                tokio::time::sleep(Duration::from_millis(150)).await;
            }
        }
    }
}

fn setup() -> ServerGuard {
    let g = spawn_server(TEST_PORT);
    wait_ready(TEST_PORT);
    g
}

#[tokio::test(flavor = "current_thread")]
async fn all_cases() {
    // Run every assertion against a single server instance so that integration
    // tests are cheap to execute under `--test-threads=1`.
    let _g = setup();
    let mut c = client(TEST_PORT).await;

    // ---- model info (tiktoken) ----
    let info = c
        .get_model_info(GetModelInfoRequest {
            model: "gpt-3.5-turbo".into(),
        })
        .await
        .expect("get_model_info")
        .into_inner();
    assert!(info.vocab_size > 0, "tiktoken vocab_size should be > 0");

    // ---- tokenize_raw_text_tiktoken ----
    let raw = c
        .tokenize(TokenizeRequest {
            model: "gpt-3.5-turbo".into(),
            messages: vec![],
            add_generation_prompt: false,
            raw_text: "Hello, world!".into(),
            ..Default::default()
        })
        .await
        .expect("tokenize raw")
        .into_inner();
    assert!(raw.num_tokens > 0);
    assert_eq!(raw.num_tokens as usize, raw.token_ids.len());

    // ---- roundtrip_tiktoken ----
    let det = c
        .detokenize(DetokenizeRequest {
            model: "gpt-3.5-turbo".into(),
            token_ids: raw.token_ids.clone(),
            skip_special_tokens: true,
        })
        .await
        .expect("detokenize")
        .into_inner();
    assert_eq!(det.text, "Hello, world!");

    // ---- detokenize_stream ----
    let mut stream = c
        .detokenize_stream(DetokenizeStreamRequest {
            model: "gpt-3.5-turbo".into(),
            token_ids: raw.token_ids.clone(),
            skip_special_tokens: true,
        })
        .await
        .expect("detokenize_stream")
        .into_inner();
    let mut acc = String::new();
    while let Some(chunk) = stream.next().await {
        let c = chunk.expect("stream chunk");
        acc.push_str(&c.delta);
    }
    assert_eq!(acc, det.text);

    // ---- tokenize_mock_raw_text ----
    let mock_raw = c
        .tokenize(TokenizeRequest {
            model: "mock".into(),
            messages: vec![],
            add_generation_prompt: false,
            raw_text: "hello world".into(),
            ..Default::default()
        })
        .await
        .expect("tokenize mock raw")
        .into_inner();
    assert!(mock_raw.num_tokens > 0);

    // ---- tokenize_mock_messages ----
    let mock_msg = c
        .tokenize(TokenizeRequest {
            model: "mock".into(),
            messages: vec![
                ChatMessage {
                    role: "user".into(),
                    content: "hello".into(),
                    ..Default::default()
                },
                ChatMessage {
                    role: "assistant".into(),
                    content: "world".into(),
                    ..Default::default()
                },
            ],
            add_generation_prompt: false,
            raw_text: "".into(),
            ..Default::default()
        })
        .await
        .expect("tokenize mock messages")
        .into_inner();
    assert!(mock_msg.num_tokens > 0);
    // mock detokenize should round-trip something non-empty
    let mock_det = c
        .detokenize(DetokenizeRequest {
            model: "mock".into(),
            token_ids: mock_raw.token_ids.clone(),
            skip_special_tokens: true,
        })
        .await
        .expect("detokenize mock")
        .into_inner();
    assert!(!mock_det.text.is_empty());

    // ---- model_not_found ----
    // A bogus kimi:// repo must fail (HF will refuse to resolve it).
    let bad = c
        .get_model_info(GetModelInfoRequest {
            model: "kimi://does-not-exist/definitely-not-a-real-repo".into(),
        })
        .await;
    assert!(bad.is_err(), "bogus kimi:// repo should fail to load");
    let err = bad.unwrap_err();
    let code = err.code();
    assert!(
        matches!(
            code,
            tonic::Code::NotFound | tonic::Code::Internal | tonic::Code::Unimplemented
        ),
        "unexpected gRPC code for bogus kimi:// repo: {:?}",
        code
    );
}

/// Offline `model://kimi-k2-instruct`: re-tokenize the verify text from the
/// archive manifest and confirm the ids match the recorded golden values.
#[tokio::test(flavor = "current_thread")]
async fn model_uri_offline_kimi() {
    const PORT: u16 = 50071;
    let _g = spawn_server(PORT);
    wait_ready(PORT);
    let mut c = client(PORT).await;
    let resp = c
        .tokenize(TokenizeRequest {
            model: "model://kimi-k2-instruct".into(),
            messages: vec![],
            add_generation_prompt: false,
            raw_text: "Hello, world! 你好。".into(),
            ..Default::default()
        })
        .await
        .expect("tokenize kimi offline")
        .into_inner();
    assert_eq!(resp.token_ids, vec![19180, 11, 2695, 0, 220, 33845, 292]);

    let info = c
        .get_model_info(GetModelInfoRequest {
            model: "model://kimi-k2-instruct".into(),
        })
        .await
        .expect("get_model_info kimi offline")
        .into_inner();
    assert_eq!(info.vocab_size, 163840);
    assert!(info.has_chat_template);
}

/// Offline `model://deepseek-v3.1`.
#[tokio::test(flavor = "current_thread")]
async fn model_uri_offline_deepseek() {
    const PORT: u16 = 50072;
    let _g = spawn_server(PORT);
    wait_ready(PORT);
    let mut c = client(PORT).await;
    let resp = c
        .tokenize(TokenizeRequest {
            model: "model://deepseek-v3.1".into(),
            messages: vec![],
            add_generation_prompt: false,
            raw_text: "Hello, world! 你好。".into(),
            ..Default::default()
        })
        .await
        .expect("tokenize deepseek offline")
        .into_inner();
    assert_eq!(resp.token_ids, vec![19923, 14, 2058, 3, 223, 30594, 320]);

    let info = c
        .get_model_info(GetModelInfoRequest {
            model: "model://deepseek-v3.1".into(),
        })
        .await
        .expect("get_model_info deepseek offline")
        .into_inner();
    assert_eq!(info.vocab_size, 128000);
}

/// Offline `model://glm-4.7`.
#[tokio::test(flavor = "current_thread")]
async fn model_uri_offline_glm() {
    const PORT: u16 = 50073;
    let _g = spawn_server(PORT);
    wait_ready(PORT);
    let mut c = client(PORT).await;
    let resp = c
        .tokenize(TokenizeRequest {
            model: "model://glm-4.7".into(),
            messages: vec![],
            add_generation_prompt: false,
            raw_text: "Hello, world! 你好。".into(),
            ..Default::default()
        })
        .await
        .expect("tokenize glm offline")
        .into_inner();
    assert_eq!(resp.token_ids, vec![9703, 11, 1879, 0, 101726, 98384, 1773]);

    let info = c
        .get_model_info(GetModelInfoRequest {
            model: "model://glm-4.7".into(),
        })
        .await
        .expect("get_model_info glm offline")
        .into_inner();
    assert_eq!(info.vocab_size, 151329);
    assert!(info.has_chat_template);
}

/// Sanity check: `model://kimi-k2-instruct` (offline archive) must produce the
/// **same** token_ids as the legacy `kimi://moonshotai/Kimi-K2-Instruct` URI
/// for the same bilingual sample.
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires network access to HuggingFace"]
async fn model_uri_matches_kimi_uri() {
    const PORT: u16 = 50074;
    let _g = spawn_server(PORT);
    wait_ready(PORT);
    let mut c = client(PORT).await;
    let txt = "Hello, Kimi! 你好。";
    let offline = c
        .tokenize(TokenizeRequest {
            model: "model://kimi-k2-instruct".into(),
            messages: vec![],
            add_generation_prompt: false,
            raw_text: txt.into(),
            ..Default::default()
        })
        .await
        .expect("offline tokenize")
        .into_inner();
    let online = c
        .tokenize(TokenizeRequest {
            model: "kimi://moonshotai/Kimi-K2-Instruct".into(),
            messages: vec![],
            add_generation_prompt: false,
            raw_text: txt.into(),
            ..Default::default()
        })
        .await
        .expect("online tokenize")
        .into_inner();
    assert_eq!(
        offline.token_ids, online.token_ids,
        "offline model:// and online kimi:// must produce identical ids"
    );
}

// Real-network Kimi K2 roundtrip. Ignored by default — opt in with `--ignored`.
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires network access to HuggingFace"]
async fn kimi_k2_roundtrip() {
    const PORT: u16 = 50062;
    let _g = spawn_server(PORT);
    wait_ready(PORT);
    let mut c = client(PORT).await;

    let model = "kimi://moonshotai/Kimi-K2-Instruct";

    // 1) raw_text — bilingual sample.
    let resp = c
        .tokenize(TokenizeRequest {
            model: model.into(),
            messages: vec![],
            add_generation_prompt: false,
            raw_text: "Hello, Kimi! 你好。".into(),
            ..Default::default()
        })
        .await
        .expect("tokenize raw")
        .into_inner();
    println!(
        "[kimi raw] prompt={:?} num_tokens={} token_ids={:?}",
        resp.prompt, resp.num_tokens, resp.token_ids
    );
    assert!(resp.num_tokens > 0);
    assert_eq!(resp.num_tokens as usize, resp.token_ids.len());

    // 2) chat mode — should use the Kimi chat template.
    let resp_chat = c
        .tokenize(TokenizeRequest {
            model: model.into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: "Hi Kimi".into(),
                ..Default::default()
            }],
            add_generation_prompt: true,
            raw_text: "".into(),
            ..Default::default()
        })
        .await
        .expect("tokenize chat")
        .into_inner();
    println!(
        "[kimi chat] prompt={:?} num_tokens={}",
        resp_chat.prompt, resp_chat.num_tokens
    );
    assert!(resp_chat.num_tokens > 0);
    assert!(!resp_chat.prompt.is_empty());

    // 3) GetModelInfo should advertise the chat template.
    let info = c
        .get_model_info(GetModelInfoRequest {
            model: model.into(),
        })
        .await
        .expect("get_model_info kimi")
        .into_inner();
    println!(
        "[kimi info] vocab_size={} bos={} eos={} has_chat_template={}",
        info.vocab_size, info.bos_token, info.eos_token, info.has_chat_template
    );
    assert!(info.vocab_size >= 163840);
    assert!(info.has_chat_template);
    assert!(!info.bos_token.is_empty());
    assert!(!info.eos_token.is_empty());

    // 4) Roundtrip (no special-skip).
    let dec = c
        .detokenize(DetokenizeRequest {
            model: model.into(),
            token_ids: resp.token_ids.clone(),
            skip_special_tokens: false,
        })
        .await
        .expect("detokenize")
        .into_inner();
    assert_eq!(dec.text, resp.prompt);
}
