//! Chat template branch coverage tests for DeepSeek V3.1, Kimi K2 and GLM 4.7.
//!
//! Each model exercises the major branches of its own chat_template (tool
//! calls, multi-system / multi-turn rendering, multimodal content, reasoning
//! blocks, thinking-mode toggles, named speakers, …). All tests share a
//! single tokenizer-server process via a OnceCell so the cargo test pass is
//! fast and deterministic. The server is launched on a unique port (50081)
//! distinct from the ports used by `tests/integration.rs` so the suites can
//! run in parallel if necessary.

use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use tokenizer_proto::tokenizer_client::TokenizerClient;
use tokenizer_proto::{ChatMessage, DetokenizeRequest, TokenizeRequest};
use tonic::transport::Channel;

const PORT: u16 = 50081;

fn server_binary() -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let mut p = PathBuf::from(manifest_dir);
    p.pop();
    p.pop();
    p.push("target");
    p.push("release");
    p.push("tokenizer-server");
    p
}

fn workspace_root() -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let mut p = PathBuf::from(manifest_dir);
    p.pop();
    p.pop();
    p
}

struct ServerHandle {
    _child: Child,
}

fn spawn_server() -> ServerHandle {
    let bin = server_binary();
    assert!(
        bin.exists(),
        "server binary not found at {:?}; run `cargo build --release` first",
        bin
    );
    let models_dir = workspace_root().join("models");
    let rust_log = std::env::var("TOKENIZER_TEST_LOG").unwrap_or_else(|_| "error".to_string());
    let child = Command::new(&bin)
        .env("LISTEN_ADDR", format!("127.0.0.1:{}", PORT))
        .env("RUST_LOG", rust_log)
        .env("TOKENIZER_MODELS_DIR", &models_dir)
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn tokenizer-server");
    wait_ready();
    ServerHandle { _child: child }
}

fn wait_ready() {
    let deadline = Instant::now() + Duration::from_secs(20);
    while Instant::now() < deadline {
        if TcpStream::connect(("127.0.0.1", PORT)).is_ok() {
            std::thread::sleep(Duration::from_millis(150));
            return;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    panic!("server did not become ready on port {}", PORT);
}

// We can't store the server child in a OnceCell because Child isn't Sync, but
// because the test harness keeps the process alive for the entire run we can
// leak the guard. Use a static AtomicBool gate instead.
fn ensure_server() {
    static INIT: OnceLock<()> = OnceLock::new();
    INIT.get_or_init(|| {
        let h = spawn_server();
        // Intentionally leak: the OS will reap when the test binary exits.
        std::mem::forget(h);
    });
}

async fn make_client() -> TokenizerClient<Channel> {
    ensure_server();
    let endpoint = format!("http://127.0.0.1:{}", PORT);
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

// ---- helpers ----------------------------------------------------------------

fn msg(role: &str, content: &str) -> ChatMessage {
    ChatMessage {
        role: role.into(),
        content: content.into(),
        ..Default::default()
    }
}

fn named_msg(role: &str, name: &str, content: &str) -> ChatMessage {
    ChatMessage {
        role: role.into(),
        content: content.into(),
        name: name.into(),
        ..Default::default()
    }
}

fn tool_calls_msg(content: &str, tool_calls_json: &str) -> ChatMessage {
    ChatMessage {
        role: "assistant".into(),
        content: content.into(),
        tool_calls_json: tool_calls_json.into(),
        ..Default::default()
    }
}

fn tool_response_msg(tool_call_id: &str, content: &str) -> ChatMessage {
    ChatMessage {
        role: "tool".into(),
        content: content.into(),
        tool_call_id: tool_call_id.into(),
        ..Default::default()
    }
}

fn req(model: &str, messages: Vec<ChatMessage>, add_gen: bool) -> TokenizeRequest {
    TokenizeRequest {
        model: model.into(),
        messages,
        add_generation_prompt: add_gen,
        raw_text: String::new(),
        tools_json: String::new(),
        template_kwargs_json: String::new(),
    }
}

fn req_with_tools(
    model: &str,
    messages: Vec<ChatMessage>,
    add_gen: bool,
    tools_json: &str,
) -> TokenizeRequest {
    let mut r = req(model, messages, add_gen);
    r.tools_json = tools_json.into();
    r
}

fn req_with_kwargs(
    model: &str,
    messages: Vec<ChatMessage>,
    add_gen: bool,
    kwargs_json: &str,
) -> TokenizeRequest {
    let mut r = req(model, messages, add_gen);
    r.template_kwargs_json = kwargs_json.into();
    r
}

fn assert_contains_all(prompt: &str, needles: &[&str]) {
    for n in needles {
        assert!(
            prompt.contains(n),
            "prompt missing {:?}\n----- full prompt -----\n{}\n-----------------------",
            n,
            prompt
        );
    }
}

fn count_occurrences(haystack: &str, needle: &str) -> usize {
    haystack.matches(needle).count()
}

/// Round-trip the prompt through detokenize(skip_special_tokens=false) and
/// assert the decode at least contains a meaningful prefix of the original.
/// HuggingFace BPE tokenizers (GLM, DeepSeek) are byte-lossless under
/// skip_special=false; Kimi's tiktoken backend is similarly lossless for the
/// covered special tokens.
async fn assert_roundtrip(
    client: &mut TokenizerClient<Channel>,
    model: &str,
    prompt: &str,
    token_ids: &[u32],
) {
    let dec = client
        .detokenize(DetokenizeRequest {
            model: model.into(),
            token_ids: token_ids.to_vec(),
            skip_special_tokens: false,
        })
        .await
        .expect("detokenize")
        .into_inner();
    // For most cases dec.text == prompt. For at least the first 30 chars it
    // should always agree.
    let prefix_end = prompt
        .char_indices()
        .nth(30)
        .map(|(i, _)| i)
        .unwrap_or(prompt.len());
    let prefix = &prompt[..prefix_end];
    assert!(
        dec.text.contains(prefix),
        "roundtrip diverged for {model}\nprompt prefix={:?}\ndec.text={:?}",
        prefix,
        dec.text
    );
}

// =============================================================================
// DeepSeek V3.1
// =============================================================================

const DEEPSEEK: &str = "model://deepseek-v3.1";

#[tokio::test(flavor = "current_thread")]
async fn deepseek_multi_system_concat() {
    let mut c = make_client().await;
    let resp = c
        .tokenize(req(
            DEEPSEEK,
            vec![
                msg("system", "First system prompt."),
                msg("system", "Second system prompt."),
                msg("user", "Hi."),
            ],
            false,
        ))
        .await
        .unwrap()
        .into_inner();
    assert!(resp.num_tokens > 0);
    assert!(!resp.prompt.is_empty());
    // Both system messages must be concatenated with a \n\n separator.
    assert_contains_all(
        &resp.prompt,
        &[
            "<\u{ff5c}begin\u{2581}of\u{2581}sentence\u{ff5c}>",
            "First system prompt.",
            "Second system prompt.",
            "\n\n",
            "<\u{ff5c}User\u{ff5c}>Hi.",
        ],
    );
    assert_eq!(count_occurrences(&resp.prompt, "<\u{ff5c}User\u{ff5c}>"), 1);
    assert_roundtrip(&mut c, DEEPSEEK, &resp.prompt, &resp.token_ids).await;
}

#[tokio::test(flavor = "current_thread")]
async fn deepseek_multi_turn() {
    let mut c = make_client().await;
    let resp = c
        .tokenize(req(
            DEEPSEEK,
            vec![
                msg("system", "Be brief."),
                msg("user", "What is 1+1?"),
                msg("assistant", "2"),
                msg("user", "And 2+2?"),
            ],
            true,
        ))
        .await
        .unwrap()
        .into_inner();
    assert!(resp.num_tokens > 0);
    assert_eq!(count_occurrences(&resp.prompt, "<\u{ff5c}User\u{ff5c}>"), 2);
    assert!(
        count_occurrences(&resp.prompt, "<\u{ff5c}Assistant\u{ff5c}>") >= 1,
        "expected at least one <｜Assistant｜> marker, got {:?}",
        resp.prompt
    );
    assert!(resp
        .prompt
        .contains("<\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}>"));
    assert_contains_all(&resp.prompt, &["Be brief.", "What is 1+1?", "2", "And 2+2?"]);
    assert_roundtrip(&mut c, DEEPSEEK, &resp.prompt, &resp.token_ids).await;
}

#[tokio::test(flavor = "current_thread")]
async fn deepseek_tool_calls() {
    let mut c = make_client().await;
    let tool_calls = r#"[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Beijing\"}"}}]"#;
    let mut asst = tool_calls_msg("", tool_calls);
    asst.role = "assistant".into();
    let resp = c
        .tokenize(req(
            DEEPSEEK,
            vec![msg("user", "Weather in Beijing?"), asst],
            false,
        ))
        .await
        .unwrap()
        .into_inner();
    assert!(resp.num_tokens > 0);
    assert_contains_all(
        &resp.prompt,
        &[
            "<\u{ff5c}tool\u{2581}calls\u{2581}begin\u{ff5c}>",
            "<\u{ff5c}tool\u{2581}call\u{2581}begin\u{ff5c}>",
            "get_weather",
            "<\u{ff5c}tool\u{2581}sep\u{ff5c}>",
            "Beijing",
            "<\u{ff5c}tool\u{2581}call\u{2581}end\u{ff5c}>",
            "<\u{ff5c}tool\u{2581}calls\u{2581}end\u{ff5c}>",
        ],
    );
    assert_roundtrip(&mut c, DEEPSEEK, &resp.prompt, &resp.token_ids).await;
}

#[tokio::test(flavor = "current_thread")]
async fn deepseek_thinking_mode() {
    let mut c = make_client().await;
    // When `thinking=true` and an assistant turn has prefix=true, the template
    // injects `<think>` instead of `</think>`. We provide an assistant turn
    // with prefix=true so the thinking-mode branch fires.
    let asst = ChatMessage {
        role: "assistant".into(),
        content: "draft response".into(),
        prefix: true,
        ..Default::default()
    };
    let resp = c
        .tokenize(req_with_kwargs(
            DEEPSEEK,
            vec![msg("user", "Hello"), asst],
            false,
            r#"{"thinking": true}"#,
        ))
        .await
        .unwrap()
        .into_inner();
    assert!(resp.num_tokens > 0);
    assert!(
        resp.prompt.contains("<think>"),
        "expected <think> token in thinking mode, prompt was:\n{}",
        resp.prompt
    );
    assert_roundtrip(&mut c, DEEPSEEK, &resp.prompt, &resp.token_ids).await;
}

#[tokio::test(flavor = "current_thread")]
async fn deepseek_no_thinking_default() {
    let mut c = make_client().await;
    let resp = c
        .tokenize(req(
            DEEPSEEK,
            vec![msg("user", "Hello")],
            true, // request a generation prompt — that path emits </think>
        ))
        .await
        .unwrap()
        .into_inner();
    assert!(resp.num_tokens > 0);
    let open_count = count_occurrences(&resp.prompt, "<think>");
    let close_count = count_occurrences(&resp.prompt, "</think>");
    assert_eq!(
        open_count, 0,
        "<think> should not appear in default (thinking=false) mode; got prompt:\n{}",
        resp.prompt
    );
    assert!(
        close_count >= 1,
        "</think> expected at least once; got prompt:\n{}",
        resp.prompt
    );
    assert_roundtrip(&mut c, DEEPSEEK, &resp.prompt, &resp.token_ids).await;
}

#[tokio::test(flavor = "current_thread")]
async fn deepseek_chinese_content() {
    let mut c = make_client().await;
    let resp = c
        .tokenize(req(
            DEEPSEEK,
            vec![
                msg("system", "你是一个有用的助手。"),
                msg("user", "请用中文回答：今天的天气怎么样？"),
            ],
            true,
        ))
        .await
        .unwrap()
        .into_inner();
    assert!(resp.num_tokens > 0);
    assert!(resp.prompt.contains("你是一个有用的助手。"));
    assert!(resp.prompt.contains("请用中文回答"));
    // Roundtrip — Chinese must survive byte-level BPE.
    let dec = c
        .detokenize(DetokenizeRequest {
            model: DEEPSEEK.into(),
            token_ids: resp.token_ids.clone(),
            skip_special_tokens: false,
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(dec.text, resp.prompt, "deepseek bilingual roundtrip diverged");
}

// =============================================================================
// Kimi K2
// =============================================================================

const KIMI: &str = "model://kimi-k2-instruct";

#[tokio::test(flavor = "current_thread")]
async fn kimi_default_system_injected() {
    let mut c = make_client().await;
    let resp = c
        .tokenize(req(KIMI, vec![msg("user", "Hi Kimi")], true))
        .await
        .unwrap()
        .into_inner();
    assert!(resp.num_tokens > 0);
    assert_contains_all(
        &resp.prompt,
        &[
            "<|im_system|>system<|im_middle|>You are Kimi, an AI assistant created by Moonshot AI.<|im_end|>",
            "<|im_user|>user<|im_middle|>",
            "Hi Kimi",
            "<|im_assistant|>assistant<|im_middle|>",
        ],
    );
    assert_roundtrip(&mut c, KIMI, &resp.prompt, &resp.token_ids).await;
}

#[tokio::test(flavor = "current_thread")]
async fn kimi_explicit_system_overrides_default() {
    let mut c = make_client().await;
    let resp = c
        .tokenize(req(
            KIMI,
            vec![
                msg("system", "You are a pirate."),
                msg("user", "ahoy"),
            ],
            false,
        ))
        .await
        .unwrap()
        .into_inner();
    assert!(resp.num_tokens > 0);
    assert!(resp.prompt.contains("You are a pirate."));
    // The default Kimi system must not be injected when an explicit system
    // message is already first.
    assert!(
        !resp
            .prompt
            .contains("You are Kimi, an AI assistant created by Moonshot AI."),
        "default Kimi system leaked despite explicit system override:\n{}",
        resp.prompt
    );
    assert_roundtrip(&mut c, KIMI, &resp.prompt, &resp.token_ids).await;
}

#[tokio::test(flavor = "current_thread")]
async fn kimi_with_tools() {
    let mut c = make_client().await;
    let tools = r#"[{"name":"search","description":"web search","parameters":{"type":"object","properties":{"q":{"type":"string"}}}}]"#;
    let resp = c
        .tokenize(req_with_tools(
            KIMI,
            vec![msg("user", "search rust async")],
            true,
            tools,
        ))
        .await
        .unwrap()
        .into_inner();
    assert!(resp.num_tokens > 0);
    assert_contains_all(
        &resp.prompt,
        &[
            "<|im_system|>tool_declare<|im_middle|>",
            "\"search\"",
            "\"web search\"",
            "<|im_end|>",
        ],
    );
    assert_roundtrip(&mut c, KIMI, &resp.prompt, &resp.token_ids).await;
}

#[tokio::test(flavor = "current_thread")]
async fn kimi_assistant_tool_calls() {
    let mut c = make_client().await;
    let tool_calls = r#"[{"id":"functions.search:0","type":"function","function":{"name":"search","arguments":"{\"q\":\"rust\"}"}}]"#;
    let resp = c
        .tokenize(req(
            KIMI,
            vec![
                msg("user", "search rust"),
                tool_calls_msg("", tool_calls),
            ],
            false,
        ))
        .await
        .unwrap()
        .into_inner();
    assert!(resp.num_tokens > 0);
    assert_contains_all(
        &resp.prompt,
        &[
            "<|tool_calls_section_begin|>",
            "<|tool_call_begin|>functions.search:0",
            "<|tool_call_argument_begin|>",
            "<|tool_call_end|>",
            "<|tool_calls_section_end|>",
        ],
    );
    assert_roundtrip(&mut c, KIMI, &resp.prompt, &resp.token_ids).await;
}

#[tokio::test(flavor = "current_thread")]
async fn kimi_tool_response() {
    let mut c = make_client().await;
    let tool_calls = r#"[{"id":"call_42","type":"function","function":{"name":"add","arguments":"{\"a\":40,\"b\":2}"}}]"#;
    let resp = c
        .tokenize(req(
            KIMI,
            vec![
                msg("user", "what is 40 + 2"),
                tool_calls_msg("", tool_calls),
                tool_response_msg("call_42", "42"),
            ],
            true,
        ))
        .await
        .unwrap()
        .into_inner();
    assert!(resp.num_tokens > 0);
    assert!(
        resp.prompt.contains("## Return of call_42"),
        "expected `## Return of call_42` marker, got:\n{}",
        resp.prompt
    );
    assert!(resp.prompt.contains("42"));
    assert_roundtrip(&mut c, KIMI, &resp.prompt, &resp.token_ids).await;
}

#[tokio::test(flavor = "current_thread")]
async fn kimi_named_user() {
    let mut c = make_client().await;
    let resp = c
        .tokenize(req(
            KIMI,
            vec![named_msg("user", "alice", "Hello!")],
            true,
        ))
        .await
        .unwrap()
        .into_inner();
    assert!(resp.num_tokens > 0);
    assert!(
        resp.prompt.contains("<|im_user|>alice<|im_middle|>"),
        "expected named user marker, got:\n{}",
        resp.prompt
    );
    assert!(!resp.prompt.contains("<|im_user|>user<|im_middle|>"));
    assert_roundtrip(&mut c, KIMI, &resp.prompt, &resp.token_ids).await;
}

#[tokio::test(flavor = "current_thread")]
async fn kimi_multimodal_image() {
    let mut c = make_client().await;
    let parts = r#"[{"type":"text","text":"What is in this?"},{"type":"image","image_url":"data:image/png;base64,AAAA"}]"#;
    let user = ChatMessage {
        role: "user".into(),
        content: String::new(),
        content_parts_json: parts.into(),
        ..Default::default()
    };
    let resp = c
        .tokenize(req(KIMI, vec![user], true))
        .await
        .unwrap()
        .into_inner();
    assert!(resp.num_tokens > 0);
    assert_contains_all(
        &resp.prompt,
        &[
            "<|media_start|>image<|media_content|><|media_pad|><|media_end|>",
            "What is in this?",
        ],
    );
    assert_roundtrip(&mut c, KIMI, &resp.prompt, &resp.token_ids).await;
}

// =============================================================================
// GLM 4.7
// =============================================================================

const GLM: &str = "model://glm-4.7";

#[tokio::test(flavor = "current_thread")]
async fn glm_gmask_prefix() {
    let mut c = make_client().await;
    let resp = c
        .tokenize(req(GLM, vec![msg("user", "Hello GLM")], true))
        .await
        .unwrap()
        .into_inner();
    assert!(resp.num_tokens > 0);
    assert!(
        resp.prompt.starts_with("[gMASK]<sop>"),
        "expected [gMASK]<sop> prefix, got start={:?}",
        &resp.prompt[..resp.prompt.len().min(40)]
    );
    assert_roundtrip(&mut c, GLM, &resp.prompt, &resp.token_ids).await;
}

#[tokio::test(flavor = "current_thread")]
async fn glm_multi_turn() {
    let mut c = make_client().await;
    let resp = c
        .tokenize(req(
            GLM,
            vec![
                msg("system", "Be concise."),
                msg("user", "Q1"),
                msg("assistant", "A1"),
                msg("user", "Q2"),
            ],
            true,
        ))
        .await
        .unwrap()
        .into_inner();
    assert!(resp.num_tokens > 0);
    assert!(resp.prompt.contains("<|system|>"));
    assert_eq!(count_occurrences(&resp.prompt, "<|user|>"), 2);
    assert!(count_occurrences(&resp.prompt, "<|assistant|>") >= 1);
    assert_contains_all(&resp.prompt, &["Be concise.", "Q1", "A1", "Q2"]);
    assert_roundtrip(&mut c, GLM, &resp.prompt, &resp.token_ids).await;
}

#[tokio::test(flavor = "current_thread")]
async fn glm_with_tools() {
    let mut c = make_client().await;
    let tools = r#"[{"name":"get_time","description":"return current time","parameters":{"type":"object","properties":{}}}]"#;
    let resp = c
        .tokenize(req_with_tools(
            GLM,
            vec![msg("user", "what time is it")],
            true,
            tools,
        ))
        .await
        .unwrap()
        .into_inner();
    assert!(resp.num_tokens > 0);
    assert_contains_all(
        &resp.prompt,
        &["# Tools", "<tools>", "</tools>", "get_time"],
    );
    assert_roundtrip(&mut c, GLM, &resp.prompt, &resp.token_ids).await;
}

#[tokio::test(flavor = "current_thread")]
async fn glm_reasoning_content_last_turn() {
    let mut c = make_client().await;
    let asst = ChatMessage {
        role: "assistant".into(),
        content: "The answer is 42.".into(),
        reasoning_content: "Let me think about it.".into(),
        ..Default::default()
    };
    let resp = c
        .tokenize(req(
            GLM,
            vec![msg("user", "Tell me the answer."), asst],
            false,
        ))
        .await
        .unwrap()
        .into_inner();
    assert!(resp.num_tokens > 0);
    assert!(
        resp.prompt.contains("<think>Let me think about it.</think>"),
        "expected reasoning_content rendered as <think>...</think> in last assistant turn, prompt was:\n{}",
        resp.prompt
    );
    assert_roundtrip(&mut c, GLM, &resp.prompt, &resp.token_ids).await;
}

#[tokio::test(flavor = "current_thread")]
async fn glm_reasoning_cleared_old_turn() {
    let mut c = make_client().await;
    let old_asst = ChatMessage {
        role: "assistant".into(),
        content: "old answer".into(),
        reasoning_content: "OLD_REASONING_SHOULD_BE_DROPPED".into(),
        ..Default::default()
    };
    let resp = c
        .tokenize(req(
            GLM,
            vec![
                msg("user", "q1"),
                old_asst,
                msg("user", "q2"),
                msg("assistant", "new answer"),
            ],
            false,
        ))
        .await
        .unwrap()
        .into_inner();
    assert!(resp.num_tokens > 0);
    assert!(
        !resp.prompt.contains("OLD_REASONING_SHOULD_BE_DROPPED"),
        "old reasoning_content leaked into the prompt:\n{}",
        resp.prompt
    );
    assert!(resp.prompt.contains("new answer"));
    assert!(resp.prompt.contains("old answer"));
    assert_roundtrip(&mut c, GLM, &resp.prompt, &resp.token_ids).await;
}

#[tokio::test(flavor = "current_thread")]
async fn glm_tool_calls() {
    let mut c = make_client().await;
    // GLM expects arguments as an object (not a JSON string). We feed it raw.
    let tool_calls = r#"[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":{"city":"Beijing"}}}]"#;
    let resp = c
        .tokenize(req(
            GLM,
            vec![
                msg("user", "weather in Beijing"),
                tool_calls_msg("", tool_calls),
            ],
            false,
        ))
        .await
        .unwrap()
        .into_inner();
    assert!(resp.num_tokens > 0);
    assert_contains_all(
        &resp.prompt,
        &[
            "<tool_call>",
            "get_weather",
            // Template: `v | tojson if v is not string else v` — strings are
            // inlined raw, so "Beijing" comes out without surrounding quotes.
            "<arg_key>city</arg_key><arg_value>Beijing</arg_value>",
            "</tool_call>",
        ],
    );
    assert_roundtrip(&mut c, GLM, &resp.prompt, &resp.token_ids).await;
}

#[tokio::test(flavor = "current_thread")]
async fn glm_tool_response() {
    let mut c = make_client().await;
    let tool_calls = r#"[{"id":"call_1","type":"function","function":{"name":"add","arguments":{"a":40,"b":2}}}]"#;
    let resp = c
        .tokenize(req(
            GLM,
            vec![
                msg("user", "compute 40+2"),
                tool_calls_msg("", tool_calls),
                tool_response_msg("call_1", "42"),
            ],
            true,
        ))
        .await
        .unwrap()
        .into_inner();
    assert!(resp.num_tokens > 0);
    assert_contains_all(
        &resp.prompt,
        &["<|observation|>", "<tool_response>", "42", "</tool_response>"],
    );
    assert_roundtrip(&mut c, GLM, &resp.prompt, &resp.token_ids).await;
}

// =============================================================================
// DeepSeek V3.1 — complex / multi-branch scenarios
// =============================================================================

/// Full agent loop: user → assistant(tool_calls) → tool response → assistant(text).
/// Exercises `<｜tool▁output▁begin｜>` / `<｜tool▁output▁end｜>` plus the
/// `ns.is_tool` flag transition that switches the next assistant turn from the
/// "after-user" path to the "after-tool" path.
#[tokio::test(flavor = "current_thread")]
async fn deepseek_tool_output_full_loop() {
    let mut c = make_client().await;
    let tool_calls = r#"[{"id":"call_w","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Beijing\",\"unit\":\"celsius\"}"}}]"#;
    let resp = c
        .tokenize(req(
            DEEPSEEK,
            vec![
                msg("system", "You are a weather assistant. Be concise."),
                msg("user", "How hot is it in Beijing right now?"),
                tool_calls_msg("", tool_calls),
                tool_response_msg("call_w", "{\"temp_c\":31,\"summary\":\"sunny\"}"),
                msg("assistant", "It is 31°C and sunny in Beijing."),
            ],
            true,
        ))
        .await
        .unwrap()
        .into_inner();
    assert!(resp.num_tokens > 0);
    assert_contains_all(
        &resp.prompt,
        &[
            "You are a weather assistant. Be concise.",
            "How hot is it in Beijing right now?",
            "<\u{ff5c}tool\u{2581}calls\u{2581}begin\u{ff5c}>",
            "<\u{ff5c}tool\u{2581}call\u{2581}begin\u{ff5c}>",
            "get_weather",
            "<\u{ff5c}tool\u{2581}sep\u{ff5c}>",
            "\"city\":\"Beijing\"",
            "<\u{ff5c}tool\u{2581}call\u{2581}end\u{ff5c}>",
            "<\u{ff5c}tool\u{2581}calls\u{2581}end\u{ff5c}>",
            "<\u{ff5c}tool\u{2581}output\u{2581}begin\u{ff5c}>",
            "\"temp_c\":31",
            "<\u{ff5c}tool\u{2581}output\u{2581}end\u{ff5c}>",
            "It is 31°C and sunny in Beijing.",
        ],
    );
    // After a tool reply the assistant turn that follows must NOT re-emit the
    // `<｜Assistant｜>` header (the `ns.is_tool` branch swallows it). The final
    // generation prompt does add one fresh assistant marker.
    let assistant_markers = count_occurrences(&resp.prompt, "<\u{ff5c}Assistant\u{ff5c}>");
    assert!(
        (1..=2).contains(&assistant_markers),
        "expected 1-2 <｜Assistant｜> markers (tool branch suppresses the inner one), got {} in:\n{}",
        assistant_markers,
        resp.prompt
    );
    assert_roundtrip(&mut c, DEEPSEEK, &resp.prompt, &resp.token_ids).await;
}

/// Multiple tool calls in one assistant turn exercise the `ns.is_first` flag:
/// only the *first* call must get `<｜tool▁calls▁begin｜>`, the others reuse the
/// outer block. The shared `<｜tool▁calls▁end｜>` is emitted once at the end.
#[tokio::test(flavor = "current_thread")]
async fn deepseek_multi_tool_calls_one_turn() {
    let mut c = make_client().await;
    let tool_calls = r#"[
        {"id":"c1","type":"function","function":{"name":"search","arguments":"{\"q\":\"rust async\"}"}},
        {"id":"c2","type":"function","function":{"name":"search","arguments":"{\"q\":\"tokio runtime\"}"}},
        {"id":"c3","type":"function","function":{"name":"summarize","arguments":"{\"limit\":3}"}}
    ]"#;
    let resp = c
        .tokenize(req(
            DEEPSEEK,
            vec![
                msg("user", "Find me 3 things about Rust async."),
                tool_calls_msg("Working on it.", tool_calls),
            ],
            false,
        ))
        .await
        .unwrap()
        .into_inner();
    assert!(resp.num_tokens > 0);
    // Exactly ONE outer block, but THREE inner call begin/end pairs.
    assert_eq!(
        count_occurrences(&resp.prompt, "<\u{ff5c}tool\u{2581}calls\u{2581}begin\u{ff5c}>"),
        1,
        "expected exactly one tool_calls_begin, got prompt:\n{}",
        resp.prompt
    );
    assert_eq!(
        count_occurrences(&resp.prompt, "<\u{ff5c}tool\u{2581}calls\u{2581}end\u{ff5c}>"),
        1
    );
    assert_eq!(
        count_occurrences(&resp.prompt, "<\u{ff5c}tool\u{2581}call\u{2581}begin\u{ff5c}>"),
        3
    );
    assert_eq!(
        count_occurrences(&resp.prompt, "<\u{ff5c}tool\u{2581}call\u{2581}end\u{ff5c}>"),
        3
    );
    assert_contains_all(&resp.prompt, &["Working on it.", "rust async", "tokio runtime", "summarize"]);
    assert_roundtrip(&mut c, DEEPSEEK, &resp.prompt, &resp.token_ids).await;
}

/// When assistant `content` itself contains `<think>…</think>` the template
/// strips everything up to and including the first `</think>` before emitting
/// the cleaned content (`content.split('</think>', 1)[1]`). The reasoning
/// portion must NOT leak into the final prompt.
#[tokio::test(flavor = "current_thread")]
async fn deepseek_content_with_inline_think() {
    let mut c = make_client().await;
    let asst = msg(
        "assistant",
        "<think>scratchpad reasoning SHOULD_BE_STRIPPED</think>The capital of France is Paris.",
    );
    let resp = c
        .tokenize(req(
            DEEPSEEK,
            vec![msg("user", "What is the capital of France?"), asst],
            false,
        ))
        .await
        .unwrap()
        .into_inner();
    assert!(resp.num_tokens > 0);
    assert!(
        !resp.prompt.contains("SHOULD_BE_STRIPPED"),
        "in-content reasoning leaked despite </think> split branch:\n{}",
        resp.prompt
    );
    assert!(resp.prompt.contains("The capital of France is Paris."));
    assert_roundtrip(&mut c, DEEPSEEK, &resp.prompt, &resp.token_ids).await;
}

/// Realistic complex content: code-block fenced markdown, bilingual text,
/// quoted JSON and a multi-turn dialogue. Verifies that the BPE roundtrip
/// preserves every byte of the prompt even with this much variety.
#[tokio::test(flavor = "current_thread")]
async fn deepseek_realistic_code_review() {
    let mut c = make_client().await;
    let user_q = "请评审下面这段 Rust 代码并指出潜在 bug：\n```rust\nfn divide(a: i32, b: i32) -> i32 {\n    a / b\n}\n```";
    let asst_a = "代码存在两个问题：\n1. `b == 0` 时会 panic — 应改为返回 `Result<i32, &'static str>`。\n2. 整数除法会向零截断，浮点场景需使用 `f64`。\n示例：\n```rust\nfn divide(a: i32, b: i32) -> Result<i32, &'static str> {\n    if b == 0 { return Err(\"divide by zero\"); }\n    Ok(a / b)\n}\n```";
    let resp = c
        .tokenize(req(
            DEEPSEEK,
            vec![
                msg("system", "You are a senior Rust code reviewer."),
                msg("system", "Always answer in the user's language."),
                msg("user", user_q),
                msg("assistant", asst_a),
                msg("user", "再帮我加一个单元测试。"),
            ],
            true,
        ))
        .await
        .unwrap()
        .into_inner();
    assert!(resp.num_tokens > 30);
    assert_contains_all(
        &resp.prompt,
        &[
            "You are a senior Rust code reviewer.",
            "Always answer in the user's language.",
            // The two-system aggregation joins with \n\n
            "You are a senior Rust code reviewer.\n\nAlways answer in the user's language.",
            "fn divide(a: i32, b: i32)",
            "Result<i32, &'static str>",
            "再帮我加一个单元测试。",
            "<\u{ff5c}User\u{ff5c}>",
            "<\u{ff5c}Assistant\u{ff5c}>",
        ],
    );
    assert_eq!(count_occurrences(&resp.prompt, "<\u{ff5c}User\u{ff5c}>"), 2);
    // Full lossless roundtrip — every byte (including triple-backtick fences
    // and CJK punctuation) must survive.
    let dec = c
        .detokenize(DetokenizeRequest {
            model: DEEPSEEK.into(),
            token_ids: resp.token_ids.clone(),
            skip_special_tokens: false,
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(dec.text, resp.prompt, "deepseek code-review roundtrip diverged");
}

// =============================================================================
// Kimi K2 — complex / multi-branch scenarios
// =============================================================================

/// `role_name = message.get('name') or message['role']` also applies to the
/// assistant branch — a named assistant must render `<|im_assistant|>{name}<|im_middle|>`
/// instead of the default `<|im_assistant|>assistant<|im_middle|>`.
#[tokio::test(flavor = "current_thread")]
async fn kimi_named_assistant() {
    let mut c = make_client().await;
    let resp = c
        .tokenize(req(
            KIMI,
            vec![
                msg("user", "Who are you?"),
                named_msg("assistant", "Kimi-Pro", "I am Kimi-Pro, a specialised assistant."),
            ],
            false,
        ))
        .await
        .unwrap()
        .into_inner();
    assert!(resp.num_tokens > 0);
    assert!(
        resp.prompt.contains("<|im_assistant|>Kimi-Pro<|im_middle|>"),
        "expected named assistant marker, got:\n{}",
        resp.prompt
    );
    assert!(!resp.prompt.contains("<|im_assistant|>assistant<|im_middle|>I am Kimi-Pro"));
    assert_roundtrip(&mut c, KIMI, &resp.prompt, &resp.token_ids).await;
}

/// Multiple tool calls inside a single assistant turn — the `{% for tool_call %}`
/// loop must emit one `<|tool_call_begin|>…<|tool_call_end|>` pair per call,
/// wrapped by a single `<|tool_calls_section_begin|>…<|tool_calls_section_end|>`.
#[tokio::test(flavor = "current_thread")]
async fn kimi_multi_tool_calls_one_turn() {
    let mut c = make_client().await;
    let tool_calls = r#"[
        {"id":"functions.lookup:0","type":"function","function":{"name":"lookup","arguments":"{\"key\":\"a\"}"}},
        {"id":"functions.lookup:1","type":"function","function":{"name":"lookup","arguments":"{\"key\":\"b\"}"}},
        {"id":"functions.lookup:2","type":"function","function":{"name":"lookup","arguments":"{\"key\":\"c\"}"}}
    ]"#;
    let resp = c
        .tokenize(req(
            KIMI,
            vec![
                msg("user", "Look up keys a, b, c in parallel."),
                tool_calls_msg("Querying each key now.", tool_calls),
            ],
            false,
        ))
        .await
        .unwrap()
        .into_inner();
    assert!(resp.num_tokens > 0);
    assert_eq!(count_occurrences(&resp.prompt, "<|tool_calls_section_begin|>"), 1);
    assert_eq!(count_occurrences(&resp.prompt, "<|tool_calls_section_end|>"), 1);
    assert_eq!(count_occurrences(&resp.prompt, "<|tool_call_begin|>"), 3);
    assert_eq!(count_occurrences(&resp.prompt, "<|tool_call_end|>"), 3);
    assert_contains_all(
        &resp.prompt,
        &[
            "functions.lookup:0",
            "functions.lookup:1",
            "functions.lookup:2",
            "Querying each key now.",
            "<|tool_call_argument_begin|>",
        ],
    );
    assert_roundtrip(&mut c, KIMI, &resp.prompt, &resp.token_ids).await;
}

/// End-to-end agent loop: tools advertised → user → assistant tool_call →
/// tool response → assistant final answer. Stresses every Kimi marker plus
/// the `## Return of {id}` tool branch.
#[tokio::test(flavor = "current_thread")]
async fn kimi_full_agent_loop() {
    let mut c = make_client().await;
    let tools = r#"[{"name":"calc","description":"evaluate an arithmetic expression","parameters":{"type":"object","properties":{"expr":{"type":"string"}},"required":["expr"]}}]"#;
    let tool_calls = r#"[{"id":"functions.calc:0","type":"function","function":{"name":"calc","arguments":"{\"expr\":\"(7+3)*4\"}"}}]"#;
    let resp = c
        .tokenize(req_with_tools(
            KIMI,
            vec![
                msg("system", "You are a careful math tutor."),
                msg("user", "What is (7+3)*4? Use the calculator tool."),
                tool_calls_msg("Sure, let me compute that.", tool_calls),
                tool_response_msg("functions.calc:0", "40"),
                msg("assistant", "The answer is 40."),
            ],
            true,
            tools,
        ))
        .await
        .unwrap()
        .into_inner();
    assert!(resp.num_tokens > 0);
    assert_contains_all(
        &resp.prompt,
        &[
            "<|im_system|>tool_declare<|im_middle|>",
            "\"calc\"",
            "\"evaluate an arithmetic expression\"",
            "<|im_system|>system<|im_middle|>You are a careful math tutor.<|im_end|>",
            "<|im_user|>user<|im_middle|>",
            "What is (7+3)*4?",
            "Sure, let me compute that.",
            "<|tool_calls_section_begin|>",
            "<|tool_call_begin|>functions.calc:0",
            "<|tool_call_argument_begin|>",
            "<|tool_calls_section_end|>",
            "<|im_system|>tool<|im_middle|>## Return of functions.calc:0",
            "40",
            "The answer is 40.",
            // Generation prompt appended.
            "<|im_assistant|>assistant<|im_middle|>",
        ],
    );
    // The default Kimi system must not be injected because we provided one.
    assert!(!resp
        .prompt
        .contains("You are Kimi, an AI assistant created by Moonshot AI."));
    assert_roundtrip(&mut c, KIMI, &resp.prompt, &resp.token_ids).await;
}

/// Mixed multimodal user message: text → image → text → image. The macro must
/// iterate every part and emit one `<|media_…|>` block per image while
/// inlining the text fragments in order.
#[tokio::test(flavor = "current_thread")]
async fn kimi_mixed_multimodal_parts() {
    let mut c = make_client().await;
    let parts = r#"[
        {"type":"text","text":"Here is the first picture:"},
        {"type":"image","image_url":"data:image/png;base64,AAA1"},
        {"type":"text","text":"And the second one (note the difference):"},
        {"type":"image","image_url":"data:image/png;base64,AAA2"},
        {"type":"text","text":"What changed between them?"}
    ]"#;
    let user = ChatMessage {
        role: "user".into(),
        content: String::new(),
        content_parts_json: parts.into(),
        ..Default::default()
    };
    let resp = c
        .tokenize(req(KIMI, vec![user], true))
        .await
        .unwrap()
        .into_inner();
    assert!(resp.num_tokens > 0);
    assert_eq!(
        count_occurrences(
            &resp.prompt,
            "<|media_start|>image<|media_content|><|media_pad|><|media_end|>"
        ),
        2,
        "expected two image media blocks, prompt:\n{}",
        resp.prompt
    );
    assert_contains_all(
        &resp.prompt,
        &[
            "Here is the first picture:",
            "And the second one (note the difference):",
            "What changed between them?",
        ],
    );
    assert_roundtrip(&mut c, KIMI, &resp.prompt, &resp.token_ids).await;
}

/// Realistic long multi-turn conversation: system + 3 user/assistant pairs
/// with markdown, code fences, JSON snippets and mixed CJK text. Confirms
/// every turn renders with the correct role marker and survives roundtrip.
#[tokio::test(flavor = "current_thread")]
async fn kimi_realistic_long_dialogue() {
    let mut c = make_client().await;
    let resp = c
        .tokenize(req(
            KIMI,
            vec![
                msg("system", "你是 Kimi，一个乐于助人的助手。回答要简洁、引用代码用 markdown。"),
                msg("user", "How do I read a file line-by-line in Python?"),
                msg(
                    "assistant",
                    "Use a `with` block and iterate directly:\n```python\nwith open('a.txt', encoding='utf-8') as f:\n    for line in f:\n        print(line.rstrip())\n```",
                ),
                msg("user", "What if the file is gzipped?"),
                msg(
                    "assistant",
                    "Use the `gzip` module — same iterator protocol:\n```python\nimport gzip\nwith gzip.open('a.txt.gz', 'rt', encoding='utf-8') as f:\n    for line in f:\n        ...\n```",
                ),
                msg("user", "好的，最后能给我一个完整的字数统计示例吗？"),
            ],
            true,
        ))
        .await
        .unwrap()
        .into_inner();
    assert!(resp.num_tokens > 30);
    assert_eq!(count_occurrences(&resp.prompt, "<|im_user|>user<|im_middle|>"), 3);
    assert_eq!(
        count_occurrences(&resp.prompt, "<|im_assistant|>assistant<|im_middle|>"),
        // 2 historical + 1 generation-prompt tail
        3
    );
    assert_contains_all(
        &resp.prompt,
        &[
            "你是 Kimi",
            "with open('a.txt', encoding='utf-8') as f:",
            "import gzip",
            "好的，最后能给我一个完整的字数统计示例吗？",
        ],
    );
    assert_roundtrip(&mut c, KIMI, &resp.prompt, &resp.token_ids).await;
}

// =============================================================================
// GLM 4.7 — complex / multi-branch scenarios
// =============================================================================

/// `enable_thinking=false` flips the generation-prompt tail from `<think>` to
/// `</think>` (template line 85). Default is `<think>`.
#[tokio::test(flavor = "current_thread")]
async fn glm_enable_thinking_false() {
    let mut c = make_client().await;
    let resp_default = c
        .tokenize(req(GLM, vec![msg("user", "hi")], true))
        .await
        .unwrap()
        .into_inner();
    let resp_disabled = c
        .tokenize(req_with_kwargs(
            GLM,
            vec![msg("user", "hi")],
            true,
            r#"{"enable_thinking": false}"#,
        ))
        .await
        .unwrap()
        .into_inner();
    assert!(
        resp_default.prompt.ends_with("<|assistant|><think>")
            || resp_default.prompt.ends_with("<|assistant|>\n<think>")
            || resp_default.prompt.contains("<|assistant|><think>"),
        "default GLM generation prompt should end with <think>; got:\n{}",
        resp_default.prompt
    );
    assert!(
        resp_disabled.prompt.contains("<|assistant|></think>"),
        "enable_thinking=false should emit </think> after the final <|assistant|>; got:\n{}",
        resp_disabled.prompt
    );
    assert!(!resp_disabled
        .prompt
        .trim_end()
        .ends_with("<|assistant|><think>"));
    assert_roundtrip(&mut c, GLM, &resp_default.prompt, &resp_default.token_ids).await;
    assert_roundtrip(&mut c, GLM, &resp_disabled.prompt, &resp_disabled.token_ids).await;
}

/// When an assistant turn's content has an inline `<think>…</think>` block
/// (e.g. captured straight from a previous decoder output) and no explicit
/// `reasoning_content`, the template splits the content (lines 47–50) so the
/// reasoning either gets dropped (older turn) or re-emitted as a `<think>` block.
#[tokio::test(flavor = "current_thread")]
async fn glm_inline_think_in_content() {
    let mut c = make_client().await;
    let asst_with_inline = msg(
        "assistant",
        "<think>internal scratch SHOULD_NOT_LEAK</think>The answer is 42.",
    );
    let resp = c
        .tokenize(req(
            GLM,
            vec![
                msg("user", "q1"),
                asst_with_inline,
                msg("user", "q2"), // makes the assistant turn an OLD turn
                msg("assistant", "Follow-up reply."),
            ],
            false,
        ))
        .await
        .unwrap()
        .into_inner();
    assert!(resp.num_tokens > 0);
    // The old-turn reasoning must be dropped per the `loop.index0 > ns.last_user_index`
    // branch (false for old turn → emits bare `</think>`, no reasoning_content).
    assert!(
        !resp.prompt.contains("SHOULD_NOT_LEAK"),
        "inline <think>…</think> from old turn leaked:\n{}",
        resp.prompt
    );
    assert!(resp.prompt.contains("The answer is 42."));
    assert!(resp.prompt.contains("Follow-up reply."));
    assert_roundtrip(&mut c, GLM, &resp.prompt, &resp.token_ids).await;
}

/// GLM's `visible_text` macro accepts a list-of-dicts content (parts) and
/// concatenates all `type=='text'` items, silently dropping non-text parts.
/// We supply user content_parts that mix text and image-style entries.
#[tokio::test(flavor = "current_thread")]
async fn glm_visible_text_list_parts() {
    let mut c = make_client().await;
    let parts = r#"[
        {"type":"text","text":"Look at this:"},
        {"type":"image","image_url":"data:image/png;base64,XYZ"},
        {"type":"text","text":" Then summarise."}
    ]"#;
    let user = ChatMessage {
        role: "user".into(),
        content: String::new(),
        content_parts_json: parts.into(),
        ..Default::default()
    };
    let resp = c
        .tokenize(req(GLM, vec![user], true))
        .await
        .unwrap()
        .into_inner();
    assert!(resp.num_tokens > 0);
    // Both text parts must appear, concatenated; the image dict must NOT show
    // up as raw JSON (visible_text drops it).
    assert!(resp.prompt.contains("Look at this:"));
    assert!(resp.prompt.contains("Then summarise."));
    assert!(
        !resp.prompt.contains("data:image/png;base64,XYZ"),
        "image_url should not leak into visible_text output:\n{}",
        resp.prompt
    );
    assert_roundtrip(&mut c, GLM, &resp.prompt, &resp.token_ids).await;
}

/// Multi-arg tool call mixing string and structured (non-string) argument values.
/// Per-arg the template does `v | tojson(ensure_ascii=False) if v is not string else v` —
/// so a string is inlined raw, but a number or nested object must come out as JSON.
#[tokio::test(flavor = "current_thread")]
async fn glm_multi_arg_tool_call() {
    let mut c = make_client().await;
    let tool_calls = r#"[{
        "id":"call_q",
        "type":"function",
        "function":{
            "name":"query",
            "arguments":{
                "table":"users",
                "limit":50,
                "filters":{"active":true,"region":"EU"}
            }
        }
    }]"#;
    let resp = c
        .tokenize(req(
            GLM,
            vec![
                msg("user", "fetch 50 active EU users"),
                tool_calls_msg("", tool_calls),
            ],
            false,
        ))
        .await
        .unwrap()
        .into_inner();
    assert!(resp.num_tokens > 0);
    assert_contains_all(
        &resp.prompt,
        &[
            "<tool_call>query",
            // String value → inlined raw (no surrounding quotes).
            "<arg_key>table</arg_key><arg_value>users</arg_value>",
            // Number value → JSON-serialised.
            "<arg_key>limit</arg_key><arg_value>50</arg_value>",
            // Nested object value → JSON-serialised compactly.
            "<arg_key>filters</arg_key><arg_value>",
            "\"active\":true",
            "\"region\":\"EU\"",
            "</tool_call>",
        ],
    );
    assert_roundtrip(&mut c, GLM, &resp.prompt, &resp.token_ids).await;
}

/// Two consecutive tool responses must share a single `<|observation|>` header
/// — the template emits it only on `loop.first or (messages[loop.index0 - 1].role != "tool")`.
#[tokio::test(flavor = "current_thread")]
async fn glm_consecutive_tool_responses() {
    let mut c = make_client().await;
    let tool_calls = r#"[
        {"id":"c1","type":"function","function":{"name":"get","arguments":{"k":"a"}}},
        {"id":"c2","type":"function","function":{"name":"get","arguments":{"k":"b"}}}
    ]"#;
    let resp = c
        .tokenize(req(
            GLM,
            vec![
                msg("user", "fetch both"),
                tool_calls_msg("", tool_calls),
                tool_response_msg("c1", "value_for_a"),
                tool_response_msg("c2", "value_for_b"),
            ],
            true,
        ))
        .await
        .unwrap()
        .into_inner();
    assert!(resp.num_tokens > 0);
    // Exactly ONE <|observation|> for the two consecutive tool messages.
    assert_eq!(
        count_occurrences(&resp.prompt, "<|observation|>"),
        1,
        "consecutive tool turns must share a single <|observation|> header, got:\n{}",
        resp.prompt
    );
    // But TWO <tool_response>…</tool_response> blocks.
    assert_eq!(count_occurrences(&resp.prompt, "<tool_response>"), 2);
    assert_eq!(count_occurrences(&resp.prompt, "</tool_response>"), 2);
    assert!(resp.prompt.contains("value_for_a"));
    assert!(resp.prompt.contains("value_for_b"));
    assert_roundtrip(&mut c, GLM, &resp.prompt, &resp.token_ids).await;
}

/// A realistic combined scenario that simultaneously exercises: the `[gMASK]<sop>`
/// prefix, the tools `<tools>…</tools>` block, multi-turn dialogue, an old assistant
/// turn whose `reasoning_content` must be dropped, a current assistant turn whose
/// `reasoning_content` must render as `<think>…</think>`, and a final generation prompt.
#[tokio::test(flavor = "current_thread")]
async fn glm_complex_full_scenario() {
    let mut c = make_client().await;
    let tools = r#"[
        {"name":"search","description":"web search","parameters":{"type":"object","properties":{"q":{"type":"string"}},"required":["q"]}},
        {"name":"translate","description":"translate text","parameters":{"type":"object","properties":{"text":{"type":"string"},"target":{"type":"string"}}}}
    ]"#;
    let old_asst = ChatMessage {
        role: "assistant".into(),
        content: "I picked up the topic.".into(),
        reasoning_content: "OLD_REASON_DROP_ME".into(),
        ..Default::default()
    };
    let new_asst = ChatMessage {
        role: "assistant".into(),
        content: "Sure — here is a concise summary in English and Chinese.".into(),
        reasoning_content: "Plan: 1) search 2) translate 3) condense to 3 bullets.".into(),
        ..Default::default()
    };
    let resp = c
        .tokenize(req_with_tools(
            GLM,
            vec![
                msg("system", "You are a multilingual research assistant."),
                msg("user", "Tell me about the Rust language."),
                old_asst,
                msg("user", "Now summarise it in 3 bullets and translate to Chinese."),
                new_asst,
            ],
            true,
            tools,
        ))
        .await
        .unwrap()
        .into_inner();
    assert!(resp.num_tokens > 30);
    assert!(resp.prompt.starts_with("[gMASK]<sop>"));
    assert_contains_all(
        &resp.prompt,
        &[
            "# Tools",
            "<tools>",
            "</tools>",
            "\"search\"",
            "\"translate\"",
            "You are a multilingual research assistant.",
            "Tell me about the Rust language.",
            "I picked up the topic.",
            "Now summarise it in 3 bullets and translate to Chinese.",
            "Sure — here is a concise summary in English and Chinese.",
            // Current assistant's reasoning rendered as a <think> block.
            "<think>Plan: 1) search 2) translate 3) condense to 3 bullets.</think>",
        ],
    );
    // Old-turn reasoning_content must be dropped.
    assert!(
        !resp.prompt.contains("OLD_REASON_DROP_ME"),
        "old turn reasoning_content leaked into prompt:\n{}",
        resp.prompt
    );
    // Two `<|user|>` markers, exactly.
    assert_eq!(count_occurrences(&resp.prompt, "<|user|>"), 2);
    // Three `<|assistant|>` (two historical + the generation-prompt tail).
    assert_eq!(count_occurrences(&resp.prompt, "<|assistant|>"), 3);
    // Generation prompt tail (enable_thinking default → <think>).
    assert!(resp.prompt.trim_end().ends_with("<|assistant|>\n<think>")
        || resp.prompt.trim_end().ends_with("<|assistant|><think>"));
    assert_roundtrip(&mut c, GLM, &resp.prompt, &resp.token_ids).await;
}
