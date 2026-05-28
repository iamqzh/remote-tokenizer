use anyhow::{anyhow, Result};
use tokenizer_proto::tokenizer_client::TokenizerClient;
use tokenizer_proto::{
    ChatMessage, DetokenizeRequest, DetokenizeStreamRequest, GetModelInfoRequest, TokenizeRequest,
};
use tokio_stream::StreamExt;

#[tokio::main]
async fn main() -> Result<()> {
    let endpoint = std::env::var("TOKENIZER_ADDR").unwrap_or_else(|_| "http://127.0.0.1:50051".into());
    println!("[rust_client] connecting to {endpoint}");

    let mut client = TokenizerClient::connect(endpoint).await?;

    // 1. GetModelInfo
    let info = client
        .get_model_info(GetModelInfoRequest {
            model: "gpt-3.5-turbo".into(),
        })
        .await?
        .into_inner();
    println!(
        "[GetModelInfo] model={} vocab_size={} bos={:?} eos={:?} has_chat_template={}",
        info.model, info.vocab_size, info.bos_token, info.eos_token, info.has_chat_template
    );
    if info.vocab_size == 0 {
        return Err(anyhow!("vocab_size should be > 0"));
    }

    // 2. Tokenize with chat messages (no chat template for tiktoken — fall back to simple concat)
    let resp = client
        .tokenize(TokenizeRequest {
            model: "gpt-3.5-turbo".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: "Hello".into(),
                ..Default::default()
            }],
            add_generation_prompt: true,
            raw_text: "".into(),
            ..Default::default()
        })
        .await?
        .into_inner();
    println!(
        "[Tokenize chat] prompt={:?} token_ids={:?} num_tokens={}",
        resp.prompt, resp.token_ids, resp.num_tokens
    );

    // 3. Tokenize raw text
    let raw_resp = client
        .tokenize(TokenizeRequest {
            model: "gpt-3.5-turbo".into(),
            messages: vec![],
            add_generation_prompt: false,
            raw_text: "Hello, world!".into(),
            ..Default::default()
        })
        .await?
        .into_inner();
    println!(
        "[Tokenize raw] prompt={:?} token_ids={:?} num_tokens={}",
        raw_resp.prompt, raw_resp.token_ids, raw_resp.num_tokens
    );
    if raw_resp.token_ids.is_empty() {
        return Err(anyhow!("expected non-empty token_ids"));
    }

    // 4. Detokenize
    let detok = client
        .detokenize(DetokenizeRequest {
            model: "gpt-3.5-turbo".into(),
            token_ids: raw_resp.token_ids.clone(),
            skip_special_tokens: true,
        })
        .await?
        .into_inner();
    println!("[Detokenize] text={:?}", detok.text);
    if detok.text != "Hello, world!" {
        return Err(anyhow!(
            "roundtrip mismatch: got {:?}, expected {:?}",
            detok.text,
            "Hello, world!"
        ));
    }

    // 5. DetokenizeStream
    let mut stream = client
        .detokenize_stream(DetokenizeStreamRequest {
            model: "gpt-3.5-turbo".into(),
            token_ids: raw_resp.token_ids.clone(),
            skip_special_tokens: true,
        })
        .await?
        .into_inner();

    let mut acc = String::new();
    let mut chunks = 0u32;
    while let Some(chunk) = stream.next().await {
        let c = chunk?;
        acc.push_str(&c.delta);
        chunks += 1;
        println!(
            "[stream chunk] token_index={} delta={:?}",
            c.token_index, c.delta
        );
    }
    println!("[DetokenizeStream] chunks={} joined={:?}", chunks, acc);
    if acc != detok.text {
        return Err(anyhow!(
            "stream concat mismatch: got {:?}, expected {:?}",
            acc,
            detok.text
        ));
    }

    // 6. Optional Kimi roundtrip (only if KIMI_MODEL is set).
    if let Ok(model) = std::env::var("KIMI_MODEL") {
        let text = std::env::var("KIMI_TEXT")
            .unwrap_or_else(|_| "Hello, Kimi! 你好。".to_string());
        let kresp = client
            .tokenize(TokenizeRequest {
                model: model.clone(),
                messages: vec![],
                add_generation_prompt: false,
                raw_text: text.clone(),
                ..Default::default()
            })
            .await?
            .into_inner();
        println!(
            "[Tokenize kimi raw] model={} text={:?} token_ids={:?} num_tokens={}",
            model, text, kresp.token_ids, kresp.num_tokens
        );
    }

    println!("[rust_client] OK");
    Ok(())
}
