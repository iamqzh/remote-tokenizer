//! Kimi/Moonshot tokenizer extension (server side wrapper).
//!
//! The actual offline loader lives in `tokenizer_core::kimi_local` so it can be
//! reused by the `archive-model` tool. This module is just a thin async helper
//! that downloads the assets from HuggingFace and then defers to the core loader.

use anyhow::{Context, Result};
use std::sync::Arc;
use tokenizer_core::kimi_local::load_kimi_tokenizer_from_dir;
use tokenizer_core::Tokenizer as TokenizerTrait;

/// Load a Kimi tokenizer from a HuggingFace repo id (e.g. `moonshotai/Kimi-K2-Instruct`).
///
/// Returns `(tokenizer, chat_template, bos_token, eos_token)`.
pub async fn load_kimi_tokenizer(
    repo: &str,
) -> Result<(
    Arc<dyn TokenizerTrait>,
    Option<String>,
    Option<String>,
    Option<String>,
)> {
    let cache_dir = tokenizer_core::hub::download_tokenizer_from_hf(repo)
        .await
        .with_context(|| format!("downloading Kimi tokenizer from HF '{}'", repo))?;
    load_kimi_tokenizer_from_dir(&cache_dir)
}
