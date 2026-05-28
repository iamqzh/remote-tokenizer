//! Offline-archive manifest schema for tokenizer models.
//!
//! Every archived model lives at `<root>/models/<friendly-name>/` and contains
//! the tokenizer files plus a `manifest.toml` whose schema is defined here.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const SCHEMA_VERSION: u32 = 1;
pub const ARCHIVE_TOOL_VERSION: &str = "0.1.0";

/// Tokenizer backend kind. Determines which loader the server uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Backend {
    /// Standard HuggingFace `tokenizer.json` loaded via the `tokenizers` crate.
    Huggingface,
    /// Kimi-style `tiktoken.model` + reserved-special tokens (CoreBPE).
    KimiTiktoken,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileEntry {
    pub name: String,
    pub sha256: String,
    pub size: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VocabInfo {
    pub size: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bos_token: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eos_token: Option<String>,
    #[serde(default)]
    pub has_chat_template: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifyInfo {
    pub text: String,
    pub expected_token_ids: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub schema_version: u32,
    pub name: String,
    pub display_name: String,
    pub hf_repo: String,
    pub backend: Backend,
    /// RFC3339 timestamp of when this archive was produced.
    pub archived_at: String,
    pub archive_tool_version: String,

    #[serde(default)]
    pub files: Vec<FileEntry>,

    pub vocab: VocabInfo,
    pub verify: VerifyInfo,
}

impl Manifest {
    pub fn load(path: &Path) -> Result<Self> {
        let s = std::fs::read_to_string(path)
            .with_context(|| format!("reading manifest at {}", path.display()))?;
        let m: Manifest = toml::from_str(&s)
            .with_context(|| format!("parsing manifest at {}", path.display()))?;
        if m.schema_version != SCHEMA_VERSION {
            anyhow::bail!(
                "manifest at {} has unsupported schema_version {} (expected {})",
                path.display(),
                m.schema_version,
                SCHEMA_VERSION
            );
        }
        Ok(m)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let s = toml::to_string_pretty(self).context("serialising manifest to TOML")?;
        std::fs::write(path, s)
            .with_context(|| format!("writing manifest to {}", path.display()))?;
        Ok(())
    }
}
