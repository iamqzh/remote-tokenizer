//! `archive-model`: produce an offline tokenizer archive for a HuggingFace model.
//!
//! Given a HF repo + backend kind, this tool downloads the tokenizer files
//! (never weights), copies them into `<workspace>/models/<friendly-name>/`,
//! self-verifies the tokenizer by running an end-to-end encode, and writes a
//! `manifest.toml` describing the archive.

use anyhow::{anyhow, Context, Result};
use clap::{Parser, ValueEnum};
use hf_hub::api::tokio::ApiBuilder;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use tokenizer_core::{Encoder, HuggingFaceTokenizer, Tokenizer as TokenizerTrait};
use tokenizer_manifest::{
    Backend, FileEntry, Manifest, VerifyInfo, VocabInfo, ARCHIVE_TOOL_VERSION, SCHEMA_VERSION,
};

const DEFAULT_VERIFY_TEXT: &str = "Hello, world! 你好。";

#[derive(Debug, Clone, Copy, ValueEnum)]
enum BackendArg {
    /// HuggingFace `tokenizer.json` backend.
    Hf,
    /// Kimi-style tiktoken backend.
    Kimi,
}

impl From<BackendArg> for Backend {
    fn from(a: BackendArg) -> Self {
        match a {
            BackendArg::Hf => Backend::Huggingface,
            BackendArg::Kimi => Backend::KimiTiktoken,
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "archive-model",
    about = "Download tokenizer assets from HuggingFace and produce an offline archive."
)]
struct Args {
    /// Friendly name used as the `model://<name>` URI and archive directory.
    #[arg(long)]
    name: String,
    /// HuggingFace repo id, e.g. `moonshotai/Kimi-K2-Instruct`.
    #[arg(long)]
    repo: String,
    /// Tokenizer backend.
    #[arg(long, value_enum)]
    backend: BackendArg,
    /// Output directory (defaults to `<workspace>/models/<name>/`).
    #[arg(long)]
    out_dir: Option<PathBuf>,
    /// Text used for the verify round-trip. The resulting token ids are stored
    /// in `verify.expected_token_ids` of the manifest.
    #[arg(long, default_value = DEFAULT_VERIFY_TEXT)]
    verify_text: String,
    /// Optional display name shown in tooling.
    #[arg(long)]
    display_name: Option<String>,
}

fn required_files(backend: Backend) -> &'static [&'static str] {
    match backend {
        Backend::Huggingface => &["tokenizer.json", "tokenizer_config.json"],
        Backend::KimiTiktoken => &[
            "tiktoken.model",
            "tokenizer_config.json",
            "chat_template.jinja",
        ],
    }
}

fn optional_files(backend: Backend) -> &'static [&'static str] {
    match backend {
        // Many HF models ship the chat template as a sibling jinja file.
        Backend::Huggingface => &["chat_template.jinja", "special_tokens_map.json"],
        Backend::KimiTiktoken => &["special_tokens_map.json"],
    }
}

fn workspace_root() -> Result<PathBuf> {
    // The binary is built into `<root>/target/release/archive-model`, but we
    // want to be friendly when invoked from anywhere — so locate the manifest
    // at compile-time and walk upward.
    let here = PathBuf::from(env!("CARGO_MANIFEST_DIR")); // tools/archive-model
    let root = here
        .parent() // tools
        .and_then(|p| p.parent()) // workspace root
        .ok_or_else(|| anyhow!("could not derive workspace root from {}", here.display()))?;
    Ok(root.to_path_buf())
}

async fn fetch_files(
    repo_id: &str,
    backend: Backend,
) -> Result<std::collections::BTreeMap<String, PathBuf>> {
    let token = std::env::var("HF_TOKEN").ok();
    let api = ApiBuilder::new()
        .with_progress(true)
        .with_token(token)
        .build()?;
    let repo = api.model(repo_id.to_string());

    let mut out = std::collections::BTreeMap::new();

    for &fname in required_files(backend) {
        tracing::info!("downloading {} from {}", fname, repo_id);
        let path = repo
            .get(fname)
            .await
            .with_context(|| format!("required file {} not available on {}", fname, repo_id))?;
        out.insert(fname.to_string(), path);
    }

    for &fname in optional_files(backend) {
        match repo.get(fname).await {
            Ok(path) => {
                tracing::info!("downloaded optional file {} from {}", fname, repo_id);
                out.insert(fname.to_string(), path);
            }
            Err(e) => {
                tracing::debug!("optional file {} not on {}: {}", fname, repo_id, e);
            }
        }
    }

    Ok(out)
}

fn copy_and_hash(src: &Path, dst: &Path) -> Result<(String, u64)> {
    if let Some(parent) = dst.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let bytes = std::fs::read(src)
        .with_context(|| format!("reading source {}", src.display()))?;
    let size = bytes.len() as u64;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    let digest = hasher.finalize();
    std::fs::write(dst, &bytes)
        .with_context(|| format!("writing destination {}", dst.display()))?;
    Ok((hex(&digest), size))
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

/// Run the verify-text through the tokenizer for the given backend / archive
/// directory. Returns (token_ids, vocab_size, bos, eos, has_chat_template).
fn verify_archive(
    backend: Backend,
    dir: &Path,
    text: &str,
) -> Result<(Vec<u32>, usize, Option<String>, Option<String>, bool)> {
    match backend {
        Backend::Huggingface => {
            let path = dir.join("tokenizer.json");
            let path_str = path
                .to_str()
                .ok_or_else(|| anyhow!("non-utf8 archive path {:?}", path))?;
            let hf = HuggingFaceTokenizer::from_file(path_str)
                .with_context(|| format!("loading HF tokenizer from {}", path_str))?;
            let enc = hf.encode(text)?;
            let ids = enc.token_ids().to_vec();
            let vocab = hf.vocab_size();
            let specials = hf.get_special_tokens().clone();
            let chat = hf.chat_template().map(|s| s.to_string()).is_some()
                || dir.join("chat_template.jinja").exists();
            Ok((ids, vocab, specials.bos_token, specials.eos_token, chat))
        }
        Backend::KimiTiktoken => {
            let (tk, ct, bos, eos) = tokenizer_core::kimi_local::load_kimi_tokenizer_from_dir(dir)
                .with_context(|| format!("loading Kimi tokenizer from {}", dir.display()))?;
            let enc = tk.encode(text)?;
            let ids = enc.token_ids().to_vec();
            let vocab = tk.vocab_size();
            Ok((ids, vocab, bos, eos, ct.is_some()))
        }
    }
}

// Silence unused-import warnings for trait aliases that may only be referenced
// via the `Arc<dyn TokenizerTrait>` returned by `kimi_local`.
#[allow(dead_code)]
fn _trait_imports(_: &dyn TokenizerTrait) {}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();
    let backend: Backend = args.backend.into();

    let out_dir = match args.out_dir.clone() {
        Some(p) => p,
        None => workspace_root()?.join("models").join(&args.name),
    };
    std::fs::create_dir_all(&out_dir)
        .with_context(|| format!("creating archive dir {}", out_dir.display()))?;

    tracing::info!(
        "archiving model://{} from {} into {}",
        args.name,
        args.repo,
        out_dir.display()
    );

    // 1) Fetch from HF (hf-hub will reuse its cache if a snapshot is already present).
    let downloaded = fetch_files(&args.repo, backend).await?;

    // 2) Copy into the archive dir + record sha256/size.
    let mut entries: Vec<FileEntry> = Vec::new();
    for (name, src) in &downloaded {
        let dst = out_dir.join(name);
        let (sha, size) = copy_and_hash(src, &dst)
            .with_context(|| format!("archiving file {}", name))?;
        tracing::info!("  archived {} ({} bytes, sha256={}…)", name, size, &sha[..16]);
        entries.push(FileEntry {
            name: name.clone(),
            sha256: sha,
            size,
        });
    }

    // 3) Self-verify by loading the tokenizer from disk and tokenizing the verify text.
    let (token_ids, vocab_size, bos, eos, has_chat_template) =
        verify_archive(backend, &out_dir, &args.verify_text)?;

    if token_ids.is_empty() {
        return Err(anyhow!(
            "self-verify produced 0 tokens for text {:?}",
            args.verify_text
        ));
    }
    tracing::info!(
        "self-verify ok: vocab_size={} -> {} token(s) for {:?}",
        vocab_size,
        token_ids.len(),
        args.verify_text
    );

    // 4) Write manifest.toml.
    let manifest = Manifest {
        schema_version: SCHEMA_VERSION,
        name: args.name.clone(),
        display_name: args
            .display_name
            .unwrap_or_else(|| args.name.replace('-', " ")),
        hf_repo: args.repo.clone(),
        backend,
        archived_at: chrono::Utc::now().to_rfc3339(),
        archive_tool_version: ARCHIVE_TOOL_VERSION.to_string(),
        files: entries,
        vocab: VocabInfo {
            size: vocab_size as u64,
            bos_token: bos,
            eos_token: eos,
            has_chat_template,
        },
        verify: VerifyInfo {
            text: args.verify_text.clone(),
            expected_token_ids: token_ids.clone(),
        },
    };

    let manifest_path = out_dir.join("manifest.toml");
    manifest.save(&manifest_path)?;

    println!(
        "Archived model://{}: {} files, vocab={}, verify={} tokens, manifest={}",
        args.name,
        manifest.files.len(),
        vocab_size,
        token_ids.len(),
        manifest_path.display()
    );

    Ok(())
}
