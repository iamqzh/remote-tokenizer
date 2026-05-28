use anyhow::{anyhow, Context, Result};
use dashmap::DashMap;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use tokenizer_core::{
    create_huggingface_tokenizer_async, create_tokenizer_from_file, HuggingFaceTokenizer,
    MockTokenizer, Tokenizer as TokenizerTrait, TiktokenTokenizer,
};
use tokenizer_manifest::{Backend, Manifest};

use crate::extensions;

pub struct ModelEntry {
    pub uri: String,
    pub tokenizer: Arc<dyn TokenizerTrait>,
    /// Generic chat template (jinja source). Loaded from HF tokenizer_config.json
    /// or supplied by an extension (e.g. Kimi).
    pub chat_template: Option<String>,
    /// bos/eos tokens used during chat-template rendering.
    pub bos_token: Option<String>,
    pub eos_token: Option<String>,
}

/// An archived (offline) model spec discovered under `<root>/models/<name>/`.
#[derive(Debug, Clone)]
pub struct ArchivedSpec {
    pub dir: PathBuf,
    pub backend: Backend,
    pub manifest: Manifest,
}

/// Result of the startup self-verify for one archived model.
#[derive(Debug, Clone)]
pub struct VerifyReport {
    pub name: String,
    pub expected: Vec<u32>,
    pub got: Vec<u32>,
    pub matched: bool,
}

pub struct Registry {
    inner: DashMap<String, Arc<ModelEntry>>,
    /// friendly name (without `model://` prefix) -> spec.
    /// Populated by `scan_archive_dir` at startup. Wrapped in `RwLock` so
    /// scans can run against `Arc<Registry>` without requiring `&mut self`.
    archived: RwLock<HashMap<String, ArchivedSpec>>,
}

impl Default for Registry {
    fn default() -> Self {
        Self::new()
    }
}

impl Registry {
    pub fn new() -> Self {
        let reg = Self {
            inner: DashMap::new(),
            archived: RwLock::new(HashMap::new()),
        };
        // Best-effort auto-scan so that `model://<name>` URIs work without
        // requiring an explicit `scan_archive_dir` call. Honoured paths
        // (first hit wins):
        //   1. $TOKENIZER_MODELS_DIR
        //   2. `<current_exe>/../../../models`   (workspace layout)
        //   3. `./models`                        (current working dir)
        // Failures here are silent — explicit callers can still drive
        // `scan_archive_dir` themselves.
        let candidates: Vec<PathBuf> = std::iter::empty()
            .chain(std::env::var_os("TOKENIZER_MODELS_DIR").map(PathBuf::from))
            .chain(
                std::env::current_exe()
                    .ok()
                    .and_then(|p| Some(p.parent()?.parent()?.parent()?.join("models"))),
            )
            .chain(std::iter::once(PathBuf::from("models")))
            .collect();
        for cand in candidates {
            if cand.is_dir() {
                if let Err(e) = reg.scan_archive_dir(&cand) {
                    tracing::warn!("auto-scan of {} failed: {}", cand.display(), e);
                }
                break;
            }
        }
        reg
    }

    /// Names of all archived models currently known to this registry.
    pub fn archived_names(&self) -> Vec<String> {
        let map = self.archived.read().expect("archived map poisoned");
        let mut v: Vec<String> = map.keys().cloned().collect();
        v.sort();
        v
    }

    /// Scan a directory of model archives. The argument is the
    /// **`models/` directory itself** (not its parent) — i.e. each immediate
    /// child of `models_dir` is expected to be `<friendly-name>/` containing
    /// a `manifest.toml`.
    ///
    /// Per-archive errors are logged via `tracing::warn` but do not abort the
    /// scan — that way a single corrupt directory does not take the server
    /// down. The total number of successfully registered archives is returned
    /// via tracing.
    pub fn scan_archive_dir(&self, models_dir: &Path) -> Result<()> {
        if !models_dir.exists() {
            tracing::info!(
                "archive scan: {} does not exist, skipping",
                models_dir.display()
            );
            return Ok(());
        }
        let mut count = 0usize;
        let mut new_map = HashMap::new();
        for entry in std::fs::read_dir(models_dir)
            .with_context(|| format!("reading {}", models_dir.display()))?
        {
            let entry = match entry {
                Ok(e) => e,
                Err(e) => {
                    tracing::warn!("archive scan: dir entry error: {}", e);
                    continue;
                }
            };
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let manifest_path = path.join("manifest.toml");
            if !manifest_path.exists() {
                continue;
            }
            match Manifest::load(&manifest_path) {
                Ok(m) => {
                    let spec = ArchivedSpec {
                        dir: path.clone(),
                        backend: m.backend,
                        manifest: m.clone(),
                    };
                    tracing::info!(
                        "archive scan: registered model://{} ({:?}, vocab={})",
                        m.name,
                        m.backend,
                        m.vocab.size
                    );
                    new_map.insert(m.name, spec);
                    count += 1;
                }
                Err(e) => {
                    tracing::warn!(
                        "archive scan: failed to load manifest at {}: {}",
                        manifest_path.display(),
                        e
                    );
                }
            }
        }
        {
            let mut guard = self.archived.write().expect("archived map poisoned");
            *guard = new_map;
        }
        tracing::info!(
            "archive scan: {} archived model(s) registered from {}",
            count,
            models_dir.display()
        );
        Ok(())
    }

    /// Eagerly load every archived model and run its verify-text round trip.
    /// Returns one `VerifyReport` per archived model. A returned `Err` means
    /// the load itself failed (not a token-id mismatch — those are reported
    /// via `matched = false`).
    pub async fn warmup_and_verify_archived(&self) -> Result<Vec<VerifyReport>> {
        let mut reports = Vec::new();
        let names: Vec<String> = {
            let map = self.archived.read().expect("archived map poisoned");
            map.keys().cloned().collect()
        };
        for name in names {
            let uri = format!("model://{}", name);
            let entry = self
                .get_or_load(&uri)
                .await
                .with_context(|| format!("warmup loading {}", uri))?;
            let verify = {
                let map = self.archived.read().expect("archived map poisoned");
                map.get(&name)
                    .map(|s| s.manifest.verify.clone())
                    .ok_or_else(|| anyhow!("archive entry {} vanished during warmup", name))?
            };
            let enc = entry
                .tokenizer
                .encode(&verify.text)
                .with_context(|| format!("verify encode for {}", uri))?;
            let got = enc.token_ids().to_vec();
            let matched = got == verify.expected_token_ids;
            if !matched {
                tracing::error!(
                    "verify FAILED for {}: expected={:?} got={:?}",
                    uri,
                    verify.expected_token_ids,
                    got
                );
            } else {
                tracing::info!(
                    "verify OK for {}: {} tokens (vocab={})",
                    uri,
                    got.len(),
                    entry.tokenizer.vocab_size()
                );
            }
            reports.push(VerifyReport {
                name: name.clone(),
                expected: verify.expected_token_ids.clone(),
                got,
                matched,
            });
        }
        Ok(reports)
    }

    pub async fn get_or_load(&self, model_uri: &str) -> Result<Arc<ModelEntry>> {
        if let Some(e) = self.inner.get(model_uri) {
            return Ok(e.clone());
        }

        let entry = self.load(model_uri).await?;
        self.inner.insert(model_uri.to_string(), entry.clone());
        Ok(entry)
    }

    async fn load(&self, model_uri: &str) -> Result<Arc<ModelEntry>> {
        // model://<friendly-name> — fully offline, served from a pre-scanned archive.
        if let Some(name) = model_uri.strip_prefix("model://") {
            let spec = {
                let map = self.archived.read().expect("archived map poisoned");
                map.get(name).cloned().ok_or_else(|| {
                    anyhow!(
                        "model://{} not in archive registry (did you run `archive-model` \
                         and is `models/` on the search path?)",
                        name
                    )
                })?
            };
            return load_archived(&model_uri.to_string(), &spec).await;
        }

        // Built-in mock tokenizer for tests — no network, 8-token vocab.
        if model_uri == "mock" || model_uri == "test" {
            let tk = Arc::new(MockTokenizer::new());
            return Ok(Arc::new(ModelEntry {
                uri: model_uri.to_string(),
                tokenizer: tk,
                chat_template: None,
                bos_token: None,
                eos_token: None,
            }));
        }

        // Tiktoken / OpenAI-style model names — no HF download.
        if is_openai_name(model_uri) {
            let tk = TiktokenTokenizer::from_model_name(model_uri)?;
            let special = tk.get_special_tokens().clone();
            return Ok(Arc::new(ModelEntry {
                uri: model_uri.to_string(),
                tokenizer: Arc::new(tk),
                chat_template: None,
                bos_token: special.bos_token,
                eos_token: special.eos_token,
            }));
        }

        // Kimi extension.
        if let Some(rest) = model_uri.strip_prefix("kimi://") {
            let (tk, chat_template, bos, eos) = extensions::kimi::load_kimi_tokenizer(rest).await?;
            return Ok(Arc::new(ModelEntry {
                uri: model_uri.to_string(),
                tokenizer: tk,
                chat_template,
                bos_token: bos,
                eos_token: eos,
            }));
        }

        // file:///abs/path — local tokenizer.json
        if let Some(path) = model_uri.strip_prefix("file://") {
            let tk = create_tokenizer_from_file(path)?;
            // Also try a HuggingFaceTokenizer load for chat-template support.
            let (chat_template, bos, eos) = match HuggingFaceTokenizer::from_file(path) {
                Ok(hf) => {
                    let ct = hf.chat_template().map(|s| s.to_string());
                    let s = hf.get_special_tokens().clone();
                    (ct, s.bos_token, s.eos_token)
                }
                Err(_) => (None, None, None),
            };
            return Ok(Arc::new(ModelEntry {
                uri: model_uri.to_string(),
                tokenizer: tk,
                chat_template,
                bos_token: bos,
                eos_token: eos,
            }));
        }

        // hf://<repo> or bare repo id.
        let repo = model_uri
            .strip_prefix("hf://")
            .unwrap_or(model_uri)
            .to_string();

        let hf = create_huggingface_tokenizer_async(&repo)
            .await
            .map_err(|e| anyhow!("failed to load HF tokenizer '{}': {}", repo, e))?;
        let chat_template = hf.chat_template().map(|s| s.to_string());
        let special = hf.get_special_tokens().clone();
        let tokenizer: Arc<dyn TokenizerTrait> = hf;
        Ok(Arc::new(ModelEntry {
            uri: model_uri.to_string(),
            tokenizer,
            chat_template,
            bos_token: special.bos_token,
            eos_token: special.eos_token,
        }))
    }
}

/// Load a model from an archived directory according to its declared backend.
/// The whole code path is synchronous filesystem IO and stays off the network.
async fn load_archived(uri: &str, spec: &ArchivedSpec) -> Result<Arc<ModelEntry>> {
    let dir = spec.dir.clone();
    let backend = spec.backend;
    let uri_owned = uri.to_string();

    // Tokenizer construction can be CPU-heavy (parsing tiktoken.model is ~5MB
    // of base64). Run on a blocking thread so we don't stall the runtime.
    let entry = tokio::task::spawn_blocking(move || -> Result<ModelEntry> {
        match backend {
            Backend::Huggingface => {
                let json = dir.join("tokenizer.json");
                if !json.exists() {
                    return Err(anyhow!(
                        "tokenizer.json missing in archive {}",
                        dir.display()
                    ));
                }
                let json_str = json.to_str().ok_or_else(|| {
                    anyhow!("non-utf8 path for tokenizer.json: {:?}", json)
                })?;
                let hf = HuggingFaceTokenizer::from_file(json_str)
                    .with_context(|| format!("loading HF tokenizer from {}", json_str))?;
                let chat_template = hf
                    .chat_template()
                    .map(|s| s.to_string())
                    .or_else(|| {
                        let jinja = dir.join("chat_template.jinja");
                        if jinja.exists() {
                            std::fs::read_to_string(&jinja).ok()
                        } else {
                            None
                        }
                    });
                let specials = hf.get_special_tokens().clone();
                // Fall back to tokenizer_config.json bos/eos when the HF
                // tokenizer's hardcoded heuristic missed them (DeepSeek's
                // `<｜begin▁of▁sentence｜>` is not in the default list).
                let (cfg_bos, cfg_eos) = read_tokenizer_config_specials(&dir);
                let tk: Arc<dyn TokenizerTrait> = Arc::new(hf);
                Ok(ModelEntry {
                    uri: uri_owned,
                    tokenizer: tk,
                    chat_template,
                    bos_token: specials.bos_token.or(cfg_bos),
                    eos_token: specials.eos_token.or(cfg_eos),
                })
            }
            Backend::KimiTiktoken => {
                let (tk, ct, bos, eos) =
                    tokenizer_core::kimi_local::load_kimi_tokenizer_from_dir(&dir)
                        .with_context(|| {
                            format!("loading Kimi tokenizer from {}", dir.display())
                        })?;
                Ok(ModelEntry {
                    uri: uri_owned,
                    tokenizer: tk,
                    chat_template: ct,
                    bos_token: bos,
                    eos_token: eos,
                })
            }
        }
    })
    .await
    .map_err(|e| anyhow!("archive load join error: {}", e))??;

    Ok(Arc::new(entry))
}

/// Read `bos_token` / `eos_token` strings out of `tokenizer_config.json`
/// shipped alongside an archived HF model. Each may be either a plain string
/// or an object of shape `{"content": "...", "__type": "AddedToken"}` — we
/// handle both. Missing or unreadable file → `(None, None)`.
fn read_tokenizer_config_specials(dir: &Path) -> (Option<String>, Option<String>) {
    let p = dir.join("tokenizer_config.json");
    if !p.exists() {
        return (None, None);
    }
    let txt = match std::fs::read_to_string(&p) {
        Ok(s) => s,
        Err(_) => return (None, None),
    };
    let v: serde_json::Value = match serde_json::from_str(&txt) {
        Ok(v) => v,
        Err(_) => return (None, None),
    };
    fn extract(v: &serde_json::Value, key: &str) -> Option<String> {
        match v.get(key)? {
            serde_json::Value::String(s) => Some(s.clone()),
            serde_json::Value::Object(o) => o
                .get("content")
                .and_then(|c| c.as_str())
                .map(|s| s.to_string()),
            _ => None,
        }
    }
    (extract(&v, "bos_token"), extract(&v, "eos_token"))
}

fn is_openai_name(name: &str) -> bool {
    // Match the same heuristic used in tokenizer-core::factory.
    name.contains("gpt-")
        || name.contains("davinci")
        || name.contains("curie")
        || name.contains("babbage")
        || name.contains("ada")
}
