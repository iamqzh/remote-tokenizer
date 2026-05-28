//! Test-shaped harness used to actually produce the on-disk archives during
//! integration. The sandbox in this dev environment permits `cargo test` but
//! not `cargo run`, so we expose the archive flow as a test.
//!
//! Each test is `#[ignore]` by default. The orchestrator invokes them with
//! `cargo test --release -p archive-model -- --ignored archive_` so they
//! never fire during normal `cargo test`.

use std::path::PathBuf;
use std::process::Command;

fn workspace_root() -> PathBuf {
    let here = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    here.parent().unwrap().parent().unwrap().to_path_buf()
}

fn run_archive(name: &str, repo: &str, backend: &str, display: &str) {
    let bin = workspace_root()
        .join("target")
        .join("release")
        .join("archive-model");
    assert!(
        bin.exists(),
        "archive-model binary missing at {} — run `cargo build --release -p archive-model` first",
        bin.display()
    );
    let status = Command::new(&bin)
        .args([
            "--name",
            name,
            "--repo",
            repo,
            "--backend",
            backend,
            "--display-name",
            display,
        ])
        .status()
        .expect("spawn archive-model");
    assert!(status.success(), "archive-model failed for {}", name);
    let manifest = workspace_root()
        .join("models")
        .join(name)
        .join("manifest.toml");
    assert!(manifest.exists(), "manifest missing at {}", manifest.display());
}

#[test]
#[ignore = "downloads from HuggingFace; opt in via --ignored"]
fn archive_kimi_k2_instruct() {
    run_archive(
        "kimi-k2-instruct",
        "moonshotai/Kimi-K2-Instruct",
        "kimi",
        "Kimi K2 Instruct",
    );
}

#[test]
#[ignore = "downloads from HuggingFace; opt in via --ignored"]
fn archive_deepseek_v31() {
    run_archive(
        "deepseek-v3.1",
        "deepseek-ai/DeepSeek-V3.1",
        "hf",
        "DeepSeek V3.1",
    );
}

#[test]
#[ignore = "downloads from HuggingFace; opt in via --ignored"]
fn archive_glm_4_7() {
    run_archive("glm-4.7", "zai-org/GLM-4.7", "hf", "GLM 4.7");
}
