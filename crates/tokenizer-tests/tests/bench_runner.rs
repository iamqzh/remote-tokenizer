//! Test-driven bench runner: spawns tokenizer-server, drives the `bench`
//! binary at multiple concurrency levels, and prints a comparison table.
//!
//! Enabled via `--ignored` so a normal `cargo test` run is unaffected.

use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const PORT: u16 = 50081;

fn workspace_target() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop(); // crates/
    p.pop(); // workspace root
    p.push("target");
    p.push("release");
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

fn spawn_server() -> ServerGuard {
    let bin = workspace_target().join("tokenizer-server");
    assert!(bin.exists(), "missing server binary at {:?}", bin);
    let child = Command::new(&bin)
        .env("LISTEN_ADDR", format!("127.0.0.1:{}", PORT))
        .env("RUST_LOG", "error")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn server");
    let deadline = Instant::now() + Duration::from_secs(15);
    while Instant::now() < deadline {
        if TcpStream::connect(("127.0.0.1", PORT)).is_ok() {
            std::thread::sleep(Duration::from_millis(200));
            return ServerGuard { child };
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    panic!("server never came up on port {PORT}");
}

fn run_bench(model: &str, concurrency: usize, duration_secs: u64, text: &str) -> String {
    let bin = workspace_target().join("bench");
    assert!(bin.exists(), "missing bench binary at {:?}", bin);
    let out = Command::new(&bin)
        .args([
            "--endpoint",
            &format!("http://127.0.0.1:{PORT}"),
            "--model",
            model,
            "--rpc",
            "tokenize",
            "--concurrency",
            &concurrency.to_string(),
            "--duration-secs",
            &duration_secs.to_string(),
            "--text",
            text,
        ])
        .output()
        .expect("run bench");
    assert!(
        out.status.success(),
        "bench failed: {}\nstderr: {}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    let s = String::from_utf8_lossy(&out.stdout).to_string();
    println!("{s}");
    s
}

fn extract_summary(stdout: &str) -> String {
    stdout
        .lines()
        .find(|l| l.starts_with("SUMMARY "))
        .unwrap_or("(no SUMMARY line)")
        .to_string()
}

#[test]
#[ignore = "bench: opt in with --ignored"]
fn bench_compare() {
    let _g = spawn_server();
    let text =
        "Hello, world! 你好。这是一段较长的测试文本，用来评估高并发下的 tokenizer 吞吐性能。";

    let mut summaries = Vec::new();
    for &c in &[1usize, 8, 32, 128, 256] {
        println!("\n========== concurrency={c} ==========");
        let out = run_bench("gpt-3.5-turbo", c, 5, text);
        summaries.push(extract_summary(&out));
    }

    println!("\n\n=============== AGGREGATE ===============");
    for s in &summaries {
        println!("{s}");
    }
}
