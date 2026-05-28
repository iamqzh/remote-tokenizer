mod extensions;
mod registry;
mod service;

use std::sync::Arc;
use std::time::Duration;

use tonic::transport::Server;

use tokenizer_proto::tokenizer_server::TokenizerServer;

use crate::registry::Registry;
use crate::service::TokenizerService;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    // Build a tuned multi-thread runtime. Default worker count is num_cpus, but
    // both the worker count and the blocking pool size are tunable via env vars
    // so we can dial the server up/down without recompiling.
    let worker_threads = std::env::var("TOKENIZER_WORKERS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or_else(num_cpus::get);
    let max_blocking_threads = std::env::var("TOKENIZER_BLOCKING_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1024);

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(worker_threads)
        .max_blocking_threads(max_blocking_threads)
        .enable_all()
        .thread_name("tokenizer-worker")
        .build()?;

    runtime.block_on(async move { run().await })
}

async fn run() -> anyhow::Result<()> {
    let addr_str = std::env::var("LISTEN_ADDR").unwrap_or_else(|_| "0.0.0.0:50051".to_string());
    let addr = addr_str.parse()?;

    let registry = Arc::new(Registry::new());

    // Registry::new() already auto-scans `$TOKENIZER_MODELS_DIR` /
    // `<exe>/../../../models` / `./models`. We only need to run verify here.
    match registry.warmup_and_verify_archived().await {
        Ok(reports) => {
            for r in &reports {
                if r.matched {
                    tracing::info!("model://{}: verify ok ({} tokens)", r.name, r.got.len());
                } else {
                    tracing::error!(
                        "model://{}: verify FAILED expected={:?} got={:?}",
                        r.name,
                        r.expected,
                        r.got
                    );
                }
            }
            if reports.iter().any(|r| !r.matched) {
                anyhow::bail!("model archive verification failed");
            }
        }
        Err(e) => tracing::warn!("warmup verification skipped: {}", e),
    }

    let svc = TokenizerService::new(registry);

    tracing::info!("Tokenizer gRPC server listening on {}", addr);
    Server::builder()
        // single-request execution budget; long downloads use shorter ops, but
        // a hard timeout keeps a wedged tokenizer from monopolising threads.
        .timeout(Duration::from_secs(60))
        .http2_keepalive_interval(Some(Duration::from_secs(30)))
        .http2_keepalive_timeout(Some(Duration::from_secs(10)))
        .tcp_keepalive(Some(Duration::from_secs(30)))
        .tcp_nodelay(true)
        .concurrency_limit_per_connection(1024)
        .max_concurrent_streams(Some(2048))
        .add_service(TokenizerServer::new(svc))
        .serve(addr)
        .await?;
    Ok(())
}
