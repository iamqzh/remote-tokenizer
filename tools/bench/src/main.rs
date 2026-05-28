//! High-concurrency load generator for the tokenizer gRPC server.
//!
//! Spawns N async workers (each with its own client over a shared `Channel`),
//! drives them for `--duration-secs`, records per-task HdrHistograms and
//! aggregates them at the end. Reports qps, p50/p95/p99/p999, max latency and
//! error rate.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::Parser;
use hdrhistogram::Histogram;
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tonic::transport::Endpoint;

use tokenizer_proto::tokenizer_client::TokenizerClient;
use tokenizer_proto::{DetokenizeRequest, DetokenizeStreamRequest, TokenizeRequest};

#[derive(Parser, Debug, Clone)]
#[command(name = "bench", about = "tokenizer-service load generator")]
struct Args {
    /// Server endpoint, e.g. http://127.0.0.1:50051
    #[arg(long, default_value = "http://127.0.0.1:50051")]
    endpoint: String,

    /// Model URI (e.g. gpt-3.5-turbo, model://kimi-k2-instruct, mock).
    #[arg(long)]
    model: String,

    /// RPC to drive.
    #[arg(long, value_enum, default_value_t = Rpc::Tokenize)]
    rpc: Rpc,

    /// Number of concurrent workers.
    #[arg(long, default_value_t = 32)]
    concurrency: usize,

    /// How long to run, in seconds.
    #[arg(long, default_value_t = 10)]
    duration_secs: u64,

    /// Input text for tokenize / detokenize-stream prompts. detokenize uses
    /// a fixed numeric warmup pulled from a pre-tokenize round trip.
    #[arg(
        long,
        default_value = "Hello, world! 你好。Hello, world! 你好。Hello, world! 你好。"
    )]
    text: String,

    /// Print per-second progress (qps so far).
    #[arg(long, default_value_t = true)]
    progress: bool,
}

#[derive(clap::ValueEnum, Clone, Debug)]
enum Rpc {
    Tokenize,
    Detokenize,
    DetokenizeStream,
}

fn new_hist() -> Histogram<u64> {
    // 1 µs .. 60 s in nanoseconds, 3 sig figs.
    Histogram::<u64>::new_with_bounds(1_000, 60_000_000_000, 3).unwrap()
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let args = Args::parse();

    println!(
        "bench: endpoint={} model={} rpc={:?} concurrency={} duration={}s",
        args.endpoint, args.model, args.rpc, args.concurrency, args.duration_secs
    );

    let endpoint = Endpoint::from_shared(args.endpoint.clone())?
        .tcp_nodelay(true)
        .http2_keep_alive_interval(Duration::from_secs(30))
        .keep_alive_timeout(Duration::from_secs(10));
    let channel = endpoint
        .connect()
        .await
        .with_context(|| format!("connect to {}", args.endpoint))?;

    // For detokenize variants we need real token ids. Do a single warmup
    // tokenize so we exercise the right RPC.
    let prep_ids: Vec<u32> = match args.rpc {
        Rpc::Detokenize | Rpc::DetokenizeStream => {
            let mut c = TokenizerClient::new(channel.clone());
            let resp = c
                .tokenize(TokenizeRequest {
                    model: args.model.clone(),
                    messages: vec![],
                    add_generation_prompt: false,
                    raw_text: args.text.clone(),
                    ..Default::default()
                })
                .await
                .context("warmup tokenize to obtain ids for detokenize bench")?
                .into_inner();
            resp.token_ids
        }
        Rpc::Tokenize => Vec::new(),
    };

    let stop = Arc::new(AtomicBool::new(false));
    let total_ok = Arc::new(AtomicU64::new(0));
    let total_err = Arc::new(AtomicU64::new(0));

    // Stop the workers after `duration_secs`.
    {
        let stop = stop.clone();
        let secs = args.duration_secs;
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs(secs)).await;
            stop.store(true, Ordering::SeqCst);
        });
    }

    // Per-second progress printer.
    if args.progress {
        let stop = stop.clone();
        let ok = total_ok.clone();
        let err = total_err.clone();
        tokio::spawn(async move {
            let mut last = 0u64;
            let mut last_err = 0u64;
            while !stop.load(Ordering::SeqCst) {
                tokio::time::sleep(Duration::from_secs(1)).await;
                let now = ok.load(Ordering::Relaxed);
                let now_err = err.load(Ordering::Relaxed);
                eprintln!(
                    "  [tick] +{} ok / +{} err  (total ok={}, err={})",
                    now - last,
                    now_err - last_err,
                    now,
                    now_err
                );
                last = now;
                last_err = now_err;
            }
        });
    }

    // Per-task histograms collected back to main via mpsc.
    let (hist_tx, mut hist_rx) = mpsc::unbounded_channel::<Histogram<u64>>();

    let mut handles = Vec::with_capacity(args.concurrency);
    let wall_start = Instant::now();
    for _ in 0..args.concurrency {
        let ch = channel.clone();
        let stop = stop.clone();
        let total_ok = total_ok.clone();
        let total_err = total_err.clone();
        let text = args.text.clone();
        let model = args.model.clone();
        let rpc = args.rpc.clone();
        let prep_ids = prep_ids.clone();
        let hist_tx = hist_tx.clone();
        handles.push(tokio::spawn(async move {
            let mut hist = new_hist();
            let mut client = TokenizerClient::new(ch);
            while !stop.load(Ordering::SeqCst) {
                let t0 = Instant::now();
                let ok = match rpc {
                    Rpc::Tokenize => client
                        .tokenize(TokenizeRequest {
                            model: model.clone(),
                            messages: vec![],
                            add_generation_prompt: false,
                            raw_text: text.clone(),
                            ..Default::default()
                        })
                        .await
                        .map(|_| ())
                        .is_ok(),
                    Rpc::Detokenize => client
                        .detokenize(DetokenizeRequest {
                            model: model.clone(),
                            token_ids: prep_ids.clone(),
                            skip_special_tokens: true,
                        })
                        .await
                        .map(|_| ())
                        .is_ok(),
                    Rpc::DetokenizeStream => {
                        match client
                            .detokenize_stream(DetokenizeStreamRequest {
                                model: model.clone(),
                                token_ids: prep_ids.clone(),
                                skip_special_tokens: true,
                            })
                            .await
                        {
                            Ok(resp) => {
                                let mut s = resp.into_inner();
                                let mut all_ok = true;
                                while let Some(chunk) = s.next().await {
                                    if chunk.is_err() {
                                        all_ok = false;
                                        break;
                                    }
                                }
                                all_ok
                            }
                            Err(_) => false,
                        }
                    }
                };
                let dt = t0.elapsed().as_nanos() as u64;
                if ok {
                    // Histogram is bounded at 60s; saturate just in case.
                    let _ = hist.record(dt.min(60_000_000_000));
                    total_ok.fetch_add(1, Ordering::Relaxed);
                } else {
                    total_err.fetch_add(1, Ordering::Relaxed);
                }
            }
            let _ = hist_tx.send(hist);
        }));
    }
    drop(hist_tx);

    for h in handles {
        let _ = h.await;
    }

    // Aggregate per-task histograms.
    let mut agg = new_hist();
    while let Some(h) = hist_rx.recv().await {
        agg.add(&h).ok();
    }

    let wall = wall_start.elapsed();
    let ok = total_ok.load(Ordering::Relaxed);
    let err = total_err.load(Ordering::Relaxed);
    let total = ok + err;
    let qps = ok as f64 / wall.as_secs_f64();
    let err_rate = if total == 0 {
        0.0
    } else {
        err as f64 / total as f64
    };

    let p = |q: f64| ns_to_ms(agg.value_at_quantile(q));
    let max_ms = ns_to_ms(agg.max());

    println!();
    println!("================ BENCH RESULT ================");
    println!(
        "model={} rpc={:?} concurrency={} duration={:.2}s",
        args.model,
        args.rpc,
        args.concurrency,
        wall.as_secs_f64()
    );
    println!("ok       = {}", ok);
    println!("errors   = {}  ({:.4}%)", err, err_rate * 100.0);
    println!("qps      = {:.1}", qps);
    println!("p50  ms  = {:.3}", p(0.50));
    println!("p95  ms  = {:.3}", p(0.95));
    println!("p99  ms  = {:.3}", p(0.99));
    println!("p999 ms  = {:.3}", p(0.999));
    println!("max  ms  = {:.3}", max_ms);
    println!("==============================================");

    // Single-line summary for scripted scrapes.
    println!(
        "SUMMARY model={} rpc={:?} concurrency={} qps={:.1} p50_ms={:.3} p95_ms={:.3} p99_ms={:.3} p999_ms={:.3} max_ms={:.3} errors={}",
        args.model,
        args.rpc,
        args.concurrency,
        qps,
        p(0.50),
        p(0.95),
        p(0.99),
        p(0.999),
        max_ms,
        err,
    );

    Ok(())
}

fn ns_to_ms(ns: u64) -> f64 {
    ns as f64 / 1_000_000.0
}
