mod health;
mod monitor;
mod qa_bench;
mod runner;

use clap::{CommandFactory, Parser, Subcommand};
use serde::Serialize;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

#[derive(Parser)]
#[command(
    name = "sparkleforge",
    version = env!("CARGO_PKG_VERSION"),
    about = "SparkleForge Rust-centric CLI & Orchestration Monitor",
    long_about = None
)]
struct Cli {
    /// Enable verbose output and bypass quiet-boot filtering
    #[arg(short, long, global = true)]
    verbose: bool,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Fast sub-5ms liveness check
    Ping {
        /// Emit JSON output
        #[arg(long)]
        json: bool,
    },
    /// Print CLI version metadata (sub-5ms)
    Version {
        /// Emit JSON output
        #[arg(long)]
        json: bool,
    },
    /// Fast Rust-native health and environment check
    Health {
        /// Emit JSON output
        #[arg(long)]
        json: bool,
        /// Exit non-zero if Python runtime or API keys are missing
        #[arg(long)]
        strict: bool,
    },
    /// Rust-native orchestration monitor inspecting .sparkleforge state
    Monitor {
        /// Path to session/state directory (defaults to .sparkleforge)
        #[arg(long, value_name = "PATH")]
        session_dir: Option<PathBuf>,
        /// Emit JSON output
        #[arg(long)]
        json: bool,
    },
    /// Rust-native CLI usability benchmark runner and UX scorecard
    #[command(name = "qa-bench")]
    QaBench {
        /// Command(s) to benchmark (defaults to built-in smoke suite)
        #[arg(long = "command", value_name = "CMD")]
        commands: Vec<String>,
        /// Latency budget threshold in milliseconds
        #[arg(long, default_value_t = 2000)]
        max_latency_ms: u64,
        /// Per-command timeout in milliseconds
        #[arg(long, default_value_t = 30000)]
        timeout_ms: u64,
        /// Exit non-zero if overall score is below threshold (0.0-1.0)
        #[arg(long)]
        fail_under: Option<f64>,
        /// Emit JSON output
        #[arg(long)]
        json: bool,
    },
    /// Execute an agent run via Python entry (quiet-boot enabled unless --verbose)
    #[command(disable_help_flag = true)]
    Run {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Execute a query via Python entry (quiet-boot enabled unless --verbose)
    #[command(disable_help_flag = true)]
    Query {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Passthrough subcommands delegated to Python CLI entry
    #[command(external_subcommand)]
    External(Vec<String>),
}

#[derive(Serialize)]
struct PingPayload {
    status: &'static str,
    pong: bool,
    version: &'static str,
    elapsed_ms: f64,
}

#[derive(Serialize)]
struct VersionPayload {
    name: &'static str,
    version: &'static str,
    runtime: &'static str,
}

fn should_delegate_top_level(raw_args: &[String]) -> bool {
    const RUST_SUBCOMMANDS: [&str; 5] = ["ping", "version", "health", "monitor", "qa-bench"];
    const PYTHON_TOP_FLAGS: [&str; 9] = [
        "--query",
        "--request",
        "--prompt",
        "--interactive",
        "--web",
        "--mcp-server",
        "--check-mcp-servers",
        "--daemon",
        "--debug-bootstrap",
    ];

    let first_non_flag = raw_args
        .iter()
        .find(|a| !a.starts_with('-'))
        .map(String::as_str);
    if let Some(cmd) = first_non_flag {
        if RUST_SUBCOMMANDS.contains(&cmd) {
            return false;
        }
    }

    raw_args.iter().any(|arg| {
        PYTHON_TOP_FLAGS
            .iter()
            .any(|flag| arg == flag || arg.starts_with(&format!("{flag}=")))
    })
}

fn run_ping(start: Instant, json: bool) -> i32 {
    let elapsed_ms = ((start.elapsed().as_secs_f64() * 1000.0) * 1000.0).round() / 1000.0;
    if json {
        let payload = PingPayload {
            status: "ok",
            pong: true,
            version: env!("CARGO_PKG_VERSION"),
            elapsed_ms,
        };
        if let Ok(text) = serde_json::to_string(&payload) {
            println!("{text}");
        }
    } else {
        println!(
            "pong (sparkleforge v{}, elapsed: {:.3}ms)",
            env!("CARGO_PKG_VERSION"),
            elapsed_ms
        );
    }
    0
}

fn run_version(json: bool) -> i32 {
    if json {
        let payload = VersionPayload {
            name: "sparkleforge",
            version: env!("CARGO_PKG_VERSION"),
            runtime: "rust",
        };
        if let Ok(text) = serde_json::to_string(&payload) {
            println!("{text}");
        }
    } else {
        println!("sparkleforge {} (rust-cli)", env!("CARGO_PKG_VERSION"));
    }
    0
}

fn build_tokio_runtime() -> anyhow::Result<tokio::runtime::Runtime> {
    Ok(tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?)
}

fn dispatch(raw_args: &[String]) -> i32 {
    let start = Instant::now();

    // Sub-5ms fast path for ping/version without constructing clap parser metadata
    match raw_args {
        [cmd] if cmd == "ping" => return run_ping(start, false),
        [cmd, flag] if cmd == "ping" && flag == "--json" => return run_ping(start, true),
        [cmd] if cmd == "version" => return run_version(false),
        [cmd, flag] if cmd == "version" && flag == "--json" => return run_version(true),
        _ => {}
    }

    if should_delegate_top_level(raw_args) {
        return match build_tokio_runtime() {
            Ok(rt) => rt.block_on(runner::run_delegated_cli(raw_args, false)),
            Err(err) => {
                eprintln!("[FAIL] Runtime init error: {err}");
                1
            }
        };
    }

    let mut argv_with_prog = Vec::with_capacity(raw_args.len() + 1);
    argv_with_prog.push("sparkleforge".to_string());
    argv_with_prog.extend(raw_args.iter().cloned());

    let cli = match Cli::try_parse_from(argv_with_prog) {
        Ok(c) => c,
        Err(err) => {
            let code = err.exit_code();
            let _ = err.print();
            return code;
        }
    };

    match cli.command {
        Some(Commands::Ping { json }) => run_ping(start, json),
        Some(Commands::Version { json }) => run_version(json),
        Some(Commands::Health { json, strict }) => health::run_health(None, json, strict),
        Some(Commands::Monitor { session_dir, json }) => {
            monitor::run_monitor(session_dir.as_deref(), json)
        }
        Some(Commands::QaBench {
            commands,
            max_latency_ms,
            timeout_ms,
            fail_under,
            json,
        }) => match build_tokio_runtime() {
            Ok(rt) => rt.block_on(qa_bench::run_qa_bench(
                commands,
                max_latency_ms,
                timeout_ms,
                fail_under,
                json,
            )),
            Err(err) => {
                eprintln!("[FAIL] Runtime init error: {err}");
                1
            }
        },
        Some(Commands::Run { args }) => {
            let mut forward = Vec::with_capacity(args.len() + 1);
            forward.push("run".to_string());
            forward.extend(args);
            match build_tokio_runtime() {
                Ok(rt) => rt.block_on(runner::run_delegated_cli(&forward, cli.verbose)),
                Err(err) => {
                    eprintln!("[FAIL] Runtime init error: {err}");
                    1
                }
            }
        }
        Some(Commands::Query { args }) => {
            let mut forward = Vec::with_capacity(args.len() + 1);
            forward.push("query".to_string());
            forward.extend(args);
            match build_tokio_runtime() {
                Ok(rt) => rt.block_on(runner::run_delegated_cli(&forward, cli.verbose)),
                Err(err) => {
                    eprintln!("[FAIL] Runtime init error: {err}");
                    1
                }
            }
        }
        Some(Commands::External(args)) => match build_tokio_runtime() {
            Ok(rt) => rt.block_on(runner::run_delegated_cli(&args, cli.verbose)),
            Err(err) => {
                eprintln!("[FAIL] Runtime init error: {err}");
                1
            }
        },
        None => {
            let mut cmd = Cli::command();
            let _ = cmd.print_help();
            println!();
            0
        }
    }
}

fn main() -> ExitCode {
    let raw_args: Vec<String> = std::env::args().skip(1).collect();
    let code = dispatch(&raw_args);
    let clamped = u8::try_from(code).unwrap_or(1);
    ExitCode::from(clamped)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_level_delegation_detects_query_and_legacy_flags() {
        assert!(should_delegate_top_level(&[
            "--query".to_string(),
            "hello".to_string()
        ]));
        assert!(should_delegate_top_level(&[
            "--verbose".to_string(),
            "--query=hello".to_string()
        ]));
        assert!(should_delegate_top_level(&["--request".to_string(), "q".to_string()]));

        assert!(!should_delegate_top_level(&["ping".to_string()]));
        assert!(!should_delegate_top_level(&["health".to_string(), "--json".to_string()]));
        assert!(!should_delegate_top_level(&["monitor".to_string(), "--json".to_string()]));
    }

    #[test]
    fn dispatch_ping_and_version_succeed_quickly() {
        assert_eq!(dispatch(&["ping".to_string()]), 0);
        assert_eq!(dispatch(&["ping".to_string(), "--json".to_string()]), 0);
        assert_eq!(dispatch(&["version".to_string()]), 0);
        assert_eq!(dispatch(&["version".to_string(), "--json".to_string()]), 0);
        assert_eq!(dispatch(&[]), 0);
    }
}
