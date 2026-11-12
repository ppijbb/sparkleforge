use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "sparkleforge")]
#[command(about = "SparkleForge Rust CLI", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Example subcommand
    Ping,
}

#[tokio::main]
async fn main() {
    let _cli = Cli::parse();
    println!("SparkleForge Rust CLI v0.1.0 (Cold start ~0ms)");
}
