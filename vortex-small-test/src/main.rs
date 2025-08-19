// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use anyhow::Result;
use clap::Parser;
use vortex_small_test::{BenchmarkConfig, Scenario};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to output directory for test files
    #[arg(short, long, default_value = "./test-data")]
    output_dir: String,

    /// Number of small files to generate
    #[arg(long, default_value = "100")]
    num_small_files: usize,

    /// Size of each small file in rows
    #[arg(long, default_value = "10000")]
    small_file_rows: usize,

    /// Size of large file in rows
    #[arg(long, default_value = "1000000")]
    large_file_rows: usize,

    /// Run scenario 1: many small files
    #[arg(long, action)]
    small_files: bool,

    /// Run scenario 2: single large file
    #[arg(long, action)]
    large_file: bool,

    /// Enable verbose logging
    #[arg(short, long, action)]
    verbose: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logging
    if args.verbose {
        tracing_subscriber::fmt::init();
    }

    println!("Vortex Small File Performance Testing");
    println!("=====================================");

    let config = BenchmarkConfig {
        output_dir: args.output_dir,
        num_small_files: args.num_small_files,
        small_file_rows: args.small_file_rows,
        large_file_rows: args.large_file_rows,
    };

    if args.small_files {
        println!("\nRunning scenario 1: Many small files");
        let result =
            vortex_small_test::benchmark::run_benchmark(&config, Scenario::ManySmallFiles).await?;
        for r in &result {
            println!(
                "  {} {} ({} files, {} rows): write={}ms, read={}ms, random_access={}ms, size={} bytes",
                r.scenario,
                r.format,
                r.file_count,
                r.total_rows,
                r.write_time_ms,
                r.read_time_ms,
                r.random_access_time_ms.unwrap_or(0),
                r.file_size_bytes
            );
        }
    }

    if args.large_file {
        println!("\nRunning scenario 2: Single large file");
        let result =
            vortex_small_test::benchmark::run_benchmark(&config, Scenario::SingleLargeFile).await?;
        for r in &result {
            println!(
                "  {} {} ({} files, {} rows): write={}ms, read={}ms, random_access={}ms, size={} bytes",
                r.scenario,
                r.format,
                r.file_count,
                r.total_rows,
                r.write_time_ms,
                r.read_time_ms,
                r.random_access_time_ms.unwrap_or(0),
                r.file_size_bytes
            );
        }
    }

    if !args.small_files && !args.large_file {
        println!("No scenario selected. Use --small-files or --large-file to run tests.");
    }

    Ok(())
}
