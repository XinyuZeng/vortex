// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmark module for comparing Parquet and Vortex performance

use std::fs;
use std::path::Path;
use std::time::Instant;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::data_gen::{generate_test_batch, test_schema};
use crate::file_ops::{
    read_parquet_file, read_parquet_file_rows, read_vortex_file, read_vortex_file_rows,
    write_parquet_file, write_vortex_file,
};

#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    pub output_dir: String,
    pub num_small_files: usize,
    pub small_file_rows: usize,
    pub large_file_rows: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub scenario: String,
    pub format: String,
    pub file_count: usize,
    pub total_rows: usize,
    pub write_time_ms: u128,
    pub read_time_ms: u128,
    pub random_access_time_ms: Option<u128>,
    pub file_size_bytes: u64,
}

#[derive(Debug, Clone)]
pub enum Scenario {
    ManySmallFiles,
    SingleLargeFile,
}

/// Run a benchmark for the given scenario and format
pub async fn run_benchmark(
    config: &BenchmarkConfig,
    scenario: Scenario,
) -> Result<Vec<BenchmarkResult>> {
    // Create output directory
    fs::create_dir_all(&config.output_dir)?;

    let mut results = Vec::new();

    // Run Parquet benchmark
    let parquet_result = run_format_benchmark(config, scenario.clone(), "parquet").await?;
    results.push(parquet_result);

    // Run Vortex benchmark
    let vortex_result = run_format_benchmark(config, scenario, "vortex").await?;
    results.push(vortex_result);

    Ok(results)
}

/// Helper function to select a random row index
fn select_random_row(total_rows: usize) -> usize {
    use rand::Rng;
    let mut rng = rand::rng();
    rng.random_range(0..total_rows)
}

async fn run_format_benchmark(
    config: &BenchmarkConfig,
    scenario: Scenario,
    format: &str,
) -> Result<BenchmarkResult> {
    match scenario {
        Scenario::ManySmallFiles => run_many_small_files_benchmark(config, format).await,
        Scenario::SingleLargeFile => run_single_large_file_benchmark(config, format).await,
    }
}

async fn run_many_small_files_benchmark(
    config: &BenchmarkConfig,
    format: &str,
) -> Result<BenchmarkResult> {
    let format_dir = format!("{}/{}", config.output_dir, format);
    fs::create_dir_all(&format_dir)?;

    // Generate data
    let schema = test_schema();
    let batches: Vec<_> = (0..config.num_small_files)
        .map(|_| generate_test_batch(config.small_file_rows))
        .collect();

    // Write files
    let write_start = Instant::now();
    let mut file_paths = Vec::new();

    for (i, batch) in batches.iter().enumerate() {
        let file_path = format!("{}/file_{}.{}", format_dir, i, format);
        let path = Path::new(&file_path);

        if format == "parquet" {
            write_parquet_file(std::slice::from_ref(batch), schema.clone(), path)?;
        } else {
            write_vortex_file(vec![batch.clone()], schema.clone(), path).await?;
        }

        file_paths.push(file_path);
    }

    let write_time = write_start.elapsed();

    // Read files
    let read_start = Instant::now();
    let mut total_rows = 0;
    let mut file_row_counts = Vec::new();

    for file_path in &file_paths {
        let path = Path::new(file_path);
        if format == "parquet" {
            let batches = read_parquet_file(path).await?;
            let rows = batches.iter().map(|b| b.num_rows()).sum::<usize>();
            file_row_counts.push(rows);
            total_rows += rows;
        } else {
            let batches = read_vortex_file(path, schema.clone()).await?;
            let rows = batches.iter().map(|a| a.num_rows()).sum::<usize>();
            file_row_counts.push(rows);
            total_rows += rows;
        }
    }

    let read_time = read_start.elapsed();

    // Random access benchmark for multiple files
    let random_access_time = {
        // Select a random row from the total rows
        let random_row = select_random_row(total_rows);

        // Determine which file contains this row and the local row index
        let mut current_row = 0;
        let mut target_file_index = 0;
        let mut local_row_index = 0;

        for (i, &file_rows) in file_row_counts.iter().enumerate() {
            if random_row >= current_row && random_row < current_row + file_rows {
                target_file_index = i;
                local_row_index = random_row - current_row;
                break;
            }

            current_row += file_rows;
        }

        // Perform the random access
        let random_start = Instant::now();
        if format == "parquet" {
            let _batch = read_parquet_file_rows(
                Path::new(&file_paths[target_file_index]),
                &[local_row_index as u64],
            )
            .await?;
        } else {
            let _batches = read_vortex_file_rows(
                Path::new(&file_paths[target_file_index]),
                schema.clone(),
                &[local_row_index as u64],
            )
            .await?;
        }
        Some(random_start.elapsed().as_millis())
    };

    // Calculate total file size
    let total_file_size: u64 = file_paths
        .iter()
        .map(|path| fs::metadata(path).map(|m| m.len()).unwrap_or(0))
        .sum();

    Ok(BenchmarkResult {
        scenario: "many_small_files".to_string(),
        format: format.to_string(),
        file_count: config.num_small_files,
        total_rows,
        write_time_ms: write_time.as_millis(),
        read_time_ms: read_time.as_millis(),
        random_access_time_ms: random_access_time,
        file_size_bytes: total_file_size,
    })
}

async fn run_single_large_file_benchmark(
    config: &BenchmarkConfig,
    format: &str,
) -> Result<BenchmarkResult> {
    let format_dir = format!("{}/{}", config.output_dir, format);
    fs::create_dir_all(&format_dir)?;

    // Generate data
    let batch = generate_test_batch(config.large_file_rows);
    let schema = test_schema();
    let file_path = format!("{}/large_file.{}", format_dir, format);
    let path = Path::new(&file_path);

    // Write file
    let write_start = Instant::now();
    if format == "parquet" {
        write_parquet_file(std::slice::from_ref(&batch), schema.clone(), path)?;
    } else {
        write_vortex_file(vec![batch.clone()], schema.clone(), path).await?;
    }
    let write_time = write_start.elapsed();

    // Read file
    let read_start = Instant::now();
    let total_rows = if format == "parquet" {
        let batches = read_parquet_file(path).await?;
        batches.iter().map(|b| b.num_rows()).sum::<usize>()
    } else {
        let arrays = read_vortex_file(path, schema.clone()).await?;
        arrays.iter().map(|a| a.num_rows()).sum::<usize>()
    };
    let read_time = read_start.elapsed();

    // Random access benchmark
    let random_row = select_random_row(total_rows);
    let random_access_time = if format == "parquet" {
        let random_start = Instant::now();
        let _batch = read_parquet_file_rows(path, &[random_row as u64]).await?;
        Some(random_start.elapsed().as_millis())
    } else {
        let random_start = Instant::now();
        let _batches = read_vortex_file_rows(path, schema.clone(), &[random_row as u64]).await?;
        Some(random_start.elapsed().as_millis())
    };

    // Get file size
    let file_size = fs::metadata(path).map(|m| m.len()).unwrap_or(0);

    Ok(BenchmarkResult {
        scenario: "single_large_file".to_string(),
        format: format.to_string(),
        file_count: 1,
        total_rows,
        write_time_ms: write_time.as_millis(),
        read_time_ms: read_time.as_millis(),
        random_access_time_ms: random_access_time,
        file_size_bytes: file_size,
    })
}
