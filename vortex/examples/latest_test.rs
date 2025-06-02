use std::env;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use arrow::array::RecordBatchReader;
use itertools::Itertools;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use tokio::fs;
use vortex::arrays::ChunkedArray;
use vortex::arrow::compute::to_arrow_preferred;
use vortex::dtype::arrow::FromArrowType;
use vortex::dtype::DType;
use vortex::{Array, IntoArray};
use vortex_array::stream::ArrayStreamExt;
use vortex_array::TryIntoArray;
use vortex_error::VortexResult;
use vortex_file::{VortexOpenOptions, VortexWriteOptions};

#[tokio::main]
async fn main() -> VortexResult<()> {
    let args: Vec<String> = env::args().collect();

    let datasets = vec![
        "/mnt/nvme0n1/xinyu/tpch/parquet/lineitem_duckdb_double.parquet",
        "/mnt/nvme0n1/xinyu/clickbench/parquet/hits_8M.parquet",
        "/mnt/nvme0n1/xinyu/data/parquet/core.parquet",
        "/mnt/nvme0n1/xinyu/data/parquet/bi.parquet",
        "/mnt/nvme0n1/xinyu/data/parquet/classic.parquet",
        "/mnt/nvme0n1/xinyu/data/parquet/geo.parquet",
        "/mnt/nvme0n1/xinyu/data/parquet/log.parquet",
        "/mnt/nvme0n1/xinyu/data/parquet/ml.parquet",
    ];

    if args.len() > 1 && args[1] == "convert" {
        // Convert Parquet to Vortex and record sizes
        convert_parquet_to_vortex(&datasets).await?;
    } else if args.len() > 1 && args[1] == "benchmark" {
        // Benchmark read performance
        benchmark_read_performance(&datasets).await?;
    } else {
        println!("Usage: {} [convert|benchmark]", args[0]);
    }

    Ok(())
}

async fn convert_parquet_to_vortex(datasets: &[&str]) -> VortexResult<()> {
    println!("Converting Parquet files to Vortex format...");

    let mut csv_file = File::create("file_sizes.csv")?;
    writeln!(csv_file, "dataset,parquet_size_bytes,vortex_size_bytes")?;

    for dataset_path in datasets {
        let path = Path::new(dataset_path);
        let dataset_name = path.file_stem().unwrap().to_str().unwrap();

        // Replace 'parquet' with 'vtxlatest' in the entire path
        let output_path = dataset_path.replace("parquet", "vtxlatest");
        let output_path = PathBuf::from(output_path);

        // Ensure parent directory exists
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent).await?;
        }

        println!("Processing: {}", dataset_path);
        println!("Output to: {}", output_path.display());

        // Get Parquet file size
        let parquet_metadata = std::fs::metadata(dataset_path)?;
        let parquet_size = parquet_metadata.len();

        // Read Parquet file
        let reader =
            ParquetRecordBatchReaderBuilder::try_new(File::open(dataset_path)?)?.build()?;

        let dtype = DType::from_arrow(reader.schema());
        let chunks = reader
            .map(|record_batch| record_batch?.try_into_array())
            .try_collect()?;
        let vortex_array = ChunkedArray::try_new(chunks, dtype)?.into_array();
        let file = fs::File::create(&output_path).await?;
        // Write Vortex file
        if let Ok(_) = VortexWriteOptions::default()
            .write(file, vortex_array.to_array_stream())
            .await
        {
            // Get Vortex file size
            let vortex_metadata = std::fs::metadata(&output_path)?;
            let vortex_size = vortex_metadata.len();

            writeln!(
                csv_file,
                "{},{},{}",
                dataset_name, parquet_size, vortex_size
            )?;

            println!("Converted {} to {}", dataset_path, output_path.display());
            println!("  Parquet size: {} bytes", parquet_size);
            println!("  Vortex size: {} bytes", vortex_size);
        } else {
            println!("Failed to create file: {}", output_path.display());
        }
    }

    println!("File sizes recorded in file_sizes.csv");
    Ok(())
}

async fn benchmark_read_performance(datasets: &[&str]) -> VortexResult<()> {
    println!("Benchmarking read performance...");

    let mut csv_file = File::create("read_times.csv")?;
    writeln!(
        csv_file,
        "dataset,vortex_read_time_ms,vortex_throughput_rows_per_sec"
    )?;

    for dataset_path in datasets {
        let path = Path::new(dataset_path);
        let dataset_name = path.file_stem().unwrap().to_str().unwrap();

        // Replace 'parquet' with 'vtxlatest' in the entire path
        let vortex_path = dataset_path.replace("parquet", "vtxlatest");

        if !Path::new(&vortex_path).exists() {
            println!("Vortex file {} not found, skipping benchmark", vortex_path);
            continue;
        }

        println!("Benchmarking: {}", dataset_name);
        println!("Parquet path: {}", dataset_path);
        println!("Vortex path: {}", vortex_path);

        // Benchmark Vortex read time
        let start = Instant::now();
        let vortex_array_stream_result = VortexOpenOptions::file()
            .open(&vortex_path)
            .await
            .and_then(|file| file.scan())
            .and_then(|scanner| scanner.into_array_stream());

        if let Ok(vortex_array_stream) = vortex_array_stream_result {
            let vortex_array = vortex_array_stream.read_all().await?;
            let arrow_array = to_arrow_preferred(&vortex_array)?;
            let vortex_read_time = start.elapsed().as_millis();

            // Calculate throughput in rows per second
            let row_count = vortex_array.len();
            let throughput = if vortex_read_time > 0 {
                (row_count as f64) / (vortex_read_time as f64 / 1000.0)
            } else {
                0.0
            };

            println!(
                "  Vortex read time: {}ms, rows: {}, throughput: {:.2} rows/sec",
                vortex_read_time, row_count, throughput
            );
            println!("{:?}", arrow_array.data_type());

            writeln!(
                csv_file,
                "{},{},{:.2}",
                dataset_name, vortex_read_time, throughput
            )?;
        } else {
            println!("Failed to read Vortex file: {}", vortex_path);
        }
    }

    println!("Read times recorded in read_times.csv");
    Ok(())
}
