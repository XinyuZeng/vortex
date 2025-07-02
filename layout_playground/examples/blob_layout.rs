use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use arrow::array::RecordBatchReader;
use itertools::Itertools;
use layout_playground::BlobLayoutStrategy;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use tokio::fs;
use vortex::arrays::ChunkedArray;
use vortex::dtype::arrow::FromArrowType;
use vortex::dtype::DType;
use vortex::error::VortexResult;
use vortex::file::scan::LocalExecutor;
use vortex::file::VortexWriteOptions;
use vortex::stream::ArrayStream;
use vortex::{ArrayRef, IntoArray, TryIntoArray};

struct BlobLayoutConverter {
    input_parquet_path: PathBuf,
    output_dir: PathBuf,
    blob_col_idx: usize,
}

impl BlobLayoutConverter {
    fn new(input_path: &str, output_dir: &str, blob_col_idx: usize) -> Self {
        Self {
            input_parquet_path: PathBuf::from(input_path),
            output_dir: PathBuf::from(output_dir),
            blob_col_idx,
        }
    }

    async fn convert(&self) -> VortexResult<()> {
        println!("Converting Parquet to Vortex using BlobLayoutStrategy...");
        println!("Input: {}", self.input_parquet_path.display());

        // Ensure output directory exists
        fs::create_dir_all(&self.output_dir).await?;

        // Create output filename
        let input_stem = self
            .input_parquet_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");
        let output_path = self.output_dir.join(format!("{}_blob.vortex", input_stem));

        println!("Output: {}", output_path.display());

        // Read Parquet file
        let start_time = Instant::now();
        let parquet_data = self.read_parquet_file().await?;
        let read_time = start_time.elapsed();
        println!("✅ Read Parquet file in {:?}", read_time);

        // Get row count for reporting
        let row_count = parquet_data.len();
        println!("📊 Rows: {}", row_count);

        // Convert using BlobLayoutStrategy
        let convert_start = Instant::now();
        let stream = parquet_data.to_array_stream();
        let dtype = stream.dtype().clone();
        let strategy =
            BlobLayoutStrategy::with_executor(Arc::new(LocalExecutor), dtype, self.blob_col_idx);

        let file = fs::File::create(&output_path).await?;
        VortexWriteOptions::default()
            .with_strategy(strategy)
            .write(file, stream)
            .await?;

        let convert_time = convert_start.elapsed();
        println!("✅ Converted to Vortex in {:?}", convert_time);

        // Get file size
        let file_metadata = fs::metadata(&output_path).await?;
        let file_size = file_metadata.len();
        println!("📁 Output file size: {} bytes", file_size);

        println!("🎉 Conversion completed successfully!");

        Ok(())
    }

    async fn read_parquet_file(&self) -> VortexResult<ArrayRef> {
        let reader =
            ParquetRecordBatchReaderBuilder::try_new(File::open(&self.input_parquet_path)?)?
                .build()?;

        let dtype = DType::from_arrow(reader.schema());
        let chunks = reader
            .map(|record_batch| record_batch?.try_into_array())
            .try_collect()?;

        Ok(ChunkedArray::try_new(chunks, dtype)?.into_array())
    }
}

#[tokio::main]
async fn main() -> VortexResult<()> {
    let input_path = "/home/xinyu/r2-data-catalog-notebook/data/DUDE_filter.parquet";
    let output_dir = "./vortex_layout_test_output";

    let converter = BlobLayoutConverter::new(input_path, output_dir, 3);
    converter.convert().await?;

    Ok(())
}
