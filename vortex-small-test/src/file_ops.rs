// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! File operations for Parquet and Vortex formats

use std::fs::File;
use std::path::Path;

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use futures::StreamExt;
use itertools::Itertools;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use tokio::fs::File as TokioFile;
use vortex::ArrayRef;
use vortex::arrow::FromArrowArray;
use vortex::dtype::arrow::FromArrowType;
use vortex::file::VortexOpenOptions;
use vortex::stream::ArrayStreamAdapter;

/// Write RecordBatches to a Parquet file
pub fn write_parquet_file(
    batches: &[RecordBatch],
    schema: SchemaRef,
    path: &Path,
) -> anyhow::Result<usize> {
    let file = File::create(path)?;
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;

    let mut total_bytes = 0;
    for batch in batches {
        writer.write(batch)?;
        total_bytes += batch.get_array_memory_size();
    }

    writer.close()?;
    Ok(total_bytes)
}

/// Read a Parquet file
pub async fn read_parquet_file(path: &Path) -> anyhow::Result<Vec<RecordBatch>> {
    use parquet::arrow::ParquetRecordBatchStreamBuilder;
    use tokio::fs::File;

    let file = File::open(path).await?;
    let builder = ParquetRecordBatchStreamBuilder::new(file).await?;
    let mut stream = builder.build()?;

    let mut batches = Vec::new();
    while let Some(batch) = stream.next().await {
        batches.push(batch?);
    }

    Ok(batches)
}

/// Write RecordBatches to a Vortex file
pub async fn write_vortex_file(
    batches: Vec<RecordBatch>,
    schema: SchemaRef,
    path: &Path,
) -> anyhow::Result<usize> {
    let file = TokioFile::create(path).await?;
    let dtype = vortex::dtype::DType::from_arrow(schema.as_ref());

    let arrays: Vec<ArrayRef> = batches
        .into_iter()
        .map(|batch| ArrayRef::from_arrow(&batch, false))
        .collect();

    let stream = ArrayStreamAdapter::new(dtype, futures::stream::iter(arrays.into_iter().map(Ok)));

    let options = vortex::file::VortexWriteOptions::default();
    let file = options.write(file, stream).await?;

    Ok(usize::try_from(file.metadata().await?.len())?)
}

/// Read a Vortex file
pub async fn read_vortex_file(path: &Path, schema: SchemaRef) -> anyhow::Result<Vec<RecordBatch>> {
    let vortex_file = VortexOpenOptions::file().open(path).await?;
    let rr = vortex_file.scan()?.into_record_batch_reader(schema)?;
    let batches = rr.try_collect()?;
    // let stream = vortex_file.scan()?.into_tokio_array_stream()?;
    // use futures::TryStreamExt;
    // let arrays: Vec<ArrayRef> = stream.try_collect::<Vec<_>>().await?;
    Ok(batches)
}

/// Read specific rows from a Vortex file
pub async fn read_vortex_file_rows(
    path: &Path,
    schema: SchemaRef,
    row_indices: &[u64],
) -> anyhow::Result<Vec<RecordBatch>> {
    let vortex_file = VortexOpenOptions::file().open(path).await?;
    // Convert u64 indices to u64 buffer
    let indices_buffer = vortex::buffer::Buffer::from_iter(row_indices.to_vec());
    let rr = vortex_file
        .scan()?
        .with_row_indices(indices_buffer)
        .into_record_batch_reader(schema)?;
    let batches = rr.try_collect()?;
    Ok(batches)
}

/// Read specific rows from a Parquet file
#[allow(clippy::cast_possible_truncation)]
pub async fn read_parquet_file_rows(
    path: &Path,
    row_indices: &[u64],
) -> anyhow::Result<RecordBatch> {
    use arrow_select::concat::concat_batches;
    use parquet::arrow::ParquetRecordBatchStreamBuilder;
    use parquet::arrow::arrow_reader::{ArrowReaderOptions, RowSelection};
    use tokio::fs::File;

    let file = File::open(path).await?;
    let builder = ParquetRecordBatchStreamBuilder::new_with_options(
        file,
        ArrowReaderOptions::new().with_page_index(true),
    )
    .await?;

    // Get total number of rows in the file
    let total_rows = builder
        .metadata()
        .row_groups()
        .iter()
        .map(|rg| usize::try_from(rg.num_rows()))
        .sum::<Result<usize, _>>()?;

    // Convert u64 indices to usize indices and create ranges for RowSelection
    let ranges: Vec<std::ops::Range<usize>> = row_indices
        .iter()
        .map(|&idx| usize::try_from(idx))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .map(|idx| idx..idx + 1)
        .collect();

    // Create RowSelection from the ranges
    let row_selection = RowSelection::from_consecutive_ranges(ranges.into_iter(), total_rows);

    // Apply row selection to the builder
    let schema = builder.schema().clone();
    let mut stream = builder.with_row_selection(row_selection).build()?;

    let mut all_batches = Vec::new();
    while let Some(batch) = stream.next().await {
        all_batches.push(batch?);
    }

    // Concatenate all selected batches into a single batch
    let result = concat_batches(&schema, &all_batches)?;

    Ok(result)
}
