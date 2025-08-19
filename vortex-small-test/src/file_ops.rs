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

    Ok(file.metadata().await?.len() as usize)
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
