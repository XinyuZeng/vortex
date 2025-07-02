use std::sync::Arc;

use arcref::ArcRef;
use vortex::dtype::DType;
use vortex::error::{vortex_bail, vortex_panic};
use vortex::file::scan::TaskExecutor;
use vortex::layout::layouts::buffered::BufferedStrategy;
use vortex::layout::layouts::chunked::writer::ChunkedLayoutStrategy;
use vortex::layout::layouts::compressed::BtrBlocksCompressedStrategy;
use vortex::layout::layouts::dict::writer::DictStrategy;
use vortex::layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex::layout::layouts::repartition::{RepartitionStrategy, RepartitionWriterOptions};
use vortex::layout::layouts::struct_::writer::StructStrategy;
use vortex::layout::layouts::zoned::writer::{ZonedLayoutOptions, ZonedStrategy};
use vortex::layout::LayoutStrategy;
use vortex::stats::PRUNING_STATS;

const ROW_BLOCK_SIZE: usize = 8192;

pub struct ParquetLayoutStrategy;

impl ParquetLayoutStrategy {
    pub fn with_executor(executor: Arc<dyn TaskExecutor>) -> ArcRef<dyn LayoutStrategy> {
        // 7. for each chunk create a flat layout
        let chunked = arcref(ChunkedLayoutStrategy::default());
        // 6. buffer chunks so they end up with closer segment ids physically
        let buffered = arcref(BufferedStrategy::new(chunked, 2 << 20)); // 2MB
                                                                        // 5. compress each chunk
        let compressing = arcref(BtrBlocksCompressedStrategy::new(
            buffered,
            executor.clone(),
            16,
        ));

        // 4. prior to compression, coalesce up to a minimum size
        let coalescing = arcref(RepartitionStrategy::new(
            compressing,
            RepartitionWriterOptions {
                block_size_minimum: 1 << 20,
                block_len_multiple: ROW_BLOCK_SIZE,
            },
        ));

        // 2.1. | 3.1. compress stats tables and dict values.
        let compress_then_flat = arcref(BtrBlocksCompressedStrategy::new(
            arcref(FlatLayoutStrategy::default()),
            executor.clone(),
            1,
        ));

        // 3. apply dict encoding or fallback
        let dict = arcref(DictStrategy::new(
            coalescing.clone(),
            compress_then_flat.clone(),
            coalescing,
            Default::default(),
            executor.clone(),
        ));

        // 2. calculate stats for each row group
        let stats = arcref(ZonedStrategy::new(
            dict,
            compress_then_flat.clone(),
            ZonedLayoutOptions {
                block_size: ROW_BLOCK_SIZE,
                stats: PRUNING_STATS.into(),
                max_variable_length_statistics_size: 64,
                parallelism: 16,
            },
            executor.clone(),
        ));

        // 1. repartition each column to fixed row counts
        let repartition = arcref(RepartitionStrategy::new(
            stats,
            RepartitionWriterOptions {
                // No minimum block size in bytes
                block_size_minimum: 0,
                // Always repartition into 8K row blocks
                block_len_multiple: ROW_BLOCK_SIZE,
            },
        ));

        // 0. start with splitting columns
        let struct_strategy = StructStrategy::new(repartition);

        // -1: chunked for row groups
        let chunked = ChunkedLayoutStrategy {
            chunk_strategy: ArcRef::new_arc(Arc::new(struct_strategy)),
        };
        arcref(RepartitionStrategy::new(
            arcref(chunked),
            RepartitionWriterOptions {
                block_size_minimum: 0,
                block_len_multiple: 1 << 20,
            },
        ))
    }
}

pub struct BlobLayoutStrategy;

impl BlobLayoutStrategy {
    pub fn with_executor(
        executor: Arc<dyn TaskExecutor>,
        dtype: DType,
        blob_col_idx: usize,
    ) -> ArcRef<dyn LayoutStrategy> {
        let create_column_strategy = |block_len_multiple: usize, compress: bool| {
            // 7. for each chunk create a flat layout
            let chunked = arcref(ChunkedLayoutStrategy::default());
            // 6. buffer chunks so they end up with closer segment ids physically
            let buffered = arcref(BufferedStrategy::new(chunked, 2 << 20)); // 2MB
                                                                            // 5. compress each chunk
            let compressing = if compress {
                arcref(BtrBlocksCompressedStrategy::new(
                    buffered,
                    executor.clone(),
                    16,
                ))
            } else {
                buffered
            };

            // 4. prior to compression, coalesce up to a minimum size
            let coalescing = arcref(RepartitionStrategy::new(
                compressing,
                RepartitionWriterOptions {
                    block_size_minimum: 1 << 20,
                    block_len_multiple,
                },
            ));

            // 2.1. | 3.1. compress stats tables and dict values.
            let compress_then_flat = arcref(BtrBlocksCompressedStrategy::new(
                arcref(FlatLayoutStrategy::default()),
                executor.clone(),
                1,
            ));

            // 3. apply dict encoding or fallback
            let dict = arcref(DictStrategy::new(
                coalescing.clone(),
                compress_then_flat.clone(),
                coalescing,
                Default::default(),
                executor.clone(),
            ));

            // 2. calculate stats for each row group
            let stats = arcref(ZonedStrategy::new(
                dict,
                compress_then_flat.clone(),
                ZonedLayoutOptions {
                    block_size: ROW_BLOCK_SIZE,
                    stats: PRUNING_STATS.into(),
                    max_variable_length_statistics_size: 64,
                    parallelism: 16,
                },
                executor.clone(),
            ));

            // 1. repartition each column to fixed row counts
            let repartition = arcref(RepartitionStrategy::new(
                stats,
                RepartitionWriterOptions {
                    // No minimum block size in bytes
                    block_size_minimum: 0,
                    // Always repartition into 8K row blocks
                    block_len_multiple: ROW_BLOCK_SIZE,
                },
            ));
            repartition
        };

        let repartition_blob = create_column_strategy(2, false);
        let repartition = create_column_strategy(ROW_BLOCK_SIZE, true);

        // 0. start with splitting columns
        let Some(struct_dtype) = dtype.as_struct().cloned() else {
            vortex_panic!("BlobLayoutStrategy requires struct dtype");
        };
        let children: Vec<_> = (0..struct_dtype.nfields())
            .map(|i| {
                if i == blob_col_idx {
                    repartition_blob.clone()
                } else {
                    repartition.clone()
                }
            })
            .collect();
        arcref(MultiColumnStructStrategy::new(children))
    }
}

fn arcref(item: impl LayoutStrategy) -> ArcRef<dyn LayoutStrategy> {
    ArcRef::new_arc(Arc::new(item))
}

pub struct MultiColumnStructStrategy {
    children: Vec<ArcRef<dyn LayoutStrategy>>,
}

/// A [`LayoutStrategy`] that splits a StructArray batch into child layout writers,
/// applying different strategies to each column
impl MultiColumnStructStrategy {
    pub fn new(children: Vec<ArcRef<dyn LayoutStrategy>>) -> Self {
        Self { children }
    }
}

impl LayoutStrategy for MultiColumnStructStrategy {
    fn write_stream(
        &self,
        ctx: &vortex::ArrayContext,
        sequence_writer: vortex::layout::segments::SequenceWriter,
        stream: vortex::layout::SendableSequentialStream,
    ) -> vortex::layout::SendableLayoutWriter {
        use futures::future::try_join_all;
        use futures::StreamExt;
        use itertools::Itertools;
        use vortex::error::{vortex_bail, VortexExpect as _};
        use vortex::layout::layouts::struct_::writer::transpose_stream;
        use vortex::layout::layouts::struct_::StructLayout;
        use vortex::layout::{IntoLayout as _, SequentialStreamAdapter, SequentialStreamExt};
        use vortex::utils::aliases::hash_set::HashSet;
        use vortex::utils::aliases::DefaultHashBuilder;
        use vortex::ToCanonical;

        let dtype = stream.dtype().clone();
        let Some(struct_dtype) = stream.dtype().as_struct().cloned() else {
            vortex_panic!("MultiColumnStructStrategy requires struct dtype");
        };

        if HashSet::<_, DefaultHashBuilder>::from_iter(struct_dtype.names().iter()).len()
            != struct_dtype.names().len()
        {
            return Box::pin(async { vortex_bail!("StructLayout must have unique field names") });
        }

        if self.children.len() != struct_dtype.nfields() {
            return Box::pin(async {
                vortex_bail!("Number of child strategies must match number of struct fields ")
            });
        }

        let stream = stream.map(|chunk| {
            let (sequence_id, chunk) = chunk?;
            if !chunk.all_valid()? {
                vortex_bail!("Cannot push struct chunks with top level invalid values");
            };
            Ok((sequence_id, chunk))
        });

        // stream<struct_chunk> -> stream<vec<column_chunk>>
        let columns_vec_stream = stream.map(|chunk| {
            let (sequence_id, chunk) = chunk?;
            let mut sequence_pointer = sequence_id.descend();
            let struct_chunk = chunk.to_struct()?;
            let columns: Vec<_> = (0..struct_chunk.struct_fields().nfields())
                .map(|idx| {
                    (
                        sequence_pointer.advance(),
                        struct_chunk.fields()[idx].to_array(),
                    )
                })
                .collect();
            Ok(columns)
        });

        // stream<vec<column_chunk>> -> vec<stream<column_chunk>>
        let column_streams = transpose_stream(columns_vec_stream, struct_dtype.nfields());

        let column_dtypes = (0..struct_dtype.nfields()).map(move |idx| {
            struct_dtype
                .field_by_index(idx)
                .vortex_expect("bound checked")
        });

        let ctx = ctx.clone();
        let layout_futures = column_dtypes
            .zip_eq(column_streams)
            .zip_eq(self.children.clone().into_iter())
            .map(move |((dtype, stream), child_strategy)| {
                let column_stream = SequentialStreamAdapter::new(dtype, stream).sendable();
                child_strategy.write_stream(&ctx, sequence_writer.clone(), column_stream)
            });

        Box::pin(async move {
            let column_layouts = try_join_all(layout_futures).await?;
            // TODO(os): transposed stream could count row counts as well,
            // This must hold though, all columns must have the same row count of the struct layout
            let row_count = column_layouts.first().map(|l| l.row_count()).unwrap_or(0);
            Ok(StructLayout::new(row_count, dtype, column_layouts).into_layout())
        })
    }
}
