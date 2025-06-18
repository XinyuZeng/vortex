use std::sync::Arc;

use arcref::ArcRef;
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
        ));

        // 2. calculate stats for each row group
        let stats = arcref(ZonedStrategy::new(
            dict,
            compress_then_flat.clone(),
            ZonedLayoutOptions {
                block_size: ROW_BLOCK_SIZE,
                stats: PRUNING_STATS.into(),
                max_variable_length_statistics_size: 64,
            },
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

fn arcref(item: impl LayoutStrategy) -> ArcRef<dyn LayoutStrategy> {
    ArcRef::new_arc(Arc::new(item))
}
