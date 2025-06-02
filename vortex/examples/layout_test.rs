#![allow(dead_code)]
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::Instant;

use arrow::array::RecordBatchReader;
use itertools::Itertools;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use tokio::fs;
use vortex::arrays::ChunkedArray;
use vortex::dtype::arrow::FromArrowType;
use vortex::dtype::DType;
use vortex::{ArrayRef, IntoArray};
use vortex_array::stream::ArrayStreamExt;
use vortex_array::TryIntoArray;
use vortex_error::VortexResult;
use vortex_file::{VortexLayoutStrategy, VortexOpenOptions, VortexWriteOptions};
use vortex_layout::layouts::chunked::writer::ChunkedLayoutStrategy;
use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex_layout::{LayoutStrategy as LayoutStrategyTrait, StructStrategy};

/// Trait for defining different layout strategies with metadata
trait LayoutTestStrategy: Send + Sync {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn create_strategy(&self) -> Box<dyn LayoutStrategyTrait>;
}

/// Flat layout strategy - stores data in simple flat chunks
struct FlatLayoutTestStrategy;

impl LayoutTestStrategy for FlatLayoutTestStrategy {
    fn name(&self) -> &'static str {
        "flat"
    }

    fn description(&self) -> &'static str {
        "Flat layout with simple chunking, no compression or optimization"
    }

    fn create_strategy(&self) -> Box<dyn LayoutStrategyTrait> {
        Box::new(FlatLayoutStrategy::default())
    }
}

/// Chunked layout strategy - groups data into chunks with flat storage per chunk
struct ChunkedLayoutTestStrategy;

impl LayoutTestStrategy for ChunkedLayoutTestStrategy {
    fn name(&self) -> &'static str {
        "chunked"
    }

    fn description(&self) -> &'static str {
        "Chunked layout that organizes data into separate chunks"
    }

    fn create_strategy(&self) -> Box<dyn LayoutStrategyTrait> {
        Box::new(ChunkedLayoutStrategy::default())
    }
}

/// Struct layout strategy - preserves struct arrays and uses flat for everything else
struct StructLayoutTestStrategy;

impl LayoutTestStrategy for StructLayoutTestStrategy {
    fn name(&self) -> &'static str {
        "struct"
    }

    fn description(&self) -> &'static str {
        "Struct-preserving layout that maintains struct boundaries"
    }

    fn create_strategy(&self) -> Box<dyn LayoutStrategyTrait> {
        Box::new(StructStrategy)
    }
}

/// Default Vortex layout strategy - the production default
struct VortexLayoutTestStrategy;

impl LayoutTestStrategy for VortexLayoutTestStrategy {
    fn name(&self) -> &'static str {
        "vortex_default"
    }

    fn description(&self) -> &'static str {
        "Default Vortex layout strategy with compression and optimization"
    }

    fn create_strategy(&self) -> Box<dyn LayoutStrategyTrait> {
        Box::new(VortexLayoutStrategy::default())
    }
}

/// Registry of all available layout strategies
struct LayoutStrategyRegistry {
    strategies: HashMap<String, Box<dyn LayoutTestStrategy>>,
}

impl LayoutStrategyRegistry {
    fn new() -> Self {
        let mut strategies: HashMap<String, Box<dyn LayoutTestStrategy>> = HashMap::new();

        // Register all available strategies
        let test_strategies: Vec<Box<dyn LayoutTestStrategy>> = vec![
            Box::new(FlatLayoutTestStrategy),
            Box::new(ChunkedLayoutTestStrategy),
            Box::new(StructLayoutTestStrategy),
            Box::new(VortexLayoutTestStrategy),
        ];

        for strategy in test_strategies {
            strategies.insert(strategy.name().to_string(), strategy);
        }

        Self { strategies }
    }

    fn get_strategy(&self, name: &str) -> Option<&dyn LayoutTestStrategy> {
        self.strategies.get(name).map(|s| s.as_ref())
    }

    fn list_strategies(&self) -> Vec<&dyn LayoutTestStrategy> {
        self.strategies.values().map(|s| s.as_ref()).collect()
    }
}

/// Configuration for the layout test
#[derive(Clone)]
struct LayoutTestConfig {
    input_parquet_path: PathBuf,
    output_dir: PathBuf,
    strategies_to_test: Vec<String>,
    benchmark_reads: bool,
}

impl LayoutTestConfig {
    fn new(input_path: &str) -> Self {
        let input_parquet_path = PathBuf::from(input_path);
        let input_stem = input_parquet_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");

        let output_dir = PathBuf::from(format!("./vortex_layout_test_output/{}", input_stem));

        Self {
            input_parquet_path,
            output_dir,
            strategies_to_test: vec![], // Will be populated with all strategies by default
            benchmark_reads: true,
        }
    }

    fn with_strategies(mut self, strategies: Vec<String>) -> Self {
        self.strategies_to_test = strategies;
        self
    }

    fn with_output_dir(mut self, output_dir: PathBuf) -> Self {
        self.output_dir = output_dir;
        self
    }

    fn with_benchmark_reads(mut self, benchmark: bool) -> Self {
        self.benchmark_reads = benchmark;
        self
    }
}

/// Results from testing a layout strategy
#[derive(Debug)]
struct LayoutTestResult {
    strategy_name: String,
    write_time_ms: u128,
    file_size_bytes: u64,
    read_time_ms: Option<u128>,
    read_throughput_rows_per_sec: Option<f64>,
    success: bool,
    error_message: Option<String>,
}

/// Main layout test runner
struct LayoutTester {
    config: LayoutTestConfig,
    registry: LayoutStrategyRegistry,
}

impl LayoutTester {
    fn new(config: LayoutTestConfig) -> Self {
        Self {
            config,
            registry: LayoutStrategyRegistry::new(),
        }
    }

    async fn run_tests(&self) -> VortexResult<Vec<LayoutTestResult>> {
        println!("🚀 Starting layout strategy tests");
        println!("📁 Input: {}", self.config.input_parquet_path.display());
        println!("📂 Output directory: {}", self.config.output_dir.display());

        // Create output directory
        fs::create_dir_all(&self.config.output_dir).await?;

        // Read the parquet file once
        println!("📖 Reading Parquet file...");
        let parquet_data = self.read_parquet_file().await?;
        let row_count = parquet_data.len();
        println!("✅ Loaded {} rows", row_count);

        // Determine which strategies to test
        let strategies_to_test = if self.config.strategies_to_test.is_empty() {
            self.registry
                .list_strategies()
                .iter()
                .map(|s| s.name().to_string())
                .collect()
        } else {
            self.config.strategies_to_test.clone()
        };

        println!("🧪 Testing {} layout strategies", strategies_to_test.len());

        let mut results = Vec::new();

        for strategy_name in &strategies_to_test {
            if let Some(strategy) = self.registry.get_strategy(strategy_name) {
                println!(
                    "\n🔄 Testing strategy: {} ({})",
                    strategy.name(),
                    strategy.description()
                );

                let result = self.test_strategy(strategy, &parquet_data, row_count).await;
                results.push(result);
            } else {
                println!("⚠️  Unknown strategy: {}", strategy_name);
                results.push(LayoutTestResult {
                    strategy_name: strategy_name.clone(),
                    write_time_ms: 0,
                    file_size_bytes: 0,
                    read_time_ms: None,
                    read_throughput_rows_per_sec: None,
                    success: false,
                    error_message: Some(format!("Unknown strategy: {}", strategy_name)),
                });
            }
        }

        self.print_results(&results, row_count);
        self.save_results_csv(&results).await?;

        Ok(results)
    }

    async fn read_parquet_file(&self) -> VortexResult<ArrayRef> {
        let reader =
            ParquetRecordBatchReaderBuilder::try_new(File::open(&self.config.input_parquet_path)?)?
                .build()?;

        let dtype = DType::from_arrow(reader.schema());
        let chunks = reader
            .map(|record_batch| record_batch?.try_into_array())
            .try_collect()?;

        Ok(ChunkedArray::try_new(chunks, dtype)?.into_array())
    }

    async fn test_strategy(
        &self,
        strategy: &dyn LayoutTestStrategy,
        parquet_data: &ArrayRef,
        row_count: usize,
    ) -> LayoutTestResult {
        let strategy_name = strategy.name().to_string();
        let output_path = self
            .config
            .output_dir
            .join(format!("{}.vortex", strategy_name));

        // Test writing
        let write_start = Instant::now();
        let write_result = self
            .write_with_strategy(strategy, parquet_data, &output_path)
            .await;
        let write_time_ms = write_start.elapsed().as_millis();

        match write_result {
            Ok(()) => {
                let file_size_bytes = match fs::metadata(&output_path).await {
                    Ok(metadata) => metadata.len(),
                    Err(_) => 0,
                };

                println!(
                    "  ✅ Write: {}ms, Size: {} bytes",
                    write_time_ms, file_size_bytes
                );

                // Test reading if enabled
                let (read_time_ms, read_throughput_rows_per_sec) = if self.config.benchmark_reads {
                    match self.benchmark_read(&output_path, row_count).await {
                        Ok((time, throughput)) => {
                            println!(
                                "  ✅ Read: {}ms, Throughput: {:.2} rows/sec",
                                time, throughput
                            );
                            (Some(time), Some(throughput))
                        }
                        Err(e) => {
                            println!("  ❌ Read failed: {}", e);
                            (None, None)
                        }
                    }
                } else {
                    (None, None)
                };

                LayoutTestResult {
                    strategy_name,
                    write_time_ms,
                    file_size_bytes,
                    read_time_ms,
                    read_throughput_rows_per_sec,
                    success: true,
                    error_message: None,
                }
            }
            Err(e) => {
                println!("  ❌ Write failed: {}", e);
                LayoutTestResult {
                    strategy_name,
                    write_time_ms,
                    file_size_bytes: 0,
                    read_time_ms: None,
                    read_throughput_rows_per_sec: None,
                    success: false,
                    error_message: Some(e.to_string()),
                }
            }
        }
    }

    async fn write_with_strategy(
        &self,
        strategy: &dyn LayoutTestStrategy,
        data: &ArrayRef,
        output_path: &Path,
    ) -> VortexResult<()> {
        let file = fs::File::create(output_path).await?;

        // Convert Box<dyn LayoutStrategy> to a concrete type by dereferencing and re-boxing
        // This is needed because VortexWriteOptions expects the strategy to implement LayoutStrategy directly
        match strategy.name() {
            "flat" => {
                VortexWriteOptions::default()
                    .with_strategy(FlatLayoutStrategy::default())
                    .write(file, data.to_array_stream())
                    .await?;
            }
            "chunked" => {
                VortexWriteOptions::default()
                    .with_strategy(ChunkedLayoutStrategy::default())
                    .write(file, data.to_array_stream())
                    .await?;
            }
            "struct" => {
                VortexWriteOptions::default()
                    .with_strategy(StructStrategy)
                    .write(file, data.to_array_stream())
                    .await?;
            }
            "vortex_default" => {
                VortexWriteOptions::default()
                    .with_strategy(VortexLayoutStrategy::default())
                    .write(file, data.to_array_stream())
                    .await?;
            }
            _ => {
                return Err(vortex_error::vortex_err!(
                    "Unknown strategy: {}",
                    strategy.name()
                ));
            }
        }

        Ok(())
    }

    async fn benchmark_read(
        &self,
        vortex_path: &Path,
        row_count: usize,
    ) -> VortexResult<(u128, f64)> {
        let start = Instant::now();

        let vortex_array_stream = VortexOpenOptions::file()
            .open(vortex_path)
            .await?
            .scan()?
            .into_array_stream()?;

        let _vortex_array = vortex_array_stream.read_all().await?;
        let read_time_ms = start.elapsed().as_millis();

        let throughput = if read_time_ms > 0 {
            (row_count as f64) / (read_time_ms as f64 / 1000.0)
        } else {
            0.0
        };

        Ok((read_time_ms, throughput))
    }

    fn print_results(&self, results: &[LayoutTestResult], row_count: usize) {
        println!("\n📊 RESULTS SUMMARY");
        println!("==================");
        println!("Input rows: {}", row_count);

        // Find best performing strategies
        let mut successful_results: Vec<_> = results.iter().filter(|r| r.success).collect();

        if !successful_results.is_empty() {
            successful_results.sort_by_key(|r| r.file_size_bytes);
            println!(
                "\n🏆 Best compression: {} ({} bytes)",
                successful_results[0].strategy_name, successful_results[0].file_size_bytes
            );

            if self.config.benchmark_reads {
                successful_results.sort_by(|a, b| {
                    b.read_throughput_rows_per_sec
                        .partial_cmp(&a.read_throughput_rows_per_sec)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                if let Some(fastest) = successful_results.first() {
                    if let Some(throughput) = fastest.read_throughput_rows_per_sec {
                        println!(
                            "🚀 Fastest read: {} ({:.2} rows/sec)",
                            fastest.strategy_name, throughput
                        );
                    }
                }
            }
        }

        println!("\nDetailed Results:");
        println!(
            "{:<15} {:<8} {:<12} {:<10} {:<15} {:<20}",
            "Strategy", "Success", "Size (bytes)", "Write (ms)", "Read (ms)", "Throughput (rows/s)"
        );
        println!("{}", "-".repeat(85));

        for result in results {
            let success_icon = if result.success { "✅" } else { "❌" };
            let read_time = result
                .read_time_ms
                .map_or("N/A".to_string(), |t| t.to_string());
            let throughput = result
                .read_throughput_rows_per_sec
                .map_or("N/A".to_string(), |t| format!("{:.2}", t));

            println!(
                "{:<15} {:<8} {:<12} {:<10} {:<15} {:<20}",
                result.strategy_name,
                success_icon,
                result.file_size_bytes,
                result.write_time_ms,
                read_time,
                throughput
            );

            if let Some(error) = &result.error_message {
                println!("  └─ Error: {}", error);
            }
        }
    }

    async fn save_results_csv(&self, results: &[LayoutTestResult]) -> VortexResult<()> {
        let csv_path = self.config.output_dir.join("layout_test_results.csv");
        let mut csv_content = String::new();

        csv_content.push_str("strategy_name,success,write_time_ms,file_size_bytes,read_time_ms,read_throughput_rows_per_sec,error_message\n");

        for result in results {
            csv_content.push_str(&format!(
                "{},{},{},{},{},{},{}\n",
                result.strategy_name,
                result.success,
                result.write_time_ms,
                result.file_size_bytes,
                result
                    .read_time_ms
                    .map_or("".to_string(), |t| t.to_string()),
                result
                    .read_throughput_rows_per_sec
                    .map_or("".to_string(), |t| format!("{:.2}", t)),
                result.error_message.as_deref().unwrap_or("")
            ));
        }

        fs::write(&csv_path, csv_content).await?;
        println!("\n💾 Results saved to: {}", csv_path.display());

        Ok(())
    }

    fn list_available_strategies(&self) {
        println!("Available Layout Strategies:");
        println!("============================");
        for strategy in self.registry.list_strategies() {
            println!("• {}: {}", strategy.name(), strategy.description());
        }
    }
}

#[tokio::main]
async fn main() -> VortexResult<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        println!(
            "Usage: {} <parquet_file_path> [strategy1,strategy2,...]",
            args[0]
        );
        println!("       {} --list-strategies", args[0]);
        return Ok(());
    }

    if args[1] == "--list-strategies" {
        let tester = LayoutTester::new(LayoutTestConfig::new("dummy"));
        tester.list_available_strategies();
        return Ok(());
    }

    let parquet_path = &args[1];
    let mut config = LayoutTestConfig::new(parquet_path);

    // Parse specific strategies if provided
    if args.len() > 2 {
        let strategies: Vec<String> = args[2].split(',').map(|s| s.trim().to_string()).collect();
        config = config.with_strategies(strategies);
    }

    // Verify input file exists
    if !Path::new(parquet_path).exists() {
        eprintln!("❌ Error: Parquet file '{}' not found", parquet_path);
        return Ok(());
    }

    let tester = LayoutTester::new(config);
    tester.run_tests().await?;

    Ok(())
}
