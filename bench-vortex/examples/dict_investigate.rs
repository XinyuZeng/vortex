#![allow(unused_imports)]
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use bench_vortex::parquet_reader::parquet_to_vortex;
use futures::StreamExt;
use tokio::fs;
use vortex::arrays::PrimitiveArray;
use vortex::error::VortexResult;
use vortex::nbytes::NBytes;
use vortex::{ArrayRef, ToCanonical};
use vortex_btrblocks::integer::SparseScheme;
use vortex_btrblocks::{Compressor, IntCompressor, Scheme};

struct DictionaryStats {
    len_sum: u64,
    nbytes_sum: f64,
    percentage_sum: f64,
    count: u32,
}

impl DictionaryStats {
    fn new() -> Self {
        Self {
            len_sum: 0,
            nbytes_sum: 0.0,
            percentage_sum: 0.0,
            count: 0,
        }
    }

    fn add(&mut self, len: u64, nbytes: f64, percentage: f64) {
        self.len_sum += len;
        self.nbytes_sum += nbytes;
        self.percentage_sum += percentage;
        self.count += 1;
    }

    fn print_averages(&self, dataset_name: &str, idx: usize) {
        if self.count == 0 {
            return;
        }

        let avg_len = self.len_sum as f64 / self.count as f64;
        let avg_nbytes = self.nbytes_sum / self.count as f64;
        let avg_percentage = self.percentage_sum / self.count as f64;

        println!(
            "Dataset: {}, Column: {}, Dictionary values averages - len: {:.2}, nbytes: {:.2} B, percentage: {:.4}%, count: {}",
            dataset_name, idx, avg_len, avg_nbytes, avg_percentage, self.count
        );
    }
}

// Parse a string like "8 B" or "73.74 kB" to extract the numeric value in bytes
fn parse_nbytes(nbytes_str: &str) -> f64 {
    let parts: Vec<&str> = nbytes_str.split_whitespace().collect();
    if parts.len() != 2 {
        return 0.0;
    }

    let value: f64 = parts[0].parse().unwrap_or(0.0);
    let unit = parts[1];

    match unit {
        "B" => value,
        "kB" => value * 1024.0,
        "MB" => value * 1024.0 * 1024.0,
        "GB" => value * 1024.0 * 1024.0 * 1024.0,
        _ => value,
    }
}

#[tokio::main]
async fn main() {
    // env_logger::init();
    // log::set_max_level(log::LevelFilter::Trace);
    let base_path = "/mnt/nvme0n1/xinyu/PBI";
    let mut entries = fs::read_dir(base_path).await.unwrap();

    while let Some(entry) = entries.next_entry().await.unwrap() {
        let dataset_name = entry.file_name().to_string_lossy().to_string();
        let parquet_path = PathBuf::from(format!(
            "{}/{}/parquet/{}_1.parquet",
            base_path, dataset_name, dataset_name
        ));

        if !parquet_path.exists() {
            continue;
        }

        let mut stats_per_col = HashMap::new();
        let mut stream = parquet_to_vortex(parquet_path.clone()).await.unwrap();

        while let Some(Ok(array)) = stream.next().await {
            let struct_array = array.to_struct().unwrap();
            for (idx, field) in struct_array.fields().iter().enumerate() {
                if let Some(array) = field.as_primitive_typed() {
                    if array.ptype() <= vortex::dtype::PType::I64 {
                        let primitive_array: PrimitiveArray = array.to_primitive().unwrap();
                        let cascade_3_array =
                            IntCompressor::compress(&primitive_array, false, 3, &[]).unwrap();
                        let tree = tree_display(cascade_3_array);

                        // Check if root is a dictionary and extract values information
                        if tree.starts_with("root: vortex.dict") {
                            if let Some(values_line) =
                                tree.lines().find(|line| line.starts_with("  values:"))
                            {
                                // Extract length from pattern like "len=1"
                                if let Some(len_start) = values_line.find("len=") {
                                    let len_substr = &values_line[len_start + 4..];
                                    if let Some(len_end) = len_substr.find(')') {
                                        let len_value = &len_substr[..len_end];
                                        let len = len_value.parse::<u64>().unwrap_or(0);

                                        // Extract nbytes from pattern like "nbytes=8 B"
                                        if let Some(nbytes_start) = values_line.find("nbytes=") {
                                            let nbytes_substr = &values_line[nbytes_start + 7..];
                                            if let Some(nbytes_end) = nbytes_substr.find('(') {
                                                let nbytes_value =
                                                    nbytes_substr[..nbytes_end].trim();
                                                let nbytes = parse_nbytes(nbytes_value);

                                                // Extract percentage from pattern like "(0.01%)"
                                                if let Some(percent_start) = nbytes_substr.find("(")
                                                {
                                                    let percent_substr =
                                                        &nbytes_substr[percent_start + 1..];
                                                    if let Some(percent_end) =
                                                        percent_substr.find('%')
                                                    {
                                                        let percent_value =
                                                            &percent_substr[..percent_end];
                                                        let percentage = percent_value
                                                            .parse::<f64>()
                                                            .unwrap_or(0.0);

                                                        stats_per_col
                                                            .entry(idx)
                                                            .or_insert(DictionaryStats::new())
                                                            .add(len, nbytes, percentage);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        for (idx, stats) in stats_per_col.iter() {
            stats.print_averages(&dataset_name, *idx);
        }
    }
}

fn tree_display(array: ArrayRef) -> String {
    let tree = array.tree_display();
    format!("{}", tree)
}
