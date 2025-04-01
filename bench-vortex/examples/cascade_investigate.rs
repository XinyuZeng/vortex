use std::collections::HashSet;
use std::path::PathBuf;

use bench_vortex::parquet_reader::parquet_to_vortex;
use futures::StreamExt;
use tokio::fs;
use vortex::arrays::PrimitiveArray;
use vortex::error::VortexResult;
use vortex::nbytes::NBytes;
use vortex::{ArrayRef, ToCanonical};
use vortex_btrblocks::integer::{DictScheme, SparseScheme};
use vortex_btrblocks::{Compressor, IntCompressor, Scheme};

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
        let mut columns = HashSet::new();
        let mut stream = parquet_to_vortex(parquet_path.clone()).await.unwrap();
        while let Some(Ok(array)) = stream.next().await {
            let struct_array = array.to_struct().unwrap();
            for (col_idx, field) in struct_array.fields().iter().enumerate() {
                // if col_idx != 24 {
                //     continue;
                // }
                if let Some(array) = field.as_primitive_typed() {
                    if array.ptype() <= vortex::dtype::PType::I64 {
                        let primitive_array: PrimitiveArray = array.to_primitive().unwrap();
                        let cr = cr_cascade(primitive_array);
                        if cr.is_err() {
                            println!(
                                "File: {}, Column: {}, Error: {:?}",
                                dataset_name,
                                col_idx,
                                cr.err().unwrap()
                            );
                            continue;
                        }
                        let cr = cr.unwrap();

                        if cr.cr_3 < cr.cr_1 * 0.4 {
                            if columns.contains(&col_idx) {
                                continue;
                            }
                            println!(
                                "File: {}, Column: {}, CR(level 1): {:.4}, CR(level 3): {:.4}\nTree(level 1): {}\nTree(level 3): {}",
                                dataset_name,
                                col_idx,
                                cr.cr_1,
                                cr.cr_3,
                                cr.tree_1,
                                cr.tree_3
                            );
                            columns.insert(col_idx);
                        }
                    }
                }
                // if col_idx == 24 {
                //     return;
                // }
            }
        }
    }
}

struct CascadeResult {
    cr_1: f64,
    cr_3: f64,
    tree_1: String,
    tree_3: String,
}

// Given a canonical array of integer, return the CR when cascade level is 1 and cascade level is 3.
fn cr_cascade(array: PrimitiveArray) -> VortexResult<CascadeResult> {
    let exclude = vec![SparseScheme.code(), DictScheme.code()];
    let cascade_1_array = IntCompressor::compress(&array, false, 1, &exclude)?;
    let cascade_3_array = IntCompressor::compress(&array, false, 3, &exclude)?;
    let cr_cascade_1 = cascade_1_array.nbytes() as f64 / array.nbytes() as f64;
    let cr_cascade_3 = cascade_3_array.nbytes() as f64 / array.nbytes() as f64;
    Ok(CascadeResult {
        cr_1: cr_cascade_1,
        cr_3: cr_cascade_3,
        tree_1: tree_display(cascade_1_array),
        tree_3: tree_display(cascade_3_array),
    })
}

fn tree_display(array: ArrayRef) -> String {
    let tree = array.tree_display();
    format!("{}", tree)
}
