# vortex-small-test

Performance testing for Vortex and Parquet with small files vs large files.

This crate provides benchmarks to compare the performance of Vortex and Parquet under two scenarios:
1. Many small files
2. A single large file

Tests include:
- Scan performance
- Filtered scan performance
- Random access performance

## Usage

To run the benchmarks:

```bash
# Run scenario 1: many small files
cargo run --bin vortex-small-test -- --small-files

# Run scenario 2: single large file
cargo run --bin vortex-small-test -- --large-file

# Run both scenarios
cargo run --bin vortex-small-test -- --small-files --large-file

# Customize parameters
cargo run --bin vortex-small-test -- \
  --small-files \
  --num-small-files 50 \
  --small-file-rows 5000 \
  --large-file-rows 500000 \
  --output-dir ./my-test-data
```

## Configuration Options

- `--output-dir`: Path to output directory for test files (default: `./test-data`)
- `--num-small-files`: Number of small files to generate (default: `100`)
- `--small-file-rows`: Size of each small file in rows (default: `10000`)
- `--large-file-rows`: Size of large file in rows (default: `1000000`)
- `--small-files`: Run scenario 1: many small files
- `--large-file`: Run scenario 2: single large file
- `--verbose`: Enable verbose logging

## Output

The benchmark results will be printed to the console and include:

- Write time (ms)
- Read time (ms)
- File size (bytes)
- Total rows processed

Files are generated in the specified output directory under subdirectories for each format (`parquet` and `vortex`).