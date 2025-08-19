// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Performance testing for Vortex and Parquet with small files vs large files
//!
//! This crate provides benchmarks to compare the performance of Vortex and Parquet
//! under two scenarios:
//! 1. Many small files
//! 2. A single large file
//!
//! Tests include:
//! - Scan performance
//! - Filtered scan performance
//! - Random access performance

pub mod benchmark;
pub mod data_gen;
pub mod file_ops;

/// Re-export commonly used types
pub use benchmark::{BenchmarkConfig, BenchmarkResult, Scenario};