// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Data generation utilities for creating test datasets

use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray, UInt32Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use rand::Rng;

/// Generate a schema for our test data
pub fn test_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("value", DataType::UInt32, false),
        Field::new("category", DataType::Utf8, false),
        Field::new("timestamp", DataType::Int64, false),
    ]))
}

/// Generate a RecordBatch with random test data
pub fn generate_test_batch(size: usize) -> RecordBatch {
    let mut rng = rand::rng();

    let ids: Vec<i64> = (0..size as i64).collect();
    let values: Vec<u32> = (0..size).map(|_| rng.random_range(0..1000000u32)).collect();
    let categories: Vec<String> = (0..size)
        .map(|_| {
            let categories = ["A", "B", "C", "D", "E"];
            categories[rng.random_range(0..categories.len())].to_string()
        })
        .collect();
    let timestamps: Vec<i64> = (0..size).map(|i| 1640995200 + i as i64 * 10).collect();

    let id_array = Arc::new(Int64Array::from(ids)) as ArrayRef;
    let value_array = Arc::new(UInt32Array::from(values)) as ArrayRef;
    let category_array = Arc::new(StringArray::from(categories)) as ArrayRef;
    let timestamp_array = Arc::new(Int64Array::from(timestamps)) as ArrayRef;

    RecordBatch::try_new(
        test_schema(),
        vec![id_array, value_array, category_array, timestamp_array],
    )
    .expect("Failed to create RecordBatch")
}

/// Generate multiple RecordBatches
pub fn generate_test_batches(batch_size: usize, num_batches: usize) -> Vec<RecordBatch> {
    (0..num_batches)
        .map(|_| generate_test_batch(batch_size))
        .collect()
}
