use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::arrays::primitive::PrimitiveArray;
use crate::arrays::{varbin_scalar, VarBinArray, VarBinVTable};
use crate::vtable::{OperationsVTable, ValidityHelper};
use crate::{Array, ArrayRef, IntoArray, ToCanonical};

impl OperationsVTable<VarBinVTable> for VarBinVTable {
    fn slice(array: &VarBinArray, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        // Get the start and end byte positions from the offsets
        let start_byte_pos = array.offset_at(start)?;
        let end_byte_pos = array.offset_at(stop)?;

        // Slice the bytes from start_byte_pos to end_byte_pos
        let sliced_bytes = array.bytes().slice(start_byte_pos..end_byte_pos);
        println!("sliced_bytes_size: {:?}", sliced_bytes.len());
        // Slice the offsets array and adjust them relative to start_byte_pos
        let original_offsets = array.offsets().slice(start, stop + 1)?;

        // Create new offsets by subtracting start_byte_pos from each offset
        let new_offsets = if let Ok(primitive_offsets) = original_offsets.to_primitive() {
            // Handle based on the offset type
            match primitive_offsets.ptype() {
                vortex_dtype::PType::U8 => {
                    let offsets_slice = primitive_offsets.as_slice::<u8>();
                    let adjusted_offsets: Vec<u8> = offsets_slice
                        .iter()
                        .map(|&offset| offset - start_byte_pos as u8)
                        .collect();
                    PrimitiveArray::from_iter(adjusted_offsets).into_array()
                }
                vortex_dtype::PType::U16 => {
                    let offsets_slice = primitive_offsets.as_slice::<u16>();
                    let adjusted_offsets: Vec<u16> = offsets_slice
                        .iter()
                        .map(|&offset| offset - start_byte_pos as u16)
                        .collect();
                    PrimitiveArray::from_iter(adjusted_offsets).into_array()
                }
                vortex_dtype::PType::U32 => {
                    let offsets_slice = primitive_offsets.as_slice::<u32>();
                    let adjusted_offsets: Vec<u32> = offsets_slice
                        .iter()
                        .map(|&offset| offset - start_byte_pos as u32)
                        .collect();
                    PrimitiveArray::from_iter(adjusted_offsets).into_array()
                }
                vortex_dtype::PType::U64 => {
                    let offsets_slice = primitive_offsets.as_slice::<u64>();
                    let adjusted_offsets: Vec<u64> = offsets_slice
                        .iter()
                        .map(|&offset| offset - start_byte_pos as u64)
                        .collect();
                    PrimitiveArray::from_iter(adjusted_offsets).into_array()
                }
                vortex_dtype::PType::I32 => {
                    let offsets_slice = primitive_offsets.as_slice::<i32>();
                    let adjusted_offsets: Vec<i32> = offsets_slice
                        .iter()
                        .map(|&offset| offset - start_byte_pos as i32)
                        .collect();
                    PrimitiveArray::from_iter(adjusted_offsets).into_array()
                }
                vortex_dtype::PType::I64 => {
                    let offsets_slice = primitive_offsets.as_slice::<i64>();
                    let adjusted_offsets: Vec<i64> = offsets_slice
                        .iter()
                        .map(|&offset| offset - start_byte_pos as i64)
                        .collect();
                    PrimitiveArray::from_iter(adjusted_offsets).into_array()
                }
                _ => {
                    // For unsupported types, fall back to the original method
                    return Ok(VarBinArray::try_new(
                        array.offsets().slice(start, stop + 1)?,
                        array.bytes().clone(),
                        array.dtype().clone(),
                        array.validity().slice(start, stop)?,
                    )?
                    .into_array());
                }
            }
        } else {
            panic!("offset dtype: {:?}", array.offsets().dtype());
            // If we can't get it as primitive, fall back to original method
            // return Ok(VarBinArray::try_new(
            //     array.offsets().slice(start, stop + 1)?,
            //     array.bytes().clone(),
            //     array.dtype().clone(),
            //     array.validity().slice(start, stop)?,
            // )?
            // .into_array());
        };

        VarBinArray::try_new(
            new_offsets,
            sliced_bytes,
            array.dtype().clone(),
            array.validity().slice(start, stop)?,
        )
        .map(|a| a.into_array())
    }

    fn scalar_at(array: &VarBinArray, index: usize) -> VortexResult<Scalar> {
        Ok(varbin_scalar(array.bytes_at(index)?, array.dtype()))
    }
}
