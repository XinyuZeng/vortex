#![cfg(feature = "monoio")]

use bytes::BytesMut;
use flatbuffers::{root, root_unchecked};
use monoio::buf::{IoBufMut, IoVecBufMut, VecBuf};
use monoio::io::{AsyncReadRent, AsyncReadRentExt};
use vortex_buffer::Buffer;
use vortex_error::VortexResult;

use crate::codecs::message_reader::MessageReader;
use crate::flatbuffers::ipc::Message;

pub struct MonoIoMessageReader<R: AsyncReadRent + Unpin> {
    // TODO(ngates): swap this for our own mutable aligned buffer so we can support direct reads.
    read: R,
    message: BytesMut,
    prev_message: BytesMut,
    finished: bool,
}

impl<R: AsyncReadRent + Unpin> MonoIoMessageReader<R> {
    pub async fn try_new(read: R) -> VortexResult<Self> {
        let mut reader = Self {
            read,
            message: BytesMut::new(),
            prev_message: BytesMut::new(),
            finished: false,
        };
        reader.load_next_message().await?;
        Ok(reader)
    }

    async fn load_next_message(&mut self) -> VortexResult<bool> {
        // FIXME(ngates): how do we read into a stack allocated thing?
        let len_buf = self.read.read_exact_into(Vec::with_capacity(4)).await?;

        let len = u32::from_le_bytes(len_buf.as_slice().try_into()?);
        if len == u32::MAX {
            // Marker for no more messages.
            return Ok(false);
        }

        // TODO(ngates): we may be able to use self.message.split() and then swap back after.

        let message = self
            .read
            .read_exact_into(BytesMut::with_capacity(len as usize))
            .await?;

        // Validate that the message is valid a flatbuffer.
        let _ = root::<Message>(message.as_ref())?;

        self.message = message;

        Ok(true)
    }
}

impl<R: AsyncReadRent + Unpin> MessageReader for MonoIoMessageReader<R> {
    fn peek(&self) -> Option<Message> {
        if self.finished {
            return None;
        }
        // The message has been validated by the next() call.
        Some(unsafe { root_unchecked::<Message>(&self.message) })
    }

    async fn next(&mut self) -> VortexResult<Message> {
        if self.finished {
            panic!("StreamMessageReader is finished - should've checked peek!");
        }
        self.prev_message = self.message.split();
        if !self.load_next_message().await? {
            self.finished = true;
        }
        Ok(unsafe { root_unchecked::<Message>(&self.prev_message) })
    }

    async fn next_raw(&mut self) -> VortexResult<Buffer> {
        if self.finished {
            panic!("StreamMessageReader is finished - should've checked peek!");
        }
        self.prev_message = self.message.split();
        if !self.load_next_message().await? {
            self.finished = true;
        }
        Ok(Buffer::from(self.prev_message.clone().freeze()))
    }

    async fn read_into(&mut self, buffers: Vec<Vec<u8>>) -> VortexResult<Vec<Vec<u8>>> {
        Ok(self
            .read
            .readv_exact_into(VecBuf::from(buffers))
            .await?
            .into())
    }
}

trait AsyncReadRentMoreExt: AsyncReadRentExt {
    /// Same as read_exact except unwraps the BufResult into a regular IO result.
    async fn read_exact_into<B: IoBufMut>(&mut self, buf: B) -> std::io::Result<B> {
        match self.read_exact(buf).await {
            (Ok(_), buf) => Ok(buf),
            (Err(e), _) => Err(e),
        }
    }

    /// Same as read_vectored_exact except unwraps the BufResult into a regular IO result.
    async fn readv_exact_into<B: IoVecBufMut>(&mut self, buf: B) -> std::io::Result<B> {
        match self.read_vectored_exact(buf).await {
            (Ok(_), buf) => Ok(buf),
            (Err(e), _) => Err(e),
        }
    }
}

impl<R: AsyncReadRentExt> AsyncReadRentMoreExt for R {}

#[cfg(test)]
mod tests {
    use futures_util::TryStreamExt;
    use vortex::encoding::EncodingRef;
    use vortex::Context;
    use vortex_alp::ALPEncoding;
    use vortex_fastlanes::BitPackedEncoding;

    use super::*;
    use crate::codecs::array_reader::ArrayReader;
    use crate::codecs::ipc_reader::IPCReader;
    use crate::codecs::message_reader::test::create_stream;

    #[monoio::test]
    async fn test_something() -> VortexResult<()> {
        let buffer = create_stream();

        let ctx =
            Context::default().with_encodings([&ALPEncoding as EncodingRef, &BitPackedEncoding]);
        let mut messages = MonoIoMessageReader::try_new(buffer.as_slice()).await?;

        let mut reader = IPCReader::try_from_messages(&ctx, &mut messages).await?;
        while let Some(array) = reader.next().await? {
            futures_util::pin_mut!(array);
            println!("ARRAY {}", array.dtype());

            while let Some(chunk) = array.try_next().await? {
                println!("chunk {:?}", chunk);
            }
        }

        Ok(())
    }

    #[monoio::test]
    async fn test_array_stream() -> VortexResult<()> {
        let buffer = create_stream();

        let ctx =
            Context::default().with_encodings([&ALPEncoding as EncodingRef, &BitPackedEncoding]);
        let mut messages = MonoIoMessageReader::try_new(buffer.as_slice()).await?;

        let mut reader = IPCReader::try_from_messages(&ctx, &mut messages).await?;
        while let Some(array) = reader.next().await? {
            futures_util::pin_mut!(array);
            while let Some(array) = array.try_next().await? {
                println!("chunk {:?}", array);
            }
        }

        Ok(())
    }
}
