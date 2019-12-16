//! Buffer API.

use std::marker::PhantomData;

use crate::backend::buffer::{Buffer as BufferBackend, BufferError};
use crate::context::GraphicsContext;

pub struct Buffer<S, T>
where
  S: BufferBackend,
{
  repr: S::Repr,
  _t: PhantomData<T>,
}

impl<S, T> Buffer<S, T>
where
  S: BufferBackend,
{
  pub fn new<C>(ctx: &mut C, len: usize) -> Result<Self, BufferError>
  where
    C: GraphicsContext<Backend = S>,
  {
    let repr = unsafe { ctx.backend().new_buffer::<T>(len)? };

    Ok(Buffer {
      repr,
      _t: PhantomData,
    })
  }

  pub fn from_slice<C, X>(ctx: &mut C, slice: X) -> Result<Self, BufferError>
  where
    C: GraphicsContext<Backend = S>,
    X: AsRef<[T]>,
  {
    let repr = unsafe { ctx.backend().from_slice(slice)? };

    Ok(Buffer {
      repr,
      _t: PhantomData,
    })
  }
}
