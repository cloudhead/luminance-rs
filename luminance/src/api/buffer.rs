//! Buffer API.

use std::marker::PhantomData;

use crate::backend::buffer::{Buffer as BufferBackend, BufferError};
use crate::context::GraphicsContext;

pub struct Buffer<S, T>
where
  S: BufferBackend<T>,
{
  repr: S::Repr,
  _t: PhantomData<T>,
}

impl<S, T> Buffer<S, T>
where
  S: BufferBackend<T>,
{
  pub fn new<C>(ctx: &mut C, len: usize) -> Result<Self, BufferError>
  where
    C: GraphicsContext<Backend = S>,
  {
    let repr = unsafe { ctx.backend().new_buffer(len)? };

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

  pub fn repeat<C>(ctx: &mut C, len: usize, value: T) -> Result<Self, BufferError>
  where
    C: GraphicsContext<Backend = S>,
    T: Copy,
  {
    let repr = unsafe { ctx.backend().repeat(len, value)? };

    Ok(Buffer {
      repr,
      _t: PhantomData,
    })
  }

  pub fn at(&self, i: usize) -> Option<T>
  where
    T: Copy,
  {
    unsafe { S::at(&self.repr, i) }
  }

  pub fn whole(&self) -> Vec<T>
  where
    T: Copy,
  {
    unsafe { S::whole(&self.repr) }
  }

  pub fn set(&mut self, i: usize, x: T) -> Result<(), BufferError>
  where
    T: Copy,
  {
    unsafe { S::set(&mut self.repr, i, x) }
  }

  pub fn write_whole(&mut self, values: &[T]) -> Result<(), BufferError> {
    unsafe { S::write_whole(&mut self.repr, values) }
  }

  pub fn clear(&mut self, x: T) -> Result<(), BufferError>
  where
    T: Copy,
  {
    unsafe { S::clear(&mut self.repr, x) }
  }
}
