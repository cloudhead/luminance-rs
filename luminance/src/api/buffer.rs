//! Buffer API.

use std::marker::PhantomData;

use crate::backend::buffer::{
  Buffer as BufferBackend, BufferError, BufferSlice as BufferSliceBackend,
};
use crate::context::GraphicsContext;

#[derive(Debug)]
pub struct Buffer<S, T>
where
  S: BufferBackend<T>,
{
  repr: S::BufferRepr,
  _t: PhantomData<T>,
}

impl<S, T> Drop for Buffer<S, T>
where
  S: BufferBackend<T>,
{
  fn drop(&mut self) {
    let _ = unsafe { <S as BufferBackend<T>>::destroy_buffer(&mut self.repr) };
  }
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

  #[inline(always)]
  pub fn len(&self) -> usize {
    unsafe { S::len(&self.repr) }
  }

  #[inline(always)]
  pub fn is_empty(&self) -> bool {
    self.len() == 0
  }
}

impl<S, T> Buffer<S, T>
where
  S: BufferSliceBackend<T>,
{
  pub fn slice(&mut self) -> Result<BufferSlice<S, T>, BufferError> {
    unsafe {
      S::slice_buffer(&mut self.repr).map(|slice| BufferSlice {
        slice,
        _a: PhantomData,
      })
    }
  }
}

#[derive(Debug)]
pub struct BufferSlice<'a, S, T>
where
  S: BufferSliceBackend<T>,
{
  slice: S::SliceRepr,
  _a: PhantomData<&'a mut ()>,
}

impl<'a, S, T> Drop for BufferSlice<'a, S, T>
where
  S: BufferSliceBackend<T>,
{
  fn drop(&mut self) {
    let _ = unsafe { S::destroy_buffer_slice(&mut self.slice) };
  }
}

impl<'a, S, T> BufferSlice<'a, S, T>
where
  S: BufferSliceBackend<T>,
{
  pub fn as_slice(&self) -> Result<&[T], BufferError> {
    unsafe { S::obtain_slice(&self.slice) }
  }

  pub fn as_slice_mut(&mut self) -> Result<&mut [T], BufferError> {
    unsafe { S::obtain_slice_mut(&mut self.slice) }
  }
}
