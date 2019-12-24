//! Tessellation API.

use crate::backend::tess::{
  Mode, Tess as TessBackend, TessBuilder as TessBuilderBackend, TessError, TessIndex, TessMapError,
  TessSlice as TessSliceBackend,
};
use crate::context::GraphicsContext;
use crate::vertex::Vertex;

#[derive(Debug)]
pub struct TessBuilder<S>
where
  S: TessBuilderBackend,
{
  repr: S::TessBuilderRepr,
}

impl<S> TessBuilder<S>
where
  S: TessBuilderBackend,
{
  pub fn new<C>(ctx: &mut C) -> Result<Self, TessError>
  where
    C: GraphicsContext<Backend = S>,
  {
    unsafe {
      ctx
        .backend()
        .new_tess_builder()
        .map(|repr| TessBuilder { repr })
    }
  }

  pub fn add_vertices<V, W>(mut self, vertices: W) -> Result<Self, TessError>
  where
    W: AsRef<[V]>,
    V: Vertex,
  {
    unsafe { S::add_vertices(&mut self.repr, vertices).map(move |_| self) }
  }

  pub fn add_instances<V, W>(mut self, instances: W) -> Result<Self, TessError>
  where
    W: AsRef<[V]>,
    V: Vertex,
  {
    unsafe { S::add_instances(&mut self.repr, instances).map(move |_| self) }
  }

  pub fn set_indices<T, I>(mut self, indices: T) -> Result<Self, TessError>
  where
    T: AsRef<[I]>,
    I: TessIndex,
  {
    unsafe { S::set_indices(&mut self.repr, indices).map(move |_| self) }
  }

  pub fn set_mode(mut self, mode: Mode) -> Result<Self, TessError> {
    unsafe { S::set_mode(&mut self.repr, mode).map(move |_| self) }
  }

  pub fn set_vertex_nb(mut self, nb: usize) -> Result<Self, TessError> {
    unsafe { S::set_vertex_nb(&mut self.repr, nb).map(move |_| self) }
  }

  pub fn set_instance_nb(mut self, nb: usize) -> Result<Self, TessError> {
    unsafe { S::set_instance_nb(&mut self.repr, nb).map(move |_| self) }
  }

  pub fn set_primitive_restart_index<T>(mut self, index: T) -> Result<Self, TessError>
  where
    T: Into<Option<u32>>,
  {
    unsafe { S::set_primitive_restart_index(&mut self.repr, index.into()).map(move |_| self) }
  }
}

impl<S> TessBuilder<S>
where
  S: TessBackend,
{
  pub fn build(self) -> Result<Tess<S>, TessError> {
    unsafe { S::build(self.repr).map(|repr| Tess { repr }) }
  }
}

#[derive(Debug)]
pub struct Tess<S>
where
  S: TessBackend,
{
  repr: S::TessRepr,
}

impl<S> Drop for Tess<S>
where
  S: TessBackend,
{
  fn drop(&mut self) {
    let _ = unsafe { S::destroy_tess(&mut self.repr) };
  }
}

impl<S> Tess<S>
where
  S: TessBackend,
{
  pub fn slice_vertices<T>(&self) -> Result<TessSlice<S, T>, TessMapError>
  where
    S: TessSliceBackend<T>,
    T: Vertex,
  {
    unsafe { S::slice_vertices(&self.repr).map(|repr| TessSlice { repr }) }
  }

  pub fn slice_vertices_mut<T>(&mut self) -> Result<TessSlice<S, T>, TessMapError>
  where
    S: TessSliceBackend<T>,
    T: Vertex,
  {
    unsafe { S::slice_vertices_mut(&mut self.repr).map(|repr| TessSlice { repr }) }
  }

  pub fn slice_indices<T>(&self) -> Result<TessSlice<S, T>, TessMapError>
  where
    S: TessSliceBackend<T>,
    T: TessIndex,
  {
    unsafe { S::slice_indices(&self.repr).map(|repr| TessSlice { repr }) }
  }

  pub fn slice_indices_mut<T>(&mut self) -> Result<TessSlice<S, T>, TessMapError>
  where
    S: TessSliceBackend<T>,
    T: TessIndex,
  {
    unsafe { S::slice_indices_mut(&mut self.repr).map(|repr| TessSlice { repr }) }
  }

  pub fn slice_instances<T>(&self) -> Result<TessSlice<S, T>, TessMapError>
  where
    S: TessSliceBackend<T>,
    T: Vertex,
  {
    unsafe { S::slice_instances(&self.repr).map(|repr| TessSlice { repr }) }
  }

  pub fn slice_instances_mut<T>(&mut self) -> Result<TessSlice<S, T>, TessMapError>
  where
    S: TessSliceBackend<T>,
    T: Vertex,
  {
    unsafe { S::slice_instances_mut(&mut self.repr).map(|repr| TessSlice { repr }) }
  }
}

#[derive(Debug)]
pub struct TessSlice<S, T>
where
  S: TessSliceBackend<T>,
{
  repr: S::SliceRepr,
}

impl<S, T> Drop for TessSlice<S, T>
where
  S: TessSliceBackend<T>,
{
  fn drop(&mut self) {
    let _ = unsafe { S::destroy_tess_slice(&mut self.repr) };
  }
}

impl<S, T> TessSlice<S, T>
where
  S: TessSliceBackend<T>,
{
  pub fn as_slice(&self) -> Result<&[T], TessMapError> {
    unsafe { S::obtain_slice(&self.repr) }
  }
}

#[derive(Debug)]
pub struct TessSliceMut<S, T>
where
  S: TessSliceBackend<T>,
{
  repr: S::SliceRepr,
}

impl<S, T> Drop for TessSliceMut<S, T>
where
  S: TessSliceBackend<T>,
{
  fn drop(&mut self) {
    let _ = unsafe { S::destroy_tess_slice(&mut self.repr) };
  }
}

impl<S, T> TessSliceMut<S, T>
where
  S: TessSliceBackend<T>,
{
  pub fn as_slice(&self) -> Result<&[T], TessMapError> {
    unsafe { S::obtain_slice(&self.repr) }
  }

  pub fn as_slice_mut(&mut self) -> Result<&mut [T], TessMapError> {
    unsafe { S::obtain_slice_mut(&mut self.repr) }
  }
}
