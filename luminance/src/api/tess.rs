//! Tessellation API.

use crate::backend::tess::{Mode, TessBuilder as TessBuilderBackend, TessError, TessIndex};
use crate::context::GraphicsContext;
use crate::vertex::Vertex;

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
