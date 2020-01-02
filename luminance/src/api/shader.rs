//! Shader API.

use crate::backend::shader::Shader;
use crate::backend::shader_stage::{StageError, StageType};
use crate::context::GraphicsContext;

pub struct Stage<S>
where
  S: Shader,
{
  repr: S::StageRepr,
}

impl<S> Stage<S>
where
  S: Shader,
{
  pub fn new<C, R>(ctx: &mut C, ty: StageType, src: R) -> Result<Self, StageError>
  where
    C: GraphicsContext<Backend = S>,
    R: AsRef<str>,
  {
    unsafe {
      ctx
        .backend()
        .new_stage(ty, src.as_ref())
        .map(|repr| Stage { repr })
    }
  }
}

impl<S> Drop for Stage<S>
where
  S: Shader,
{
  fn drop(&mut self) {
    unsafe { S::destroy_stage(&mut self.repr) }
  }
}
