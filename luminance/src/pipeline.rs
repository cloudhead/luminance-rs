use crate::context::GraphicsContext;
use crate::framebuffer::Framebuffer;
use crate::render_state::RenderState;
use crate::shader::program::Program;
use crate::tess::{Tess, TessSlice};
use crate::texture::{Dimensionable, Layerable};

pub trait Builder<'ctx, 'a, C>
where
  C: GraphicsContext,
{
  type ShadingGate: ShadingGate<'a, C>;

  /// Create a new `Builder`.
  ///
  /// Even though you can call this function by yourself, you’re likely to prefer using
  /// `GraphicsContext::pipeline_builder` instead.
  fn new(ctx: &'ctx mut C) -> Self;

  fn pipeline<L, D, CS, DS, Fr, F>(
    &'a mut self,
    framebuffer: &Self::Framebuffer,
    clear_color: [f32; 4],
    f: F,
  ) where
    Self: PipelineFramebuffer<'ctx, 'a, C, L, D, CS, DS>,
    L: Layerable,
    D: Dimensionable,
    F:
      FnOnce(<Self as PipelineFramebuffer<'ctx, 'a, C, L, D, CS, DS>>::Pipeline, Self::ShadingGate),
  {
    <Self as PipelineFramebuffer<'ctx, 'a, C, L, D, CS, DS>>::run_pipeline(
      self,
      framebuffer,
      clear_color,
      f,
    )
  }
}

pub trait PipelineFramebuffer<'ctx, 'a, C, L, D, CS, DS>: Builder<'ctx, 'a, C>
where
  C: GraphicsContext,
  L: Layerable,
  D: Dimensionable,
{
  type Pipeline: Pipeline<'a>;

  type Framebuffer: Framebuffer<C, L, D>;

  fn run_pipeline<F>(&'a mut self, framebuffer: &Self::Framebuffer, clear_color: [f32; 4], f: F)
  where
    F: FnOnce(Self::Pipeline, Self::ShadingGate);
}

pub trait Pipeline<'a> {
  fn bind<T>(&'a self, resource: &'a T) -> Result<Self::Bound, Self::Err>
  where
    Self: Bind<'a, T>,
  {
    <Self as Bind<'a, T>>::bind(self, resource)
  }
}

pub trait Bind<'a, T> {
  type Bound;

  type Err;

  fn bind(&'a self, resource: &'a T) -> Result<Self::Bound, Self::Err>;
}

pub trait ShadingGate<'a, C> {
  type RenderGate: RenderGate<'a, C>;

  fn shade<S, Out, Uni, F>(&'a mut self, program: &'a Self::Program, f: F)
  where
    Self: ShadingGateProgram<'a, C, S, Out, Uni>,
    F: FnOnce(<Self::Program as Program<'a, S, Out, Uni>>::ProgramInterface, Self::RenderGate),
  {
    <Self as ShadingGateProgram<'a, C, S, Out, Uni>>::shade_with_program(self, program, f)
  }
}

pub trait ShadingGateProgram<'a, C, S, Out, Uni>: ShadingGate<'a, C> {
  type Program: Program<'a, S, Out, Uni>;

  fn shade_with_program<F>(&'a mut self, program: &'a Self::Program, f: F)
  where
    F: FnOnce(<Self::Program as Program<'a, S, Out, Uni>>::ProgramInterface, Self::RenderGate);
}

pub trait RenderGate<'a, C> {
  type TessGate: TessGate<'a, C>;

  fn render<F>(&'a mut self, rdr_st: RenderState, f: F)
  where
    F: FnOnce(Self::TessGate);
}

pub trait TessGate<'a, C> {
  type Tess: Tess<C>;

  fn render<T>(&'a mut self, tess_slice: T)
  where
    T: TessSlice<'a, C, Self::Tess>;
}
