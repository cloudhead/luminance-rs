use crate::context::GraphicsContext;
use crate::framebuffer::{ColorSlot, DepthSlot, Framebuffer};
use crate::pixel::Pixel;
use crate::render_state::RenderState;
use crate::shader::program2::Program;
use crate::tess::{Tess, TessSlice};
use crate::texture::{Dimensionable, Layerable, Texture};

pub trait Builder<'a, C>
where
  C: GraphicsContext,
{
  type ShadingGate: ShadingGate<'a, C>;

  /// Create a new `Builder`.
  ///
  /// Even though you can call this function by yourself, you’re likely to prefer using
  /// `GraphicsContext::pipeline_builder` instead.
  fn new(ctx: &'a mut C) -> Self;

  //fn pipeline<'b, L, D, CS, DS, Fr, F>(&'b mut self, framebuffer: &Fr, clear_color: [f32; 4], f: F)
  //where
  //  Fr: Framebuffer<C::State, L, D>,
  //  L: Layerable,
  //  D: Dimensionable,
  //  CS: ColorSlot<C::State, L, D, Fr::Textures>,
  //  DS: DepthSlot<C::State, L, D, Fr::Textures>,
  //  F: FnOnce(Pipeline<'b>, ShadingGate<'b, C>);
}

pub trait PipelineFramebuffer<'a, C, L, D, CS, DS> {
  //type Framebuffer: Framebuffer<
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

  fn shade<S, Out, Uni, F>(&'a mut self, program: &Self::Program, f: F)
  where
    Self: ShadingGateProgram<'a, C, S, Out, Uni>,
    F: FnOnce(<Self::Program as Program<'a, S, Out, Uni>>::ProgramInterface, Self::RenderGate),
  {
    <Self as ShadingGateProgram<'a, C, S, Out, Uni>>::shade_with_program(self, program, f)
  }
}

pub trait ShadingGateProgram<'a, C, S, Out, Uni>: ShadingGate<'a, C> {
  type Program: Program<'a, S, Out, Uni>;

  fn shade_with_program<F>(&'a mut self, program: &Self::Program, f: F)
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
