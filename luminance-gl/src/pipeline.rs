use gl;
use gl::types::*;
use luminance::blending::BlendingState;
use luminance::context::GraphicsContext;
use luminance::depth_test::DepthTest;
use luminance::face_culling::FaceCullingState;
use luminance::framebuffer::{ColorSlot, DepthSlot};
use luminance::pipeline2::{
  Bind, Builder as BuilderBackend, Pipeline as PipelineBackend, PipelineFramebuffer,
  RenderGate as RenderGateBackend, ShadingGate as ShadingGateBackend, ShadingGateProgram,
  TessGate as TessGateBackend,
};
use luminance::pixel::{Pixel, SamplerType, Type as PxType};
use luminance::render_state::RenderState;
use luminance::shader::program2::{
  Program as ProgramBackend, Type as UniformType, UniformInterface, Uniformable,
};
use luminance::tess::TessSlice;
use luminance::texture::{Dim, Dimensionable, Layerable};
use luminance::vertex::Semantics;
use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;

use crate::buffer::Buffer;
use crate::framebuffer::{Framebuffer, ReifyState};
use crate::shader::program::{Program, ProgramInterface, Uniform};
use crate::state::GraphicsState;
use crate::tess::Tess;
use crate::texture::Texture;

// A stack of bindings.
//
// This type implements a stacking system for effective resource bindings by allocating new
// bindings points only when no recycled resource is available. It helps have a better memory
// footprint in the resource space.
struct BindingStack {
  state: Rc<RefCell<GraphicsState>>,
  next_texture_unit: u32,
  free_texture_units: Vec<u32>,
  next_buffer_binding: u32,
  free_buffer_bindings: Vec<u32>,
}

impl BindingStack {
  // Create a new, empty binding stack.
  fn new(state: Rc<RefCell<GraphicsState>>) -> Self {
    BindingStack {
      state,
      next_texture_unit: 0,
      free_texture_units: Vec::new(),
      next_buffer_binding: 0,
      free_buffer_bindings: Vec::new(),
    }
  }
}

pub struct Builder<'a, C>
where
  C: ?Sized,
{
  ctx: &'a mut C,
  binding_stack: Rc<RefCell<BindingStack>>,
  _borrow: PhantomData<&'a mut ()>,
}

impl<'a, C> Builder<'a, C>
where
  C: GraphicsContext<State = GraphicsState>,
{
  /// Create a new `Builder`.
  ///
  /// Even though you call this function by yourself, you’re likely to prefer using
  /// `GraphicsContext::pipeline_builder` instead.
  pub fn new(ctx: &'a mut C) -> Self {
    let state = ctx.state().clone();

    Builder {
      ctx,
      binding_stack: Rc::new(RefCell::new(BindingStack::new(state))),
      _borrow: PhantomData,
    }
  }

  pub fn pipeline<'b, L, D, CS, DS, F>(
    &'b mut self,
    framebuffer: &Framebuffer<L, D, CS, DS>,
    clear_color: [f32; 4],
    f: F,
  ) where
    L: Layerable,
    D: Dimensionable,
    CS: ColorSlot<GraphicsState, L, D, ReifyState>,
    DS: DepthSlot<GraphicsState, L, D, ReifyState>,
    F: FnOnce(Pipeline<'b>, ShadingGate<'b, C>),
  {
    unsafe {
      self
        .ctx
        .state()
        .borrow_mut()
        .bind_draw_framebuffer(framebuffer.handle());

      let dim = framebuffer.dim();
      gl::Viewport(0, 0, D::width(dim) as GLint, D::height(dim) as GLint);
      gl::ClearColor(
        clear_color[0],
        clear_color[1],
        clear_color[2],
        clear_color[3],
      );
      gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);
    }

    let binding_stack = &self.binding_stack;
    let p = Pipeline { binding_stack };
    let shd_gt = ShadingGate {
      ctx: self.ctx,
      binding_stack,
    };

    f(p, shd_gt);
  }
}

impl<'ctx, 'a, C> BuilderBackend<'ctx, 'a, C> for Builder<'ctx, C>
where
  C: 'a + GraphicsContext<State = GraphicsState>,
{
  type ShadingGate = ShadingGate<'a, C>;

  fn new(ctx: &'ctx mut C) -> Self {
    Builder::new(ctx)
  }
}

impl<'ctx, 'a, C, L, D, CS, DS> PipelineFramebuffer<'ctx, 'a, C, L, D, CS, DS> for Builder<'ctx, C>
where
  C: 'a + GraphicsContext<State = GraphicsState>,
  L: Layerable,
  D: Dimensionable,
  CS: ColorSlot<GraphicsState, L, D, ReifyState>,
  DS: DepthSlot<GraphicsState, L, D, ReifyState>,
{
  type Pipeline = Pipeline<'a>;

  type Framebuffer = Framebuffer<L, D, CS, DS>;

  fn run_pipeline<F>(&'a mut self, framebuffer: &Self::Framebuffer, clear_color: [f32; 4], f: F)
  where
    F: FnOnce(Self::Pipeline, Self::ShadingGate),
  {
    self.pipeline(framebuffer, clear_color, f)
  }
}

/// A dynamic pipeline.
///
/// Such a pipeline enables you to call shading commands, bind textures, bind uniform buffers, etc.
/// in a scoped-binding way.
pub struct Pipeline<'a> {
  binding_stack: &'a Rc<RefCell<BindingStack>>,
}

impl<'a> PipelineBackend<'a> for Pipeline<'a> {}

impl<'a, L, D, P> Bind<'a, Texture<L, D, P>> for Pipeline<'a>
where
  L: 'a + Layerable,
  D: 'a + Dimensionable,
  P: 'a + Pixel,
{
  type Bound = BoundTexture<'a, L, D, P::SamplerType>;

  type Err = ();

  fn bind(&'a self, texture: &'a Texture<L, D, P>) -> Result<Self::Bound, Self::Err> {
    let mut bstack = self.binding_stack.borrow_mut();

    let unit = bstack.free_texture_units.pop().unwrap_or_else(|| {
      // no more free units; reserve one
      let unit = bstack.next_texture_unit;
      bstack.next_texture_unit += 1;
      unit
    });

    unsafe {
      let mut state = bstack.state.borrow_mut();
      state.set_texture_unit(unit);
      state.bind_texture(texture.target(), texture.handle());
    }

    Ok(BoundTexture::new(self.binding_stack, unit))
  }
}

impl<'a, T> Bind<'a, Buffer<T>> for Pipeline<'a>
where
  T: 'a,
{
  type Bound = BoundBuffer<'a, T>;

  type Err = ();

  fn bind(&'a self, buffer: &'a Buffer<T>) -> Result<Self::Bound, Self::Err> {
    let mut bstack = self.binding_stack.borrow_mut();

    let binding = bstack.free_buffer_bindings.pop().unwrap_or_else(|| {
      // no more free bindings; reserve one
      let binding = bstack.next_buffer_binding;
      bstack.next_buffer_binding += 1;
      binding
    });

    unsafe {
      bstack
        .state
        .borrow_mut()
        .bind_buffer_base(buffer.handle(), binding);
    }

    Ok(BoundBuffer::new(self.binding_stack, binding))
  }
}

/// An opaque type representing a bound texture in a `Builder`. You may want to pass such an object
/// to a shader’s uniform’s update.
pub struct BoundTexture<'a, L, D, S>
where
  L: 'a + Layerable,
  D: 'a + Dimensionable,
  S: 'a + SamplerType,
{
  unit: u32,
  binding_stack: &'a Rc<RefCell<BindingStack>>,
  _phantom: PhantomData<&'a (L, D, S)>,
}

impl<'a, L, D, S> BoundTexture<'a, L, D, S>
where
  L: 'a + Layerable,
  D: 'a + Dimensionable,
  S: 'a + SamplerType,
{
  fn new(binding_stack: &'a Rc<RefCell<BindingStack>>, unit: u32) -> Self {
    BoundTexture {
      unit,
      binding_stack,
      _phantom: PhantomData,
    }
  }
}

impl<'a, L, D, S> Drop for BoundTexture<'a, L, D, S>
where
  L: 'a + Layerable,
  D: 'a + Dimensionable,
  S: 'a + SamplerType,
{
  fn drop(&mut self) {
    let mut bstack = self.binding_stack.borrow_mut();
    // place the unit into the free list
    bstack.free_texture_units.push(self.unit);
  }
}

unsafe impl<'a, 'b, L, D, S> Uniformable<&'b BoundTexture<'a, L, D, S>>
  for Uniform<&'b BoundTexture<'a, L, D, S>>
where
  L: 'a + Layerable,
  D: 'a + Dimensionable,
  S: 'a + SamplerType,
{
  fn ty() -> UniformType {
    match (S::sample_type(), D::dim()) {
      (PxType::NormIntegral, Dim::Dim1) => UniformType::Sampler1D,
      (PxType::NormUnsigned, Dim::Dim1) => UniformType::Sampler1D,
      (PxType::Integral, Dim::Dim1) => UniformType::ISampler1D,
      (PxType::Unsigned, Dim::Dim1) => UniformType::UISampler1D,
      (PxType::Floating, Dim::Dim1) => UniformType::Sampler1D,

      (PxType::NormIntegral, Dim::Dim2) => UniformType::Sampler2D,
      (PxType::NormUnsigned, Dim::Dim2) => UniformType::Sampler2D,
      (PxType::Integral, Dim::Dim2) => UniformType::ISampler2D,
      (PxType::Unsigned, Dim::Dim2) => UniformType::UISampler2D,
      (PxType::Floating, Dim::Dim2) => UniformType::Sampler2D,

      (PxType::NormIntegral, Dim::Dim3) => UniformType::Sampler3D,
      (PxType::NormUnsigned, Dim::Dim3) => UniformType::Sampler3D,
      (PxType::Integral, Dim::Dim3) => UniformType::ISampler3D,
      (PxType::Unsigned, Dim::Dim3) => UniformType::UISampler3D,
      (PxType::Floating, Dim::Dim3) => UniformType::Sampler3D,

      (PxType::NormIntegral, Dim::Cubemap) => UniformType::Cubemap,
      (PxType::NormUnsigned, Dim::Cubemap) => UniformType::Cubemap,
      (PxType::Integral, Dim::Cubemap) => UniformType::ICubemap,
      (PxType::Unsigned, Dim::Cubemap) => UniformType::UICubemap,
      (PxType::Floating, Dim::Cubemap) => UniformType::Cubemap,
    }
  }

  fn update(self, texture: &BoundTexture<'a, L, D, S>) {
    unsafe { gl::Uniform1i(self.index(), texture.unit as GLint) }
  }
}

/// An opaque type representing a bound buffer in a `Builder`. You may want to pass such an object
/// to a shader’s uniform’s update.
pub struct BoundBuffer<'a, T> {
  binding: u32,
  binding_stack: &'a Rc<RefCell<BindingStack>>,
  _t: PhantomData<&'a Buffer<T>>,
}

impl<'a, T> BoundBuffer<'a, T> {
  fn new(binding_stack: &'a Rc<RefCell<BindingStack>>, binding: u32) -> Self {
    BoundBuffer {
      binding,
      binding_stack,
      _t: PhantomData,
    }
  }
}

impl<'a, T> Drop for BoundBuffer<'a, T> {
  fn drop(&mut self) {
    let mut bstack = self.binding_stack.borrow_mut();
    // place the binding into the free list
    bstack.free_buffer_bindings.push(self.binding);
  }
}

unsafe impl<'a, 'b, T> Uniformable<&'b BoundBuffer<'a, T>> for Uniform<&'b BoundBuffer<'a, T>> {
  fn ty() -> UniformType {
    UniformType::BufferBinding
  }

  fn update(self, buffer: &BoundBuffer<'a, T>) {
    unsafe {
      gl::UniformBlockBinding(
        self.program(),
        self.index() as GLuint,
        buffer.binding as GLuint,
      )
    }
  }
}

/// A shading gate provides you with a way to run shaders on rendering commands.
pub struct ShadingGate<'a, C>
where
  C: ?Sized,
{
  ctx: &'a mut C,
  binding_stack: &'a Rc<RefCell<BindingStack>>,
}

impl<'a, C> ShadingGate<'a, C>
where
  C: ?Sized + GraphicsContext<State = GraphicsState>,
{
  /// Run a shader on a set of rendering commands.
  pub fn shade<In, Out, Uni, F>(&'a mut self, program: &'a Program<In, Out, Uni>, f: F)
  where
    In: Semantics,
    Uni: 'a + UniformInterface,
    F: FnOnce(ProgramInterface<'a, Uni>, RenderGate<'a, C>),
  {
    unsafe {
      let bstack = self.binding_stack.borrow_mut();
      bstack.state.borrow_mut().use_program(program.handle());
    };

    let render_gate = RenderGate {
      ctx: self.ctx,
      binding_stack: self.binding_stack,
    };

    let program_interface = program.interface();
    f(program_interface, render_gate);
  }
}

impl<'a, C> ShadingGateBackend<'a, C> for ShadingGate<'a, C>
where
  C: GraphicsContext<State = GraphicsState>,
{
  type RenderGate = RenderGate<'a, C>;
}

impl<'a, C, S, Out, Uni> ShadingGateProgram<'a, C, S, Out, Uni> for ShadingGate<'a, C>
where
  C: GraphicsContext<State = GraphicsState>,
  S: Semantics,
  Uni: 'a + UniformInterface,
{
  type Program = Program<S, Out, Uni>;

  fn shade_with_program<F>(&'a mut self, program: &'a Self::Program, f: F)
  where
    F: FnOnce(
      <Self::Program as ProgramBackend<'a, S, Out, Uni>>::ProgramInterface,
      Self::RenderGate,
    ),
  {
    ShadingGate::shade(self, program, f)
  }
}

pub struct RenderGate<'a, C>
where
  C: ?Sized,
{
  ctx: &'a mut C,
  binding_stack: &'a Rc<RefCell<BindingStack>>,
}

impl<'a, C> RenderGate<'a, C>
where
  C: GraphicsContext<State = GraphicsState>,
{
  /// Alter the render state and draw tessellations.
  pub fn render<'b, F>(&'b mut self, rdr_st: RenderState, f: F)
  where
    F: FnOnce(TessGate<'b, C>),
  {
    unsafe {
      let bstack = self.binding_stack.borrow_mut();
      let mut gfx_state = bstack.state.borrow_mut();

      match rdr_st.blending() {
        Some((equation, src_factor, dst_factor)) => {
          gfx_state.set_blending_state(BlendingState::On);
          gfx_state.set_blending_equation(equation);
          gfx_state.set_blending_func(src_factor, dst_factor);
        }
        None => {
          gfx_state.set_blending_state(BlendingState::Off);
        }
      }

      if let Some(depth_comparison) = rdr_st.depth_test() {
        gfx_state.set_depth_test(DepthTest::On);
        gfx_state.set_depth_test_comparison(depth_comparison);
      } else {
        gfx_state.set_depth_test(DepthTest::Off);
      }

      match rdr_st.face_culling() {
        Some(face_culling) => {
          gfx_state.set_face_culling_state(FaceCullingState::On);
          gfx_state.set_face_culling_order(face_culling.order());
          gfx_state.set_face_culling_mode(face_culling.mode());
        }
        None => {
          gfx_state.set_face_culling_state(FaceCullingState::Off);
        }
      }
    }

    let tess_gate = TessGate { ctx: self.ctx };

    f(tess_gate);
  }
}

impl<'a, C> RenderGateBackend<'a, C> for RenderGate<'a, C>
where
  C: GraphicsContext<State = GraphicsState>,
{
  type TessGate = TessGate<'a, C>;

  fn render<F>(&'a mut self, rdr_st: RenderState, f: F)
  where
    F: FnOnce(Self::TessGate),
  {
    RenderGate::render(self, rdr_st, f)
  }
}

/// Render tessellations.
pub struct TessGate<'a, C> {
  ctx: &'a mut C,
}

impl<'a, C> TessGateBackend<'a, C> for TessGate<'a, C>
where
  C: GraphicsContext<State = GraphicsState>,
{
  type Tess = Tess;

  fn render<T>(&'a mut self, tess_slice: T)
  where
    T: TessSlice<'a, C, Self::Tess>,
  {
    tess_slice.render(self.ctx);
  }
}
