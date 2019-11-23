use crate::context::GraphicsContext;
use crate::framebuffer::{ColorSlot, DepthSlot, Framebuffer};
use crate::pixel::Pixel;
use crate::texture::{Dimensionable, Layerable, Texture};

pub trait Builder<'a, C>
where
  C: GraphicsContext,
{
  /// Create a new `Builder`.
  ///
  /// Even though you can call this function by yourself, you’re likely to prefer using
  /// `GraphicsContext::pipeline_builder` instead.
  fn new(ctx: &'a mut C) -> Self;

  // /// Create a new [`Pipeline`] and consume it immediately.
  // ///
  // /// A dynamic rendering pipeline is responsible of rendering into a [`Framebuffer`].
  // ///
  // /// `L` refers to the [`Layering`] of the underlying [`Framebuffer`].
  // ///
  // /// `D` refers to the `Dim` of the underlying `Framebuffer`.
  // ///
  // /// `CS` and `DS` are – respectively – the *color* and *depth* `Slot`(s) of the underlying
  // /// [`Framebuffer`].
  // ///
  // /// Pipelines also have a *clear color*, used to clear the framebuffer.
  // fn pipeline<'b, L, D, CS, DS, Fr, F>(&'b mut self, framebuffer: &Fr, clear_color: [f32; 4], f: F)
  // where
  //   Fr: Framebuffer<C::State, L, D>,
  //   L: Layerable,
  //   D: Dimensionable,
  //   CS: ColorSlot<C::State, L, D, Fr::Textures>,
  //   DS: DepthSlot<C::State, L, D, Fr::Textures>,
  //   F: FnOnce(Pipeline<'b>, ShadingGate<'b, C>);
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

//pub trait BindTexture<C, L, D, P>
//where
//  D: Dimensionable,
//  L: Layerable,
//  P: Pixel,
//{
//  type Texture: Texture<C, L, D, P>;
//
//  type BoundTexture;
//
//  type Err;
//
//  fn bind_texture<'a>(
//    &'a self,
//    texture: &'a Self::Texture,
//  ) -> Result<Self::BoundTexture, Self::Err>;
//}
