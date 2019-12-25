//! Texture API.

use crate::backend::texture::{
  Dimensionable, GenMipmaps, Layerable, Sampler, Texture as TextureBackend, TextureError,
};
use crate::context::GraphicsContext;
use crate::pixel::Pixel;

pub struct Texture<S, L, D, P>
where
  S: TextureBackend<L, D, P>,
  L: Layerable,
  D: Dimensionable,
  P: Pixel,
{
  repr: S::TextureRepr,
}

impl<S, L, D, P> Drop for Texture<S, L, D, P>
where
  S: TextureBackend<L, D, P>,
  L: Layerable,
  D: Dimensionable,
  P: Pixel,
{
  fn drop(&mut self) {
    unsafe { S::destroy_texture(&mut self.repr) }
  }
}

impl<S, L, D, P> Texture<S, L, D, P>
where
  S: TextureBackend<L, D, P>,
  L: Layerable,
  D: Dimensionable,
  P: Pixel,
{
  pub fn new<C>(
    ctx: &mut C,
    size: D::Size,
    mipmaps: usize,
    sampler: Sampler,
  ) -> Result<Self, TextureError>
  where
    C: GraphicsContext<Backend = S>,
  {
    unsafe {
      ctx
        .backend()
        .new_texture(size, mipmaps, sampler)
        .map(|repr| Texture { repr })
    }
  }

  pub fn mipmaps(&self) -> usize {
    unsafe { S::mipmaps(&self.repr) }
  }

  pub fn size(&self) -> D::Size {
    unsafe { S::size(&self.repr) }
  }

  pub fn clear_part(
    &mut self,
    gen_mipmaps: GenMipmaps,
    offset: D::Offset,
    size: D::Size,
    pixel: P::Encoding,
  ) -> Result<(), TextureError> {
    unsafe { S::clear_part(&mut self.repr, gen_mipmaps, offset, size, pixel) }
  }

  pub fn clear(&mut self, gen_mipmaps: GenMipmaps, pixel: P::Encoding) -> Result<(), TextureError> {
    unsafe { S::clear(&mut self.repr, gen_mipmaps, pixel) }
  }

  pub fn upload_part(
    &mut self,
    gen_mipmaps: GenMipmaps,
    offset: D::Offset,
    size: D::Size,
    texels: &[P::Encoding],
  ) -> Result<(), TextureError> {
    unsafe { S::upload_part(&mut self.repr, gen_mipmaps, offset, size, texels) }
  }

  pub fn upload(
    &mut self,
    gen_mipmaps: GenMipmaps,
    texels: &[P::Encoding],
  ) -> Result<(), TextureError> {
    unsafe { S::upload(&mut self.repr, gen_mipmaps, texels) }
  }

  pub fn upload_part_raw(
    &mut self,
    gen_mipmaps: GenMipmaps,
    offset: D::Offset,
    size: D::Size,
    texels: &[P::RawEncoding],
  ) -> Result<(), TextureError> {
    unsafe { S::upload_part_raw(&mut self.repr, gen_mipmaps, offset, size, texels) }
  }

  pub fn upload_raw(
    &mut self,
    gen_mipmaps: GenMipmaps,
    texels: &[P::RawEncoding],
  ) -> Result<(), TextureError> {
    unsafe { S::upload_raw(&mut self.repr, gen_mipmaps, texels) }
  }

  pub fn get_raw_texels(&mut self) -> Result<Vec<P::RawEncoding>, TextureError>
  where
    P::RawEncoding: Copy + Default,
  {
    unsafe { S::get_raw_texels(&mut self.repr) }
  }
}
