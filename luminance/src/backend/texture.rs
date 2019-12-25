//! Texture backend.

use std::fmt;

use crate::depth_test::DepthComparison;
use crate::pixel::{Pixel, PixelFormat};

/// How to wrap texture coordinates while sampling textures?
#[derive(Clone, Copy, Debug)]
pub enum Wrap {
  /// If textures coordinates lay outside of *[0;1]*, they will be clamped to either *0* or *1* for
  /// every components.
  ClampToEdge,
  /// Textures coordinates are repeated if they lay outside of *[0;1]*. Picture this as:
  ///
  /// ```ignore
  /// // given the frac function returning the fractional part of a floating number:
  /// coord_ith = frac(coord_ith); // always between [0;1]
  /// ```
  Repeat,
  /// Same as `Repeat` but it will alternatively repeat between *[0;1]* and *[1;0]*.
  MirroredRepeat,
}

/// Minification filter.
#[derive(Clone, Copy, Debug)]
pub enum MinFilter {
  /// Nearest interpolation.
  Nearest,
  /// Linear interpolation between surrounding pixels.
  Linear,
  /// This filter will select the nearest mipmap between two samples and will perform a nearest
  /// interpolation afterwards.
  NearestMipmapNearest,
  /// This filter will select the nearest mipmap between two samples and will perform a linear
  /// interpolation afterwards.
  NearestMipmapLinear,
  /// This filter will linearly interpolate between two mipmaps, which selected texels would have
  /// been interpolated with a nearest filter.
  LinearMipmapNearest,
  /// This filter will linearly interpolate between two mipmaps, which selected texels would have
  /// been linarily interpolated as well.
  LinearMipmapLinear,
}

/// Magnification filter.
#[derive(Clone, Copy, Debug)]
pub enum MagFilter {
  /// Nearest interpolation.
  Nearest,
  /// Linear interpolation between surrounding pixels.
  Linear,
}

/// Reify a type into a `Dim`.
pub trait Dimensionable {
  /// Size type of a dimension (used to caracterize dimensions’ areas).
  type Size: Copy;

  /// Offset type of a dimension (used to caracterize addition and subtraction of sizes, mostly).
  type Offset: Copy;

  /// Zero offset.
  const ZERO_OFFSET: Self::Offset;

  /// Dimension.
  fn dim() -> Dim;

  /// Width of the associated `Size`.
  fn width(size: Self::Size) -> u32;

  /// Height of the associated `Size`. If it doesn’t have one, set it to 1.
  fn height(_: Self::Size) -> u32 {
    1
  }

  /// Depth of the associated `Size`. If it doesn’t have one, set it to 1.
  fn depth(_: Self::Size) -> u32 {
    1
  }

  /// X offset.
  fn x_offset(offset: Self::Offset) -> u32;

  /// Y offset. If it doesn’t have one, set it to 0.
  fn y_offset(_: Self::Offset) -> u32 {
    1
  }

  /// Z offset. If it doesn’t have one, set it to 0.
  fn z_offset(_: Self::Offset) -> u32 {
    1
  }

  /// Amount of pixels this size represents.
  ///
  /// For 2D sizes, it represents the area; for 3D sizes, the volume; etc.
  /// For cubemaps, it represents the side length of the cube.
  fn count(size: Self::Size) -> usize;
}

// Capacity of the dimension, which is the product of the width, height and depth.
fn dim_capacity<D>(size: D::Size) -> u32
where
  D: Dimensionable,
{
  D::width(size) * D::height(size) * D::depth(size)
}

/// Dimension of a texture.
#[derive(Clone, Copy, Debug)]
pub enum Dim {
  /// 1D.
  Dim1,
  /// 2D.
  Dim2,
  /// 3D.
  Dim3,
  /// Cubemap (i.e. a cube defining 6 faces — akin to 4D).
  Cubemap,
}

/// 1D dimension.
#[derive(Clone, Copy, Debug)]
pub struct Dim1;

impl Dimensionable for Dim1 {
  type Offset = u32;
  type Size = u32;

  const ZERO_OFFSET: Self::Offset = 0;

  fn dim() -> Dim {
    Dim::Dim1
  }

  fn width(w: Self::Size) -> u32 {
    w
  }

  fn x_offset(off: Self::Offset) -> u32 {
    off
  }

  fn count(size: Self::Size) -> usize {
    size as usize
  }
}

/// 2D dimension.
#[derive(Clone, Copy, Debug)]
pub struct Dim2;

impl Dimensionable for Dim2 {
  type Offset = [u32; 2];
  type Size = [u32; 2];

  const ZERO_OFFSET: Self::Offset = [0, 0];

  fn dim() -> Dim {
    Dim::Dim2
  }

  fn width(size: Self::Size) -> u32 {
    size[0]
  }

  fn height(size: Self::Size) -> u32 {
    size[1]
  }

  fn x_offset(off: Self::Offset) -> u32 {
    off[0]
  }

  fn y_offset(off: Self::Offset) -> u32 {
    off[1]
  }

  fn count([width, height]: Self::Size) -> usize {
    width as usize * height as usize
  }
}

/// 3D dimension.
#[derive(Clone, Copy, Debug)]
pub struct Dim3;

impl Dimensionable for Dim3 {
  type Offset = [u32; 3];
  type Size = [u32; 3];

  const ZERO_OFFSET: Self::Offset = [0, 0, 0];

  fn dim() -> Dim {
    Dim::Dim3
  }

  fn width(size: Self::Size) -> u32 {
    size[0]
  }

  fn height(size: Self::Size) -> u32 {
    size[1]
  }

  fn depth(size: Self::Size) -> u32 {
    size[2]
  }

  fn x_offset(off: Self::Offset) -> u32 {
    off[0]
  }

  fn y_offset(off: Self::Offset) -> u32 {
    off[1]
  }

  fn z_offset(off: Self::Offset) -> u32 {
    off[2]
  }

  fn count([width, height, depth]: Self::Size) -> usize {
    width as usize * height as usize * depth as usize
  }
}

/// Cubemap dimension.
#[derive(Clone, Copy, Debug)]
pub struct Cubemap;

impl Dimensionable for Cubemap {
  type Offset = ([u32; 2], CubeFace);
  type Size = u32;

  const ZERO_OFFSET: Self::Offset = ([0, 0], CubeFace::PositiveX);

  fn dim() -> Dim {
    Dim::Cubemap
  }

  fn width(s: Self::Size) -> u32 {
    s
  }

  fn height(s: Self::Size) -> u32 {
    s
  }

  fn depth(_: Self::Size) -> u32 {
    6
  }

  fn x_offset(off: Self::Offset) -> u32 {
    off.0[0]
  }

  fn y_offset(off: Self::Offset) -> u32 {
    off.0[1]
  }

  fn z_offset(off: Self::Offset) -> u32 {
    match off.1 {
      CubeFace::PositiveX => 0,
      CubeFace::NegativeX => 1,
      CubeFace::PositiveY => 2,
      CubeFace::NegativeY => 3,
      CubeFace::PositiveZ => 4,
      CubeFace::NegativeZ => 5,
    }
  }

  fn count(size: Self::Size) -> usize {
    let size = size as usize;
    size * size
  }
}

/// Faces of a cubemap.
#[derive(Clone, Copy, Debug)]
pub enum CubeFace {
  /// The +X face of the cube.
  PositiveX,
  /// The -X face of the cube.
  NegativeX,
  /// The +Y face of the cube.
  PositiveY,
  /// The -Y face of the cube.
  NegativeY,
  /// The +Z face of the cube.
  PositiveZ,
  /// The -Z face of the cube.
  NegativeZ,
}

/// Trait used to reify a type into a `Layering`.
pub trait Layerable {
  /// Reify to `Layering`.
  fn layering() -> Layering;
}

/// Texture layering. If a texture is layered, it has an extra coordinate to access the layer.
#[derive(Clone, Copy, Debug)]
pub enum Layering {
  /// Non-layered.
  Flat,
  /// Layered.
  Layered,
}

/// Flat texture hint.
///
/// A flat texture means it doesn’t have the concept of layers.
#[derive(Clone, Copy, Debug)]
pub struct Flat;

impl Layerable for Flat {
  fn layering() -> Layering {
    Layering::Flat
  }
}

/// Layered texture hint.
///
/// A layered texture has an extra coordinate to access the layer and can be thought of as an array
/// of textures.
#[derive(Clone, Copy, Debug)]
pub struct Layered;

impl Layerable for Layered {
  fn layering() -> Layering {
    Layering::Layered
  }
}

/// A `Sampler` object gives hint on how a `Texture` should be sampled.
#[derive(Clone, Copy, Debug)]
pub struct Sampler {
  /// How should we wrap around the *r* sampling coordinate?
  pub wrap_r: Wrap,
  /// How should we wrap around the *s* sampling coordinate?
  pub wrap_s: Wrap,
  /// How should we wrap around the *t* sampling coordinate?
  pub wrap_t: Wrap,
  /// Minification filter.
  pub min_filter: MinFilter,
  /// Magnification filter.
  pub mag_filter: MagFilter,
  /// For depth textures, should we perform depth comparison and if so, how?
  pub depth_comparison: Option<DepthComparison>,
}

/// Default value is as following:
impl Default for Sampler {
  fn default() -> Self {
    Sampler {
      wrap_r: Wrap::ClampToEdge,
      wrap_s: Wrap::ClampToEdge,
      wrap_t: Wrap::ClampToEdge,
      min_filter: MinFilter::NearestMipmapLinear,
      mag_filter: MagFilter::Linear,
      depth_comparison: None,
    }
  }
}

/// Whether mipmaps should be generated.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GenMipmaps {
  /// Mipmaps should be generated.
  ///
  /// Mipmaps are generated when creating textures but also when uploading texels, clearing, etc.
  Yes,
  /// Never generate mipmaps.
  No,
}

/// Errors that might happen when working with textures.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TextureError {
  /// A texture’s storage failed to be created.
  ///
  /// The carried [`String`] gives the reason of the failure.
  TextureStorageCreationFailed(String),
  /// Not enough pixel data provided for the given area asked.
  ///
  /// The first [`usize`] is the number of expected bytes to be uploaded and the second [`usize`] is
  /// the number you provided. You must provide at least as many pixels as expected by the area in
  /// the texture you’re uploading to.
  NotEnoughPixels(usize, usize),
  /// Unsupported pixel format.
  ///
  /// Sometimes, some hardware might not support a given pixel format (or the format exists on
  /// the interface side but doesn’t in the implementation). That error represents such a case.
  UnsupportedPixelFormat(PixelFormat),
}

impl fmt::Display for TextureError {
  fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    match *self {
      TextureError::TextureStorageCreationFailed(ref e) => {
        write!(f, "texture storage creation failed: {}", e)
      }

      TextureError::NotEnoughPixels(expected, provided) => write!(
        f,
        "not enough texels provided: expected {} bytes, provided {} bytes",
        expected, provided
      ),

      TextureError::UnsupportedPixelFormat(fmt) => write!(f, "unsupported pixel format: {:?}", fmt),
    }
  }
}

pub unsafe trait Texture<L, D, P>
where
  D: Dimensionable,
  L: Layerable,
  P: Pixel,
{
  type TextureRepr;

  unsafe fn new_texture(
    &mut self,
    size: D::Size,
    mipmaps: usize,
    sampler: Sampler,
  ) -> Result<Self::TextureRepr, TextureError>;

  unsafe fn destroy_texture(texture: &mut Self::TextureRepr);

  unsafe fn mipmaps(texture: &Self::TextureRepr) -> usize;

  unsafe fn size(texture: &Self::TextureRepr) -> D::Size;

  unsafe fn clear_part(
    texture: &mut Self::TextureRepr,
    gen_mipmaps: GenMipmaps,
    offset: D::Offset,
    size: D::Size,
    pixel: P::Encoding,
  ) -> Result<(), TextureError>;

  unsafe fn clear(
    texture: &mut Self::TextureRepr,
    gen_mipmaps: GenMipmaps,
    pixel: P::Encoding,
  ) -> Result<(), TextureError>;

  unsafe fn upload_part(
    texture: &mut Self::TextureRepr,
    gen_mipmaps: GenMipmaps,
    offset: D::Offset,
    size: D::Size,
    texels: &[P::Encoding],
  ) -> Result<(), TextureError>;

  unsafe fn upload(
    texture: &mut Self::TextureRepr,
    gen_mipmaps: GenMipmaps,
    texels: &[P::Encoding],
  ) -> Result<(), TextureError>;

  unsafe fn upload_part_raw(
    texture: &mut Self::TextureRepr,
    gen_mipmaps: GenMipmaps,
    offset: D::Offset,
    size: D::Size,
    texels: &[P::RawEncoding],
  ) -> Result<(), TextureError>;

  unsafe fn upload_raw(
    texture: &mut Self::TextureRepr,
    gen_mipmaps: GenMipmaps,
    texels: &[P::RawEncoding],
  ) -> Result<(), TextureError>;

  unsafe fn get_raw_texels(
    texture: &Self::TextureRepr,
  ) -> Result<Vec<P::RawEncoding>, TextureError>
  where
    P::RawEncoding: Copy + Default;
}
