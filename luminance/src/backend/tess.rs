//! Tessellation backend.

use std::fmt;

use crate::backend::buffer::BufferError;
use crate::vertex::{Vertex, VertexDesc};

/// Vertices can be connected via several modes.
///
/// Some modes allow for _primitive restart_. Primitive restart is a cool feature that allows to
/// _break_ the building of a primitive to _start over again_. For instance, when making a curve,
/// you can imagine gluing segments next to each other. If at some point, you want to start a new
/// line, you have two choices:
///
///   - Either you stop your draw call and make another one.
///   - Or you just use the _primitive restart_ feature to ask to create another line from scratch.
///
/// That feature is encoded with a special _vertex index_. You can setup the value of the _primitive
/// restart index_ with [`TessBuilder::set_primitive_restart_index`]. Whenever a vertex index is set
/// to the same value as the _primitive restart index_, the value is not interpreted as a vertex
/// index but just a marker / hint to start a new primitive.
#[derive(Copy, Clone, Debug)]
pub enum Mode {
  /// A single point.
  ///
  /// Points are left unconnected from each other and represent a _point cloud_. This is the typical
  /// primitive mode you want to do, for instance, particles rendering.
  Point,
  /// A line, defined by two points.
  ///
  /// Every pair of vertices are connected together to form a straight line.
  Line,
  /// A strip line, defined by at least two points and zero or many other ones.
  ///
  /// The first two vertices create a line, and every new vertex flowing in the graphics pipeline
  /// (starting from the third, then) well extend the initial line, making a curve composed of
  /// several segments.
  ///
  /// > This kind of primitive mode allows the usage of _primitive restart_.
  LineStrip,
  /// A triangle, defined by three points.
  Triangle,
  /// A triangle fan, defined by at least three points and zero or many other ones.
  ///
  /// Such a mode is easy to picture: a cooling fan is a circular shape, with blades.
  /// [`Mode::TriangleFan`] is kind of the same. The first vertex is at the center of the fan, then
  /// the second vertex creates the first edge of the first triangle. Every time you add a new
  /// vertex, a triangle is created by taking the first (center) vertex, the very previous vertex
  /// and the current vertex. By specifying vertices around the center, you actually create a
  /// fan-like shape.
  ///
  /// > This kind of primitive mode allows the usage of _primitive restart_.
  TriangleFan,
  /// A triangle strip, defined by at least three points and zero or many other ones.
  ///
  /// This mode is a bit different from [`Mode::TriangleFan`]. The first two vertices define the
  /// first edge of the first triangle. Then, for each new vertex, a new triangle is created by
  /// taking the very previous vertex and the last to very previous vertex. What it means is that
  /// every time a triangle is created, the next vertex will share the edge that was created to
  /// spawn the previous triangle.
  ///
  /// This mode is useful to create long ribbons / strips of triangles.
  ///
  /// > This kind of primitive mode allows the usage of _primitive restart_.
  TriangleStrip,
  /// A general purpose primitive with _n_ vertices, for use in tessellation shaders.
  /// For example, `Mode::Patch(3)` represents triangle patches, so every three vertices in the
  /// buffer form a patch.
  /// If you want to employ tessellation shaders, this is the only primitive mode you can use.
  Patch(usize),
}

/// Error that can occur while trying to map GPU tessellation to host code.
#[derive(Debug, Eq, PartialEq)]
pub enum TessMapError {
  /// The CPU mapping failed due to buffer errors.
  VertexBufferMapFailed(BufferError),
  /// The CPU mapping failed due to buffer errors.
  IndexBufferMapFailed(BufferError),
  /// Vertex target type is not the same as the one stored in the buffer.
  VertexTypeMismatch(VertexDesc, VertexDesc),
  /// Index target type is not the same as the one stored in the buffer.
  IndexTypeMismatch(TessIndexType, TessIndexType),
  /// The CPU mapping failed because you cannot map an attributeless tessellation since it doesn’t
  /// have any vertex attribute.
  ForbiddenAttributelessMapping,
  /// The CPU mapping failed because currently, mapping deinterleaved buffers is not supported via
  /// a single slice.
  ForbiddenDeinterleavedMapping,
}

impl fmt::Display for TessMapError {
  fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    match *self {
      TessMapError::VertexBufferMapFailed(ref e) => {
        write!(f, "cannot map tessellation vertex buffer: {}", e)
      }
      TessMapError::IndexBufferMapFailed(ref e) => {
        write!(f, "cannot map tessellation index buffer: {}", e)
      }
      TessMapError::VertexTypeMismatch(ref a, ref b) => write!(
        f,
        "cannot map tessellation: vertex type mismatch between {:?} and {:?}",
        a, b
      ),
      TessMapError::IndexTypeMismatch(ref a, ref b) => write!(
        f,
        "cannot map tessellation: index type mismatch between {:?} and {:?}",
        a, b
      ),
      TessMapError::ForbiddenAttributelessMapping => {
        f.write_str("cannot map an attributeless buffer")
      }
      TessMapError::ForbiddenDeinterleavedMapping => {
        f.write_str("cannot map a deinterleaved buffer as interleaved")
      }
    }
  }
}

/// Possible errors that might occur when dealing with [`Tess`].
#[derive(Debug)]
pub enum TessError {
  /// Error related to attributeless tessellation and/or render.
  AttributelessError(String),
  /// Length incoherency in vertex, index or instance buffers.
  LengthIncoherency(usize),
  /// Overflow when accessing underlying buffers.
  Overflow(usize, usize),
}

/// Possible tessellation index types.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TessIndexType {
  /// 8-bit unsigned integer.
  U8,
  /// 16-bit unsigned integer.
  U16,
  /// 32-bit unsigned integer.
  U32,
}

/// Class of tessellation indexes.
///
/// Values which types implement this trait are allowed to be used to index tessellation in *indexed
/// draw commands*.
///
/// You shouldn’t have to worry to much about that trait. Have a look at the current implementors
/// for an exhaustive list of types you can use.
///
/// > Implementing this trait is `unsafe`.
pub unsafe trait TessIndex {
  /// Type of the underlying index.
  ///
  /// You are limited in which types you can use as indexes. Feel free to have a look at the
  /// documentation of the [`TessIndexType`] trait for further information.
  const INDEX_TYPE: TessIndexType;
}

unsafe impl TessIndex for u8 {
  const INDEX_TYPE: TessIndexType = TessIndexType::U8;
}

unsafe impl TessIndex for u16 {
  const INDEX_TYPE: TessIndexType = TessIndexType::U16;
}

unsafe impl TessIndex for u32 {
  const INDEX_TYPE: TessIndexType = TessIndexType::U32;
}

pub unsafe trait TessBuilder {
  type TessBuilderRepr;

  unsafe fn new_tess_builder(&mut self) -> Result<Self::TessBuilderRepr, TessError>;

  unsafe fn add_vertices<V, W>(
    tess_builder: &mut Self::TessBuilderRepr,
    vertices: W,
  ) -> Result<(), TessError>
  where
    W: AsRef<[V]>,
    V: Vertex;

  unsafe fn add_instances<V, W>(
    tess_builder: &mut Self::TessBuilderRepr,
    instances: W,
  ) -> Result<(), TessError>
  where
    W: AsRef<[V]>,
    V: Vertex;

  unsafe fn set_indices<T, I>(
    tess_builder: &mut Self::TessBuilderRepr,
    indices: T,
  ) -> Result<(), TessError>
  where
    T: AsRef<[I]>,
    I: TessIndex;

  unsafe fn set_mode(tess_builder: &mut Self::TessBuilderRepr, mode: Mode)
    -> Result<(), TessError>;

  unsafe fn set_vertex_nb(
    tess_builder: &mut Self::TessBuilderRepr,
    nb: usize,
  ) -> Result<(), TessError>;

  unsafe fn set_instance_nb(
    tess_builder: &mut Self::TessBuilderRepr,
    nb: usize,
  ) -> Result<(), TessError>;

  unsafe fn set_primitive_restart_index(
    tess_builder: &mut Self::TessBuilderRepr,
    index: Option<u32>,
  ) -> Result<(), TessError>;
}

pub unsafe trait Tess: TessBuilder {
  type TessRepr;

  unsafe fn build(tess_builder: Self::TessBuilderRepr) -> Result<Self::TessRepr, TessError>;

  unsafe fn destroy_tess(tess: &mut Self::TessRepr) -> Result<(), TessError>;

  unsafe fn render(
    tess: &Self::TessRepr,
    start_index: usize,
    vert_nb: usize,
    inst_nb: usize,
  ) -> Result<(), TessError>;
}

pub unsafe trait TessSlice<T>: Tess {
  type SliceRepr;

  unsafe fn destroy_tess_slice(slice: &mut Self::SliceRepr) -> Result<(), TessMapError>;

  unsafe fn slice_vertices(tess: &Self::TessRepr) -> Result<Self::SliceRepr, TessMapError>
  where
    T: Vertex;

  unsafe fn slice_vertices_mut(tess: &mut Self::TessRepr) -> Result<Self::SliceRepr, TessMapError>
  where
    T: Vertex;

  unsafe fn slice_indices(tess: &Self::TessRepr) -> Result<Self::SliceRepr, TessMapError>
  where
    T: TessIndex;

  unsafe fn slice_indices_mut(tess: &mut Self::TessRepr) -> Result<Self::SliceRepr, TessMapError>
  where
    T: TessIndex;

  unsafe fn slice_instances(tess: &Self::TessRepr) -> Result<Self::SliceRepr, TessMapError>
  where
    T: Vertex;

  unsafe fn slice_instances_mut(tess: &mut Self::TessRepr) -> Result<Self::SliceRepr, TessMapError>
  where
    T: Vertex;

  unsafe fn obtain_slice(slice: &Self::SliceRepr) -> Result<&[T], TessMapError>;

  unsafe fn obtain_slice_mut(slice: &mut Self::SliceRepr) -> Result<&mut [T], TessMapError>;
}
