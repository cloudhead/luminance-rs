//! Shader backend.

use std::fmt;

/// A shader stage type.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StageType {
  /// Tessellation control shader.
  TessellationControlShader,
  /// Tessellation evaluation shader.
  TessellationEvaluationShader,
  /// Vertex shader.
  VertexShader,
  /// Geometry shader.
  GeometryShader,
  /// Fragment shader.
  FragmentShader,
}

impl fmt::Display for StageType {
  fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    match *self {
      StageType::TessellationControlShader => f.write_str("tessellation control shader"),
      StageType::TessellationEvaluationShader => f.write_str("tessellation evaluation shader"),
      StageType::VertexShader => f.write_str("vertex shader"),
      StageType::GeometryShader => f.write_str("geometry shader"),
      StageType::FragmentShader => f.write_str("fragment shader"),
    }
  }
}

/// Errors that shader stages can emit.
#[derive(Clone, Debug)]
pub enum StageError {
  /// Occurs when a shader fails to compile.
  CompilationFailed(StageType, String),
  /// Occurs when you try to create a shader which type is not supported on the current hardware.
  UnsupportedType(StageType),
}

impl fmt::Display for StageError {
  fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    match *self {
      StageError::CompilationFailed(ref ty, ref r) => write!(f, "{} compilation error: {}", ty, r),

      StageError::UnsupportedType(ty) => write!(f, "unsupported {}", ty),
    }
  }
}

pub unsafe trait Shader {
  type StageRepr;

  unsafe fn new_stage(&mut self, ty: StageType, src: &str) -> Result<Self::StageRepr, StageError>;

  unsafe fn destroy_stage(stage: &mut Self::StageRepr);
}
