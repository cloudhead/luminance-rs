//! Depth test related features.

use crate::metagl::*;

/// Whether or not depth test should be enabled.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum DepthTest {
  /// The depth test is enabled.
  On,
  /// The depth test is disabled.
  Off,
}

/// Depth comparison to perform while depth test. `a` is the incoming fragment’s depth and b is the
/// fragment’s depth that is already stored.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum DepthComparison {
  /// Depth test never succeeds.
  Never,
  /// Depth test always succeeds.
  Always,
  /// Depth test succeeds if `a == b`.
  Equal,
  /// Depth test succeeds if `a != b`.
  NotEqual,
  /// Depth test succeeds if `a < b`.
  Less,
  /// Depth test succeeds if `a <= b`.
  LessOrEqual,
  /// Depth test succeeds if `a > b`.
  Greater,
  /// Depth test succeeds if `a >= b`.
  GreaterOrEqual,
}

impl DepthComparison {
  pub(crate) fn to_glenum(self) -> GLenum {
    match self {
      DepthComparison::Never => gl::NEVER,
      DepthComparison::Always => gl::ALWAYS,
      DepthComparison::Equal => gl::EQUAL,
      DepthComparison::NotEqual => gl::NOTEQUAL,
      DepthComparison::Less => gl::LESS,
      DepthComparison::LessOrEqual => gl::LEQUAL,
      DepthComparison::Greater => gl::GREATER,
      DepthComparison::GreaterOrEqual => gl::GEQUAL,
    }
  }
}
