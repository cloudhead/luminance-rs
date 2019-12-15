//! OpenGL module provider.
//!
//! This module provides OpenGL types and functions that are used to implement the rest of this
//! crate.

mod meta {
  pub(crate) use gl;
  pub(crate) use gl::types::*;
}

pub(crate) use self::meta::*;
