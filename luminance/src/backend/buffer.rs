//! Buffer backend interface.
//!
//! This interface defines the low-level API buffers must implement to be usable.

use std::fmt;

/// Buffer backend.
///
/// You want to implement that trait on your backend type to support buffers.
pub unsafe trait Buffer<T> {
  /// The inner representation of the buffer for this backend.
  type Repr;

  /// Create a new buffer with a given number of uninitialized elements.
  unsafe fn new_buffer(&mut self, len: usize) -> Result<Self::Repr, BufferError>;

  unsafe fn destroy_buffer(buffer: &mut Self::Repr) -> Result<(), BufferError>;

  /// Create a new buffer from a slice.
  unsafe fn from_slice<S>(&mut self, slice: S) -> Result<Self::Repr, BufferError>
  where
    S: AsRef<[T]>;

  unsafe fn repeat(&mut self, len: usize, value: T) -> Result<Self::Repr, BufferError>
  where
    T: Copy;

  unsafe fn at(buffer: &Self::Repr, i: usize) -> Option<T>
  where
    T: Copy;

  unsafe fn whole(buffer: &Self::Repr) -> Vec<T>
  where
    T: Copy;

  unsafe fn set(buffer: &mut Self::Repr, i: usize, x: T) -> Result<(), BufferError>
  where
    T: Copy;

  unsafe fn write_whole(buffer: &mut Self::Repr, values: &[T]) -> Result<(), BufferError>;

  unsafe fn clear(buffer: &mut Self::Repr, x: T) -> Result<(), BufferError>
  where
    T: Copy;
}

/// Buffer errors.
#[derive(Debug, Eq, PartialEq)]
pub enum BufferError {
  /// Overflow when setting a value with a specific index.
  ///
  /// Contains the index and the size of the buffer.
  Overflow { index: usize, buffer_len: usize },

  /// Too few values were passed to fill a buffer.
  ///
  /// Contains the number of passed value and the size of the buffer.
  TooFewValues {
    provided_len: usize,
    buffer_len: usize,
  },

  /// Too many values were passed to fill a buffer.
  ///
  /// Contains the number of passed value and the size of the buffer.
  TooManyValues {
    provided_len: usize,
    buffer_len: usize,
  },

  /// Mapping the buffer failed.
  MapFailed,
}

impl fmt::Display for BufferError {
  fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    match *self {
      BufferError::Overflow { index, buffer_len } => write!(
        f,
        "buffer overflow (index = {}, size = {})",
        index, buffer_len
      ),

      BufferError::TooFewValues {
        provided_len,
        buffer_len,
      } => write!(
        f,
        "too few values passed to the buffer (nb = {}, size = {})",
        provided_len, buffer_len
      ),

      BufferError::TooManyValues {
        provided_len,
        buffer_len,
      } => write!(
        f,
        "too many values passed to the buffer (nb = {}, size = {})",
        provided_len, buffer_len
      ),

      BufferError::MapFailed => write!(f, "buffer mapping failed"),
    }
  }
}
