//! Buffer backend interface.
//!
//! This interface defines the low-level API buffers must implement to be usable.

/// Buffer backend.
///
/// You want to implement that trait on your backend type to support buffers.
pub unsafe trait Buffer<T> {
  /// The inner representation of the buffer for this backend.
  type Repr;

  /// Create a new buffer with a given number of uninitialized elements.
  unsafe fn new_buffer(&mut self, len: usize) -> Result<Self::Repr, BufferError>;

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
  Overflow(usize, usize),
  /// Too few values were passed to fill a buffer.
  ///
  /// Contains the number of passed value and the size of the buffer.
  TooFewValues(usize, usize),
  /// Too many values were passed to fill a buffer.
  ///
  /// Contains the number of passed value and the size of the buffer.
  TooManyValues(usize, usize),
  /// Mapping the buffer failed.
  MapFailed,
}
