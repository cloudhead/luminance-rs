//! OpenGL buffer implementation.

use gl;
use gl::types::*;
use std::cell::RefCell;
use std::mem;
use std::os::raw::c_void;
use std::ptr;
use std::rc::Rc;

use crate::backend::buffer::{Buffer, BufferError};
use crate::backend::gl::state::{Bind, GLState};
use crate::backend::gl::GL;

/// OpenGL buffer.
pub struct RawBuffer {
  handle: GLuint,
  bytes: usize,
  len: usize,
  state: Rc<RefCell<GLState>>,
}

unsafe impl<T> Buffer<T> for GL {
  type Repr = RawBuffer;

  unsafe fn new_buffer(&mut self, len: usize) -> Result<Self::Repr, BufferError> {
    let mut buffer: GLuint = 0;
    let bytes = mem::size_of::<T>() * len;

    // generate a buffer and force binding the handle; this prevent side-effects from previous bound
    // resources to prevent binding the buffer
    gl::GenBuffers(1, &mut buffer);
    self
      .state
      .borrow_mut()
      .bind_array_buffer(buffer, Bind::Forced);

    gl::BufferData(
      gl::ARRAY_BUFFER,
      bytes as isize,
      ptr::null(),
      gl::STREAM_DRAW,
    );

    Ok(RawBuffer {
      handle: buffer,
      bytes,
      len,
      state: self.state.clone(),
    })
  }

  unsafe fn from_slice<S>(&mut self, slice: S) -> Result<Self::Repr, BufferError>
  where
    S: AsRef<[T]>,
  {
    let mut buffer: GLuint = 0;
    let slice = slice.as_ref();
    let len = slice.len();
    let bytes = mem::size_of::<T>() * len;

    unsafe {
      gl::GenBuffers(1, &mut buffer);
      self
        .state
        .borrow_mut()
        .bind_array_buffer(buffer, Bind::Cached);
      gl::BufferData(
        gl::ARRAY_BUFFER,
        bytes as isize,
        slice.as_ptr() as *const c_void,
        gl::STREAM_DRAW,
      );
    }

    Ok(RawBuffer {
      handle: buffer,
      bytes,
      len,
      state: self.state.clone(),
    })
  }

  // unsafe fn repeat<T>(&mut self, buffer: &mut Self::Repr, len: usize, value: T) -> Self
  // where
  //   T: Copy;
  // {
  //   let mut buf = unsafe { Self::new(ctx, len) };

  //   buf.clear(value).unwrap();
  //   buf
  // }
}
