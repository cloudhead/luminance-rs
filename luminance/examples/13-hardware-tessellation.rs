mod common;

use common::{Semantics, Vertex, VertexColor, VertexPosition};
use luminance::shader::program::{Program, Uniform};
use luminance_derive::UniformInterface;
use luminance_glfw::{GlfwSurface, Surface, WindowDim, WindowOpt};

#[derive(Debug, UniformInterface)]
pub struct Iface {
  time: Uniform<f32>,
}

const VS_SRC: &str = include_str!("./simple-vs.glsl");
const TCS_SRC: &str = include_str!("./simple-tcs.glsl");
const TES_SRC: &str = include_str!("./simple-tes.glsl");

fn main() {
  let mut surface = GlfwSurface::new(
    WindowDim::Windowed(960, 540),
    "Hello, world!",
    WindowOpt::default(),
  )
  .expect("GLFW surface creation");

  let program = Program::from_strings(Some((TCS_SRC, TES_SRC)), VS_SRC, None, FS_SRC)
    .unwrap()
    .ignore_warnings();
}
