mod common;

use common::{Semantics, Vertex, VertexColor, VertexPosition};
use luminance::shader::program::Uniform;
use luminance_derive::UniformInterface;

#[derive(Debug, UniformInterface)]
pub struct Iface {
  time: Uniform<f32>,
}

fn main() {
  let mut surface = GlfwSurface::new(
    WindowDim::Windowed(960, 540),
    "Hello, world!",
    WindowOpt::default(),
  )
  .expect("GLFW surface creation");
}
