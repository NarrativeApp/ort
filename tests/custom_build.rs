//! Linkage and inference checks for a locally built ONNX Runtime.
//!
//! ```shell
//! ORT_LIB_LOCATION=<onnxruntime>/build/<Platform>/Release cargo test --test custom_build
//! ```

use std::path::{Path, PathBuf};

use ndarray::Array4;
use ort::{inputs, session::Session, value::TensorRef};

fn model_path() -> PathBuf {
	Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("data").join("upsample.onnx")
}

fn infer(session: &mut Session) -> ort::Result<Vec<usize>> {
	let input: Array4<f32> = Array4::zeros((1, 224, 224, 3));
	let outputs = session.run(inputs![TensorRef::from_array_view(&input)?])?;
	Ok(outputs[0].try_extract_array::<f32>()?.shape().to_vec())
}

#[test]
fn links_and_infers() -> ort::Result<()> {
	println!("{}", ort::info());
	let mut session = Session::builder()?.commit_from_file(model_path())?;
	assert_eq!(infer(&mut session)?, [1, 448, 448, 3]);
	Ok(())
}

/// EP registration failure is non-fatal by default, so a build with CoreML missing is
/// indistinguishable from a working one -- it just silently runs on CPU. `error_on_failure` is what
/// makes this test meaningful.
#[cfg(all(target_vendor = "apple", feature = "coreml"))]
#[test]
fn coreml_registers() -> ort::Result<()> {
	let ep = ort::ep::CoreML::default().build().error_on_failure();
	let mut session = Session::builder()?.with_execution_providers([ep])?.commit_from_file(model_path())?;
	assert_eq!(infer(&mut session)?, [1, 448, 448, 3]);
	Ok(())
}
