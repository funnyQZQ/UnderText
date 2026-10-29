// macOS: find the sherpa-onnx dylibs copied next to the binary (target/<profile>/).
fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-arg-bins=-Wl,-rpath,@executable_path");
    }
}
