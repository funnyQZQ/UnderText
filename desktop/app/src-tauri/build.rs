fn main() {
    // macOS: sherpa-onnx dylibs sit next to the binary for `cargo run` (target/<profile>/)
    // and in Contents/Frameworks inside the .app bundle (see bundle.macOS.frameworks).
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-arg-bins=-Wl,-rpath,@executable_path");
        println!("cargo:rustc-link-arg-bins=-Wl,-rpath,@executable_path/../Frameworks");
    }
    tauri_build::build()
}
