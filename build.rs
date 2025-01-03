use std::process::Command;

fn main() {
    // Instruct Cargo to re-run this script if the shader file changes.
    println!("cargo:rerun-if-changed=sumshader.metal");

    // 1. Compile sumshader.metal into sumshader.air
    let status = Command::new("xcrun")
        .args(["metal", "-c", "sumshader.metal", "-o", "sumshader.air"])
        .status()
        .expect("Failed to run xcrun metal command.");
    if !status.success() {
        panic!("Metal shader compilation (sumshader.metal -> sumshader.air) failed!");
    }

    // 2. Link sumshader.air into sumshader.metallib
    let status = Command::new("xcrun")
        .args(["metallib", "sumshader.air", "-o", "sumshader.metallib"])
        .status()
        .expect("Failed to run xcrun metallib command.");
    if !status.success() {
        panic!("Linking (sumshader.air -> sumshader.metallib) failed!");
    }
}
