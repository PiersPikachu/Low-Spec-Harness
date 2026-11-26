// Compile kernels.cu to PTX with nvcc when it is installed (CUDA toolkit; on Windows run from a shell where cl.exe is on
// PATH, e.g. after vcvars64.bat). Without nvcc the PTX file stays empty and the kernels are compiled by NVRTC at run time.
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/kernels.cu");
    println!("cargo:rerun-if-env-changed=CUBE_NVCC");
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("kernels.ptx");
    let nvcc = std::env::var("CUBE_NVCC").unwrap_or_else(|_| "nvcc".into());
    let ok = Command::new(&nvcc)
        .args(["--ptx", "-O3", "-arch=compute_70", "-o"])
        .arg(&out)
        .arg("src/kernels.cu")
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !ok {
        std::fs::write(&out, "").unwrap();
        println!("cargo:warning=nvcc not available: the GPU kernels will be compiled by NVRTC at run time");
    }
}
