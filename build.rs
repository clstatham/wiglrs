use std::{fs, path::Path};

fn main() {
    let path = "target/debug/build/torch-sys-9a41e038e61c2ff3/out/libtorch/libtorch/lib/";
    for entry in fs::read_dir(path).unwrap() {
        let entry = entry.unwrap();
        fs::copy(
            entry.path(),
            Path::new("target/debug/").join(entry.file_name()),
        )
        .unwrap();
    }
    println!("cargo:rerun-if-changed=build.rs");
}
