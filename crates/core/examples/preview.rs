//! Writes before/after preview images of one frame as PPM files.
//!
//!   cargo run --release --example preview -- <clip> <frame> <quality 1..4> <original|6k|4k> <out prefix> [x y w h]

use bluecompress::{Backend, Resolution};

fn write_ppm(path: &str, w: u32, h: u32, rgba: &[u8]) -> std::io::Result<()> {
    let mut data = format!("P6\n{w} {h}\n255\n").into_bytes();
    for px in rgba.as_chunks::<4>().0 {
        data.extend_from_slice(&px[..3]);
    }
    std::fs::write(path, data)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 6 {
        eprintln!("usage: preview <clip> <frame> <quality> <original|6k|4k> <out prefix>");
        std::process::exit(2);
    }
    let res = match a[4].as_str() {
        "6k" => Resolution::K6,
        "4k" => Resolution::K4,
        _ => Resolution::Original,
    };
    let crop: Vec<f64> = a[6..].iter().map(|v| v.parse().unwrap()).collect();
    let crop = if crop.len() == 4 { [crop[0], crop[1], crop[2], crop[3]] } else { [0.0, 0.0, 1.0, 1.0] };
    let t = std::time::Instant::now();
    let p = bluecompress::preview_region(
        a[1].as_ref(),
        a[2].parse().unwrap(),
        a[3].parse().unwrap(),
        res,
        Backend::Auto,
        crop,
        1600,
    )
    .unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1)
    });
    println!(
        "{}x{} in {:.2} s, frame {} -> {} bytes ({:.1}%), {} frames",
        p.width,
        p.height,
        t.elapsed().as_secs_f64(),
        p.source_frame_bytes,
        p.output_frame_bytes,
        100.0 * p.output_frame_bytes as f64 / p.source_frame_bytes as f64,
        p.frames
    );
    write_ppm(&format!("{}-before.ppm", a[5]), p.width, p.height, &p.before_rgba).unwrap();
    write_ppm(&format!("{}-after.ppm", a[5]), p.width, p.height, &p.after_rgba).unwrap();
}
