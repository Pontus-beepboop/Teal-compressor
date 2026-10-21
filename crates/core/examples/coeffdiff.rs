//! Decodes frame N of two clips and compares coefficients (dev tool).
use bluecompress::bmff::{InputFile, Movie};
use bluecompress::frame::{decode_frame, Coeffs};
use bluecompress::nraw::NrawFrame;
use bluecompress::tico::{Layout, TicoHeader};

fn coeffs(path: &str, n: usize) -> (Vec<i32>, Layout) {
    let f = InputFile::open(path.as_ref()).unwrap();
    let m = Movie::parse(&f).unwrap();
    let s = m.video().samples[n];
    let data = f.read(s.offset, s.size as usize).unwrap();
    let nf = NrawFrame::parse(&data).unwrap();
    let cs = &data[nf.codestream_offset..];
    let (h, _) = TicoHeader::parse(cs).unwrap();
    let l = Layout::make(&h, h.width as usize, h.height as usize).unwrap();
    let mut c = vec![0; Coeffs::len(&l)];
    decode_frame(cs, &h, &l, &mut c).unwrap();
    (c, l)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let n = a.get(3).map_or(0, |v| v.parse().unwrap());
    let (x, l) = coeffs(&a[1], n);
    let (y, _) = coeffs(&a[2], n);
    let plane = l.precincts * l.cw;
    for p in 0..8 {
        let r = p * plane..(p + 1) * plane;
        let d: Vec<usize> = r.clone().filter(|&i| x[i] != y[i]).collect();
        let maxd = d.iter().map(|&i| (x[i] - y[i]).abs()).max().unwrap_or(0);
        let first = d.first().map(|&i| ((i - r.start) / l.cw, (i - r.start) % l.cw));
        println!("plane {p}: {} differ, max |d| {maxd}, first (row, col) {first:?}", d.len());
    }
}
